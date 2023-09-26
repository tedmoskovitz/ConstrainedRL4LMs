from functools import partial
from typing import Any, Dict, List, Optional
import numpy as np
import torch
import random
import pdb

from rl4lms.data_pools.text_generation_pool import Sample
from rl4lms.envs.text_generation.env import TextGenEnv
from rl4lms.envs.text_generation.constrained_env import ConstrainedTextGenEnv
from rl4lms.envs.text_generation.evaluation_utils import evaluate_on_samples
from rl4lms.envs.text_generation.utils_supervised import evaluate_on_samples as evaluate_supervised
from rl4lms.envs.text_generation.logging_utils import Tracker
from rl4lms.envs.text_generation.registry import (DataPoolRegistry,
                                                   MetricRegistry,
                                                   RewardFunctionRegistry,
                                                   PolicyRegistry,
                                                   AlgorithmRegistry,
                                                   WrapperRegistry)
from rl4lms.envs.text_generation.reward import RewardFunction
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from transformers import (AutoTokenizer,
                          AutoModelForCausalLM,
                          AutoModelForSeq2SeqLM,
                          Trainer,
                          TrainingArguments,
                          DataCollatorForLanguageModeling,
                          DataCollatorForSeq2Seq)
from rl4lms.envs.text_generation.utils_supervised import (get_datasets_for_causal,
                                                           get_datasets_for_seq2seq,
                                                           tokenize_causal,
                                                           tokenize_seq2seq,
                                                           EvalCallack)
from rl4lms.envs.text_generation.warm_start import TrainerWarmStartMixin


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def build_tokenizer(tokenizer_config: Dict[str, Any]):
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_config["model_name"])
    if tokenizer.pad_token is None and tokenizer_config.get("pad_token_as_eos_token", True):
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = tokenizer_config.get(
        "padding_side", "left")
    tokenizer.truncation_side = tokenizer_config.get(
        "truncation_side", "left")
    return tokenizer


def build_reward_fn(reward_config: Dict[str, Any]):
    reward_fn = RewardFunctionRegistry.get(reward_config["id"],
                                           reward_config.get("args", {}))
    return reward_fn


def build_metrics(metric_configs: List[Dict[str, Any]]):
    metrics = [MetricRegistry.get(metric_config["id"], metric_config.get("args", {}))
               for metric_config in metric_configs]
    return metrics


def build_datapool(datapool_config: Dict[str, Any]):

    def _get_datapool_by_split(split: str):
        kwargs = datapool_config.get("args", {})
        kwargs["split"] = split
        dp_split = DataPoolRegistry.get(datapool_config["id"], kwargs)
        return dp_split

    train_datapool = _get_datapool_by_split("train")
    val_datapool = _get_datapool_by_split("val")
    test_datapool = _get_datapool_by_split("test")

    samples_by_split = {
        "train": [(sample, weight)
                  for sample, weight in train_datapool],
        "val": [sample for sample, _ in val_datapool],
        "test": [sample for sample, _ in test_datapool]
    }
    return samples_by_split


def build_env(env_config: Dict[str, Any],
              reward_fn: RewardFunction,
              tokenizer: AutoTokenizer,
              train_samples: List[Sample],
              multiprocess: bool = True,):
    # vectoried env
    env_kwargs = {
        "reward_function": reward_fn,
        "tokenizer": tokenizer,
        "samples": train_samples,
    }
    env_config_args = env_config.get("args", {})
    env_kwargs = {**env_kwargs, **env_config_args}
    constrained = env_config.get("constrained", False)
    env_cls = ConstrainedTextGenEnv if constrained else TextGenEnv
    env = make_vec_env(env_cls,
                       n_envs=env_config.get("n_envs", 1),
                       vec_env_cls=SubprocVecEnv if multiprocess else None,
                       env_kwargs=env_kwargs)
    return env


def build_alg(alg_config: Dict[str, Any],
              env: TextGenEnv,
              tracker: Tracker,
              policy_state: Dict[str, Any],
              alg_state: Dict[str, Any]):
    # TBD - move these to a registry once the experimentation is done
    # Also switch to Sb3 algos when possible with minimal code adaptations
    policy_config = alg_config["policy"]
    policy_cls = PolicyRegistry.get(policy_config["id"])
    alg_cls = AlgorithmRegistry.get(alg_config["id"])

    policy_args = policy_config["args"]
    policy_args["state_dict"] = policy_state
    alg_kwargs = {
        "policy": policy_cls,
        "env": env,
        "policy_kwargs": policy_args,
    }
    alg_kwargs = {**alg_kwargs, **alg_config.get("args")}
    wrapper = WrapperRegistry.get(alg_config["id"])
    alg = wrapper(alg_cls, alg_kwargs,
                  alg_config["kl_div"]["coeff"], tracker,
                  alg_config["kl_div"].get("target_kl", None),
                  alg_config["kl_div"].get("norm_reward", False))
    alg.load_from_dict(alg_state)
    return alg


class OnPolicyTrainer(TrainerWarmStartMixin):
    """
    A generic trainer for training LMs with onpolicy algorithms from SB3
    """

    def __init__(self,
                 tokenizer_config: Dict[str, Any],
                 datapool_config: Dict[str, Any],
                 reward_config: Dict[str, Any],
                 env_config: Dict[str, Any],
                 on_policy_alg_config: Dict[str, Any],
                 train_eval_config: Dict[str, Any],
                 tracker: Tracker = None,
                 experiment_name: str = '',
                 disable_multiprocess: bool = False,
                 seed: int = 0
                 ):
        self._tokenizer_config = tokenizer_config
        self._datapool_config = datapool_config
        self._reward_config = reward_config
        self._env_config = env_config
        self._on_policy_alg_config = on_policy_alg_config
        self._train_eval_config = train_eval_config
        self._tracker = tracker
        self._experiment_name = experiment_name
        self._seed = seed
        self._disable_multiprocess = disable_multiprocess
        self._setup()

    def _setup(self):
        # load trainer state from available previous checkpoint if available
        self.load_trainer_state(self._tracker)
        # set random seed
        set_seed_everywhere(self._seed)

        # build components
        self._tokenizer = build_tokenizer(self._tokenizer_config)
        self._reward_fn = build_reward_fn(self._reward_config)
        self._metrics = build_metrics(
            self._train_eval_config.get("metrics", []))
        self._samples_by_split = build_datapool(
            self._datapool_config)
        self._env = build_env(self._env_config, self._reward_fn,
                              self._tokenizer, self._samples_by_split["train"],
                              multiprocess=not self._disable_multiprocess,)
        self._alg = build_alg(self._on_policy_alg_config,
                              self._env, self._tracker,
                              self._policy_state_dict,
                              self._alg_state_dict)

        # extract train params
        self._max_episode_length = self._env_config["args"]["max_episode_length"]
        self._max_prompt_length = self._env_config["args"]["max_prompt_length"]
        self._eval_batch_size = self._train_eval_config["eval_batch_size"]
        self._n_iters = int(self._train_eval_config["n_iters"])
        self._n_steps_per_iter = self._env.num_envs * self._alg.n_steps

        # gen kwargs for evaluation (if it is different from rollout gen kwargs)
        self._eval_gen_kwargs = self._train_eval_config.get(
            "generation_kwargs", None)

    def _evaluate_on_datapools(self, epoch: int,
                               splits: List[str] = ["val", "test"]):
        for split in splits:
            evaluate_on_samples(policy=self._alg.policy,
                                tokenizer=self._tokenizer,
                                samples=self._samples_by_split[split],
                                batch_size=self._eval_batch_size,
                                max_prompt_length=self._max_prompt_length,
                                metrics=self._metrics,
                                epoch=epoch,
                                split_name=split,
                                tracker=self._tracker,
                                gen_kwargs=self._eval_gen_kwargs)

    def train_and_eval(self):
        # evaluate on val and test set before fine-tuning once
        iter_start = self._trainer_state["current_iter"]
        self._evaluate_on_datapools(epoch=iter_start)

        # train for given number of iters
        for epoch in range(iter_start, self._n_iters):
            # current state
            self._trainer_state["current_iter"] = epoch

            # inner rollout and learn loop for on-policy algorithm
            self._alg.learn(self._n_steps_per_iter)

            # save the policy checkpoint
            # if (epoch + 1) % self._train_eval_config.get("save_every", 20) == 0:
            #     self.save_trainer_state(
            #         self._tracker, self._alg.policy, self._trainer_state)

            # evaluate on val set in the given intervals
            if (epoch + 1) % self._train_eval_config["eval_every"] == 0:
                self._evaluate_on_datapools(epoch=epoch)  #, splits=["val"])  # TODO: change back

        # finally evaluate on val and test samples
        self._evaluate_on_datapools(epoch=epoch)

        # save model here - we save only the language model
        if self._tracker is not None:
            self._tracker.save_auto_model(
                self._alg.policy.get_language_model())


class NelderMeadFunc:
    def __init__(self, func, tolerance=5e-3):
        """A function class that caches previously computed values."""
        self.func = func
        self.threshold2eval_cache = {}
        self.tolerance = tolerance

    def find_nearby_key(self, x):
        for key in self.threshold2eval_cache.keys():
            existing_array = np.array(key)
            if np.all(np.abs(existing_array - x) < self.tolerance):
                return key
        return None

    def __call__(self, x):
        # First try exact match
        exact_key = tuple(x)
        if exact_key in self.threshold2eval_cache:
            return self.threshold2eval_cache[exact_key]

        # Try approximate match
        nearby_key = self.find_nearby_key(x)
        if nearby_key is not None:
            return self.threshold2eval_cache[nearby_key]

        # Compute and cache if no match found
        out = self.func(x)
        self.threshold2eval_cache[exact_key] = out
        return out


class NelderMeadTrainer(TrainerWarmStartMixin):
    """
    A Nelder-Mead trainer for training LMs with onpolicy algorithms from SB3
    """

    def __init__(self,
                 tokenizer_config: Dict[str, Any],
                 datapool_config: Dict[str, Any],
                 reward_config: Dict[str, Any],
                 env_config: Dict[str, Any],
                 on_policy_alg_config: Dict[str, Any],
                 train_eval_config: Dict[str, Any],
                 nelder_mead_config: Dict[str, Any],
                 tracker: Tracker = None,
                 experiment_name: str = '',
                 disable_multiprocess: bool = False,
                 seed: int = 0
                 ):
        self._tokenizer_config = tokenizer_config
        self._datapool_config = datapool_config
        self._reward_config = reward_config
        self._env_config = env_config
        self._on_policy_alg_config = on_policy_alg_config
        self._train_eval_config = train_eval_config
        self._nelder_mead_config = nelder_mead_config["args"]
        self._tracker = tracker
        self._experiment_name = experiment_name
        self._seed = seed
        self._disable_multiprocess = disable_multiprocess
        self._num_evaluations = 0
        self._setup()

    def _setup(self):
        # load trainer state from available previous checkpoint if available
        self.load_trainer_state(self._tracker)
        # set random seed
        set_seed_everywhere(self._seed)

        # build components
        self._tokenizer = build_tokenizer(self._tokenizer_config)
        self._reward_fn = build_reward_fn(self._reward_config)
        self._metrics = build_metrics(
            self._train_eval_config.get("metrics", []))
        self._samples_by_split = build_datapool(
            self._datapool_config)
        self._env = build_env(self._env_config, self._reward_fn,
                              self._tokenizer, self._samples_by_split["train"],
                              multiprocess=not self._disable_multiprocess,)
        self._alg = build_alg(self._on_policy_alg_config,
                              self._env, self._tracker,
                              self._policy_state_dict,
                              self._alg_state_dict)

        # extract train params
        self._max_episode_length = self._env_config["args"]["max_episode_length"]
        self._max_prompt_length = self._env_config["args"]["max_prompt_length"]
        self._eval_batch_size = self._train_eval_config["eval_batch_size"]
        self._n_iters = int(self._train_eval_config["n_iters"])
        self._n_steps_per_iter = self._env.num_envs * self._alg.n_steps

        # gen kwargs for evaluation (if it is different from rollout gen kwargs)
        self._eval_gen_kwargs = self._train_eval_config.get(
            "generation_kwargs", None)
        

    def _evaluate_on_datapools(self,
                               epoch: int,
                               increment_counter: bool = True,
                               splits: List[str] = ["val", "test"]):
        split2metrics = {}
        for split in splits:
            metrics = evaluate_on_samples(policy=self._alg.policy,
                                tokenizer=self._tokenizer,
                                samples=self._samples_by_split[split],
                                batch_size=self._eval_batch_size,
                                max_prompt_length=self._max_prompt_length,
                                metrics=self._metrics,
                                epoch=epoch,
                                split_name=split,
                                tracker=self._tracker,
                                gen_kwargs=self._eval_gen_kwargs)
            split2metrics[split] = metrics

        self._tracker.log_metrics(
            epoch, "test", {"num_evaluations": self._num_evaluations})

        pdb.set_trace()
        out = {
            "eval_score": split2metrics['test']["lexical/CRLHFEval_Score"],
            "meteor": split2metrics['test']["lexical/meteor"],
            "intent": split2metrics['test']["intent/accuracy"],
        }

        return out

    def evaluate_thresholds(self, thresholds: np.ndarray) -> float:
        # train for given number of iters
        iter_start = self._trainer_state["current_iter"]
        task_threshold, constraint_threshold = thresholds
        reached = lambda x, thresh: x >= 0.95 * thresh and x <= 1.05 * thresh
        self._alg.task_threshold = task_threshold
        self._alg.constraint_threshold = constraint_threshold
        for epoch in range(iter_start, self._n_iters):
            # current state
            self._trainer_state["current_iter"] = epoch

            # inner rollout and learn loop for on-policy algorithm
            self._alg.learn(self._n_steps_per_iter)

            # evaluate on val set in the given intervals
            if (epoch + 1) % self._train_eval_config["eval_every"] == 0:
                scores = self._evaluate_on_datapools(
                    epoch=epoch, increment_counter=False)
                
                if reached(scores['meteor'], task_threshold) and reached(scores['intent'], constraint_threshold):
                    self._num_evaluations += 1
                    self._tracker.log_metrics(
                        epoch, "NelderMead", {"num_evaluations": self._num_evaluations})
                    self._trainer_state["current_iter"] = epoch
                    return scores['eval_score']

        self._trainer_state["current_iter"] = epoch
        if scores is not None:
            return scores['eval_score']
        return 0.0

            


    def train_and_eval(self):
        # evaluate on val and test set before fine-tuning once
        iter_start = self._trainer_state["current_iter"]
        self._evaluate_on_datapools(epoch=iter_start)  #TODO re-enable

        # initialize simplex - 3 pairs of task and constraint thresholds
        _METEOR_MIN, _METEOR_MAX = 0.00037604571643093187, 0.24810026760745868
        _INTENT_MIN, _INTENT_MAX = 0.2504002561639449, 0.5283381364073007
        _METEOR_MID = (_METEOR_MIN + _METEOR_MAX) / 2
        _INTENT_MID = (_INTENT_MIN + _INTENT_MAX) / 2
        _METEOR_RANGE = _METEOR_MAX - _METEOR_MIN
        _INTENT_RANGE = _INTENT_MAX - _INTENT_MIN
        simplex = np.array([
            [_METEOR_MID + np.random.uniform(-0.1 * _METEOR_RANGE, 0.1 * _METEOR_RANGE),
             _INTENT_MID + np.random.uniform(-0.1 * _INTENT_RANGE, 0.1 * _INTENT_RANGE)] for _ in range(3)])
        

        num_vars = simplex.shape[1]  # Number of variables (2 in this case)
        iterates = []

        func = NelderMeadFunc(self.evaluate_thresholds)

        for _ in range(self._nelder_mead_config['max_iters']):
            iterates.append(simplex[-1])
            # Order the simplex based on function values
            simplex = sorted(simplex, key=func)
            # log the current simplex
            self._tracker.log_simplex(
                self._trainer_state["current_iter"], "NelderMead", simplex.tolist())

            # Compute the centroid of the n best points
            centroid = np.mean(simplex[:-1], axis=0)

            # Reflect the worst point
            reflected = centroid + self._nelder_mead_config['alpha'] * (centroid - simplex[-1])
            
            if func(simplex[0]) <= func(reflected) < func(simplex[-2]):
                simplex[-1] = reflected
                
                continue
                

            # Expand
            if func(reflected) < func(simplex[0]):
                expanded = centroid + self._nelder_mead_config['gamma'] * (reflected - centroid)                
                if func(expanded) < func(simplex[0]):
                    simplex[-1] = expanded
                    continue
                else:
                    simplex[-1] = reflected
                    continue
            
            # Contract
            contracted = centroid + self._nelder_mead_config['rho'] * (simplex[-1] - centroid)
            
            if func(contracted) < func(simplex[-1]):
                simplex[-1] = contracted
                continue
            
            # Shrink
            for i in range(1, num_vars + 1):
                shrunk = simplex[0] + self._nelder_mead_config['sigma'] * (simplex[i] - simplex[0])
                simplex[i] = shrunk
            
            # Check for convergence (using the standard deviation of function values)
            if np.std([func(v) for v in simplex]) < self._nelder_mead_config['tol']:
                break

            if self._trainer_state["current_iter"] >= self._n_iters:
                break


        # finally evaluate on val and test samples
        epoch = self._trainer_state["current_iter"]
        self._evaluate_on_datapools(epoch=epoch)
        # log final simplex
        self._tracker.log_simplex(epoch, "NelderMead", simplex.tolist())






class SupervisedTrainer:
    """
    A supervised trainer to train LMs (causal and seq2seq) on text generation tasks (wrapper on HF trainer)
    """

    def __init__(self,
                 tokenizer_config: Dict[str, Any],
                 datapool_config: Dict[str, Any],
                 train_eval_config: Dict[str, Any],
                 alg_config: Dict[str, Any],
                 tracker: Tracker = None
                 ):
        self._tokenizer_config = tokenizer_config
        self._datapool_config = datapool_config
        self._train_eval_config = train_eval_config
        self._alg_config = alg_config
        self._tracker = tracker
        self._setup()

    def _evaluate_on_datapools(self, epoch: int,
                               splits: List[str] = ["val", "test"]):
        for split in splits:
            evaluate_supervised(model=self._model,
                                tokenizer=self._tokenizer,
                                samples=self._samples_by_split[split],
                                batch_size=self._eval_batch_size,
                                max_prompt_length=self._max_prompt_length,
                                metrics_config_dict=self._metrics_config_dict,
                                epoch=epoch,
                                split_name=split,
                                tracker=self._tracker,
                                generation_kwargs=self._gen_kwargs
                                )

    def _setup(self):
        self._tokenizer = build_tokenizer(self._tokenizer_config)
        self._metrics_config_dict = self._train_eval_config.get("metrics")
        self._samples_by_split = build_datapool(
            self._datapool_config)
        self._train_dataset = get_datasets_for_causal(
            self._samples_by_split["train"]) if self._alg_config[
            "model_type"] == "causal" else get_datasets_for_seq2seq(self._samples_by_split["train"])
        preprocess_fn = tokenize_causal if self._alg_config[
            "model_type"] == "causal" else tokenize_seq2seq
        preprocess_fn = partial(preprocess_fn, tokenizer=self._tokenizer)
        self._tokenized_dataset = self._train_dataset.map(
            preprocess_fn, batched=True,
            remove_columns=self._train_dataset.column_names)
        model_cls = AutoModelForCausalLM if self._alg_config[
            "model_type"] == "causal" else AutoModelForSeq2SeqLM
        self._gen_kwargs = self._alg_config["generation_kwargs"]
        self._model = model_cls.from_pretrained(self._alg_config["model_name"])
        self._model.parallelize()
        self._eval_batch_size = self._train_eval_config["eval_batch_size"]

        # setting max prompt length
        self._max_prompt_length = self._tokenizer_config.get(
            "max_length",  self._tokenizer.model_max_length)

        if (self._alg_config["model_type"] == "causal") and ((self._max_prompt_length + self._gen_kwargs["max_new_tokens"]) > self._tokenizer.model_max_length):
            self._max_prompt_length = self._max_prompt_length - \
                self._gen_kwargs["max_new_tokens"]

        self._eval_callback = EvalCallack(self._samples_by_split["val"],
                                          self._gen_kwargs,
                                          self._eval_batch_size,
                                          self._tokenizer,
                                          self._metrics_config_dict,
                                          self._max_prompt_length,
                                          self._tracker)
        train_args = self._alg_config["training_args"]
        train_args["output_dir"] = self._tracker.checkpoint_base_path
        train_args["seed"] = np.random.randint(1e+2)  # random seed
        self._train_args = TrainingArguments(**train_args)
        data_collator = DataCollatorForLanguageModeling(self._tokenizer, mlm=False) if self._alg_config[
            "model_type"] == "causal" else DataCollatorForSeq2Seq(self._tokenizer, self._model)
        self._trainer = Trainer(model=self._model,
                                tokenizer=self._tokenizer,
                                args=self._train_args,
                                data_collator=data_collator,
                                train_dataset=self._tokenized_dataset,
                                callbacks=[self._eval_callback])

    def train_and_eval(self):
        # evaluate on val and test set before fine-tuning once
        self._evaluate_on_datapools(epoch=0)

        # train using HF trainer
        self._trainer.train()

        # finally evaluate on val and test samples
        self._evaluate_on_datapools(epoch=self._train_args.num_train_epochs)

        # save model here - we save only the language model
        if self._tracker is not None:
            self._tracker.save_auto_model(
                self._model)
