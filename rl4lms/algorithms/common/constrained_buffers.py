from typing import Dict, Generator, Optional, Union, NamedTuple

import numpy as np
import torch as th
from gymnasium import spaces

from stable_baselines3.common.buffers import BaseBuffer
from stable_baselines3.common.vec_env import VecNormalize

try:
    # Check memory used by replay buffer when possible
    import psutil
except ImportError:
    psutil = None

TensorDict = Dict[str, th.Tensor]

class ConstrainedRolloutBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    old_task_values: th.Tensor
    old_constraint_values: th.Tensor
    old_log_prob: th.Tensor
    task_advantages: th.Tensor
    constraint_advantages: th.Tensor
    task_returns: th.Tensor
    constraint_returns: th.Tensor


class ConstrainedDictRolloutBufferSamples(NamedTuple):
    observations: TensorDict
    actions: th.Tensor
    old_task_values: th.Tensor
    old_constraint_values: th.Tensor
    old_log_prob: th.Tensor
    task_advantages: th.Tensor
    constraint_advantages: th.Tensor
    task_returns: th.Tensor
    constraint_returns: th.Tensor



class ConstrainedRolloutBuffer(BaseBuffer):
    """
    Rollout buffer used in on-policy algorithms like A2C/PPO.
    It corresponds to ``buffer_size`` transitions collected
    using the current policy.
    This experience will be discarded after the policy update.
    In order to use PPO objective, we also store the current value of each state
    and the log probability of each taken action.

    The term rollout here refers to the model-free notion and should not
    be used with the concept of rollout used in model-based RL or planning.
    Hence, it is only involved in policy and value function training but not action selection.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        Equivalent to classic advantage when set to 1.
    :param gamma: Discount factor
    :param n_envs: Number of parallel environments
    """

    observations: np.ndarray
    actions: np.ndarray
    task_rewards: np.ndarray
    constraint_rewards: np.ndarray
    task_advantages: np.ndarray
    constraint_advantages: np.ndarray
    task_returns: np.ndarray
    constraint_returns: np.ndarray
    episode_starts: np.ndarray
    log_probs: np.ndarray
    task_values: np.ndarray
    constraint_values: np.ndarray

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
    ):
        super().__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.generator_ready = False
        self.reset()

    def reset(self) -> None:
        self.observations = np.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=np.float32)
        self.actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=np.float32)
        self.task_rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.constraint_rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.task_returns = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.constraint_returns = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.episode_starts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.task_values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.constraint_values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.log_probs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.task_advantages = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.constraint_advantages = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.generator_ready = False
        super().reset()

    def compute_returns_and_advantage(self, last_task_values: th.Tensor, last_constraint_values: th.Tensor, dones: np.ndarray) -> None:
        """
        Post-processing step: compute the lambda-return (TD(lambda) estimate)
        and GAE(lambda) advantage.

        Uses Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)
        to compute the advantage. To obtain Monte-Carlo advantage estimate (A(s) = R - V(S))
        where R is the sum of discounted reward with value bootstrap
        (because we don't always have full episode), set ``gae_lambda=1.0`` during initialization.

        The TD(lambda) estimator has also two special cases:
        - TD(1) is Monte-Carlo estimate (sum of discounted rewards)
        - TD(0) is one-step estimate with bootstrapping (r_t + gamma * v(s_{t+1}))

        For more information, see discussion in https://github.com/DLR-RM/stable-baselines3/pull/375.

        :param last_values: state value estimation for the last step (one for each env)
        :param dones: if the last step was a terminal step (one bool for each env).
        """
        # Convert to numpy
        last_task_values = last_task_values.clone().cpu().numpy().flatten()
        last_constraint_values = last_constraint_values.clone().cpu().numpy().flatten()

        last_gae_lam = 0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones
                next_task_values = last_task_values
                next_constraint_values = last_constraint_values
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_task_values = self.task_values[step + 1]
                next_constraint_values = self.constraint_values[step + 1]
            task_delta = self.task_rewards[step] + self.gamma * next_task_values * next_non_terminal - self.task_values[step]
            constraint_delta = self.constraint_rewards[step] + self.gamma * next_constraint_values * next_non_terminal - self.constraint_values[step]
            last_task_gae_lam = task_delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            last_constraint_gae_lam = constraint_delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            self.task_advantages[step] = last_task_gae_lam
            self.constraint_advantages[step] = last_constraint_gae_lam
        # TD(lambda) estimator, see Github PR #375 or "Telescoping in TD(lambda)"
        # in David Silver Lecture 4: https://www.youtube.com/watch?v=PnHCvfgC_ZA
        self.task_returns = self.task_advantages + self.task_values
        self.constraint_returns = self.constraint_advantages + self.constraint_values

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        task_reward: np.ndarray,
        constraint_reward: np.ndarray,
        episode_start: np.ndarray,
        task_value: th.Tensor,
        constraint_value: th.Tensor,
        log_prob: th.Tensor,
    ) -> None:
        """
        :param obs: Observation
        :param action: Action
        :param reward:
        :param episode_start: Start of episode signal.
        :param value: estimated value of the current state
            following the current policy.
        :param log_prob: log probability of the action
            following the current policy.
        """
        if len(log_prob.shape) == 0:
            # Reshape 0-d tensor to avoid error
            log_prob = log_prob.reshape(-1, 1)

        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs, *self.obs_shape))

        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action = action.reshape((self.n_envs, self.action_dim))

        self.observations[self.pos] = np.array(obs).copy()
        self.actions[self.pos] = np.array(action).copy()
        self.task_rewards[self.pos] = np.array(task_reward).copy()
        self.constraint_rewards[self.pos] = np.array(constraint_reward).copy()
        self.episode_starts[self.pos] = np.array(episode_start).copy()
        self.task_values[self.pos] = task_value.clone().cpu().numpy().flatten()
        self.constraint_values[self.pos] = constraint_value.clone().cpu().numpy().flatten()
        self.log_probs[self.pos] = log_prob.clone().cpu().numpy()
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def get(self, batch_size: Optional[int] = None) -> Generator[ConstrainedRolloutBufferSamples, None, None]:
        assert self.full, ""
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        # Prepare the data
        if not self.generator_ready:
            _tensor_names = [
                "observations",
                "actions",
                "task_values",
                "constraint_values",
                "log_probs",
                "task_advantages",
                "constraint_advantages",
                "task_returns",
                "constraint_returns",
            ]

            for tensor in _tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(
        self,
        batch_inds: np.ndarray,
        env: Optional[VecNormalize] = None,
    ) -> ConstrainedRolloutBufferSamples:  # type: ignore[signature-mismatch] #FIXME
        data = (
            self.observations[batch_inds],
            self.actions[batch_inds],
            self.task_values[batch_inds].flatten(),
            self.constraint_values[batch_inds].flatten(),
            self.log_probs[batch_inds].flatten(),
            self.task_advantages[batch_inds].flatten(),
            self.constraint_advantages[batch_inds].flatten(),
            self.task_returns[batch_inds].flatten(),
            self.constraint_returns[batch_inds].flatten(),
        )
        return ConstrainedRolloutBufferSamples(*tuple(map(self.to_torch, data)))

class ConstrainedDictRolloutBuffer(ConstrainedRolloutBuffer):
    """
    Dict Rollout buffer used in on-policy algorithms like A2C/PPO.
    Extends the RolloutBuffer to use dictionary observations

    It corresponds to ``buffer_size`` transitions collected
    using the current policy.
    This experience will be discarded after the policy update.
    In order to use PPO objective, we also store the current value of each state
    and the log probability of each taken action.

    The term rollout here refers to the model-free notion and should not
    be used with the concept of rollout used in model-based RL or planning.
    Hence, it is only involved in policy and value function training but not action selection.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        Equivalent to Monte-Carlo advantage estimate when set to 1.
    :param gamma: Discount factor
    :param n_envs: Number of parallel environments
    """

    observations: Dict[str, np.ndarray]

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
    ):
        super(ConstrainedRolloutBuffer, self).__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)

        assert isinstance(self.obs_shape, dict), "DictRolloutBuffer must be used with Dict obs space only"

        self.gae_lambda = gae_lambda
        self.gamma = gamma

        self.generator_ready = False
        self.reset()

    def reset(self) -> None:
        assert isinstance(self.obs_shape, dict), "DictRolloutBuffer must be used with Dict obs space only"
        self.observations = {}
        for key, obs_input_shape in self.obs_shape.items():
            self.observations[key] = np.zeros((self.buffer_size, self.n_envs, *obs_input_shape), dtype=np.float32)
        self.actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=np.float32)
        self.task_rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.constraint_rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.task_returns = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.constraint_returns = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.episode_starts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.task_values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.constraint_values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.log_probs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.task_advantages = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.constraint_advantages = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.generator_ready = False
        super(ConstrainedRolloutBuffer, self).reset()

    def add(
        self,
        obs: Dict[str, np.ndarray],
        action: np.ndarray,
        task_reward: np.ndarray,
        constraint_reward: np.ndarray,
        episode_start: np.ndarray,
        task_value: th.Tensor,
        constraint_value: th.Tensor,
        log_prob: th.Tensor,
    ) -> None:  # pytype: disable=signature-mismatch
        """
        :param obs: Observation
        :param action: Action
        :param reward:
        :param episode_start: Start of episode signal.
        :param value: estimated value of the current state
            following the current policy.
        :param log_prob: log probability of the action
            following the current policy.
        """
        if len(log_prob.shape) == 0:
            # Reshape 0-d tensor to avoid error
            log_prob = log_prob.reshape(-1, 1)

        for key in self.observations.keys():
            obs_ = np.array(obs[key]).copy()
            # Reshape needed when using multiple envs with discrete observations
            # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
            if isinstance(self.observation_space.spaces[key], spaces.Discrete):
                obs_ = obs_.reshape((self.n_envs,) + self.obs_shape[key])
            self.observations[key][self.pos] = obs_

        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action = action.reshape((self.n_envs, self.action_dim))

        self.actions[self.pos] = np.array(action).copy()
        self.task_rewards[self.pos] = np.array(task_reward).copy()
        self.constraint_rewards[self.pos] = np.array(constraint_reward).copy()
        self.episode_starts[self.pos] = np.array(episode_start).copy()
        self.task_values[self.pos] = task_value.clone().cpu().numpy().flatten()
        self.constraint_values[self.pos] = constraint_value.clone().cpu().numpy().flatten()
        self.log_probs[self.pos] = log_prob.clone().cpu().numpy()
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def get(
        self,
        batch_size: Optional[int] = None,
    ) -> Generator[ConstrainedDictRolloutBufferSamples, None, None]:  # type: ignore[signature-mismatch] #FIXME
        assert self.full, ""
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        # Prepare the data
        if not self.generator_ready:
            for key, obs in self.observations.items():
                self.observations[key] = self.swap_and_flatten(obs)

            _tensor_names = [
                "actions",
                "task_values",
                "constraint_values",
                "log_probs",
                "task_advantages",
                "constraint_advantages",
                "task_returns",
                "constraint_returns",]

            for tensor in _tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(
        self,
        batch_inds: np.ndarray,
        env: Optional[VecNormalize] = None,
    ) -> ConstrainedDictRolloutBufferSamples:  # type: ignore[signature-mismatch] #FIXME
        return ConstrainedDictRolloutBufferSamples(
            observations={key: self.to_torch(obs[batch_inds]) for (key, obs) in self.observations.items()},
            actions=self.to_torch(self.actions[batch_inds]),
            old_task_values=self.to_torch(self.task_values[batch_inds].flatten()),
            old_constraint_values=self.to_torch(self.constraint_values[batch_inds].flatten()),
            old_log_prob=self.to_torch(self.log_probs[batch_inds].flatten()),
            task_advantages=self.to_torch(self.task_advantages[batch_inds].flatten()),
            constraint_advantages=self.to_torch(self.constraint_advantages[batch_inds].flatten()),
            task_returns=self.to_torch(self.task_returns[batch_inds].flatten()),
            constraint_returns=self.to_torch(self.constraint_returns[batch_inds].flatten()),
        )