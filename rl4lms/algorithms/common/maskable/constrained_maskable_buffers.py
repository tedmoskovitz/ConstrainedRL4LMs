"""
Code adapted from https://github.com/DLR-RM/stable-baselines3
"""



from typing import Generator, NamedTuple, Optional, Union

import numpy as np
import torch
import torch as th
from gym import spaces
# from stable_baselines3.common.buffers import DictRolloutBuffer, RolloutBuffer
from stable_baselines3.common.type_aliases import TensorDict
from stable_baselines3.common.vec_env import VecNormalize
from rl4lms.algorithms.common.constrained_buffers import ConstrainedDictRolloutBuffer, ConstrainedRolloutBuffer

class MaskableConstrainedRolloutBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    old_task_values: th.Tensor
    old_constraint_values: th.Tensor
    old_kl_values: th.Tensor
    old_log_prob: th.Tensor
    task_advantages: th.Tensor
    constraint_advantages: th.Tensor
    kl_advantages: th.Tensor
    task_returns: th.Tensor
    constraint_returns: th.Tensor
    kl_returns: th.Tensor
    ep_task_reward_togo: th.Tensor
    ep_constraint_reward_togo: th.Tensor
    ep_kl_reward_togo: th.Tensor
    action_masks: th.Tensor


class MaskableConstrainedDictRolloutBufferSamples(MaskableConstrainedRolloutBufferSamples):
    observations: TensorDict
    actions: th.Tensor
    old_task_values: th.Tensor
    old_constraint_values: th.Tensor
    old_kl_values: th.Tensor
    old_log_prob: th.Tensor
    task_advantages: th.Tensor
    constraint_advantages: th.Tensor
    kl_advantages: th.Tensor
    task_returns: th.Tensor
    constraint_returns: th.Tensor
    kl_returns: th.Tensor
    ep_task_reward_togo: th.Tensor
    ep_constraint_reward_togo: th.Tensor
    ep_kl_reward_togo: th.Tensor
    action_masks: th.Tensor


class MaskableConstrainedRolloutBuffer(ConstrainedRolloutBuffer):
    """
    Rollout buffer that also stores the invalid action masks associated with each observation.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device:
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        Equivalent to classic advantage when set to 1.
    :param gamma: Discount factor
    :param n_envs: Number of parallel environments
    """

    def __init__(self, *args, **kwargs):
        self.action_masks = None
        super().__init__(*args, **kwargs)

    def reset(self) -> None:
        if isinstance(self.action_space, spaces.Discrete):
            mask_dims = self.action_space.n
        elif isinstance(self.action_space, spaces.MultiDiscrete):
            mask_dims = sum(self.action_space.nvec)
        elif isinstance(self.action_space, spaces.MultiBinary):
            mask_dims = 2 * self.action_space.n  # One mask per binary outcome
        else:
            raise ValueError(
                f"Unsupported action space {type(self.action_space)}")

        self.mask_dims = mask_dims
        self.action_masks = np.ones(
            (self.buffer_size, self.n_envs, self.mask_dims), dtype=np.float32)

        super().reset()

    def add(self, *args, action_masks: Optional[np.ndarray] = None, **kwargs) -> None:
        """
        :param action_masks: Masks applied to constrain the choice of possible actions.
        """
        if action_masks is not None:
            self.action_masks[self.pos] = action_masks.reshape(
                (self.n_envs, self.mask_dims))

        super().add(*args, **kwargs)

    def get(self, batch_size: Optional[int] = None) -> Generator[MaskableConstrainedRolloutBufferSamples, None, None]:
        assert self.full, ""
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        # Prepare the data
        if not self.generator_ready:
            for tensor in [
                "observations",
                "actions",
                "task_values",
                "constraint_values",
                "kl_values",
                "log_probs",
                "task_advantages",
                "constraint_advantages",
                "kl_advantages",
                "task_returns",
                "constraint_returns",
                "kl_returns",
                "ep_task_reward_togo",
                "ep_constraint_reward_togo",
                "ep_kl_reward_togo",
                "action_masks",
            ]:
                self.__dict__[tensor] = self.swap_and_flatten(
                    self.__dict__[tensor])
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx: start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> MaskableConstrainedRolloutBufferSamples:
        data = (
            self.observations[batch_inds],
            self.actions[batch_inds],
            self.task_values[batch_inds].flatten(),
            self.constraint_values[batch_inds], #.flatten(),
            self.kl_values[batch_inds].flatten(),
            self.log_probs[batch_inds].flatten(),
            self.task_advantages[batch_inds].flatten(),
            self.constraint_advantages[batch_inds], #.flatten(),
            self.kl_advantages[batch_inds].flatten(),
            self.task_returns[batch_inds].flatten(),
            self.constraint_returns[batch_inds], #.flatten(),
            self.kl_returns[batch_inds].flatten(),
            self.ep_task_reward_togo[batch_inds].flatten(),
            self.ep_constraint_reward_togo[batch_inds], #.flatten(),
            self.ep_kl_reward_togo[batch_inds].flatten(),
            self.action_masks[batch_inds].reshape(-1, self.mask_dims),
        )
        return MaskableConstrainedRolloutBufferSamples(*map(self.to_torch, data))


class MaskableConstrainedDictRolloutBuffer(ConstrainedDictRolloutBuffer):
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
    :param device:
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        Equivalent to classic advantage when set to 1.
    :param gamma: Discount factor
    :param n_envs: Number of parallel environments
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "cpu",
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
    ):
        self.action_masks = None
        super().__init__(buffer_size, observation_space,
                         action_space, device, gae_lambda, gamma, n_envs=n_envs)

    def reset(self) -> None:
        if isinstance(self.action_space, spaces.Discrete):
            mask_dims = self.action_space.n
        elif isinstance(self.action_space, spaces.MultiDiscrete):
            mask_dims = sum(self.action_space.nvec)
        elif isinstance(self.action_space, spaces.MultiBinary):
            mask_dims = 2 * self.action_space.n  # One mask per binary outcome
        else:
            raise ValueError(
                f"Unsupported action space {type(self.action_space)}")

        self.mask_dims = mask_dims
        self.action_masks = np.ones(
            (self.buffer_size, self.n_envs, self.mask_dims))  # .to(self.device)

        super().reset()

    def add(self, *args, action_masks: Optional[torch.Tensor] = None, **kwargs) -> None:
        """
        :param action_masks: Masks applied to constrain the choice of possible actions.
        """
        if action_masks is not None:
            self.action_masks[self.pos] = action_masks.reshape(
                (self.n_envs, self.mask_dims))

        super().add(*args, **kwargs)

    def get(self, batch_size: Optional[int] = None) -> Generator[MaskableConstrainedDictRolloutBufferSamples, None, None]:
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
                "kl_values",
                "log_probs",
                "task_advantages",
                "constraint_advantages",
                "kl_advantages",
                "task_returns",
                "constraint_returns",
                "kl_returns",
                "ep_task_reward_togo",
                "ep_constraint_reward_togo",
                "ep_kl_reward_togo",
                "action_masks"]

            for tensor in _tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(
                    self.__dict__[tensor])
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx: start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> MaskableConstrainedDictRolloutBufferSamples:

        return MaskableConstrainedDictRolloutBufferSamples(
            observations={key: self.to_torch(obs[batch_inds]) for (
                key, obs) in self.observations.items()},
            actions=self.to_torch(self.actions[batch_inds]),
            old_task_values=self.to_torch(self.task_values[batch_inds].flatten()),
            old_constraint_values=self.to_torch(self.constraint_values[batch_inds]), # .flatten()
            old_kl_values=self.to_torch(self.kl_values[batch_inds].flatten()),
            old_log_prob=self.to_torch(self.log_probs[batch_inds].flatten()),
            task_advantages=self.to_torch(self.task_advantages[batch_inds].flatten()),
            constraint_advantages=self.to_torch(self.constraint_advantages[batch_inds]), #.flatten()
            kl_advantages=self.to_torch(self.kl_advantages[batch_inds].flatten()),
            task_returns=self.to_torch(self.task_returns[batch_inds].flatten()),
            constraint_returns=self.to_torch(self.constraint_returns[batch_inds]), #.flatten()
            kl_returns=self.to_torch(self.kl_returns[batch_inds].flatten()),
            ep_task_reward_togo=self.to_torch(self.ep_task_reward_togo[batch_inds].flatten()),
            ep_constraint_reward_togo=self.to_torch(self.ep_constraint_reward_togo[batch_inds]), #.flatten()
            ep_kl_reward_togo=self.to_torch(self.ep_kl_reward_togo[batch_inds].flatten()),
            action_masks=self.to_torch(
                self.action_masks[batch_inds].reshape(-1, self.mask_dims)),
        )


class MaskableConstrainedDictRolloutBuffer(ConstrainedDictRolloutBuffer):
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
    :param device:
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        Equivalent to classic advantage when set to 1.
    :param gamma: Discount factor
    :param n_envs: Number of parallel environments
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "cpu",
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
        n_constraints: int = 1,
    ):
        self.action_masks = None
        super().__init__(buffer_size, observation_space,
                         action_space, device, gae_lambda, gamma,
                         n_envs=n_envs, n_constraints=n_constraints)

    def reset(self) -> None:
        if isinstance(self.action_space, spaces.Discrete):
            mask_dims = self.action_space.n
        elif isinstance(self.action_space, spaces.MultiDiscrete):
            mask_dims = sum(self.action_space.nvec)
        elif isinstance(self.action_space, spaces.MultiBinary):
            mask_dims = 2 * self.action_space.n  # One mask per binary outcome
        else:
            raise ValueError(
                f"Unsupported action space {type(self.action_space)}")

        self.mask_dims = mask_dims
        self.action_masks = np.ones(
            (self.buffer_size, self.n_envs, self.mask_dims))  # .to(self.device)

        super().reset()

    def add(self, *args, action_masks: Optional[torch.Tensor] = None, **kwargs) -> None:
        """
        :param action_masks: Masks applied to constrain the choice of possible actions.
        """
        if action_masks is not None:
            self.action_masks[self.pos] = action_masks.reshape(
                (self.n_envs, self.mask_dims))

        super().add(*args, **kwargs)

    def get(self, batch_size: Optional[int] = None) -> Generator[MaskableConstrainedDictRolloutBufferSamples, None, None]:
        assert self.full, ""
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        # Prepare the data
        if not self.generator_ready:

            for key, obs in self.observations.items():
                self.observations[key] = self.swap_and_flatten(obs)

            _tensor_names = ["actions",
                             "task_values",
                             "constraint_values",
                             "kl_values",
                             "log_probs",
                             "task_advantages",
                             "constraint_advantages",
                             "kl_advantages",
                             "task_returns",
                             "constraint_returns",
                             "kl_returns",
                             "ep_task_reward_togo",
                             "ep_constraint_reward_togo",
                             "ep_kl_reward_togo",
                             "action_masks"]

            for tensor in _tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(
                    self.__dict__[tensor])
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx: start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> MaskableConstrainedDictRolloutBufferSamples:

        return MaskableConstrainedDictRolloutBufferSamples(
            observations={key: self.to_torch(obs[batch_inds]) for (
                key, obs) in self.observations.items()},
            actions=self.to_torch(self.actions[batch_inds]),
            old_task_values=self.to_torch(self.task_values[batch_inds].flatten()),
            old_constraint_values=self.to_torch(self.constraint_values[batch_inds]), #.flatten()
            old_kl_values=self.to_torch(self.kl_values[batch_inds].flatten()),
            old_log_prob=self.to_torch(self.log_probs[batch_inds].flatten()),
            task_advantages=self.to_torch(self.task_advantages[batch_inds].flatten()),
            constraint_advantages=self.to_torch(self.constraint_advantages[batch_inds]), #.flatten()
            kl_advantages=self.to_torch(self.kl_advantages[batch_inds].flatten()),
            task_returns=self.to_torch(self.task_returns[batch_inds].flatten()),
            constraint_returns=self.to_torch(self.constraint_returns[batch_inds]), #.flatten()
            kl_returns=self.to_torch(self.kl_returns[batch_inds].flatten()),
            ep_task_reward_togo=self.to_torch(self.ep_task_reward_togo[batch_inds].flatten()),
            ep_constraint_reward_togo=self.to_torch(self.ep_constraint_reward_togo[batch_inds]), #.flatten()
            ep_kl_reward_togo=self.to_torch(self.ep_kl_reward_togo[batch_inds].flatten()),
            action_masks=self.to_torch(
                self.action_masks[batch_inds].reshape(-1, self.mask_dims)),
        )


