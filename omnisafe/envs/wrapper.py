# Copyright 2023 OmniSafe Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Wrapper for the environment."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
import gymnasium as gym
from omnisafe.common import Normalizer
from omnisafe.envs.core import CMDP, Wrapper


class TimeLimit(Wrapper):
    """Time limit wrapper for the environment.

    .. warning::
        The time limit wrapper only supports single environment.

    Examples:
        >>> env = TimeLimit(env, time_limit=100)

    Args:
        env (CMDP): The environment to wrap.
        time_limit (int): The time limit for each episode.
        device (torch.device): The torch device to use.

    Attributes:
        _time_limit (int): The time limit for each episode.
        _time (int): The current time step.
    """

    def __init__(self, env: CMDP, time_limit: int, device: torch.device) -> None:
        """Initialize an instance of :class:`TimeLimit`."""
        super().__init__(env=env, device=device)
        assert self.num_envs == 1, 'TimeLimit only supports single environment'
        self._time: int = 0
        self._time_limit: int = time_limit

    def reset(
            self,
            seed: int | None = None,
            options: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Reset the environment.

        .. note::
            Additionally, the time step will be reset to 0.

        Args:
            seed (int, optional): The random seed. Defaults to None.
            options (dict[str, Any], optional): The options for the environment. Defaults to None.

        Returns:
            observation: The initial observation of the space.
            info: Some information logged by the environment.
        """
        self._time = 0
        return super().reset(seed=seed, options=options)

    def step(
            self,
            action: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, Any],
    ]:
        """Run one timestep of the environment's dynamics using the agent actions.

        .. note::
            Additionally, the time step will be increased by 1.

        Args:
            action (torch.Tensor): The action from the agent or random.

        Returns:
            observation: The agent's observation of the current environment.
            reward: The amount of reward returned after previous action.
            cost: The amount of cost returned after previous action.
            terminated: Whether the episode has ended.
            truncated: Whether the episode has been truncated due to a time limit.
            info: Some information logged by the environment.
        """
        obs, reward, cost, terminated, truncated, info = super().step(action)

        self._time += 1
        truncated = torch.tensor(
            self._time >= self._time_limit,
            dtype=torch.bool,
            device=self._device,
        )

        return obs, reward, cost, terminated, truncated, info


class AutoReset(Wrapper):
    """Auto reset the environment when the episode is terminated.

    Examples:
        >>> env = AutoReset(env)

    Args:
        env (CMDP): The environment to wrap.
        device (torch.device): The torch device to use.
    """

    def __init__(self, env: CMDP, device: torch.device) -> None:
        """Initialize an instance of :class:`AutoReset`."""
        super().__init__(env=env, device=device)

        assert self.num_envs == 1, 'AutoReset only supports single environment'

    def step(
            self,
            action: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, Any],
    ]:
        """Run one timestep of the environment's dynamics using the agent actions.

        .. note::
            If the episode is terminated, the environment will be reset. The ``obs`` will be the
            first observation of the new episode. And the true final observation will be stored in
            ``info['final_observation']``.

        Args:
            action (torch.Tensor): The action from the agent or random.

        Returns:
            observation: The agent's observation of the current environment.
            reward: The amount of reward returned after previous action.
            cost: The amount of cost returned after previous action.
            terminated: Whether the episode has ended.
            truncated: Whether the episode has been truncated due to a time limit.
            info: Some information logged by the environment.
        """
        obs, reward, cost, terminated, truncated, info = super().step(action)

        if terminated or truncated:
            new_obs, new_info = self.reset()
            assert (
                    'final_observation' not in new_info
            ), 'info dict cannot contain key "final_observation" '
            assert 'final_info' not in new_info, 'info dict cannot contain key "final_info" '

            new_info['final_observation'] = obs
            new_info['final_info'] = info

            obs = new_obs
            info = new_info

        return obs, reward, cost, terminated, truncated, info


class BudgetWrapper(Wrapper):
    """Budget constraint wrapper that terminates episodes when cumulative cost exceeds a threshold.

    .. warning::
        The budget wrapper only supports single environment.

    Examples:
        >>> env = BudgetWrapper(env, device, budget_limit=25.0)

    Args:
        env (CMDP): The environment to wrap.
        device (torch.device): The torch device to use.
        budget_limit (float): The maximum cumulative cost allowed per episode.

    Attributes:
        _budget_limit (float): The maximum cumulative cost threshold.
        _cumulative_cost (float): The current cumulative cost in the episode.
    """

    def __init__(self, env: CMDP, device: torch.device, budget_limit: float = 25.0) -> None:
        """Initialize an instance of :class:`BudgetWrapper`."""
        super().__init__(env=env, device=device)
        assert self.num_envs == 1, 'BudgetWrapper only supports single environment'
        self._budget_limit: float = budget_limit
        self._cumulative_cost: float = 0.0

    def reset(
            self,
            seed: int | None = None,
            options: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Reset the environment.

        .. note::
            Additionally, the cumulative cost will be reset to 0.

        Args:
            seed (int, optional): The random seed. Defaults to None.
            options (dict[str, Any], optional): The options for the environment. Defaults to None.

        Returns:
            observation: The initial observation of the space.
            info: Some information logged by the environment.
        """
        self._cumulative_cost = 0.0
        return super().reset(seed=seed, options=options)

    def step(
            self,
            action: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, Any],
    ]:
        """Run one timestep of the environment's dynamics using the agent actions.

        .. note::
            The cumulative cost is tracked and compared against the budget limit.
            If exceeded, truncated is set to True and info['budget_exceeded'] is set.

        Args:
            action (torch.Tensor): The action from the agent or random.

        Returns:
            observation: The agent's observation of the current environment.
            reward: The amount of reward returned after previous action.
            cost: The amount of cost returned after previous action.
            terminated: Whether the episode has ended.
            truncated: Whether the episode has been truncated due to budget violation.
            info: Some information logged by the environment.
        """
        obs, reward, cost, terminated, truncated, info = super().step(action)

        # Accumulate cost
        self._cumulative_cost += cost.item()

        # Check budget violation
        if self._cumulative_cost > self._budget_limit:
            truncated = torch.tensor(True, dtype=torch.bool, device=self._device)
            info['budget_exceeded'] = True

        return obs, reward, cost, terminated, truncated, info


class ObsNormalize(Wrapper):
    """Normalize the observation.

    Examples:
        >>> env = ObsNormalize(env)
        >>> norm = Normalizer(env.observation_space.shape)  # load saved normalizer
        >>> env = ObsNormalize(env, norm)

    Args:
        env (CMDP): The environment to wrap.
        device (torch.device): The torch device to use.
        norm (Normalizer or None, optional): The normalizer to use. Defaults to None.
    """

    def __init__(self, env: CMDP, device: torch.device, norm: Normalizer | None = None) -> None:
        """Initialize an instance of :class:`ObsNormalize`."""
        super().__init__(env=env, device=device)
        assert isinstance(self.observation_space, spaces.Box), 'Observation space must be Box'
        self._obs_normalizer: Normalizer

        if norm is not None:
            self._obs_normalizer = norm.to(self._device)
        else:
            self._obs_normalizer = Normalizer(self.observation_space.shape, clip=5).to(self._device)

    def step(
            self,
            action: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, Any],
    ]:
        """Run one timestep of the environment's dynamics using the agent actions.

        .. note::
            The observation and the ``info['final_observation']`` will be normalized.

        Args:
            action (torch.Tensor): The action from the agent or random.

        Returns:
            observation: The agent's observation of the current environment.
            reward: The amount of reward returned after previous action.
            cost: The amount of cost returned after previous action.
            terminated: Whether the episode has ended.
            truncated: Whether the episode has been truncated due to a time limit.
            info: Some information logged by the environment.
        """
        obs, reward, cost, terminated, truncated, info = super().step(action)
        obs, reward, cost, terminated, truncated = (
            torch.as_tensor(x, dtype=torch.float32)
            for x in (obs, reward, cost, terminated, truncated)
        )
        if 'final_observation' in info:
            final_obs_slice = info['_final_observation'] if self.num_envs > 1 else slice(None)
            info['final_observation'] = torch.as_tensor(
                info['final_observation'], dtype=torch.float32
            )
            info['final_observation'] = info['final_observation'].to(self._device)
            info['original_final_observation'] = info['final_observation']
            info['final_observation'][final_obs_slice] = self._obs_normalizer.normalize(
                info['final_observation'][final_obs_slice],
            )
        info['original_obs'] = obs
        obs = self._obs_normalizer.normalize(obs)
        return obs, reward, cost, terminated, truncated, info

    def reset(
            self,
            seed: int | None = None,
            options: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Reset the environment and returns an initial observation.

        Args:
            seed (int, optional): The random seed. Defaults to None.
            options (dict[str, Any], optional): The options for the environment. Defaults to None.

        Returns:
            observation: The initial observation of the space.
            info: Some information logged by the environment.
        """
        obs, info = super().reset(seed=seed, options=options)
        info['original_obs'] = obs
        obs = self._obs_normalizer.normalize(obs)
        return obs, info

    def save(self) -> dict[str, torch.nn.Module]:
        """Save the observation normalizer.

        .. note::
            The saved components will be stored in the wrapped environment. If the environment is
            not wrapped, the saved components will be empty dict. common wrappers are obs_normalize,
            reward_normalize, and cost_normalize. When evaluating the saved model, the normalizer
            should be loaded.

        Returns:
            The saved components, that is the observation normalizer.
        """
        saved = super().save()
        saved['obs_normalizer'] = self._obs_normalizer
        return saved


class ModalityObsNormalize(Wrapper):
    """
    Per-modality observation normalization using OmniSafe's Normalizer per modality,
    contained entirely within the wrapper (no extra helper classes).

    - Expects flat Box obs of shape (D,) or (1, D).
    - `modality_to_span`: dict name -> (start, end) indices over the *feature* part.
    - If the last `mask_length` dims are a binary mask, they are preserved.
    - Only masked-on modalities are updated & normalized.
    - `save()` exposes state under 'obs_normalizer' (state_dict of ModuleDict).
    """

    def __init__(
            self,
            env: CMDP,
            device: torch.device,
            modality_to_span: dict[str, tuple[int, int]],
            *,
            mask_length: int = 0,
            clip_value: float = 5.0,
            update_stats: bool = True,
            # If provided, should be a state_dict previously saved under 'obs_normalizer'
            norm_per_mod_state=None,
    ) -> None:
        super().__init__(env=env, device=device)
        assert self.num_envs == 1, 'This normalizer supports single env only.'
        assert isinstance(self.observation_space, spaces.Box), 'Observation space must be Box.'
        assert len(self.observation_space.shape) == 1, 'Expect flat 1-D observation.'

        # --- config / dims ---
        self.modality_to_span = dict(sorted(modality_to_span.items(), key=lambda kv: kv[1][0]))
        self.modalities = list(self.modality_to_span.keys())
        self.mask_length = int(mask_length)
        self.clip_value = float(clip_value)
        self.update_stats_enabled = bool(update_stats)

        total_dim = int(self.observation_space.shape[0])
        feature_dim = total_dim - self.mask_length
        last_end = max(end for (_, end) in self.modality_to_span.values())
        assert last_end <= feature_dim, 'Modality spans exceed (D - mask_length).'

        # --- build per-modality normalizers, contained inside the wrapper ---
        if norm_per_mod_state is None:
            self._per_mod_norm = nn.ModuleDict()
            for mod, (s, e) in self.modality_to_span.items():
                seg_len = int(e - s)
                self._per_mod_norm[mod] = Normalizer(shape=(seg_len,), clip=self.clip_value)
            self._per_mod_norm.to(self._device)

        if norm_per_mod_state is not None:
            self._per_mod_norm = norm_per_mod_state
            self._per_mod_norm.to(self._device)
        # respect initial update mode
        self._set_train_mode(self.update_stats_enabled)

    # ---------------- public controls ----------------
    def freeze_stats(self, freeze: bool = True) -> None:
        """Freeze/unfreeze running-stat updates (call True before eval)."""
        self.update_stats_enabled = not freeze
        self._set_train_mode(not freeze)

    def _set_train_mode(self, train: bool) -> None:
        self._per_mod_norm.train(train)

    # ---------------- internals ----------------
    def _split_features_and_mask(self, obs_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.mask_length == 0:
            return obs_t, None
        return obs_t[..., :-self.mask_length], obs_t[..., -self.mask_length:]

    def _normalize_features(self, features: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        """
        Apply per-modality normalization with mask-awareness.
        features: (F,) or (B,F)
        mask:     (M,) or (B,M) binary (if None -> treat as all ones)
        """
        assert features.ndim in (1, 2)
        batched = (features.ndim == 2)
        B = features.shape[0] if batched else 1

        if mask is None:
            mask = torch.ones((B, len(self.modalities)), dtype=torch.bool, device=features.device) if batched else \
                torch.ones((len(self.modalities),), dtype=torch.bool, device=features.device)

        feats = features if batched else features.unsqueeze(0)
        ms = mask if (batched or mask.ndim == 2) else mask.unsqueeze(0)

        outs = []
        for i, mod in enumerate(self.modalities):
            s, e = self.modality_to_span[mod]
            seg = feats[..., s:e]  # (B, seg_len)
            m_i = ms[..., i]
            if m_i.dtype != torch.bool:
                m_i = m_i > 0.5

            seg_out = seg.clone()
            if m_i.any():
                seg_out[m_i] = self._per_mod_norm[mod].normalize(seg[m_i])
            # else passthrough unchanged
            outs.append(seg_out)

        out = torch.cat(outs, dim=-1)
        return out if batched else out.squeeze(0)

    def _process_observation(self, obs_t: torch.Tensor) -> torch.Tensor:
        """Normalize features; keep appended mask if present; preserve (1,D) leading dim."""
        had_leading_batch = (obs_t.ndim == 2)
        x = obs_t if not had_leading_batch else obs_t.squeeze(0)

        features_t, mask_t = self._split_features_and_mask(x)
        features_norm = self._normalize_features(features_t, mask_t)

        if mask_t is not None:
            obs_norm = torch.cat([features_norm, mask_t], dim=-1)
        else:
            obs_norm = features_norm

        return obs_norm if not had_leading_batch else obs_norm.unsqueeze(0)

    # ---------------- Wrapper overrides ----------------
    def step(self, action: torch.Tensor):
        obs, reward, cost, terminated, truncated, info = super().step(action)
        obs, reward, cost, terminated, truncated = (
            torch.as_tensor(x, dtype=torch.float32, device=self._device)
            for x in (obs, reward, cost, terminated, truncated)
        )
        info['original_obs'] = obs

        # First mask the original observation and update the parameters with it
        if 'unmasked_observation' in info:
            info['unmasked_observation'] = self._process_observation(
                torch.as_tensor(info['unmasked_observation'], dtype=torch.float32, device=self._device)
            )
        # Signal to the normalizers that we are in masked mode
        for mod in self.modalities:
            self._per_mod_norm[mod].masked = True

        # Mask the observation and don't update the parameters before doing it
        obs = self._process_observation(obs)

        # Signal to the normalizers that we are not in masked mode anymore
        for mod in self.modalities:
            self._per_mod_norm[mod].masked = False

        if 'final_observation' in info:
            fin = torch.as_tensor(info['final_observation'], dtype=torch.float32, device=self._device)
            info['original_final_observation'] = fin
            info['final_observation'] = self._process_observation(fin)

        return obs, reward, cost, terminated, truncated, info

    def reset(self, seed: int | None = None, options: dict | None = None):
        obs, info = super().reset(seed=seed, options=options)
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self._device)
        info['original_obs'] = obs
        obs = self._process_observation(obs)
        info['unmasked_observation'] = obs.clone()
        return obs, info

    # --------- save/load (compatible key) ---------
    def save(self) -> dict[str, dict]:
        """
        Return a dict that includes our per-modality normalizers under the same key name
        used elsewhere. We store a STATE DICT (not a module) for portability.
        """
        saved = super().save()
        saved['obs_normalizer'] = self._per_mod_norm
        return saved


class RewardNormalize(Wrapper):
    """Normalize the reward.

    Examples:
        >>> env = RewardNormalize(env)
        >>> norm = Normalizer(()) # load saved normalizer
        >>> env = RewardNormalize(env, norm)

    Args:
        env (CMDP): The environment to wrap.
        device (torch.device): The torch device to use.
        norm (Normalizer or None, optional): The normalizer to use. Defaults to None.
    """

    def __init__(self, env: CMDP, device: torch.device, norm: Normalizer | None = None) -> None:
        """Initialize an instance of :class:`RewardNormalize`."""
        super().__init__(env=env, device=device)
        self._reward_normalizer: Normalizer

        if norm is not None:
            self._reward_normalizer = norm.to(self._device)
        else:
            self._reward_normalizer = Normalizer((), clip=5).to(self._device)

    def step(
            self,
            action: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, Any],
    ]:
        """Run one timestep of the environment's dynamics using the agent actions.

        .. note::
            The reward will be normalized for agent training. Then the original reward will be
            stored in ``info['original_reward']`` for logging.

        Args:
            action (torch.Tensor): The action from the agent or random.

        Returns:
            observation: The agent's observation of the current environment.
            reward: The amount of reward returned after previous action.
            cost: The amount of cost returned after previous action.
            terminated: Whether the episode has ended.
            truncated: Whether the episode has been truncated due to a time limit.
            info: Some information logged by the environment.
        """
        obs, reward, cost, terminated, truncated, info = super().step(action)
        info['original_reward'] = reward
        reward = self._reward_normalizer.normalize(reward)
        return obs, reward, cost, terminated, truncated, info

    def save(self) -> dict[str, torch.nn.Module]:
        """Save the reward normalizer.

        .. note::
            The saved components will be stored in the wrapped environment. If the environment is
            not wrapped, the saved components will be empty dict. common wrappers are obs_normalize,
            reward_normalize, and cost_normalize.

        Returns:
            The saved components, that is the reward normalizer.
        """
        saved = super().save()
        saved['reward_normalizer'] = self._reward_normalizer
        return saved


class CostNormalize(Wrapper):
    """Normalize the cost.

    Examples:
        >>> env = CostNormalize(env)
        >>> norm = Normalizer(()) # load saved normalizer
        >>> env = CostNormalize(env, norm)

    Args:
        env (CMDP): The environment to wrap.
        device (torch.device): The torch device to use.
        norm (Normalizer or None, optional): The normalizer to use. Defaults to None.
    """

    def __init__(self, env: CMDP, device: torch.device, norm: Normalizer | None = None) -> None:
        """Initialize an instance of :class:`CostNormalize`."""
        super().__init__(env=env, device=device)
        self._cost_normalizer: Normalizer

        if norm is not None:
            self._cost_normalizer = norm.to(self._device)
        else:
            self._cost_normalizer = Normalizer((), clip=5).to(self._device)

    def step(
            self,
            action: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, Any],
    ]:
        """Run one timestep of the environment's dynamics using the agent actions.

        .. note::
            The cost will be normalized for agent training. Then the original reward will be stored
            in ``info['original_cost']`` for logging.

        Args:
            action (torch.Tensor): The action from the agent or random.

        Returns:
            observation: The agent's observation of the current environment.
            reward: The amount of reward returned after previous action.
            cost: The amount of cost returned after previous action.
            terminated: Whether the episode has ended.
            truncated: Whether the episode has been truncated due to a time limit.
            info: Some information logged by the environment.
        """
        obs, reward, cost, terminated, truncated, info = super().step(action)
        info['original_cost'] = cost
        cost = self._cost_normalizer.normalize(cost)
        return obs, reward, cost, terminated, truncated, info

    def save(self) -> dict[str, torch.nn.Module]:
        """Save the cost normalizer.

        .. note::
            The saved components will be stored in the wrapped environment. If the environment is
            not wrapped, the saved components will be empty dict. common wrappers are obs_normalize,
            reward_normalize, and cost_normalize.

        Returns:
            The saved components, that is the cost normalizer.
        """
        saved = super().save()
        saved['cost_normalizer'] = self._cost_normalizer
        return saved


class ActionScale(Wrapper):
    """Scale the action space to a given range.

    Examples:
        >>> env = ActionScale(env, low=-1, high=1)
        >>> env.action_space
        Box(-1.0, 1.0, (1,), float32)

    Args:
        env (CMDP): The environment to wrap.
        device (torch.device): The device to use.
        low (int or float): The lower bound of the action space.
        high (int or float): The upper bound of the action space.
    """

    def __init__(
            self,
            env: CMDP,
            device: torch.device,
            low: float,
            high: float,
    ) -> None:
        """Initialize an instance of :class:`ActionScale`."""
        super().__init__(env=env, device=device)
        # Original (env) continuous action space
        if isinstance(self.action_space, spaces.Box):
            cont_space = self.action_space
            is_tuple = False
        else:
            assert isinstance(self.action_space, spaces.Tuple), "Expected Tuple for multi-head."
            assert isinstance(self.action_space[0], spaces.Box), "First element must be Box (continuous)."
            cont_space = self.action_space[0]
            is_tuple = True

        # Cache original bounds (env domain)
        self._old_min_action = torch.tensor(cont_space.low, dtype=torch.float32, device=self._device)
        self._old_max_action = torch.tensor(cont_space.high, dtype=torch.float32, device=self._device)

        # Define the *normalized* domain that the policy should output in
        min_action = np.full(cont_space.shape, low, dtype=cont_space.dtype)
        max_action = np.full(cont_space.shape, high, dtype=cont_space.dtype)
        self._min_action = torch.tensor(min_action, dtype=torch.float32, device=self._device)
        self._max_action = torch.tensor(max_action, dtype=torch.float32, device=self._device)

        # Expose a normalized action_space to the agent
        new_cont = spaces.Box(low=min_action, high=max_action, shape=cont_space.shape, dtype=cont_space.dtype)
        if is_tuple:
            self._action_space = spaces.Tuple((new_cont, self._env.action_space[1]))
        else:
            self._action_space = new_cont

    def step(
            self,
            action: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, Any],
    ]:
        """Run one timestep of the environment's dynamics using the agent actions.

        .. note::
            The action will be scaled to the original range for agent training.

        Args:
            action (torch.Tensor): The action from the agent or random.

        Returns:
            observation: The agent's observation of the current environment.
            reward: The amount of reward returned after previous action.
            cost: The amount of cost returned after previous action.
            terminated: Whether the episode has ended.
            truncated: Whether the episode has been truncated due to a time limit.
            info: Some information logged by the environment.
        """
        if isinstance(self.action_space, spaces.Box):
            # ensure tensor
            if not torch.is_tensor(action):
                action = torch.as_tensor(action, dtype=torch.float32, device=self._device)
            a = torch.clamp(action, self._min_action, self._max_action)
            scaled = self._old_min_action + (self._old_max_action - self._old_min_action) * (
                    (a - self._min_action) / (self._max_action - self._min_action)
            )
            action_env = scaled
        else:
            # Tuple: (continuous, discrete)
            cont_action, disc_action = action
            if not torch.is_tensor(cont_action):
                cont_action = torch.as_tensor(cont_action, dtype=torch.float32, device=self._device)
            a = torch.clamp(cont_action, self._min_action, self._max_action)
            scaled = self._old_min_action + (self._old_max_action - self._old_min_action) * (
                    (a - self._min_action) / (self._max_action - self._min_action)
            )
            action_env = (scaled, disc_action)

        return super().step(action_env)


class ActionRepeat(Wrapper):
    """Repeat action given times.

    Example:
        >>> env = ActionRepeat(env, times=3)
    """

    def __init__(
            self,
            env: CMDP,
            times: int,
            device: torch.device,
    ) -> None:
        """Initialize the wrapper.

        Args:
            env: The environment to wrap.
            times: The number of times to repeat the action.
            device: The device to use.
        """
        super().__init__(env=env, device=device)
        self._times = times
        self._device = device

    def step(
            self,
            action: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, Any],
    ]:
        """Run self._times timesteps of the environment's dynamics using the agent actions.

        Args:
            action (torch.Tensor): The action from the agent or random.

        Returns:
            observation: The agent's observation of the current environment.
            reward: The amount of reward returned after previous action.
            cost: The amount of cost returned after previous action.
            terminated: Whether the episode has ended.
            truncated: Whether the episode has been truncated due to a time limit.
            info: Some information logged by the environment.
        """
        rewards, costs = torch.tensor(0.0).to(self._device), torch.tensor(0.0).to(self._device)
        for _step, _ in enumerate(range(self._times)):
            obs, reward, cost, terminated, truncated, info = super().step(action)
            rewards += reward
            costs += cost
            goal_met = info.get('goal_met', False)
            if terminated or truncated or goal_met:
                break
        info['num_step'] = _step + 1
        return obs, rewards, costs, terminated, truncated, info


class Unsqueeze(Wrapper):
    """Unsqueeze the observation, reward, cost, terminated, truncated and info.

    Examples:
        >>> env = Unsqueeze(env)
    """

    def __init__(self, env: CMDP, device: torch.device) -> None:
        """Initialize an instance of :class:`Unsqueeze`."""
        super().__init__(env=env, device=device)
        assert self.num_envs == 1, 'Unsqueeze only works with single environment'
        assert isinstance(self.observation_space, spaces.Box), 'Observation space must be Box'

    def step(
            self,
            action: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, Any],
    ]:
        """Run one timestep of the environment's dynamics using the agent actions.

        .. note::
            The vector information will be unsqueezed to (1, dim) for agent training.

        Args:
            action (torch.Tensor): The action from the agent or random.

        Returns:
            observation: The agent's observation of the current environment.
            reward: The amount of reward returned after previous action.
            cost: The amount of cost returned after previous action.
            terminated: Whether the episode has ended.
            truncated: Whether the episode has been truncated due to a time limit.
            info: Some information logged by the environment.
        """
        if isinstance(action, torch.Tensor):
            action = action.squeeze(0)
        elif isinstance(action, tuple):
            action = tuple((torch.as_tensor(x, dtype=torch.float32).squeeze(0) for x in action))
        obs, reward, cost, terminated, truncated, info = super().step(action)
        obs, reward, cost, terminated, truncated = (
            torch.as_tensor(x, dtype=torch.float32)
            for x in (obs, reward, cost, terminated, truncated)
        )
        obs, reward, cost, terminated, truncated = (
            x.unsqueeze(0) for x in (obs, reward, cost, terminated, truncated)
        )
        for k, v in info.items():
            if isinstance(v, torch.Tensor):
                info[k] = v.unsqueeze(0)

        return obs, reward, cost, terminated, truncated, info

    def reset(
            self,
            seed: int | None = None,
            options: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Reset the environment and returns a new observation.

        .. note::
            The vector information will be unsqueezed to (1, dim) for agent training.

        Args:
            seed (int, optional): The random seed. Defaults to None.
            options (dict[str, Any], optional): The options for the environment. Defaults to None.

        Returns:
            observation: The initial observation of the space.
            info: Some information logged by the environment.
        """
        obs, info = super().reset(seed=seed, options=options)
        obs = obs.unsqueeze(0)
        for k, v in info.items():
            if isinstance(v, torch.Tensor):
                info[k] = v.unsqueeze(0)

        return obs, info


class ModalityObsScale(Wrapper):
    """Scale active modality slices by M / max(1, sum(mask))."""

    def __init__(self, env: CMDP, device):
        super().__init__(env=env, device=device)
        self.spans = dict(self.mapping)  # assumes env.mapping exists
        self.names = list(self.spans.keys())
        self.M = len(self.names)

    def _scale(self, obs):
        feat_dim = obs.shape[0] - self.M
        feat, mask = obs[:feat_dim], obs[feat_dim:]
        m = mask.sum()
        if m.item() == 0:
            return obs  # nothing active, skip scaling
        scale = self.M / m
        for j, name in enumerate(self.names):
            if mask[j] > 0:  # active
                s, e = self.spans[name]
                feat[s:e] *= scale
        return obs
        # (above keeps it torch-y without extra allocations; feel free to just do torch.cat([feat, mask]))

    def step(self, action):
        obs, r, c, term, trunc, info = super().step(action)
        obs = self._scale(obs)
        return obs, r, c, term, trunc, info

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        obs = self._scale(obs)
        return obs, info
