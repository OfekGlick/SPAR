"""
Budget-Aware Feature Selection (BAFS) Base Class

This abstract base class provides shared functionality for budget-aware wrappers
that implement modality-level observation masking with per-modality costs.

Subclasses must implement:
    - _build_modalities(): Environment-specific modality construction
    - _get_raw_observation(): Environment-specific observation retrieval
"""

from abc import ABC, abstractmethod
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Optional, Tuple, Any, Union
import random
import torch

from omnisafe.envs.core import CMDP


class BudgetAwareBase(gym.Wrapper, CMDP, ABC):
    """Abstract base class for budget-aware environment wrappers.

    This class implements the core BAFS logic:
    - Modality-level masking with per-modality costs
    - CMDP interface (constrained MDP with cost signals)
    - Action space: Tuple(env_action, modality_mask) or env_action (if use_all_obs=True)

    Subclasses provide environment-specific:
    - Modality construction (_build_modalities)
    - Observation retrieval (_get_raw_observation)
    """

    # Subclasses should override these for OmniSafe discovery
    _support_envs: List[str] = []
    need_auto_reset_wrapper = True
    need_time_limit_wrapper = True

    def __init__(
        self,
        env: gym.Env,
        *,
        use_all_obs: bool = False,
        sensor_dropout_rescale: bool = True,
        cast_dtype: np.dtype = np.float32,
        max_episode_steps: Optional[int] = None,
        modality_costs: Optional[Dict[str, float]] = None,
        available_sensors: Optional[List[str]] = None,
        num_envs: int = 1,
        device: Optional[str] = None,
        seed: int = 42,
        cost_penalty_coef: float = 0.0,
        **kwargs: Any,
    ):
        """
        Initialize Budget-Aware base wrapper.

        Args:
            env: Wrapped gymnasium environment
            use_all_obs: If True, no masking (all observations always available)
            sensor_dropout_rescale: If True, rescale observations to maintain signal magnitude
            cast_dtype: Data type for observations
            max_episode_steps: Episode horizon
            modality_costs: Dict mapping modality names to costs (default: all 1.0)
            available_sensors: List of sensor names to make available (None = all sensors)
            num_envs: Number of parallel environments (for vectorized envs)
            device: PyTorch device for tensor operations
            seed: Random seed
            cost_penalty_coef: Coefficient for cost penalty (0=no penalty, 1=full cost subtracted from reward)
            **kwargs: Additional environment-specific arguments
        """
        super().__init__(env)

        # ── User configuration ─────────────────────────────────────────────────
        self.use_all_obs = bool(use_all_obs)
        self.sensor_dropout_rescale = bool(sensor_dropout_rescale)
        self.cast_dtype = cast_dtype
        self._num_envs = num_envs
        self._device = device if device else ('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.seed_value = seed
        self.cost_penalty_coef = float(cost_penalty_coef)
        self._available_sensors = available_sensors  # Stored for subclasses to validate/filter

        # ── Build modalities (environment-specific) ────────────────────────────
        # Subclass implements this to construct:
        # - mapping: Dict[modality_name, (start_idx, end_idx)]
        # - mod_sizes: Dict[modality_name, size]
        # - obs_names: List[modality_name]
        # - _base_low, _base_high: numpy arrays (flattened observation bounds)
        # - _flat_dim: int (total observation dimension)
        modality_info = self._build_modalities()
        self.mapping: Dict[str, Tuple[int, int]] = modality_info['mapping']
        self.mod_sizes: Dict[str, int] = modality_info['mod_sizes']
        self.obs_names: List[str] = modality_info['obs_names']
        self._base_low: np.ndarray = modality_info['base_low']
        self._base_high: np.ndarray = modality_info['base_high']
        self._flat_dim: int = modality_info['flat_dim']
        self._num_modalities = len(self.obs_names)

        # ── Observation space ──────────────────────────────────────────────────
        # Append modality mask bits to observation: [flat_obs, modality_mask]
        low = np.concatenate([self._base_low, np.zeros(self._num_modalities, dtype=self._base_low.dtype)], axis=0)
        high = np.concatenate([self._base_high, np.ones(self._num_modalities, dtype=self._base_high.dtype)], axis=0)
        self.observation_space = spaces.Box(low=low, high=high, dtype=low.dtype)

        # ── Action space ───────────────────────────────────────────────────────
        self.original_action_space = self.env.action_space
        self.is_continuous = isinstance(self.env.action_space, gym.spaces.Box)

        if self.use_all_obs:
            new_action_space = self.env.action_space
        else:
            env_action_space = self.env.action_space
            sensor_mask_space = gym.spaces.MultiBinary(self._num_modalities)

            # Store references for actor building
            if self.is_continuous:
                self.cont_act_space = env_action_space
                self.disc_act_space = sensor_mask_space
            else:
                self.disc_act_space = env_action_space
                self.disc_mask_space = sensor_mask_space

            new_action_space = gym.spaces.Tuple((env_action_space, sensor_mask_space))

        self.action_space = new_action_space
        self._action_space = new_action_space

        # ── Per-modality costs ─────────────────────────────────────────────────
        if modality_costs is None:
            modality_costs = {n: 1.0 for n in self.obs_names}
        self._modality_costs = {n: float(modality_costs.get(n, 1.0)) for n in self.obs_names}
        self.costs = np.array([self._modality_costs[n] for n in self.obs_names], dtype=np.float32)

        # ── Episode horizon ────────────────────────────────────────────────────
        self._max_episode_steps = max_episode_steps

        self.set_seed(seed)

    # ══════════════════════════════════════════════════════════════════════════
    # Abstract Methods (subclass must implement)
    # ══════════════════════════════════════════════════════════════════════════

    @abstractmethod
    def _build_modalities(self) -> Dict[str, Any]:
        """Build modality mapping and observation space bounds (environment-specific).

        Returns:
            Dict containing:
                - 'mapping': Dict[str, Tuple[int, int]] - modality name -> (start_idx, end_idx)
                - 'mod_sizes': Dict[str, int] - modality name -> size
                - 'obs_names': List[str] - ordered list of modality names
                - 'base_low': np.ndarray - lower bounds of flattened observation
                - 'base_high': np.ndarray - upper bounds of flattened observation
                - 'flat_dim': int - total flattened observation dimension
        """
        raise NotImplementedError

    @abstractmethod
    def _get_raw_observation(self) -> np.ndarray:
        """Get raw flattened observation from environment (environment-specific).

        Returns:
            Flattened observation array matching the modality mapping.
        """
        raise NotImplementedError

    # ══════════════════════════════════════════════════════════════════════════
    # Properties
    # ══════════════════════════════════════════════════════════════════════════

    @property
    def num_modalities(self) -> int:
        """Number of observation modalities."""
        return self._num_modalities

    @property
    def max_episode_steps(self) -> Optional[int]:
        """Maximum episode length."""
        return self._max_episode_steps

    @max_episode_steps.setter
    def max_episode_steps(self, value: int):
        self._max_episode_steps = value

    @property
    def obs_mapping(self) -> Dict[str, Tuple[int, int]]:
        """Mapping from modality names to observation indices."""
        return dict(self.mapping)

    @property
    def obs_modalities(self) -> List[str]:
        """List of modality names."""
        return list(self.obs_names)

    # ══════════════════════════════════════════════════════════════════════════
    # Utility Methods
    # ══════════════════════════════════════════════════════════════════════════

    def set_seed(self, seed: int) -> None:
        """Set random seed for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def _to_numpy(self, x) -> np.ndarray:
        """Convert torch tensor or array-like to numpy array."""
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    def _expand_modality_mask_to_features(self, m_mod: np.ndarray) -> np.ndarray:
        """Convert modality mask (len M) -> feature-level mask (len flat_dim).

        Args:
            m_mod: Modality-level binary mask (shape: [num_modalities])

        Returns:
            Feature-level binary mask (shape: [flat_dim])
        """
        m_feat = np.zeros(self._flat_dim, dtype=np.float32)
        for i, name in enumerate(self.obs_names):
            s, e = self.mapping[name]
            m_feat[s:e] = float(m_mod[i])
        return m_feat

    def _apply_mask(self, flat: np.ndarray, feat_mask01: np.ndarray, m_mod01: np.ndarray) -> np.ndarray:
        """Apply feature-level mask to observation with optional rescaling.

        Args:
            flat: Flattened observation (feature-level)
            feat_mask01: Feature-level binary mask (expanded from modality mask)
            m_mod01: Modality-level binary mask

        Returns:
            Gated (and optionally rescaled) observation
        """
        feat_mask01 = feat_mask01.astype(flat.dtype, copy=False)
        gated = flat * feat_mask01
        return gated

    def _mask_cost(self, m_mod01: np.ndarray) -> float:
        """Calculate cost based on active modalities.

        Args:
            m_mod01: Modality-level binary mask

        Returns:
            Total cost (sum of costs for active modalities)
        """
        total = 0.0
        for i, name in enumerate(self.obs_names):
            if m_mod01[i] > 0.5:
                total += self._modality_costs.get(name, 1.0)
        return float(total)

    # ══════════════════════════════════════════════════════════════════════════
    # CMDP / Gymnasium API
    # ══════════════════════════════════════════════════════════════════════════

    def reset(self, **kwargs):
        """Reset environment and return initial observation.

        Returns:
            Tuple of (observation, info) where observation includes modality mask bits.
            CMDP signature: (obs, info)
        """
        # Reset wrapped environment (subclass handles specifics)
        obs, info = self.env.reset(**kwargs)

        # Get raw observation (environment-specific)
        flat = self._get_raw_observation()

        # Initial mask: all modalities active
        mask0 = np.ones(self._num_modalities, dtype=flat.dtype)
        out = np.concatenate([flat, mask0], axis=0).astype(self.observation_space.dtype, copy=False)
        out = torch.as_tensor(out, dtype=torch.float32).to(self._device)

        info['unmasked_observation'] = out.clone()
        info['sensor_mask'] = mask0

        return out, info

    def step(self, action: Union[Tuple[Any, Any], Any]):
        """Execute one environment step.

        Args:
            action: Either env_action (if use_all_obs=True) or (env_action, modality_mask)

        Returns:
            CMDP step signature: (obs, reward, cost, terminated, truncated, info)
        """
        if self.use_all_obs:
            return self._step_no_masking(action)
        else:
            return self._step_with_masking(action)

    def _step_no_masking(self, action: Any):
        """Execute step without observation masking (all modalities active).

        Args:
            action: Environment action

        Returns:
            CMDP step signature: (obs, reward, cost, terminated, truncated, info)
        """
        inner = self._to_numpy(action)
        base_obs, r, term, trunc, info = self.env.step(inner)

        # Get raw observation with all modalities active
        flat = self._get_raw_observation()
        m_mod01 = np.ones(self._num_modalities, dtype=flat.dtype)
        out = np.concatenate([flat, m_mod01.astype(flat.dtype)], axis=0).astype(self.observation_space.dtype)

        # Calculate reward and cost
        rew = float(r)
        cost = self._mask_cost(m_mod01)

        # Populate info dictionary
        info = dict(info)
        info['sensor_mask'] = m_mod01
        info['original_reward'] = torch.as_tensor(rew, dtype=torch.float32)
        info['original_cost'] = torch.as_tensor(cost, dtype=torch.float32)

        returned_obs = torch.as_tensor(out, dtype=torch.float32).to(self._device)
        info['unmasked_observation'] = returned_obs.clone()

        return (
            returned_obs,
            torch.as_tensor(rew, dtype=torch.float32).to(self._device),
            torch.as_tensor(cost, dtype=torch.float32).to(self._device),
            torch.as_tensor(term, dtype=torch.bool).to(self._device),
            torch.as_tensor(trunc, dtype=torch.bool).to(self._device),
            info,
        )

    def _step_with_masking(self, action: Tuple[Any, Any]):
        """Execute step with observation masking based on modality selection.

        Args:
            action: Tuple of (env_action, modality_mask)

        Returns:
            CMDP step signature: (obs, reward, cost, terminated, truncated, info)

        Raises:
            ValueError: If action is not a 2-tuple
        """
        # Validate action format
        if not (isinstance(action, (tuple, list)) and len(action) == 2):
            raise ValueError("Action must be Tuple((env_action, modality_mask)) when use_all_obs=False")

        # Parse action components
        env_action = self._to_numpy(action[0])
        mod_mask = self._to_numpy(action[1]).astype(np.int64).reshape(-1)

        # Step base environment
        base_obs, r, term, trunc, info = self.env.step(env_action)

        # Get raw observation and apply masking
        flat = self._get_raw_observation()
        mod_mask_float = mod_mask.astype(np.float32)
        feat_mask01 = self._expand_modality_mask_to_features(mod_mask_float)
        gated = self._apply_mask(flat, feat_mask01, mod_mask_float)

        # Calculate cost based on selected modalities
        step_cost = self._mask_cost(mod_mask.astype(np.float32))

        # Compose final observation with mask bits
        out = np.concatenate(
            [gated, mod_mask.astype(gated.dtype)], axis=0
        ).astype(self.observation_space.dtype, copy=False)

        # Populate info dictionary
        info = dict(info)
        info["sensor_mask"] = mod_mask.astype(np.float32)
        info["original_step_reward"] = float(r)
        info["original_step_cost"] = float(step_cost)

        # Store unmasked observation for reference
        original_obs_flat = np.concatenate([flat, np.ones(self._num_modalities, dtype=flat.dtype)])
        returned_obs = torch.as_tensor(original_obs_flat, dtype=torch.float32).to(self._device)
        info["unmasked_observation"] = returned_obs.clone()

        return (
            torch.as_tensor(out, dtype=torch.float32).to(self._device),
            torch.as_tensor(float(r), dtype=torch.float32).to(self._device),
            torch.as_tensor(float(step_cost), dtype=torch.float32).to(self._device),
            torch.as_tensor(term, dtype=torch.bool).to(self._device),
            torch.as_tensor(trunc, dtype=torch.bool).to(self._device),
            info,
        )