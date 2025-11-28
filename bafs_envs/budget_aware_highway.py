import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Optional, Tuple, Any

from base_envs.highway_env.envs.common.observation import observation_factory

# OmniSafe CMDP registration
from omnisafe.envs.core import env_register
from bafs_envs.budget_aware_base import BudgetAwareBase


@env_register
class BudgetAwareHighway(BudgetAwareBase):
    """Highway-env wrapper with BAFS-style modality-level masking for OmniSafe.

    Action space:
        - If use_all_obs=True: env.action_space (no masking).
        - Else: Tuple( env.action_space, MultiBinary(M) ), M = number of modalities.

    Mask is applied to the *returned* observation each step (i.e., mask_t gates obs_{t+1}).
    Costs are per-modality. We do NOT expose a feature-level masking mode.
    """

    # OmniSafe discovery helpers (used by make()/Evaluator)
    _support_envs = [
        "budget-aware-highway-fast-v0",
        "budget-aware-roundabout-v0",
        "budget-aware-intersection-v1",
    ]
    need_auto_reset_wrapper = True
    need_time_limit_wrapper = True

    DEFAULT_TYPES = [
        "Kinematics",
        "LidarObservation",
        "OccupancyGrid",
        "TimeToCollision",
    ]
    SKIP_TYPES = {
        "MultiAgentObservation", "TupleObservation",
        "AttributesObservation", "KinematicsGoal", "ExitObservation"
    }

    def __init__(
            self,
            env_id: Optional[str] = None,
            observation_types: List[str] = None,
            observation_configs: Dict[str, Dict] = None,
            seed=42,
            *,
            use_all_obs: bool = False,
            sensor_dropout_rescale: bool = True,
            modality_costs: Dict[str, float] = None,
            cast_dtype: np.dtype = np.float32,
            available_sensors: Optional[List[str]] = None,
            **kwargs: Any,
    ):
        if env_id is None:
            raise ValueError("env_id must be provided")

        # Allow "BudgetAware..." aliases: strip prefix for the underlying env id
        base_id = env_id.replace("budget-aware-", "")

        # Filter out OmniSafe-specific parameters that highway-env doesn't understand
        env_kwargs = kwargs.copy()
        num_envs = env_kwargs.pop('num_envs', 1)
        device = env_kwargs.pop('device', None)

        try:
            render_mode = env_kwargs.pop('render_mode')
        except KeyError:
            render_mode = None

        env = gym.make(
            base_id,
            config=env_kwargs,
            render_mode=render_mode,
        )
        max_episode_steps = env.unwrapped.config.get("duration", 40)

        # Store observation configuration for _build_modalities
        self.observation_types = observation_types or list(self.DEFAULT_TYPES)
        self.observation_configs = observation_configs or {}

        # Initialize base class
        super().__init__(
            env=env,
            use_all_obs=use_all_obs,
            sensor_dropout_rescale=sensor_dropout_rescale,
            cast_dtype=cast_dtype,
            max_episode_steps=max_episode_steps,
            modality_costs=modality_costs,
            available_sensors=available_sensors,
            num_envs=num_envs,
            device=device,
            seed=seed,
        )

    @property
    def max_episode_steps(self) -> int | None:
        return self.unwrapped.config["duration"]

    @max_episode_steps.setter
    def max_episode_steps(self, value):
        self._max_episode_steps = value

    # ══════════════════════════════════════════════════════════════════════════
    # Abstract Method Implementations (Highway-specific)
    # ══════════════════════════════════════════════════════════════════════════

    def _build_modalities(self) -> Dict[str, Any]:
        """Build multi-observation from highway-env observation factory."""

        # ── Validate and filter available sensors ────────────────────────────
        if self._available_sensors is not None:
            available_set = set(self._available_sensors)
            valid_sensors = set(self.DEFAULT_TYPES)
            invalid = available_set - valid_sensors

            if invalid:
                raise ValueError(
                    f"Invalid sensor names in available_sensors: {invalid}. "
                    f"Valid Highway sensors: {valid_sensors}"
                )

            # Filter observation_types to only available sensors
            self.observation_types = [
                t for t in self.observation_types if t in available_set
            ]

            if not self.observation_types:
                raise ValueError(
                    f"No valid observation types after filtering with "
                    f"available_sensors={self._available_sensors}"
                )

        # Build observation objects using highway-env's observation_factory
        obs_names: List[str] = []
        self._obs_objects = []
        shapes: Dict[str, Tuple[int, ...]] = {}
        _spaces: Dict[str, spaces.Box] = {}

        for name in self.observation_types:
            if name in self.SKIP_TYPES:
                continue
            cfg = {"type": name}
            cfg.update(self.observation_configs.get(name, {}))
            if name == "GrayscaleObservation" and "weights" not in cfg:
                cfg["weights"] = [0.2989, 0.5870, 0.1140]

            obj = observation_factory(self.env.unwrapped, cfg)
            sp = obj.space()
            if isinstance(sp, spaces.Box):
                obs_names.append(name)
                self._obs_objects.append(obj)
                _spaces[name] = sp
                shapes[name] = tuple(sp.shape)

        # Store for later use
        self.shapes = shapes
        self._spaces = _spaces

        # Build mapping and flattened bounds
        lows, highs = [], []
        mapping: Dict[str, Tuple[int, int]] = {}
        mod_sizes: Dict[str, int] = {}
        cursor = 0

        for name in obs_names:
            sp = _spaces[name]
            lo = np.asarray(sp.low).flatten()
            hi = np.asarray(sp.high).flatten()
            K = lo.size
            lows.append(lo)
            highs.append(hi)
            mapping[name] = (cursor, cursor + K)
            mod_sizes[name] = K
            cursor += K

        base_low = np.concatenate(lows, axis=0) if lows else np.array([], dtype=np.float32)
        base_high = np.concatenate(highs, axis=0) if highs else np.array([], dtype=np.float32)

        if self.cast_dtype is not None:
            base_low = base_low.astype(self.cast_dtype, copy=False)
            base_high = base_high.astype(self.cast_dtype, copy=False)

        return {
            'mapping': mapping,
            'mod_sizes': mod_sizes,
            'obs_names': obs_names,
            'base_low': base_low,
            'base_high': base_high,
            'flat_dim': base_low.size,
        }

    def _get_raw_observation(self) -> np.ndarray:
        """Get raw observation from highway-env observation objects."""
        chunks = []
        for name, obj in zip(self.obs_names, self._obs_objects):
            sub = np.asarray(obj.observe()).flatten()
            chunks.append(sub)
        out = np.concatenate(chunks, axis=0) if chunks else np.array([], dtype=self._base_low.dtype)
        if self.cast_dtype is not None and out.dtype != self.cast_dtype:
            out = out.astype(self.cast_dtype, copy=False)
        return out

    # ══════════════════════════════════════════════════════════════════════════
    # Convenience Methods
    # ══════════════════════════════════════════════════════════════════════════

    def split(self, flat_obs: np.ndarray, obs_name: str, reshape: bool = True) -> np.ndarray:
        """Split observation by modality name and optionally reshape."""
        s, e = self.mapping[obs_name]
        part = flat_obs[s:e]
        return part.reshape(self.shapes[obs_name]) if reshape else part
