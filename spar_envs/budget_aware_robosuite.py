"""
Budget-Aware Robosuite Wrapper

Similar to BudgetAwareHighway, this wrapper adds modality-level sensor masking
to robosuite environments with per-modality costs.

Action space:
    - If use_all_obs=True: env.action_space (no masking)
    - Else: Tuple(env.action_space, MultiBinary(M)), M = number of modalities

Observation modalities (for robosuite):
    - Robot Proprioception: joint positions, velocities, gripper state, end-effector pose
    - Object States: object positions, orientations
    - Task Features: gripper-to-object distances, task-specific observations
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any

from base_envs import robosuite as suite
from base_envs.robosuite.wrappers import GymWrapper

# OmniSafe CMDP registration
from omnisafe.envs.core import env_register
from spar_envs.budget_aware_base import BudgetAwareBase


@env_register
class BudgetAwareRobosuite(BudgetAwareBase):
    """Robosuite wrapper with SPAR-style modality-level masking for OmniSafe.

    Action space:
        - If use_all_obs=True: env.action_space (continuous, no masking).
        - Else: Tuple(env.action_space, MultiBinary(M)), M = number of modalities.

    Mask is applied to the returned observation each step (i.e., mask_t gates obs_{t+1}).
    Costs are per-modality.
    """

    # OmniSafe discovery helpers
    _support_envs = [
        "budget-aware-Lift",
        "budget-aware-Door",
    ]
    need_auto_reset_wrapper = True
    need_time_limit_wrapper = True

    def __init__(
            self,
            env_id: Optional[str] = None,
            robot: str = "Panda",
            controller_configs: Optional[Dict] = None,
            modality_groups: Optional[Dict[str, List[str]]] = None,
            modality_costs: Optional[Dict[str, float]] = None,
            seed: int = 42,
            *,
            use_all_obs: bool = False,
            cast_dtype: np.dtype = np.float32,
            max_episode_steps: Optional[int] = None,
            sensor_dropout_rescale: bool = True,
            use_camera: bool = True,
            camera_name: str = "agentview",
            camera_height: int = 16,
            camera_width: int = 16,
            **kwargs: Any,
    ):
        """
        Initialize Budget-Aware Robosuite environment.

        Args:
            env_id: Environment name (e.g., "budget-aware-lift")
            robot: Robot type (default: "Panda")
            controller_configs: Controller configuration (None uses default OSC_POSE)
            modality_groups: Dict mapping modality names to list of observation keys
            modality_costs: Dict mapping modality names to costs (default: all 1.0)
            seed: Random seed
            use_all_obs: If True, no masking (all observations always available)
            cast_dtype: Data type for observations
            max_episode_steps: Episode horizon (default: 500 for robosuite benchmark)
            **kwargs: Additional arguments passed to robosuite.make()
        """
        if env_id is None:
            raise ValueError("env_id must be provided")

        # Strip "budget-aware-" prefix to get base environment name
        base_id = env_id.replace("budget-aware-", "").capitalize()

        # Filter out OmniSafe-specific parameters
        env_kwargs = kwargs.copy()
        num_envs = env_kwargs.pop('num_envs', 1)
        device = env_kwargs.pop('device', None)
        env_kwargs.pop('render_mode', None)

        # Store camera settings
        self.use_camera = use_camera
        self.camera_name = camera_name
        self.camera_height = camera_height
        self.camera_width = camera_width
        # Grayscale camera: H × W (not H × W × 3)
        self._camera_dim = camera_height * camera_width if use_camera else 0

        # Set default parameters for robosuite
        horizon = max_episode_steps or 500
        env_kwargs.setdefault('horizon', horizon)
        env_kwargs.setdefault('use_camera_obs', use_camera)
        env_kwargs.setdefault('use_object_obs', True)
        env_kwargs.setdefault('has_renderer', False)

        # Only set has_offscreen_renderer default if not explicitly provided
        if 'has_offscreen_renderer' not in env_kwargs:
            env_kwargs['has_offscreen_renderer'] = use_camera

        env_kwargs.setdefault('reward_shaping', True)
        env_kwargs.setdefault('control_freq', 20)

        # Camera configuration
        if use_camera:
            env_kwargs.setdefault('camera_names', camera_name)
            env_kwargs.setdefault('camera_heights', camera_height)
            env_kwargs.setdefault('camera_widths', camera_width)
            env_kwargs.setdefault('camera_depths', False)

        # Create base robosuite environment
        robosuite_env = suite.make(
            env_name=base_id,
            robots=robot,
            controller_configs=controller_configs,
            **env_kwargs,
        )

        # Get all available observation keys from the environment
        obs_spec = robosuite_env.observation_spec()
        all_obs_keys = list(obs_spec.keys())

        # Filter keys: keep robot proprio-state, individual object/task keys, exclude object-state
        # Include camera images only if use_camera=True
        filtered_keys = []
        for key in all_obs_keys:
            # Handle camera images based on use_camera flag
            if 'image' in key:
                if use_camera:
                    filtered_keys.append(key)
                continue
            # Keep robot proprio-state (pre-aggregated)
            if key.endswith('proprio-state'):
                filtered_keys.append(key)
            if 'robot' in key:
                continue
            # Skip object-state (we'll use individual keys instead)
            elif key == 'object-state':
                continue
            # Keep all other individual observation keys
            elif not key.endswith('-state'):
                filtered_keys.append(key)

        # Build or use provided modality groups
        if modality_groups is None:
            # Get sample observation to determine modality groups
            temp_obs_dict = robosuite_env._get_observations()
            modality_groups = BudgetAwareRobosuite._auto_group_observations_static(temp_obs_dict)

        # Wrap with GymWrapper, passing key_groups for modality-grouped flattening
        gym_env = GymWrapper(robosuite_env, keys=filtered_keys, flatten_obs=True, key_groups=modality_groups)

        # Store references for _build_modalities
        self.robosuite_env = robosuite_env
        self.modality_groups = modality_groups
        self.gym_env = gym_env
        self.obs_keys = gym_env.keys

        # Initialize base class
        super().__init__(
            env=gym_env,
            use_all_obs=use_all_obs,
            sensor_dropout_rescale=sensor_dropout_rescale,
            cast_dtype=cast_dtype,
            max_episode_steps=horizon,
            modality_costs=modality_costs,
            num_envs=num_envs,
            device=device,
            seed=seed,
        )

        # Set metadata for gymnasium compatibility
        self.metadata = {
            'render_fps': int(robosuite_env.control_freq),
            'render_modes': ['rgb_array'],
        }

    # ══════════════════════════════════════════════════════════════════════════
    # Abstract Method Implementations (Robosuite-specific)
    # ══════════════════════════════════════════════════════════════════════════

    def _build_modalities(self) -> Dict[str, Any]:
        """Build modality groups from robosuite observations.

        NOTE: GymWrapper now iterates by key_groups, guaranteeing contiguous modality ranges.
        """
        # Get a sample observation to determine sizes
        temp_obs_dict = self.robosuite_env._get_observations()

        # modality_groups was already built in __init__
        # Build mapping with contiguous ranges (guaranteed by GymWrapper's key_groups iteration)
        mapping: Dict[str, Tuple[int, int]] = {}
        mod_sizes: Dict[str, int] = {}
        self._obs_keys_by_modality: Dict[str, List[str]] = {}

        # Iterate through modality groups in order (matches GymWrapper._flatten_obs iteration)
        cursor = 0
        for mod_name, mod_keys in self.modality_groups.items():
            # Special handling for camera modality
            if mod_name == "camera":
                if self.use_camera and self._camera_dim > 0:
                    modality_start = cursor
                    # Camera is flattened grayscale: H × W
                    cursor += self._camera_dim
                    mapping[mod_name] = (modality_start, cursor)
                    mod_sizes[mod_name] = self._camera_dim
                    self._obs_keys_by_modality[mod_name] = mod_keys
                continue

            # Calculate size for non-camera modalities by summing all their keys
            modality_start = cursor
            keys_in_modality = []

            for key in mod_keys:
                # Skip keys not in obs_keys or temp_obs_dict
                if key not in self.obs_keys or key not in temp_obs_dict:
                    continue

                # Get size of this observation
                val = temp_obs_dict[key]
                if isinstance(val, np.ndarray):
                    key_size = val.size
                elif isinstance(val, (int, float)):
                    key_size = 1
                else:
                    continue

                keys_in_modality.append(key)
                cursor += key_size

            # Store mapping for this modality (if non-empty)
            if keys_in_modality:
                mapping[mod_name] = (modality_start, cursor)
                mod_sizes[mod_name] = cursor - modality_start
                self._obs_keys_by_modality[mod_name] = keys_in_modality

        # Final modality names (only non-empty ones, in order)
        obs_names = list(mapping.keys())

        # Get bounds from GymWrapper's observation space
        base_low = self.gym_env.observation_space.low
        base_high = self.gym_env.observation_space.high

        if self.cast_dtype is not None:
            base_low = base_low.astype(self.cast_dtype, copy=False)
            base_high = base_high.astype(self.cast_dtype, copy=False)

        # Override bounds for camera modality: [0.0, 1.0] (normalized grayscale)
        if "camera" in mapping:
            cam_start, cam_end = mapping["camera"]
            base_low[cam_start:cam_end] = 0.0
            base_high[cam_start:cam_end] = 1.0

        return {
            'mapping': mapping,
            'mod_sizes': mod_sizes,
            'obs_names': obs_names,
            'base_low': base_low,
            'base_high': base_high,
            'flat_dim': base_low.size,
        }

    def _get_raw_observation(self) -> np.ndarray:
        """Get raw observation from robosuite environment (including camera if enabled)."""
        # Get observation dict from robosuite and flatten using GymWrapper
        obs_dict = self.robosuite_env._get_observations()
        flat = np.asarray(self.gym_env._flatten_obs(obs_dict))

        return flat

    # ══════════════════════════════════════════════════════════════════════════
    # Robosuite-Specific Methods
    # ══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _auto_group_observations_static(obs_dict: Dict[str, np.ndarray]) -> Dict[str, List[str]]:
        """Automatically group observations into 4 modalities (static version).

        Strategy:
        - robot_proprioception: Use robot{idx}_proprio-state (pre-aggregated by robosuite)
        - object_states: Individual object observation keys (pos, quat, qpos)
        - task_features: Individual task-related keys (gripper_to_*, *_to_*)
        - camera: Camera image observations (agentview, etc.)

        Note: We explicitly exclude object-state because it bundles both object_states
        and task_features together, which we want to keep separate.
        """
        groups = {
            "robot_proprioception": [],
            "object_states": [],
            "task_features": [],
            "camera": [],
        }

        # Step 1: Assign robot proprio-state (pre-aggregated)
        assigned_keys = set()
        for key in obs_dict.keys():
            if key.endswith('proprio-state'):
                groups["robot_proprioception"].append(key)
                assigned_keys.add(key)

        # Step 2: Assign camera images
        for key in obs_dict.keys():
            if 'image' in key:
                groups["camera"].append(key)
                assigned_keys.add(key)

        # Step 3: Assign individual object and task keys
        for key in obs_dict.keys():
            # Skip already assigned keys and any remaining -state keys
            if key in assigned_keys or key.endswith('-state'):
                continue

            # Classify based on key patterns
            if BudgetAwareRobosuite._is_task_feature_key(key):
                groups["task_features"].append(key)
            elif BudgetAwareRobosuite._is_object_state_key(key):
                groups["object_states"].append(key)

        # Return only non-empty groups
        return {k: v for k, v in groups.items() if v}

    def _auto_group_observations(self, obs_dict: Dict[str, np.ndarray]) -> Dict[str, List[str]]:
        """Instance method wrapper for _auto_group_observations_static."""
        return self._auto_group_observations_static(obs_dict)

    @staticmethod
    def _is_task_feature_key(key: str) -> bool:
        """Check if key represents a task feature (e.g., gripper_to_cube_pos)."""
        return any(pattern in key for pattern in ["to_", "_to_"])

    @staticmethod
    def _is_object_state_key(key: str) -> bool:
        """Check if key represents object state data (pos, quat, qpos).

        Excludes robot-related keys (joint, eef, gripper).
        """
        # Must have position/orientation markers
        has_state_marker = any(marker in key for marker in ["_pos", "_quat", "_qpos"])
        # Must not be robot-related
        is_not_robot = not any(robot_keyword in key for robot_keyword in ["joint", "eef", "gripper"])

        return has_state_marker and is_not_robot

    def render(self, mode='rgb_array', **kwargs):
        """Override render to use sim.render() for offscreen rendering.

        When has_offscreen_renderer=True but has_renderer=False (evaluation mode),
        robosuite doesn't create a viewer, so env.render() fails. Instead, we use
        sim.render() directly which uses MuJoCo's offscreen renderer.
        """
        # Check if using offscreen rendering without viewer
        has_offscreen = getattr(self.robosuite_env, 'has_offscreen_renderer', False)
        has_viewer = getattr(self.robosuite_env, 'has_renderer', False)

        if has_offscreen and not has_viewer:
            # Use sim.render() for offscreen rendering (evaluation video recording)
            # Determine camera parameters
            camera_name = kwargs.get('camera_name', 'frontview')
            if hasattr(self.robosuite_env, 'camera_names') and self.robosuite_env.camera_names:
                camera_name = self.robosuite_env.camera_names[0]

            height = kwargs.get('height', 256)
            if hasattr(self.robosuite_env, 'camera_heights') and self.robosuite_env.camera_heights:
                height = self.robosuite_env.camera_heights[0]

            width = kwargs.get('width', 256)
            if hasattr(self.robosuite_env, 'camera_widths') and self.robosuite_env.camera_widths:
                width = self.robosuite_env.camera_widths[0]

            # Render using MuJoCo sim offscreen renderer
            pixels = self.robosuite_env.sim.render(
                width=width,
                height=height,
                camera_name=camera_name,
            )
            # sim.render returns image upside down, flip it
            return pixels[::-1, :, :]
        else:
            # Use default viewer rendering (training mode or on-screen rendering)
            return self.env.render()
