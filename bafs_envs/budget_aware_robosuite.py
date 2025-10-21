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
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Optional, Tuple, Any, Union
import random
import torch

import robosuite as suite
from robosuite.wrappers import GymWrapper

# OmniSafe CMDP registration
from omnisafe.envs.core import CMDP, env_register


@env_register
class BudgetAwareRobosuite(gym.Wrapper, CMDP):
    """Robosuite wrapper with BAFS-style modality-level masking for OmniSafe.

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

    # Default modality groupings for robosuite
    DEFAULT_MODALITIES = {
        "robot_proprioception": [
            "robot0_joint_pos",
            "robot0_joint_vel",
            "robot0_eef_pos",
            "robot0_eef_quat",
            "robot0_gripper_qpos",
        ],
        "object_states": [
            "cube_pos",
            "cube_quat",
            "door_pos",
            "handle_pos",
            "hinge_qpos",
            "handle_qpos",
        ],
        "task_features": [
            "gripper_to_cube_pos",
            "handle_to_eef_pos",
            "door_to_eef_pos",
        ],
    }

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
        use_camera: bool = False,
        camera_name: str = "agentview",
        camera_height: int = 32,
        camera_width: int = 32,
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
        if 'num_envs' in env_kwargs:
            self._num_envs = env_kwargs.pop('num_envs')
        else:
            self._num_envs = 1

        if 'device' in env_kwargs:
            self._device = env_kwargs.pop('device')
        else:
            self._device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        # Filter out gymnasium-specific parameters that robosuite doesn't accept
        env_kwargs.pop('render_mode', None)

        # Store camera settings
        self.use_camera = use_camera
        self.camera_name = camera_name
        self.camera_height = camera_height
        self.camera_width = camera_width
        self._camera_dim = camera_height * camera_width if use_camera else 0

        # Set default parameters for robosuite
        horizon = max_episode_steps or 500
        env_kwargs.setdefault('horizon', horizon)
        env_kwargs.setdefault('use_camera_obs', use_camera)
        env_kwargs.setdefault('use_object_obs', True)
        env_kwargs.setdefault('has_renderer', False)

        # Only set has_offscreen_renderer default if not explicitly provided
        # This allows evaluator to override for video recording during evaluation only
        if 'has_offscreen_renderer' not in env_kwargs:
            env_kwargs['has_offscreen_renderer'] = use_camera

        env_kwargs.setdefault('reward_shaping', True)
        env_kwargs.setdefault('control_freq', 20)

        # Camera configuration
        if use_camera:
            env_kwargs.setdefault('camera_names', camera_name)
            env_kwargs.setdefault('camera_heights', camera_height)
            env_kwargs.setdefault('camera_widths', camera_width)
            env_kwargs.setdefault('camera_depths', False)  # No depth, just RGB

        # Create base robosuite environment
        robosuite_env = suite.make(
            env_name=base_id,
            robots=robot,
            controller_configs=controller_configs,
            **env_kwargs,
        )

        # Get all available observation keys from the environment
        # This ensures we include task-specific features (e.g., cube position in Lift)
        obs_spec = robosuite_env.observation_spec()
        all_obs_keys = list(obs_spec.keys())

        # Wrap with GymWrapper to make it gymnasium-compatible
        # Use all available keys (not just defaults) and flatten_obs=True
        gym_env = GymWrapper(robosuite_env, keys=all_obs_keys, flatten_obs=True)

        super().__init__(gym_env)

        # ── User configuration ─────────────────────────────────────────────────
        self.use_all_obs = bool(use_all_obs)
        self.cast_dtype = cast_dtype
        self._max_episode_steps = horizon
        self.seed_value = seed
        self.sensor_dropout_rescale = bool(sensor_dropout_rescale)

        # ── Build modality groups from GymWrapper's keys ──────────────────────
        # GymWrapper stores the keys it uses in self.env.keys
        obs_keys = gym_env.keys  # List of observation keys in flattened order

        # Get a sample observation to determine sizes
        temp_obs_dict = robosuite_env._get_observations()

        # Use provided modality groups or build default
        if modality_groups is None:
            modality_groups = self._auto_group_observations(temp_obs_dict)

        self.modality_groups = modality_groups

        # Build mapping: track which keys belong to which modality
        # and their indices in the flattened observation
        self.mapping: Dict[str, Tuple[int, int]] = {}
        self.mod_sizes: Dict[str, int] = {}
        self._obs_keys_by_modality: Dict[str, List[str]] = {}

        cursor = 0
        for mod_name, mod_keys in modality_groups.items():
            # Find keys that exist in both the modality group and GymWrapper's keys
            existing_keys = []
            mod_size = 0

            for key in mod_keys:
                # Only include keys that GymWrapper is actually using
                if key in obs_keys and key in temp_obs_dict:
                    val = temp_obs_dict[key]
                    if isinstance(val, np.ndarray):
                        existing_keys.append(key)
                        mod_size += val.size

            if mod_size > 0:
                self._obs_keys_by_modality[mod_name] = existing_keys
                self.mapping[mod_name] = (cursor, cursor + mod_size)
                self.mod_sizes[mod_name] = mod_size
                cursor += mod_size

        # Add camera modality if enabled
        if self.use_camera:
            self._obs_keys_by_modality["camera"] = [f"{camera_name}_image"]
            self.mapping["camera"] = (cursor, cursor + self._camera_dim)
            self.mod_sizes["camera"] = self._camera_dim
            cursor += self._camera_dim

        # Store robosuite_env reference for camera access
        self.robosuite_env = robosuite_env

        # Final modality names (only non-empty ones)
        self.obs_names = list(self.mapping.keys())
        self._num_modalities = len(self.obs_names)

        # Get bounds from GymWrapper's observation space
        base_low = gym_env.observation_space.low
        base_high = gym_env.observation_space.high

        if self.cast_dtype is not None:
            base_low = base_low.astype(self.cast_dtype, copy=False)
            base_high = base_high.astype(self.cast_dtype, copy=False)

        # Expand observation space to include camera
        if self.use_camera:
            # Camera pixels are in [0, 255] range, normalize to [0, 1]
            camera_low = np.zeros(self._camera_dim, dtype=base_low.dtype)
            camera_high = np.ones(self._camera_dim, dtype=base_high.dtype)
            base_low = np.concatenate([base_low, camera_low], axis=0)
            base_high = np.concatenate([base_high, camera_high], axis=0)

        self._base_low = base_low
        self._base_high = base_high
        self._flat_dim = base_low.size

        # Observation space: [flat_obs, modality_mask]
        low = np.concatenate([base_low, np.zeros(self._num_modalities, dtype=base_low.dtype)], axis=0)
        high = np.concatenate([base_high, np.ones(self._num_modalities, dtype=base_high.dtype)], axis=0)
        self.observation_space = spaces.Box(low=low, high=high, dtype=low.dtype)

        # ── Action space ──────────────────────────────────────────────────────
        self.original_action_space = self.env.action_space
        self.is_continuous = isinstance(self.env.action_space, gym.spaces.Box)

        if self.use_all_obs:
            new_action_space = self.env.action_space
        else:
            env_action_space = self.env.action_space
            sensor_mask_space = gym.spaces.MultiBinary(self._num_modalities)

            # Store references
            self.cont_act_space = env_action_space
            self.disc_act_space = sensor_mask_space

            new_action_space = gym.spaces.Tuple((env_action_space, sensor_mask_space))

        self.action_space = new_action_space
        self._action_space = new_action_space

        # ── Per-modality costs ────────────────────────────────────────────────
        if modality_costs is None:
            modality_costs = {n: 1.0 for n in self.obs_names}
        self._modality_costs = {n: float(modality_costs.get(n, 1.0)) for n in self.obs_names}
        self.costs = np.array([self._modality_costs[n] for n in self.obs_names], dtype=np.float32)

        # Set metadata for gymnasium compatibility (robosuite's GymWrapper sets metadata=None)
        # Use control_freq as render_fps since robosuite doesn't have separate rendering fps
        self.metadata = {
            'render_fps': int(robosuite_env.control_freq),
            'render_modes': ['rgb_array'],
        }

        self.set_seed(seed)

    def _auto_group_observations(self, obs_dict: Dict[str, np.ndarray]) -> Dict[str, List[str]]:
        """Automatically group observations into modalities based on naming patterns."""
        groups = {
            "robot_proprioception": [],
            "object_states": [],
            "task_features": [],
        }

        for key in obs_dict.keys():
            if any(x in key for x in ["joint_pos", "joint_vel", "eef_pos", "eef_quat", "gripper_qpos"]):
                groups["robot_proprioception"].append(key)
            elif any(x in key for x in ["cube_pos", "cube_quat", "door_pos", "handle_pos", "hinge_qpos", "handle_qpos"]):
                groups["object_states"].append(key)
            elif any(x in key for x in ["to_", "_to_"]):
                groups["task_features"].append(key)
            else:
                # Default: put in task_features
                groups["task_features"].append(key)

        # Remove empty groups
        return {k: v for k, v in groups.items() if v}

    @property
    def max_episode_steps(self) -> int:
        return self._max_episode_steps

    @max_episode_steps.setter
    def max_episode_steps(self, value):
        self._max_episode_steps = value

    def set_seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    # ── Observation handling ──────────────────────────────────────────────────
    def _to_numpy(self, x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    def _get_camera_obs(self) -> np.ndarray:
        """Get grayscale camera observation from robosuite environment.

        Returns:
            Flattened grayscale image normalized to [0, 1], shape: (camera_height * camera_width,)
        """
        if not self.use_camera:
            return np.array([], dtype=np.float32)

        # Get observations from robosuite
        obs_dict = self.robosuite_env._get_observations()

        # Get camera image (RGB, shape: (H, W, 3))
        camera_key = f"{self.camera_name}_image"
        if camera_key not in obs_dict:
            raise KeyError(f"Camera observation '{camera_key}' not found in environment observations")

        rgb_image = obs_dict[camera_key]

        # Convert RGB to grayscale using standard luminosity weights
        # Y = 0.299*R + 0.587*G + 0.114*B
        grayscale = (0.299 * rgb_image[:, :, 0] +
                    0.587 * rgb_image[:, :, 1] +
                    0.114 * rgb_image[:, :, 2])

        # Normalize to [0, 1]
        grayscale = grayscale / 255.0

        # Flatten and cast to dtype
        grayscale_flat = grayscale.flatten().astype(self.cast_dtype)

        return grayscale_flat

    def _expand_modality_mask_to_features(self, m_mod: np.ndarray) -> np.ndarray:
        """Convert modality mask (len M) -> feature-level mask (len flat_dim)."""
        m_feat = np.zeros(self._flat_dim, dtype=np.float32)
        for i, name in enumerate(self.obs_names):
            s, e = self.mapping[name]
            m_feat[s:e] = float(m_mod[i])
        return m_feat

    def _apply_mask(self, flat: np.ndarray, feat_mask01: np.ndarray, m_mod01: np.ndarray) -> np.ndarray:
        """Gating with optional rescaling.

        Args:
            flat: Flattened observation (feature-level)
            feat_mask01: Feature-level binary mask (expanded from modality mask)
            m_mod01: Modality-level binary mask

        Returns:
            Gated (and optionally rescaled) observation
        """
        feat_mask01 = feat_mask01.astype(flat.dtype, copy=False)
        gated = flat * feat_mask01

        if self.sensor_dropout_rescale:
            # Count active modalities
            n_active = np.sum(m_mod01 > 0.5)
            if n_active > 0:
                # Rescale to maintain consistent signal magnitude
                rescale_factor = self._num_modalities / n_active
                gated = gated * rescale_factor

        return gated

    def _mask_cost(self, m_mod01: np.ndarray) -> float:
        """Calculate cost based on active modalities."""
        total = 0.0
        for i, name in enumerate(self.obs_names):
            if m_mod01[i] > 0.5:
                total += self._modality_costs.get(name, 1.0)
        return float(total)

    # ── CMDP / Gymnasium API ──────────────────────────────────────────────────
    def reset(self, **kwargs):
        # Reset the gymnasium-wrapped env (returns flattened observation)
        flat, info = self.env.reset(**kwargs)
        flat = np.asarray(flat)

        # Append camera observation if enabled
        if self.use_camera:
            camera_obs = self._get_camera_obs()
            flat = np.concatenate([flat, camera_obs], axis=0)

        # Append mask (all ones at reset)
        mask0 = np.ones(self._num_modalities, dtype=flat.dtype)
        out = np.concatenate([flat, mask0], axis=0).astype(self.observation_space.dtype, copy=False)
        out = torch.as_tensor(out, dtype=torch.float32).to(self._device)
        info['unmasked_observation'] = out.clone()
        info['sensor_mask'] = mask0
        return out, info

    def step(self, action: Union[Tuple[Any, Any], Any]):
        if self.use_all_obs:
            # No masking mode
            inner = self._to_numpy(action)
            flat, r, term, trunc, info = self.env.step(inner)
            flat = np.asarray(flat)

            # Append camera observation if enabled
            if self.use_camera:
                camera_obs = self._get_camera_obs()
                flat = np.concatenate([flat, camera_obs], axis=0)

            m_mod01 = np.ones(self._num_modalities, dtype=flat.dtype)
            out = np.concatenate([flat, m_mod01.astype(flat.dtype)], axis=0).astype(self.observation_space.dtype)

            rew = float(r)
            cost = self._mask_cost(m_mod01)
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

        # Masking mode: expect Tuple((env_action, modality_mask))
        if not (isinstance(action, (tuple, list)) and len(action) == 2):
            raise ValueError("Action must be Tuple((env_action, modality_mask)) when use_all_obs=False")

        env_action = self._to_numpy(action[0])
        mod_mask = self._to_numpy(action[1]).astype(np.int64).reshape(-1)

        # Step base environment (returns flattened observation)
        flat, r, term, trunc, info = self.env.step(env_action)
        flat = np.asarray(flat)

        # Append camera observation if enabled
        if self.use_camera:
            camera_obs = self._get_camera_obs()
            flat = np.concatenate([flat, camera_obs], axis=0)

        # Build next observation and apply gating (with optional rescaling)
        mod_mask_float = mod_mask.astype(np.float32)
        feat_mask01 = self._expand_modality_mask_to_features(mod_mask_float)
        gated = self._apply_mask(flat, feat_mask01, mod_mask_float)

        # Cost
        step_cost = self._mask_cost(mod_mask.astype(np.float32))

        # Compose final observation
        out = np.concatenate(
            [gated, mod_mask.astype(gated.dtype)], axis=0
        ).astype(self.observation_space.dtype, copy=False)

        # Info & returns (CMDP)
        info = dict(info)
        info["sensor_mask"] = mod_mask.astype(np.float32)
        info["original_step_reward"] = float(r)
        info["original_step_cost"] = float(step_cost)

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

    def create_random_action(self) -> Union[Any, Tuple[Any, np.ndarray]]:
        """Create a random action."""
        env_action = self.original_action_space.sample()

        if self.use_all_obs:
            return torch.as_tensor(env_action, dtype=torch.float32).to(self._device)
        else:
            # Random binary mask (ensure at least one modality is active)
            modality_mask = np.random.randint(0, 2, size=self._num_modalities, dtype=np.int32)
            if modality_mask.sum() == 0:
                active_idx = np.random.randint(0, self._num_modalities)
                modality_mask[active_idx] = 1

            return (
                torch.as_tensor(env_action, dtype=torch.float32).to(self._device),
                torch.as_tensor(modality_mask, dtype=torch.float32).to(self._device),
            )

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

    # Convenience properties
    @property
    def obs_mapping(self) -> Dict[str, Tuple[int, int]]:
        return dict(self.mapping)

    @property
    def obs_modalities(self) -> List[str]:
        return list(self.obs_names)


# # ============================================================================
# # TESTING / DEMO
# # ============================================================================
#
# def test_budget_aware_robosuite():
#     """Test the BudgetAwareRobosuite environment."""
#
#     print("\n" + "=" * 80)
#     print("TESTING BUDGET-AWARE ROBOSUITE ENVIRONMENT")
#     print("=" * 80 + "\n")
#
#     # Test configuration
#     env_config = {
#         "env_id": "budget-aware-lift",
#         "robot": "Panda",
#         "modality_costs": {
#             "robot_proprioception": 0.5,
#             "object_states": 1.0,
#             "task_features": 0.3,
#         },
#         "use_all_obs": False,  # Enable masking
#         "max_episode_steps": 50,
#         "seed": 42,
#     }
#
#     print("Creating environment with configuration:")
#     for key, value in env_config.items():
#         print(f"  {key}: {value}")
#     print()
#
#     # Create environment
#     env = BudgetAwareRobosuite(**env_config)
#
#     # Display environment info
#     print("=" * 80)
#     print("ENVIRONMENT INFO")
#     print("=" * 80)
#     print(f"Observation space: {env.observation_space.shape}")
#     print(f"Action space: {env.action_space}")
#     print(f"Original action space: {env.original_action_space}")
#     print(f"Number of modalities: {env._num_modalities}")
#     print(f"Modality names: {env.obs_names}")
#     print()
#
#     print("Modality mapping and costs:")
#     for mod_name, (start, end) in env.mapping.items():
#         size = end - start
#         cost = env._modality_costs[mod_name]
#         obs_keys = env._obs_keys_by_modality[mod_name]
#         print(f"  {mod_name:30s}: dims [{start:3d}:{end:3d}]  size={size:3d}  cost={cost:.2f}")
#         print(f"    Contains: {', '.join(obs_keys)}")
#     print()
#
#     print(f"Total flattened observation dimension: {env._flat_dim}")
#     print(f"Total observation dimension (with mask): {env.observation_space.shape[0]}")
#     print(f"  = {env._flat_dim} (obs) + {env._num_modalities} (mask)")
#     print()
#
#     # Run test episode
#     print("=" * 80)
#     print("RUNNING TEST EPISODE (5 steps)")
#     print("=" * 80 + "\n")
#
#     obs, info = env.reset()
#     print(f"Reset observation shape: {obs.shape}")
#     print(f"Initial sensor mask: {info['sensor_mask']}")
#     print()
#
#     total_reward = 0.0
#     total_cost = 0.0
#
#     for step in range(5):
#         # Random action
#         action = env.create_random_action()
#
#         if not env.use_all_obs:
#             env_action, modality_mask = action
#             print(f"Step {step + 1}:")
#             print(f"  Environment action: {env_action.cpu().numpy()}")
#             print(f"  Modality mask: {modality_mask.cpu().numpy().astype(int)}")
#             active_mods = [env.obs_names[i] for i in range(len(modality_mask)) if modality_mask[i] > 0.5]
#             print(f"  Active modalities: {active_mods}")
#         else:
#             print(f"Step {step + 1}: Action: {action.cpu().numpy()}")
#
#         # Step environment
#         obs, reward, cost, terminated, truncated, info = env.step(action)
#
#         print(f"  Reward: {reward.item():.4f}, Cost: {cost.item():.4f}")
#         print(f"  Terminated: {terminated.item()}, Truncated: {truncated.item()}")
#         print()
#
#         total_reward += reward.item()
#         total_cost += cost.item()
#
#         if terminated or truncated:
#             print("Episode ended early!")
#             break
#
#     print("=" * 80)
#     print("EPISODE SUMMARY")
#     print("=" * 80)
#     print(f"Total reward: {total_reward:.4f}")
#     print(f"Total cost: {total_cost:.4f}")
#     print(f"Average cost per step: {total_cost / max(1, step + 1):.4f}")
#     print()
#
#     env.close()
#
#
# def test_use_all_obs_mode():
#     """Test the environment with use_all_obs=True (no masking)."""
#
#     print("\n" + "=" * 80)
#     print("TESTING USE_ALL_OBS MODE (No Masking)")
#     print("=" * 80 + "\n")
#
#     env = BudgetAwareRobosuite(
#         env_id="budget-aware-door",
#         robot="Panda",
#         use_all_obs=True,  # No masking
#         max_episode_steps=50,
#         seed=42,
#     )
#
#     print(f"Environment: Door Opening Task")
#     print(f"Action space: {env.action_space} (continuous, no mask)")
#     print(f"Observation space: {env.observation_space.shape}")
#     print(f"Modalities: {env.obs_names}")
#     print()
#
#     obs, info = env.reset()
#     print(f"Reset observation shape: {obs.shape}")
#     print(f"Initial sensor mask (all ones): {info['sensor_mask']}")
#     print()
#
#     # In use_all_obs mode, cost should be sum of all modality costs
#     expected_cost = sum(env._modality_costs.values())
#     print(f"Expected cost per step (all modalities active): {expected_cost:.4f}")
#     print()
#
#     # Take a few random steps
#     total_cost = 0.0
#     for step in range(5):
#         action = env.create_random_action()
#         print(f"Step {step + 1}:")
#         print(f"  Action: {action.cpu().numpy()}")
#
#         obs, reward, cost, terminated, truncated, info = env.step(action)
#         print(f"  Reward: {reward.item():.4f}, Cost: {cost.item():.4f}")
#
#         # Verify cost matches expected
#         assert np.isclose(cost.item(), expected_cost), f"Cost mismatch: {cost.item()} != {expected_cost}"
#         print(f"  ✓ Cost matches expected (all modalities)")
#         print()
#
#         total_cost += cost.item()
#
#         if terminated or truncated:
#             break
#
#     print(f"Total cost over {step + 1} steps: {total_cost:.4f}")
#     env.close()
#
#
# def compare_observation_modalities():
#     """Compare observations with different modality masks."""
#
#     print("\n" + "=" * 80)
#     print("COMPARING MASKED VS UNMASKED OBSERVATIONS")
#     print("=" * 80 + "\n")
#
#     env = BudgetAwareRobosuite(
#         env_id="budget-aware-lift",
#         robot="Panda",
#         use_all_obs=False,
#         max_episode_steps=50,
#         seed=42,
#     )
#
#     obs, info = env.reset()
#
#     # Get unmasked observation (all modalities)
#     unmasked_obs = info['unmasked_observation'].cpu().numpy()
#     flat_obs = unmasked_obs[:-env._num_modalities]  # Remove mask suffix
#
#     print("Unmasked observation breakdown:")
#     for mod_name in env.obs_names:
#         start, end = env.mapping[mod_name]
#         mod_obs = flat_obs[start:end]
#         print(f"  {mod_name:30s}: shape={mod_obs.shape}, sample={mod_obs[:5]}")
#     print()
#
#     # Create action with specific mask pattern
#     # Test 1: Only robot proprioception
#     env_action = env.original_action_space.sample()
#     mask_robot_only = np.zeros(env._num_modalities, dtype=np.float32)
#     mask_robot_only[0] = 1  # Assuming first modality is robot_proprioception
#
#     action = (
#         torch.as_tensor(env_action, dtype=torch.float32).to(env._device),
#         torch.as_tensor(mask_robot_only, dtype=torch.float32).to(env._device),
#     )
#
#     obs, reward, cost, terminated, truncated, info = env.step(action)
#     masked_obs = obs.cpu().numpy()
#     flat_masked = masked_obs[:-env._num_modalities]
#
#     print(f"With mask {mask_robot_only.astype(int)} (robot only):")
#     print(f"  Cost: {cost.item():.4f}")
#     for mod_name in env.obs_names:
#         start, end = env.mapping[mod_name]
#         mod_obs = flat_masked[start:end]
#         is_zero = np.allclose(mod_obs, 0)
#         print(f"  {mod_name:30s}: {'MASKED (all zeros)' if is_zero else 'ACTIVE'}")
#     print()
#
#     env.close()
#
#
# if __name__ == "__main__":
#     # Run all tests
#     test_budget_aware_robosuite()
#     test_use_all_obs_mode()
#     compare_observation_modalities()
#
#     print("\n" + "=" * 80)
#     print("ALL TESTS COMPLETED SUCCESSFULLY!")
#     print("=" * 80 + "\n")
