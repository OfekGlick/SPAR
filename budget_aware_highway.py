import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Optional, Tuple, Any, Union
import omnisafe
import random

import torch


from highway_env.envs.common.observation import observation_factory

# OmniSafe CMDP registration
from omnisafe.envs.core import CMDP, env_register


@env_register
class BudgetAwareHighway(gym.Wrapper, CMDP):
    """Highway-env wrapper with BAFS-style *modality-level* masking for OmniSafe.

    Action space:
        - If use_all_obs=True: env.action_space (no masking).
        - Else: Tuple( env.action_space, MultiBinary(M) ), M = number of modalities.

    Mask is applied to the *returned* observation each step (i.e., mask_t gates obs_{t+1}).
    Costs are per-modality. We do NOT expose a feature-level masking mode.
    """

    # OmniSafe discovery helpers (used by make()/Evaluator)
    _support_envs = [
        # You can register multiple aliases; "BudgetAware" prefix will be stripped.
        "budget-aware-highway-fast-v0",
        "budget-aware-highway-v0",
        "budget-aware-merge-v0",
        "budget-aware-roundabout-v0",
        "budget-aware-parking-v0",
        "budget-aware-intersection-v0",
        "budget-aware-intersection-v1",
    ]
    need_auto_reset_wrapper = True
    need_time_limit_wrapper = True

    DEFAULT_TYPES = [
        "Kinematics",
        "LidarObservation",
        "OccupancyGrid",  # Discrete?
        "TimeToCollision",  # Discrete?
        # "GrayscaleObservation",
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
            append_mask_to_obs: bool = True,
            modality_costs: Dict[str, float] = None,
            cast_dtype: np.dtype = np.float32,
            max_episode_steps: int = None,
            **kwargs: Any,
    ):
        if env_id is None:
            raise ValueError("Either env or env_name must be provided")

        # Allow "BudgetAware..." aliases: strip prefix for the underlying env id
        base_id = env_id.replace("budget-aware-", "")

        # Filter out OmniSafe-specific parameters that highway-env doesn't understand
        env_kwargs = kwargs.copy()
        # Remove parameters that should not be passed to the highway-env environment
        if 'num_envs' in env_kwargs:
            self._num_envs = env_kwargs.pop('num_envs')
        if 'device' in env_kwargs:
            self._device = env_kwargs.pop('device')
        else:
            self._device = f'cuda:0' if torch.cuda.is_available() else 'cpu'
        if max_episode_steps is not None:
            env_kwargs["duration"] = max(env_kwargs.get("duration", max_episode_steps),
                                         env_kwargs.get("duration", 0))
        try:
            render_mode = env_kwargs.pop('render_mode')
        except KeyError:
            render_mode = None
        env = gym.make(
            base_id,
            config=env_kwargs,
            render_mode=render_mode,
        )
        super().__init__(env)

        # ── user-config ────────────────────────────────────────────────────────────
        self.use_all_obs = bool(use_all_obs)
        self.sensor_dropout_rescale = bool(sensor_dropout_rescale)
        self.append_mask_to_obs = not bool(use_all_obs)
        self.cast_dtype = cast_dtype
        self._num_envs = kwargs.pop('num_envs')
        # ── build multi‑observation (concat) ──────────────────────────────────────
        if observation_types is None:
            observation_types = list(self.DEFAULT_TYPES)
        observation_configs = observation_configs or {}

        self.obs_names: List[str] = []
        self._obs_objects = []
        self.shapes: Dict[str, Tuple[int, ...]] = {}
        self._spaces: Dict[str, spaces.Box] = {}

        for name in observation_types:
            if name in self.SKIP_TYPES:
                continue
            cfg = {"type": name}
            cfg.update(observation_configs.get(name, {}))
            if name == "GrayscaleObservation" and "weights" not in cfg:
                cfg["weights"] = [0.2989, 0.5870, 0.1140]
            obj = observation_factory(self.env.unwrapped, cfg)
            sp = obj.space()
            if isinstance(sp, spaces.Box):
                self.obs_names.append(name)
                self._obs_objects.append(obj)
                self._spaces[name] = sp
                self.shapes[name] = tuple(sp.shape)

        # Mapping and flattened bounds
        lows, highs = [], []
        self.mapping: Dict[str, Tuple[int, int]] = {}
        self.mod_sizes: Dict[str, int] = {}
        cursor = 0
        for name in self.obs_names:
            sp = self._spaces[name]
            lo = np.asarray(sp.low).flatten()
            hi = np.asarray(sp.high).flatten()
            K = lo.size
            lows.append(lo)
            highs.append(hi)
            self.mapping[name] = (cursor, cursor + K)
            self.mod_sizes[name] = K
            cursor += K

        base_low = np.concatenate(lows, axis=0) if lows else np.array([], dtype=np.float32)
        base_high = np.concatenate(highs, axis=0) if highs else np.array([], dtype=np.float32)
        if self.cast_dtype is not None:
            base_low = base_low.astype(self.cast_dtype, copy=False)
            base_high = base_high.astype(self.cast_dtype, copy=False)
        self._base_low = base_low
        self._base_high = base_high
        self._flat_dim = base_low.size
        self._num_modalities = len(self.obs_names)

        # Observation space (optionally append mask copy)
        if self.append_mask_to_obs:
            low = np.concatenate([base_low, np.zeros(self._num_modalities, dtype=base_low.dtype)], axis=0)
            high = np.concatenate([base_high, np.ones(self._num_modalities, dtype=base_high.dtype)], axis=0)
        else:
            low, high = base_low, base_high
        self.observation_space = spaces.Box(low=low, high=high, dtype=low.dtype)

        # ── Action space ───────────────────────────────────────────────────────────
        self.original_action_space = self.env.action_space
        if self.use_all_obs:
            new_action_space = self.env.action_space
        else:
            cont_action_space = self.env.action_space
            disc_action_space = gym.spaces.MultiBinary(self._num_modalities)
            self.cont_act_space = cont_action_space
            self.disc_act_space = disc_action_space
            new_action_space = gym.spaces.Tuple((cont_action_space, disc_action_space))
        self.action_space = new_action_space
        self._action_space = new_action_space

        # ── per‑modality costs ─────────────────────────────────────────────────────
        if modality_costs is None:
            modality_costs = {n: 1.0 for n in self.obs_names}
        self._modality_costs = {n: float(modality_costs.get(n, 1.0)) for n in self.obs_names}
        # Expose costs vector for tooling/evaluator
        self.costs = np.array([self._modality_costs[n] for n in self.obs_names], dtype=np.float32)

        # Optional: keep a max_episode_steps hint like other CMDPs
        self.max_episode_steps = max_episode_steps
        self.seed = seed
        self.set_seed(seed)

    @property
    def max_episode_steps(self) -> int | None:
        return self.unwrapped.config["duration"]

    @max_episode_steps.setter
    def max_episode_steps(self, value):
        self._max_episode_steps = value

    def set_seed(self, seed) -> None:
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

    # ── CMDP / Gymnasium API ──────────────────────────────────────────────────────
    def _concat_raw_obs(self) -> np.ndarray:
        chunks = []
        for name, obj in zip(self.obs_names, self._obs_objects):
            sub = np.asarray(obj.observe()).flatten()
            chunks.append(sub)
        out = np.concatenate(chunks, axis=0) if chunks else np.array([], dtype=self._base_low.dtype)
        if self.cast_dtype is not None and out.dtype != self.cast_dtype:
            out = out.astype(self.cast_dtype, copy=False)
        return out

    def _to_numpy(self, x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    def _expand_modality_mask_to_features(self, m_mod: np.ndarray) -> np.ndarray:
        """Convert modality mask (len M) ➜ feature-level mask (len flat_dim) for gating internal obs."""
        m_feat = np.zeros(self._flat_dim, dtype=np.float32)
        for i, name in enumerate(self.obs_names):
            s, e = self.mapping[name]
            m_feat[s:e] = float(m_mod[i])
        return m_feat

    def _apply_mask(self, flat: np.ndarray, feat_mask01: np.ndarray) -> np.ndarray:
        """Gating + optional Sensor Dropout α-rescale."""
        feat_mask01 = feat_mask01.astype(flat.dtype, copy=False)
        if self.sensor_dropout_rescale:
            kept = int(feat_mask01.sum())
            if kept == 0:
                # force keep largest modality to avoid empty input
                largest = max(self.obs_names, key=lambda n: self.mod_sizes[n])
                s, e = self.mapping[largest]
                feat_mask01[s:e] = 1.0
                kept = int(feat_mask01.sum())
            # TODO: check what is better, rescaling by the number of kept modalities or by the flat_dim
            alpha = float(self._flat_dim) / float(kept)
            gated = flat * feat_mask01 * alpha
        else:
            gated = flat * feat_mask01
        return gated

    def _mask_cost(self, m_mod01: np.ndarray) -> float:
        total = 0.0
        for i, name in enumerate(self.obs_names):
            if m_mod01[i] > 0.5:
                total += self._modality_costs.get(name, 1.0)
        return float(total)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        flat = self._concat_raw_obs()
        out = flat
        if self.append_mask_to_obs:
            mask0 = np.ones(self._num_modalities, dtype=flat.dtype)
            out = np.concatenate([flat, mask0], axis=0).astype(self.observation_space.dtype, copy=False)
        out = torch.as_tensor(out, dtype=torch.float32).to(self._device)
        # Return torch tensors as done in your existing CMDP wrapper
        return out, info

    def step(self, action: Union[Tuple[Any, Any], Any]):
        if self.use_all_obs:
            inner = self._to_numpy(action)
            base_obs, r, term, trunc, info = self.env.step(inner)
            flat = self._concat_raw_obs()
            out = flat
            m_mod01 = np.ones(self._num_modalities, dtype=flat.dtype)
            if self.append_mask_to_obs:
                out = np.concatenate([flat, m_mod01], axis=0).astype(self.observation_space.dtype, copy=False)

            # CMDP step signature: (obs, reward, cost, terminated, truncated, info)
            rew = float(r)
            cost = self._mask_cost(m_mod01)
            info = dict(info)
            info['sensor_mask'] = m_mod01  # store modality-level mask for logging/analysis
            info['original_reward'] = torch.as_tensor(rew, dtype=torch.float32)
            info['original_cost'] = torch.as_tensor(cost, dtype=torch.float32)
            returned_obs = torch.as_tensor(out, dtype=torch.float32).to(self._device)
            info['original_observation'] = returned_obs.clone()  # <--- Save original obs

            return (
                returned_obs,
                torch.as_tensor(rew, dtype=torch.float32).to(self._device),
                torch.as_tensor(cost, dtype=torch.float32).to(self._device),
                torch.as_tensor(term, dtype=torch.bool).to(self._device),
                torch.as_tensor(trunc, dtype=torch.bool).to(self._device),
                info,
            )

        # Expect Tuple((env_action, modality_mask_binary))
        if not (isinstance(action, (tuple, list)) and len(action) == 2):
            raise ValueError("Action must be Tuple((env_action, modality_binary_mask)) when use_all_obs=False")

        cont_action = self._to_numpy(action[0])
        mod_mask = self._to_numpy(action[1]).astype(np.int64).reshape(-1)

        # Safety: avoid all-zero mask ➜ keep largest modality
        if mod_mask.sum() == 0:
            largest = max(self.obs_names, key=lambda n: self.mod_sizes[n])
            i = self.obs_names.index(largest)
            mod_mask[i] = 1

        # Step base env
        base_obs, r, term, trunc, info = self.env.step(cont_action)

        # Build next obs and apply gating
        flat = self._concat_raw_obs()
        feat_mask01 = self._expand_modality_mask_to_features(mod_mask.astype(np.float32))
        gated = self._apply_mask(flat, feat_mask01)

        # Cost
        step_cost = self._mask_cost(mod_mask.astype(np.float32))

        # Compose final observation
        if self.append_mask_to_obs:
            out = np.concatenate([gated, mod_mask.astype(gated.dtype)], axis=0).astype(self.observation_space.dtype,
                                                                                       copy=False)
        else:
            out = gated.astype(self.observation_space.dtype, copy=False)

        # Info & returns (CMDP)
        info = dict(info)
        info["sensor_mask"] = mod_mask.astype(np.float32)  # store *modality*-level mask
        info["original_step_reward"] = float(r)
        info["original_step_cost"] = float(step_cost)
        returned_obs = torch.as_tensor(out, dtype=torch.float32).to(self._device)
        info["original_observation"] = returned_obs.clone()  # <--- Save original flat obs

        return (
            torch.as_tensor(out, dtype=torch.float32).to(self._device),
            torch.as_tensor(float(r), dtype=torch.float32).to(self._device),
            torch.as_tensor(float(step_cost), dtype=torch.float32).to(self._device),
            torch.as_tensor(term, dtype=torch.bool).to(self._device),
            torch.as_tensor(trunc, dtype=torch.bool).to(self._device),
            info,
        )

    def create_random_action(self) -> Union[Any, Tuple[Any, np.ndarray]]:
        """
        Create a random action for the BudgetAwareHighway environment.

        Args:
            env: The BudgetAwareHighway environment instance
            use_all_obs: Override for use_all_obs flag. If None, uses env.use_all_obs

        Returns:
            - If use_all_obs=True: Just the environment action
            - If use_all_obs=False: Tuple of (env_action, modality_mask)
        """

        # Sample random continuous action from the original environment action space
        continuous_action = self.original_action_space.sample()

        if self.use_all_obs:
            return torch.as_tensor(continuous_action, dtype=torch.float32).to(self._device)
        else:
            # Sample random binary mask for modalities (at least one modality must be active)
            num_modalities = self._num_modalities
            modality_mask = np.random.randint(0, 2, size=num_modalities, dtype=np.int32)

            # Ensure at least one modality is active
            if modality_mask.sum() == 0:
                # Randomly activate one modality
                active_idx = np.random.randint(0, num_modalities)
                modality_mask[active_idx] = 1
            return (
                torch.as_tensor(continuous_action, dtype=torch.float32).to(self._device),
                torch.as_tensor(modality_mask, dtype=torch.float32).to(self._device),
            )

    # Convenience
    @property
    def obs_mapping(self) -> Dict[str, Tuple[int, int]]:
        return dict(self.mapping)

    @property
    def obs_modalities(self) -> List[str]:
        return list(self.obs_names)

    def split(self, flat_obs: np.ndarray, obs_name: str, reshape: bool = True) -> np.ndarray:
        s, e = self.mapping[obs_name]
        part = flat_obs[s:e]
        return part.reshape(self.shapes[obs_name]) if reshape else part

# if __name__ == '__main__':
#     custom_cfgs = {
#         'train_cfgs': {
#             'total_steps': 10,
#             'vector_env_nums': 1,
#             'parallel': 1,
#             'device': f'cuda:0' if torch.cuda.is_available() else 'cpu'
#         },
#         'algo_cfgs': {
#             'steps_per_epoch': 2,
#             'update_iters': 2,
#             'batch_size': 512,
#             'kl_early_stop': True,
#         },
#         'model_cfgs': {
#             'actor_type': 'multihead'
#         },
#         'logger_cfgs': {
#             'wandb_project': 'BAFS 2.0',
#             'use_wandb': True,
#         },
#         'env_cfgs': {
#             'use_all_obs': False,
#             'max_episode_steps': 5,
#             "action": {
#                 "type": "ContinuousAction"
#             },
#             'seed': 42,
#         },
#     }
#
#     agent = omnisafe.Agent("CPO", "budget-aware-roundabout-v0", custom_cfgs=custom_cfgs)
#     # if args.use_all_obs:
#     #     agent.sample_random_actions(num_episodes=1_000, max_episode_length=500)
#     agent.learn()
#     agent.evaluate(num_episodes=5)
