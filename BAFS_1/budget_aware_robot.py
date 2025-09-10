import omnisafe
import gymnasium as gym
import numpy as np
from gymnasium.spaces import flatten_space, flatten
import torch
from omnisafe.envs.core import CMDP, env_register
from omnisafe.envs.core import env_register
from typing import Any, ClassVar
import gymnasium_robotics

gym.register_envs(gymnasium_robotics)


@env_register
class DiscreteActionWrapperCMDP(gym.Wrapper, CMDP):
    _support_envs: ClassVar[list[str]] = [
        # Fetch Environments
        "BudgetAwareFetchReachDense-v3",
        "BudgetAwareFetchPushDense-v3",
        "BudgetAwareFetchSlideDense-v3",
        "BudgetAwareFetchPickAndPlaceDense-v3",

        # Shadow Dexterous Hand Environments
        "BudgetAwareHandManipulateBlock_ContinuousTouchSensors-v1",
        "BudgetAwareHandManipulateBlockRotateZ_ContinuousTouchSensors-v1",
        "BudgetAwareHandManipulateBlockRotateParallel_ContinuousTouchSensors-v1",
        "BudgetAwareHandManipulateBlockRotateXYZ_ContinuousTouchSensors-v1",
        "BudgetAwareHandManipulateBlockFull_ContinuousTouchSensors-v1",
        "BudgetAwareHandManipulateBlock_BooleanTouchSensors-v1",
        "BudgetAwareHandManipulateBlockRotateZ_BooleanTouchSensors-v1",
        "BudgetAwareHandManipulateBlockRotateParallel_BooleanTouchSensors-v1",
        "BudgetAwareHandManipulateBlockRotateXYZ_BooleanTouchSensors-v1",
        "BudgetAwareHandManipulateBlockFull_BooleanTouchSensors-v1",

        # Maze Environments
        "BudgetAwareAntMaze_UMazeDense-v5",
        "BudgetAwarePointMaze_UMazeDense-v3",

        # Adroit Hand Environments
        "BudgetAwareAdroitHandDoor-v1",
        "BudgetAwareAdroitHandHammer-v1",
        "BudgetAwareAdroitHandPen-v1",
        "BudgetAwareAdroitHandRelocate-v1",

        # Franka Kitchen Environment
        "BudgetAwareFrankaKitchen-v1",
    ]  # Supported task names
    need_auto_reset_wrapper = True
    need_time_limit_wrapper = True

    def __init__(
            self,
            env_name: str,
            use_all_obs: bool = False,
            max_episode_steps=50,
            feature_costs: np.ndarray = None,
            features_to_use=None,
            seed=42,
            **kwargs: Any
    ) -> None:

        env_name = env_name.replace("BudgetAware", "")
        try:
            self._num_envs = kwargs.pop('num_envs')
        except KeyError:
            pass
        try:
            self._device = kwargs.pop('device')
        except KeyError:
            pass
        if env_name == 'AntMaze_UMazeDense-v5':
            kwargs['include_cfrc_ext_in_observation'] = False
        self._env = gym.make(env_name, max_episode_steps=max_episode_steps, **kwargs)
        self.max_episode_steps = max_episode_steps
        super().__init__(self._env)
        if isinstance(self._env.observation_space, gym.spaces.Dict):
            self.observation_space = flatten_space(self._env.observation_space)
        else:
            self.observation_space = self._env.observation_space
        self.seed = seed
        self.use_all_obs = use_all_obs
        self.original_action_space = self._env.action_space
        self.features_to_use = np.array(features_to_use) if features_to_use is not None else None
        if self.use_all_obs:
            new_action_space = self._env.action_space
        else:
            cont_action_space = self.env.action_space
            disc_action_space = gym.spaces.MultiBinary(self.observation_space.shape[0])
            self.cont_act_space = cont_action_space
            self.disc_act_space = disc_action_space
            new_action_space = gym.spaces.Tuple((cont_action_space, disc_action_space))
        self.action_space = new_action_space
        self._action_space = new_action_space
        self.costs = feature_costs

    @property
    def max_episode_steps(self) -> int | None:
        return self._env.spec.max_episode_steps

    def _flatten_obs(self, obs):
        if isinstance(self.env.observation_space, gym.spaces.Dict):
            obs = flatten(self.env.observation_space, obs)
        return obs

    def step(self, action):
        if self.use_all_obs:
            cont_action = action.cpu().numpy()
            disc_action = np.ones(self.observation_space.shape[0])
        else:
            cont_action = action[0].cpu().numpy() if isinstance(action[0], torch.Tensor) else action[0]
            disc_action = action[1].cpu().numpy() if isinstance(action[1], torch.Tensor) else action[1]
        obs, reward, done, truncated, info = self.env.step(cont_action)
        obs = self._flatten_obs(obs)
        if self.use_all_obs:
            disc_action = np.ones(self.observation_space.shape[0])
        if self.features_to_use is not None:
            disc_action = self.features_to_use
        modified_obs = obs * disc_action  # Zero out dimensions where discrete_action is 0
        cost = np.sum(self.costs * disc_action)
        obs, reward, cost, terminated, truncated = (
            torch.as_tensor(x, dtype=torch.float32)
            for x in (obs, reward, cost, done, truncated)
        )
        info['sensor_masking'] = disc_action
        return modified_obs, reward, cost, done, truncated, info

    def reset(self, **kwargs: Any):
        obs, info = self.env.reset(**kwargs)
        if isinstance(self.env.observation_space, gym.spaces.Dict):
            obs = flatten(self.env.observation_space, obs)
        return (
            torch.as_tensor(obs, dtype=torch.float32),
            info,
        )

    def set_seed(self, seed) -> None:
        self.seed = seed

    @max_episode_steps.setter
    def max_episode_steps(self, value):
        self._max_episode_steps = value
