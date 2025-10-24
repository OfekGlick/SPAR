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
"""Implementation of the AlgoWrapper Class."""

from __future__ import annotations

import os
import sys
from typing import Any
import wandb
import torch
from tqdm import trange
from gymnasium import spaces
from omnisafe.algorithms import ALGORITHM2TYPE, ALGORITHMS, registry
from omnisafe.algorithms.base_algo import BaseAlgo
from omnisafe.envs import support_envs
from omnisafe.evaluator import Evaluator
from omnisafe.utils import distributed
from omnisafe.utils.config import Config, check_all_configs, get_default_kwargs_yaml
from omnisafe.utils.plotter import Plotter
from omnisafe.utils.tools import recursive_check_config
import numpy as np
import pandas as pd
import json


class AlgoWrapper:
    """Algo Wrapper for algorithms.

    Args:
        algo (str): The algorithm name.
        env_id (str): The environment id.
        train_terminal_cfgs (dict[str, Any], optional): The configurations for training termination.
            Defaults to None.
        custom_cfgs (dict[str, Any], optional): The custom configurations. Defaults to None.

    Attributes:
        algo (str): The algorithm name.
        env_id (str): The environment id.
        train_terminal_cfgs (dict[str, Any]): The configurations for training termination.
        custom_cfgs (dict[str, Any]): The custom configurations.
        cfgs (Config): The configurations for the algorithm.
        algo_type (str): The algorithm type.
    """

    algo_type: str

    def __init__(
            self,
            algo: str,
            env_id: str,
            train_terminal_cfgs: dict[str, Any] | None = None,
            custom_cfgs: dict[str, Any] | None = None,
    ) -> None:
        """Initialize an instance of :class:`AlgoWrapper`."""
        self.algo: str = algo
        self.env_id: str = env_id
        # algo_type will set in _init_checks()
        self.train_terminal_cfgs: dict[str, Any] | None = train_terminal_cfgs
        self.custom_cfgs: dict[str, Any] | None = custom_cfgs
        self._evaluator: Evaluator | None = None
        self._plotter: Plotter | None = None
        self.cfgs: Config = self._init_config()
        self._init_checks()
        self._init_algo()

    def _init_config(self) -> Config:
        """Initialize config.

        Initialize the configurations for the algorithm, following the order of default
        configurations, custom configurations, and terminal configurations.

        Returns:
            The configurations for the algorithm.

        Raises:
            AssertionError: If the algorithm name is not in the supported algorithms.
        """
        assert (
                self.algo in ALGORITHMS['all']
        ), f"{self.algo} doesn't exist. Please choose from {ALGORITHMS['all']}."
        self.algo_type = ALGORITHM2TYPE.get(self.algo, '')
        if self.train_terminal_cfgs is not None:
            if self.algo_type in ['model-based', 'offline']:
                assert (
                        self.train_terminal_cfgs['vector_env_nums'] == 1
                ), 'model-based and offline only support vector_env_nums==1!'
            if self.algo_type in ['off-policy', 'model-based', 'offline']:
                assert (
                        self.train_terminal_cfgs['parallel'] == 1
                ), 'off-policy, model-based and offline only support parallel==1!'

        cfgs = get_default_kwargs_yaml(self.algo, self.env_id, self.algo_type)

        # update the cfgs from custom configurations
        if self.custom_cfgs:
            # avoid repeatedly record the env_id and algo
            if 'env_id' in self.custom_cfgs:
                self.custom_cfgs.pop('env_id')
            if 'algo' in self.custom_cfgs:
                self.custom_cfgs.pop('algo')
            # validate the keys of custom configuration
            recursive_check_config(self.custom_cfgs, cfgs)
            # update the cfgs from custom configurations
            cfgs.recurisve_update(self.custom_cfgs)
            # save configurations specified in current experiment
            cfgs.update({'exp_increment_cfgs': self.custom_cfgs})
        # update the cfgs from custom terminal configurations
        if self.train_terminal_cfgs:
            # avoid repeatedly record the env_id and algo
            if 'env_id' in self.train_terminal_cfgs:
                self.train_terminal_cfgs.pop('env_id')
            if 'algo' in self.train_terminal_cfgs:
                self.train_terminal_cfgs.pop('algo')
            # validate the keys of train_terminal_cfgs configuration
            recursive_check_config(self.train_terminal_cfgs, cfgs.train_cfgs)
            # update the cfgs.train_cfgs from train_terminal configurations
            cfgs.train_cfgs.recurisve_update(self.train_terminal_cfgs)
            # save configurations specified in current experiment
            cfgs.recurisve_update({'exp_increment_cfgs': {'train_cfgs': self.train_terminal_cfgs}})

        # the exp_name format is PPO-{SafetyPointGoal1-v0}
        use_cost = "-use_cost" if cfgs['algo_cfgs']['use_cost'] else ""
        use_all_obs = '-use_all_obs' if cfgs['env_cfgs']['use_all_obs'] else ""
        sd_reg_str = '-sd_reg' if cfgs['algo_cfgs']['sd_regulizer'] else ''
        zero_act_str = '-zero_act' if cfgs['algo_cfgs']['no_zero_act'] else ''
        random_mask_str = '-random_mask' if cfgs['model_cfgs'].get('actor_type') == 'random_mask' else ''
        if 'lagrange_cfgs' not in cfgs.keys():
            budget_str = ''
        else:
            if 'cost_limit' not in cfgs['lagrange_cfgs'].keys():
                budget_str = ''
            else:
                budget_str = f'Budget{int(cfgs["lagrange_cfgs"]["cost_limit"])}'

        cost_normalize = cfgs['algo_cfgs'].get('cost_normalize', False)
        reward_normalize = cfgs['algo_cfgs'].get('reward_normalize', False)
        if cost_normalize:
            reward_norm_str = 'costNorm'
        else:
            reward_norm_str = ''
        if reward_normalize:
            reward_norm_str += '_rewardNorm'
        else:
            reward_norm_str += ''
        exp_name = f'{self.algo}-{self.env_id.split("budget-aware-")[1]}{use_all_obs}{sd_reg_str}{zero_act_str}{random_mask_str}-{budget_str}-{reward_norm_str}-{cfgs["train_cfgs"]["total_steps"]}_steps'
        cfgs.recurisve_update({'exp_name': exp_name, 'env_id': self.env_id, 'algo': self.algo})
        cfgs.train_cfgs.recurisve_update(
            {'epochs': cfgs.train_cfgs.total_steps // cfgs.algo_cfgs.steps_per_epoch},
        )
        return cfgs

    def _init_checks(self) -> None:
        """Initial checks."""
        assert isinstance(self.algo, str), 'algo must be a string!'
        assert isinstance(self.cfgs.train_cfgs.parallel, int), 'parallel must be an integer!'
        assert self.cfgs.train_cfgs.parallel > 0, 'parallel must be greater than 0!'
        assert (
                self.env_id in support_envs()
        ), f"{self.env_id} doesn't exist. Please choose from {support_envs()}."

    def _init_algo(self) -> None:
        """Initialize the algorithm."""
        check_all_configs(self.cfgs, self.algo_type)
        if distributed.fork(
                self.cfgs.train_cfgs.parallel,
                device=self.cfgs.train_cfgs.device,
        ):
            # re-launches the current script with workers linked by MPI
            sys.exit()
        if self.cfgs.train_cfgs.device == 'cpu':
            torch.set_num_threads(self.cfgs.train_cfgs.torch_threads)
        else:
            if self.cfgs.train_cfgs.parallel > 1 and os.getenv('MASTER_ADDR') is not None:
                ddp_local_rank = int(os.environ['LOCAL_RANK'])
                self.cfgs.train_cfgs.device = f'cuda:{ddp_local_rank}'
            torch.set_num_threads(1)
            torch.cuda.set_device(self.cfgs.train_cfgs.device)
        os.environ['OMNISAFE_DEVICE'] = self.cfgs.train_cfgs.device
        self.agent: BaseAlgo = registry.get(self.algo)(
            env_id=self.env_id,
            cfgs=self.cfgs,
        )

    def sample_random_actions(self, num_episodes: int = 1_000, max_episode_length=500) -> None:
        """Sample random actions, save state values and rewards, and calculate correlation.

        Args:
            num_episodes (int, optional): The number of episodes to sample. Defaults to 10.
        """
        state_values = []
        rewards = []
        correlations = []

        for episode_idx in trange(num_episodes):
            obs, _ = self.agent._env.reset()
            done = False
            episode_rewards = []
            episode_states = []

            for _ in range(max_episode_length):
                act = torch.tensor(self.agent._env.action_space.sample()).to(self.cfgs.train_cfgs.device)
                obs, rew, cost, terminated, truncated, info = self.agent._env.step(act)
                original_obs = info['original_obs']
                episode_states.append(original_obs)
                if self.algo == 'PPOSaute':
                    rew = info['original_reward']
                episode_rewards.append(rew)
                done = terminated or truncated

            state_values.extend(episode_states)
            rewards.extend(episode_rewards)

            # Calculate correlation for the episode
            episode_states = np.array(episode_states).reshape(max_episode_length, -1)
            episode_rewards = np.array(episode_rewards).reshape(-1, 1)
            df = pd.DataFrame(episode_states)
            # if self.agent._env.use_all_obs
            df['reward'] = episode_rewards
            correlation_matrix = df.corr()
            correlations.append(correlation_matrix['reward'].drop('reward'))

        # Calculate mean and median correlation
        correlations_df = pd.DataFrame(correlations)
        mean_correlation = correlations_df.mean()
        median_correlation = correlations_df.median()
        self.feature_importance = mean_correlation

        feature_importance = np.nan_to_num(mean_correlation, nan=0.0)

        # Create a dictionary for the bar plot
        data = wandb.Table(
            data=[[f'Feature {i + 1}', abs(importance)] for i, importance in enumerate(feature_importance)],
            columns=["Feature", "Importance"],
        )

        self.agent.logger.wandb_log(
            {
                "Reward corr Based Feature Importance": wandb.plot.bar(data, "Feature", "Importance",
                                                                       title="Feature Importance")
            }
        )
        print("Mean correlation of each feature with the reward:")
        print(mean_correlation)
        print("Median correlation of each feature with the reward:")
        print(median_correlation)

    def learn(self) -> tuple[float, float, float]:
        """Agent learning.

        Returns:
            ep_ret: The episode return of the final episode.
            ep_cost: The episode cost of the final episode.
            ep_len: The episode length of the final episode.
        """
        ep_ret, ep_cost, ep_len = self.agent.learn()

        self._init_statistical_tools()

        return ep_ret, ep_cost, ep_len

    def _init_statistical_tools(self) -> None:
        """Initialize statistical tools."""
        self._evaluator = Evaluator()
        self._plotter = Plotter()

    def plot(self, smooth: int = 1) -> None:
        """Plot the training curve.

        Args:
            smooth (int, optional): window size, for smoothing the curve. Defaults to 1.

        Raises:
            AssertionError: If the :meth:`learn` method has not been called.
        """
        assert self._plotter is not None, 'Please run learn() first!'
        self._plotter.make_plots(
            [self.agent.logger.log_dir],
            None,
            'Steps',
            'Rewards',
            False,
            self.agent.cost_limit,
            smooth,
            None,
            None,
            'mean',
            self.agent.logger.log_dir,
        )

    def evaluate(self, num_episodes: int = 10, cost_criteria: float = 1.0, record_video: bool = True) -> None:
        """Agent Evaluation.

        Args:
            num_episodes (int, optional): number of episodes to evaluate. Defaults to 10.
            cost_criteria (float, optional): the cost criteria to evaluate. Defaults to 1.0.
            record_video (bool, optional): whether to record video with sensor overlays. Defaults to True.

        Raises:
            AssertionError: If the :meth:`learn` method has not been called.
        """
        assert self._evaluator is not None, 'Please run learn() first!'
        scan_dir = os.scandir(os.path.join(self.agent.logger.log_dir, 'torch_save'))
        item = [item for item in scan_dir if item.is_file() and item.name.split('.')[-1] == 'pt'][-1]
        self._evaluator.load_saved(save_dir=self.agent.logger.log_dir, model_name=item.name, render_mode='rgb_array')
        results = self._evaluator.evaluate(num_episodes=num_episodes, cost_criteria=cost_criteria, record_video=record_video)
        obs_masks = np.mean(a=results['episode_obs_masks'], axis=0)
        average_sensor_used = np.mean(results['episode_obs_masks'], axis=0).tolist()
        std_sensor_used = np.std(results['episode_obs_masks'], axis=0).tolist()
        average_episode_rewards = results['episode_rewards']
        average_episode_costs = results['episode_costs']

        # Determine observation mode for bucket naming
        if self.custom_cfgs['env_cfgs']['use_all_obs']:
            obs_mode = 'AllObs'
        elif self.custom_cfgs['model_cfgs'].get('actor_type') == 'random_mask':
            obs_mode = 'RandomMask'
        else:
            obs_mode = 'SelectedObs'

        if 'lagrange_cfgs' not in self.custom_cfgs.keys():
            self.custom_cfgs['lagrange_cfgs'] = {}
        budget_str = f'Budget{int(self.custom_cfgs['lagrange_cfgs']['cost_limit'])}' if 'cost_limit' in self.custom_cfgs[
            'lagrange_cfgs'].keys() else 'BudgetNone'
        steps_str = self.custom_cfgs['train_cfgs']['total_steps']
        self._merge_rliable_json(
            bucket=self.env_id + f'_{obs_mode}' + f'_{budget_str}' + f'_{steps_str}_steps',
            key=f'{self.algo}_reward',
            value=np.mean(average_episode_rewards),
        )
        self._merge_rliable_json(
            bucket=self.env_id + f'_{obs_mode}' + f'_{budget_str}' + f'_{steps_str}_steps',
            key=f'{self.algo}_cost',
            value=np.mean(average_episode_costs),
        )
        self._merge_rliable_json(
            bucket=self.env_id + f'_{obs_mode}' + f'_{budget_str}' + f'_{steps_str}_steps',
            key=f'{self.algo}_(sensor_used_mean)_(sensor_used_std)',
            value=[average_sensor_used, std_sensor_used],
        )

        # Log run details to manifest CSV
        from datetime import datetime
        import json
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Extract budget from wherever the algorithm stores it
        budget = None
        if 'lagrange_cfgs' in self.custom_cfgs and 'cost_limit' in self.custom_cfgs['lagrange_cfgs']:
            budget = self.custom_cfgs['lagrange_cfgs']['cost_limit']
        elif 'algo_cfgs' in self.custom_cfgs and 'cost_limit' in self.custom_cfgs['algo_cfgs']:
            budget = self.custom_cfgs['algo_cfgs']['cost_limit']
        elif 'algo_cfgs' in self.custom_cfgs and 'safety_budget' in self.custom_cfgs['algo_cfgs']:
            budget = self.custom_cfgs['algo_cfgs']['safety_budget']

        # Get action space type and observation space shape from the agent's environment
        action_space_type = 'unknown'
        obs_space_shape = 'unknown'
        try:
            env = self.agent._env
            action_space = env.action_space
            if isinstance(action_space, spaces.Tuple):
                action_space_type = f"Tuple({type(action_space[0]).__name__}, {type(action_space[1]).__name__})"
            else:
                action_space_type = type(action_space).__name__
            obs_space_shape = str(env.observation_space.shape)
        except:
            action_space_type = 'unknown'
            obs_space_shape = 'unknown'

        self._log_to_manifest({
            'timestamp': timestamp,
            'algo': self.algo,
            'env': self.env_id,
            'seed': self.custom_cfgs['env_cfgs'].get('seed', 'unknown'),
            'budget': budget if budget is not None else 'None',
            'obs_mode': obs_mode,
            'actor_type': self.custom_cfgs['model_cfgs'].get('actor_type', 'unknown'),
            'action_space_type': action_space_type,
            'obs_space_shape': obs_space_shape,
            'use_cost': self.custom_cfgs['algo_cfgs'].get('use_cost', 'unknown'),
            'use_all_obs': self.custom_cfgs['env_cfgs'].get('use_all_obs', 'unknown'),
            'sd_regulizer': self.custom_cfgs['algo_cfgs'].get('sd_regulizer', 'unknown'),
            'random_obs_selection': self.custom_cfgs['model_cfgs'].get('actor_type') == 'random_mask',
            'total_steps': steps_str,
            'num_eval_episodes': num_episodes,
            'reward_mean': float(np.mean(average_episode_rewards)),
            'reward_std': float(np.std(average_episode_rewards)),
            'cost_mean': float(np.mean(average_episode_costs)),
            'cost_std': float(np.std(average_episode_costs)),
            'episode_rewards': json.dumps([float(r) for r in average_episode_rewards]),
            'episode_costs': json.dumps([float(c) for c in average_episode_costs]),
            'status': 'success',
            'log_dir': self.agent.logger.log_dir,
            'cost_normalized': self.cfgs['algo_cfgs'].get('cost_normalize', 'unknown'),
            'reward_normalized': self.cfgs['algo_cfgs'].get('reward_normalize', 'unknown'),
            'obs_modality_normalize': self.cfgs['algo_cfgs'].get('obs_modality_normalize', 'unknown'),
        })

        self.render(num_episodes=10, render_mode="rgb_array", width=256, height=256)
        sensor_costs = results['sensor_costs']
        if not self.custom_cfgs['env_cfgs']['use_all_obs']:
            obs_mask_data = wandb.Table(
                data=[[f'Feature {i + 1}', importance] for i, importance in enumerate(obs_masks)],
                columns=["Feature", "Average Time On"],
            )
            self.agent.logger.wandb_log(
                {
                    "Feature Selection": wandb.plot.bar(
                        obs_mask_data,
                        label="Feature",
                        value="Average Time On",
                        title="Feature Selection During Episode",
                    )
                }
            )
        sensor_costs_data = wandb.Table(
            data=[[f'Feature {i + 1}', cost] for i, cost in enumerate(sensor_costs)],
            columns=["Feature", "Cost"],
        )

        self.agent.logger.wandb_log(
            {
                "Sensor Costs": wandb.plot.bar(
                    sensor_costs_data,
                    label="Feature",
                    value="Cost",
                    title="Sensor Costs",
                )
            }
        )
        scan_dir.close()

    def _merge_rliable_json(self, bucket: str, key: str, value: float | list[float]) -> None:
        """Merge episodic returns into an rliable-style JSON:
           { "<bucket>": { "<env>_<algo>": [ ... floats ... ] } }"""
        path = os.path.join(self.cfgs.logger_cfgs.rliable_json_path, "results.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = {}

        data.setdefault(bucket, {})
        data[bucket].setdefault(key, [])
        # ensure plain floats
        data[bucket][key].extend([value])

        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=True)
        os.replace(tmp, path)

        # Log what was saved
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] Saved to results.json: bucket='{bucket}', key='{key}', value={value}")
        print(f"[{timestamp}] File location: {path}")

    def _log_to_manifest(self, metadata: dict) -> None:
        """Append run details to CSV manifest for tracking and diagnosis."""
        import csv
        from datetime import datetime

        manifest_path = os.path.join(
            self.cfgs.logger_cfgs.rliable_json_path,
            "run_manifest.csv"
        )
        os.makedirs(os.path.dirname(manifest_path), exist_ok=True)

        # Check if file exists to determine if we need headers
        file_exists = os.path.exists(manifest_path)

        # Define column order
        fieldnames = [
            'timestamp', 'algo', 'env', 'seed', 'budget', 'obs_mode', 'actor_type',
            'action_space_type', 'obs_space_shape',
            'use_cost', 'use_all_obs', 'sd_regulizer', 'random_obs_selection',
            'total_steps', 'num_eval_episodes', 'reward_mean', 'reward_std',
            'cost_mean', 'cost_std', 'episode_rewards', 'episode_costs',
            'status', 'log_dir', 'cost_normalized', 'reward_normalized'
        ]

        # Write to CSV with append mode
        with open(manifest_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(metadata)

        timestamp = metadata.get('timestamp', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        print(f"[{timestamp}] Logged run to manifest: {manifest_path}")

    # pylint: disable-next=too-many-arguments
    def render(
            self,
            num_episodes: int = 10,
            render_mode: str = 'rgb_array',
            camera_name: str = 'track',
            width: int = 256,
            height: int = 256,
    ) -> None:
        """Evaluate and render some episodes.

        Args:
            num_episodes (int, optional): The number of episodes to render. Defaults to 10.
            render_mode (str, optional): The render mode, can be 'rgb_array', 'depth_array' or
                'human'. Defaults to 'rgb_array'.
            camera_name (str, optional): the camera name, specify the camera which you use to
                capture images. Defaults to 'track'.
            width (int, optional): The width of the rendered image. Defaults to 256.
            height (int, optional): The height of the rendered image. Defaults to 256.

        Raises:
            AssertionError: If the :meth:`learn` method has not been called.
        """
        assert self._evaluator is not None, 'Please run learn() first!'
        scan_dir = os.scandir(os.path.join(self.agent.logger.log_dir, 'torch_save'))
        item = [item for item in scan_dir if item.is_file() and item.name.split('.')[-1] == 'pt'][-1]
        self._evaluator.load_saved(
            save_dir=self.agent.logger.log_dir,
            model_name=item.name,
            render_mode=render_mode,
            camera_name=camera_name,
            width=width,
            height=height,
        )
        self._evaluator.render(num_episodes=num_episodes)
        scan_dir.close()
