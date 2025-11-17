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
from omnisafe.algorithms.utils.config_builder import AlgoConfigBuilder
from omnisafe.algorithms.utils.evaluation_results import EvaluationResults
from omnisafe.algorithms.utils.results_logger import ResultsLogger
from omnisafe.algorithms.utils.sample_efficiency_tracker import SampleEfficiencyTracker
from omnisafe.envs import support_envs
from omnisafe.evaluator import Evaluator
from omnisafe.utils import distributed
from omnisafe.utils.config import Config, check_all_configs, get_default_kwargs_yaml
from omnisafe.utils.plotter import Plotter
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

        # Merge custom and terminal configurations using AlgoConfigBuilder
        AlgoConfigBuilder.merge_custom_configs(cfgs, self.custom_cfgs)
        AlgoConfigBuilder.merge_terminal_configs(cfgs, self.train_terminal_cfgs)

        # Build experiment name using AlgoConfigBuilder
        exp_name = AlgoConfigBuilder.build_exp_name(self.env_id, self.algo, cfgs)
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

    def learn_with_sample_efficiency(
            self,
            eval_fraction: float = 0.05,
            eval_episodes: int = 50
    ) -> tuple[float, float, float]:
        """Train with periodic evaluations for sample efficiency tracking.

        Evaluates the policy at regular intervals (default: every 5% of training)
        and stores results in a dictionary that will be saved to the manifest CSV
        at the end of training.

        Args:
            eval_fraction: Fraction of total training steps between evaluations (default: 0.05 = 5%)
            eval_episodes: Episodes per evaluation (default: 50)

        Returns:
            ep_ret: The episode return of the final episode.
            ep_cost: The episode cost of the final episode.
            ep_len: The episode length of the final episode.
        """
        # Initialize sample efficiency tracker
        tracker = SampleEfficiencyTracker(eval_fraction, eval_episodes)

        # Setup tracking intervals and adjust save frequency
        total_steps = self.custom_cfgs['train_cfgs']['total_steps']
        steps_per_epoch = self.custom_cfgs['algo_cfgs']['steps_per_epoch']
        original_save_freq = self.cfgs.logger_cfgs.save_model_freq

        eval_interval, epochs_per_eval = tracker.setup_for_training(
            total_steps, steps_per_epoch, original_save_freq
        )

        # Adjust checkpoint save frequency to match evaluation intervals
        self.cfgs.logger_cfgs.save_model_freq = epochs_per_eval

        # Inject callback into agent
        self.agent._eval_interval = eval_interval
        self.agent._eval_callback = lambda steps: tracker.run_periodic_eval(
            self.agent.logger.log_dir, steps
        )

        # Run training (PolicyGradient.learn() will call callback at intervals)
        ep_ret, ep_cost, ep_len = self.agent.learn()

        # Restore original save frequency
        self.cfgs.logger_cfgs.save_model_freq = original_save_freq

        # Store tracking data for final evaluation
        self._sample_efficiency_data = tracker.get_results()

        # Initialize statistical tools
        self._init_statistical_tools()

        # Final evaluation (includes sample_efficiency_curve in manifest)
        print(f"\n{'=' * 80}")
        print(f"RUNNING FINAL EVALUATION")
        print(f"{'=' * 80}\n")
        self.evaluate(num_episodes=eval_episodes)

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

        # Load checkpoint and run evaluation
        scan_dir = os.scandir(os.path.join(self.agent.logger.log_dir, 'torch_save'))
        checkpoints = [item for item in scan_dir if item.is_file() and item.name.split('.')[-1] == 'pt']
        # Sort by epoch number to get the latest checkpoint (not just last in filesystem order)
        item = sorted(checkpoints, key=lambda x: int(x.name.split('-')[1].split('.')[0]))[-1]
        self._evaluator.load_saved(save_dir=self.agent.logger.log_dir, model_name=item.name, render_mode='rgb_array')
        raw_results = self._evaluator.evaluate(num_episodes=num_episodes, cost_criteria=cost_criteria,
                                               record_video=record_video)
        scan_dir.close()

        # Convert to structured results
        results = EvaluationResults.from_evaluator_results(raw_results)

        # Determine observation mode
        obs_mode = self._get_obs_mode()

        # Prepare budget and step strings
        budget_str = self._get_budget_string()
        steps_str = self.custom_cfgs['train_cfgs']['total_steps']

        # Initialize results logger
        logger = ResultsLogger(self.cfgs.logger_cfgs.rliable_json_path)

        # Log to rliable JSON
        bucket = logger.create_bucket_name(self.env_id, obs_mode, budget_str, steps_str)
        logger.log_to_rliable_json(bucket, f'{self.algo}_reward', results.get_reward_mean())
        logger.log_to_rliable_json(bucket, f'{self.algo}_cost', results.get_cost_mean())
        logger.log_to_rliable_json(
            bucket,
            f'{self.algo}_(sensor_used_mean)_(sensor_used_std)',
            [results.get_sensor_usage_mean(), results.get_sensor_usage_std()]
        )

        # Prepare manifest metadata
        metadata = self._build_manifest_metadata(
            results, obs_mode, budget_str, steps_str, num_episodes
        )

        # Log to manifest CSV
        logger.log_to_manifest(metadata)

        # Log to wandb with visualizations
        logger.log_to_wandb(
            self.agent.logger,
            results,
            use_all_obs=self.custom_cfgs['env_cfgs']['use_all_obs']
        )

        # Render episodes
        self.render(num_episodes=10, render_mode="rgb_array", width=256, height=256)

    def _get_obs_mode(self) -> str:
        """Determine observation mode from configuration."""
        if self.custom_cfgs['env_cfgs']['use_all_obs']:
            return 'AllObs'
        elif self.custom_cfgs['model_cfgs'].get('actor_type') == 'random_mask':
            return 'RandomMask'
        else:
            return 'SelectedObs'

    def _get_budget_string(self) -> str:
        """Extract budget string from configuration."""
        if 'lagrange_cfgs' not in self.custom_cfgs.keys():
            self.custom_cfgs['lagrange_cfgs'] = {}

        if 'cost_limit' in self.custom_cfgs['lagrange_cfgs'].keys():
            return f'Budget{int(self.custom_cfgs["lagrange_cfgs"]["cost_limit"])}'
        return 'BudgetNone'

    def _build_manifest_metadata(
            self,
            results: EvaluationResults,
            obs_mode: str,
            budget_str: str,
            steps_str: int,
            num_episodes: int
    ) -> dict[str, Any]:
        """Build metadata dictionary for manifest logging."""
        from datetime import datetime
        import json

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Extract budget from configuration
        budget = None
        if 'lagrange_cfgs' in self.custom_cfgs and 'cost_limit' in self.custom_cfgs['lagrange_cfgs']:
            budget = self.custom_cfgs['lagrange_cfgs']['cost_limit']
        elif 'algo_cfgs' in self.custom_cfgs and 'cost_limit' in self.custom_cfgs['algo_cfgs']:
            budget = self.custom_cfgs['algo_cfgs']['cost_limit']
        elif 'algo_cfgs' in self.custom_cfgs and 'safety_budget' in self.custom_cfgs['algo_cfgs']:
            budget = self.custom_cfgs['algo_cfgs']['safety_budget']

        # Get action space type and observation space shape
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
            pass

        # Get sample efficiency data if available
        sample_efficiency_curve = getattr(self, '_sample_efficiency_data', {})

        return {
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
            'reward_mean': results.get_reward_mean(),
            'reward_std': results.get_reward_std(),
            'cost_mean': results.get_cost_mean(),
            'cost_std': results.get_cost_std(),
            'episode_rewards': json.dumps([float(r) for r in results.episode_rewards]),
            'episode_costs': json.dumps([float(c) for c in results.episode_costs]),
            'sample_efficiency_curve': json.dumps(sample_efficiency_curve),
            'status': 'success',
            'log_dir': self.agent.logger.log_dir,
            'cost_normalized': self.cfgs['algo_cfgs'].get('cost_normalize', 'unknown'),
            'reward_normalized': self.cfgs['algo_cfgs'].get('reward_normalize', 'unknown'),
            'obs_modality_normalize': self.cfgs['algo_cfgs'].get('obs_modality_normalize', 'unknown'),
        }

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
        checkpoints = [item for item in scan_dir if item.is_file() and item.name.split('.')[-1] == 'pt']
        # Sort by epoch number to get the latest checkpoint (not just last in filesystem order)
        item = sorted(checkpoints, key=lambda x: int(x.name.split('-')[1].split('.')[0]))[-1]
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
