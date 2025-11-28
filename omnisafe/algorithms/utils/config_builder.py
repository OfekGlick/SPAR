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
"""Configuration builder for OmniSafe algorithms."""

from __future__ import annotations

from typing import Any
from omnisafe.utils.config import Config
from omnisafe.utils.tools import recursive_check_config


class AlgoConfigBuilder:
    """Helper class for building algorithm configurations.

    Separates configuration construction logic from AlgoWrapper to improve modularity.
    Handles experiment naming, custom config merging, and terminal config merging.
    """

    @staticmethod
    def build_exp_name(env_id: str, algo: str, cfgs: Config) -> str:
        """Build experiment name from configuration flags.

        Creates a structured experiment name by extracting configuration flags
        and joining non-empty components with hyphens.

        Args:
            env_id: Environment identifier
            algo: Algorithm name
            cfgs: Configuration object

        Returns:
            Formatted experiment name string (e.g., "PPO-Lift-use_all_obs-Budget10-1000000_steps")
        """
        # Extract environment short name
        env_short = env_id.split("budget-aware-")[1] if "budget-aware-" in env_id else env_id

        # Build list of name components (only non-empty strings will be included)
        components = [
            algo,
            env_short,
            AlgoConfigBuilder._get_sensor_subset_str(cfgs),  # Add sensor subset info
            AlgoConfigBuilder._get_obs_flags(cfgs),
            AlgoConfigBuilder._get_algo_flags(cfgs),
            AlgoConfigBuilder._get_budget_str(cfgs),
            AlgoConfigBuilder._get_normalization_str(cfgs),
            f"{cfgs['train_cfgs']['total_steps']}_steps",
            AlgoConfigBuilder._get_obs_modality_norm(cfgs),
        ]

        # Filter out empty strings and join with hyphens
        return '-'.join(filter(None, components))

    @staticmethod
    def _get_obs_flags(cfgs: Config) -> str:
        """Extract observation-related flags."""
        flags = []
        if cfgs['env_cfgs']['use_all_obs']:
            flags.append('use_all_obs')
        if cfgs['model_cfgs'].get('actor_type') == 'random_mask':
            flags.append('random_mask')
        return '_'.join(flags)

    @staticmethod
    def _get_algo_flags(cfgs: Config) -> str:
        """Extract algorithm-related flags."""
        flags = []
        if cfgs['algo_cfgs']['use_cost']:
            flags.append('use_cost')
        if cfgs['algo_cfgs']['sd_regulizer']:
            flags.append('sd_reg')
        # Add penalty coefficient if non-zero
        penalty_coef = cfgs['algo_cfgs'].get('penalty_coef', 0.0)
        if penalty_coef > 0.0:
            flags.append(f'pen{penalty_coef}')
        return '_'.join(flags)

    @staticmethod
    def _get_budget_str(cfgs: Config) -> str:
        """Extract budget string from configuration."""
        if 'lagrange_cfgs' in cfgs and 'cost_limit' in cfgs['lagrange_cfgs']:
            return f'Budget{int(cfgs["lagrange_cfgs"]["cost_limit"])}'
        return ''

    @staticmethod
    def _get_normalization_str(cfgs: Config) -> str:
        """Extract normalization flags."""
        flags = []
        if cfgs['algo_cfgs'].get('cost_normalize', False):
            flags.append('costNorm')
        if cfgs['algo_cfgs'].get('reward_normalize', False):
            flags.append('rewardNorm')
        return '_'.join(flags)

    @staticmethod
    def _get_obs_modality_norm(cfgs: Config) -> str:
        """Extract observation modality normalization flag."""
        return 'ObsModNorm' if cfgs['algo_cfgs'].get('obs_modality_normalize', False) else ''

    @staticmethod
    def _get_sensor_subset_str(cfgs: Config) -> str:
        """Extract sensor subset configuration with abbreviations."""
        available_sensors = cfgs['env_cfgs'].get('available_sensors')
        if available_sensors is None:
            return ''  # All sensors available (default)

        # Sensor abbreviations
        abbrev = {
            'Kinematics': 'Kin',
            'LidarObservation': 'Lid',
            'OccupancyGrid': 'Occ',
            'TimeToCollision': 'TTC',
            'robot_proprioception': 'Rob',
            'object_states': 'Obj',
            'task_features': 'Task',
            'camera': 'Cam',
        }

        # Build abbreviated sensor string
        sensor_abbrevs = [abbrev.get(s, s[:3]) for s in sorted(available_sensors)]
        return 'sens_' + '_'.join(sensor_abbrevs)

    @staticmethod
    def merge_custom_configs(cfgs: Config, custom_cfgs: dict[str, Any] | None) -> None:
        """Merge custom configurations into base config.

        Args:
            cfgs: Base configuration object (modified in-place)
            custom_cfgs: Custom configuration overrides
        """
        if custom_cfgs:
            # Avoid repeatedly recording the env_id and algo
            if 'env_id' in custom_cfgs:
                custom_cfgs.pop('env_id')
            if 'algo' in custom_cfgs:
                custom_cfgs.pop('algo')

            # Validate the keys of custom configuration
            recursive_check_config(custom_cfgs, cfgs)

            # Update the cfgs from custom configurations
            cfgs.recurisve_update(custom_cfgs)

            # Save configurations specified in current experiment
            cfgs.update({'exp_increment_cfgs': custom_cfgs})

    @staticmethod
    def merge_terminal_configs(cfgs: Config, train_terminal_cfgs: dict[str, Any] | None) -> None:
        """Merge terminal training configurations into base config.

        Args:
            cfgs: Base configuration object (modified in-place)
            train_terminal_cfgs: Terminal training configuration overrides
        """
        if train_terminal_cfgs:
            # Avoid repeatedly recording the env_id and algo
            if 'env_id' in train_terminal_cfgs:
                train_terminal_cfgs.pop('env_id')
            if 'algo' in train_terminal_cfgs:
                train_terminal_cfgs.pop('algo')

            # Validate the keys of train_terminal_cfgs configuration
            recursive_check_config(train_terminal_cfgs, cfgs.train_cfgs)

            # Update the cfgs.train_cfgs from train_terminal configurations
            cfgs.train_cfgs.recurisve_update(train_terminal_cfgs)

            # Save configurations specified in current experiment
            cfgs.recurisve_update({'exp_increment_cfgs': {'train_cfgs': train_terminal_cfgs}})
