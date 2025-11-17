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
"""Evaluation results container for OmniSafe algorithms."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import numpy as np


@dataclass
class EvaluationResults:
    """Structured container for evaluation data.

    Provides type-safe storage and convenient access methods for evaluation metrics.

    Attributes:
        episode_rewards: List of episode rewards
        episode_costs: List of episode costs
        episode_obs_masks: Array of observation masks per episode
        sensor_costs: Array of sensor costs
        additional_metrics: Dict for storing custom metrics
    """

    episode_rewards: list[float] = field(default_factory=list)
    episode_costs: list[float] = field(default_factory=list)
    episode_obs_masks: np.ndarray | None = None
    sensor_costs: np.ndarray | None = None
    additional_metrics: dict[str, Any] = field(default_factory=dict)

    def get_reward_mean(self) -> float:
        """Calculate mean episode reward."""
        return float(np.mean(self.episode_rewards))

    def get_reward_std(self) -> float:
        """Calculate standard deviation of episode rewards."""
        return float(np.std(self.episode_rewards))

    def get_cost_mean(self) -> float:
        """Calculate mean episode cost."""
        return float(np.mean(self.episode_costs))

    def get_cost_std(self) -> float:
        """Calculate standard deviation of episode costs."""
        return float(np.std(self.episode_costs))

    def get_obs_mask_mean(self) -> np.ndarray:
        """Calculate mean observation mask across episodes."""
        if self.episode_obs_masks is None:
            raise ValueError("No observation masks available")
        return np.mean(self.episode_obs_masks, axis=0)

    def get_sensor_usage_mean(self) -> list[float]:
        """Calculate mean sensor usage from observation masks."""
        return self.get_obs_mask_mean().tolist()

    def get_sensor_usage_std(self) -> list[float]:
        """Calculate standard deviation of sensor usage from observation masks."""
        if self.episode_obs_masks is None:
            raise ValueError("No observation masks available")
        return np.std(self.episode_obs_masks, axis=0).tolist()

    def to_dict(self) -> dict[str, Any]:
        """Convert results to dictionary format.

        Returns:
            Dictionary containing all metrics
        """
        result = {
            'episode_rewards': self.episode_rewards,
            'episode_costs': self.episode_costs,
            'reward_mean': self.get_reward_mean(),
            'reward_std': self.get_reward_std(),
            'cost_mean': self.get_cost_mean(),
            'cost_std': self.get_cost_std(),
        }

        # Add observation mask metrics if available
        if self.episode_obs_masks is not None:
            result['obs_mask_mean'] = self.get_obs_mask_mean().tolist()
            result['sensor_usage_mean'] = self.get_sensor_usage_mean()
            result['sensor_usage_std'] = self.get_sensor_usage_std()

        # Add sensor costs if available
        if self.sensor_costs is not None:
            result['sensor_costs'] = self.sensor_costs.tolist()

        # Add any additional custom metrics
        result.update(self.additional_metrics)

        return result

    @classmethod
    def from_evaluator_results(cls, results: dict[str, Any]) -> EvaluationResults:
        """Create EvaluationResults from raw evaluator output.

        Args:
            results: Dictionary returned by Evaluator.evaluate()

        Returns:
            EvaluationResults instance
        """
        return cls(
            episode_rewards=results.get('episode_rewards', []),
            episode_costs=results.get('episode_costs', []),
            episode_obs_masks=results.get('episode_obs_masks'),
            sensor_costs=results.get('sensor_costs'),
        )
