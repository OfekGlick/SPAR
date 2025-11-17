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
"""Results logger for OmniSafe algorithms."""

from __future__ import annotations

import os
import csv
import json
from typing import Any
from datetime import datetime
import wandb
from omnisafe.algorithms.utils.evaluation_results import EvaluationResults


class ResultsLogger:
    """Centralized logging for evaluation results.

    Handles logging to:
    - rliable JSON format (for statistical analysis)
    - CSV manifest (for tracking experiments)
    - wandb (for visualization)
    """

    def __init__(self, rliable_json_path: str):
        """Initialize results logger.

        Args:
            rliable_json_path: Base directory for rliable JSON and manifest files
        """
        self.rliable_json_path = rliable_json_path
        self.manifest_path = os.path.join(rliable_json_path, "run_manifest.csv")
        self.json_path = os.path.join(rliable_json_path, "results.json")

    def log_to_rliable_json(self, bucket: str, key: str, value: float | list[float]) -> None:
        """Merge results into an rliable-style JSON file.

        Format: { "<bucket>": { "<key>": [... values ...] } }

        Args:
            bucket: Bucket name (e.g., "env_obs_mode_budget")
            key: Key name (e.g., "PPO_reward")
            value: Value(s) to append
        """
        os.makedirs(os.path.dirname(self.json_path), exist_ok=True)

        # Load existing data
        try:
            with open(self.json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = {}

        # Update data
        data.setdefault(bucket, {})
        data[bucket].setdefault(key, [])
        data[bucket][key].extend([value])

        # Write atomically (using temp file)
        tmp = self.json_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=True)
        os.replace(tmp, self.json_path)

        # Log confirmation
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] Saved to results.json: bucket='{bucket}', key='{key}', value={value}")
        print(f"[{timestamp}] File location: {self.json_path}")

    def log_to_manifest(self, metadata: dict[str, Any]) -> None:
        """Append run details to CSV manifest for tracking and diagnosis.

        Args:
            metadata: Dictionary containing run metadata (see fieldnames for expected keys)
        """
        os.makedirs(os.path.dirname(self.manifest_path), exist_ok=True)

        # Check if file exists to determine if we need headers
        file_exists = os.path.exists(self.manifest_path)

        # Define column order
        fieldnames = [
            'timestamp', 'algo', 'env', 'seed', 'budget', 'obs_mode', 'actor_type',
            'action_space_type', 'obs_space_shape',
            'use_cost', 'use_all_obs', 'sd_regulizer', 'random_obs_selection',
            'total_steps', 'num_eval_episodes', 'reward_mean', 'reward_std',
            'cost_mean', 'cost_std', 'episode_rewards', 'episode_costs',
            'sample_efficiency_curve',
            'status', 'log_dir', 'cost_normalized', 'reward_normalized', 'obs_modality_normalize'
        ]

        # Write to CSV with append mode
        with open(self.manifest_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(metadata)

        timestamp = metadata.get('timestamp', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        print(f"[{timestamp}] Logged run to manifest: {self.manifest_path}")

    def log_to_wandb(
        self,
        logger: Any,
        results: EvaluationResults,
        use_all_obs: bool = False,
    ) -> None:
        """Log evaluation results to wandb with visualizations.

        Args:
            logger: Logger instance with wandb_log method
            results: Evaluation results to visualize
            use_all_obs: Whether all observations were used (affects which plots to create)
        """
        # Only create feature selection plot if not using all observations
        if not use_all_obs and results.episode_obs_masks is not None:
            obs_masks = results.get_obs_mask_mean()
            obs_mask_data = wandb.Table(
                data=[[f'Feature {i + 1}', importance] for i, importance in enumerate(obs_masks)],
                columns=["Feature", "Average Time On"],
            )
            logger.wandb_log(
                {
                    "Feature Selection": wandb.plot.bar(
                        obs_mask_data,
                        label="Feature",
                        value="Average Time On",
                        title="Feature Selection During Episode",
                    )
                }
            )

        # Create sensor costs plot if available
        if results.sensor_costs is not None:
            sensor_costs_data = wandb.Table(
                data=[[f'Feature {i + 1}', cost] for i, cost in enumerate(results.sensor_costs)],
                columns=["Feature", "Cost"],
            )
            logger.wandb_log(
                {
                    "Sensor Costs": wandb.plot.bar(
                        sensor_costs_data,
                        label="Feature",
                        value="Cost",
                        title="Sensor Costs",
                    )
                }
            )

    def create_bucket_name(self, env_id: str, obs_mode: str, budget_str: str, steps_str: int) -> str:
        """Create standardized bucket name for rliable JSON.

        Args:
            env_id: Environment identifier
            obs_mode: Observation mode (AllObs, RandomMask, or SelectedObs)
            budget_str: Budget string (e.g., "Budget10" or "BudgetNone")
            steps_str: Total training steps

        Returns:
            Formatted bucket name
        """
        return f'{env_id}_{obs_mode}_{budget_str}_{steps_str}_steps'
