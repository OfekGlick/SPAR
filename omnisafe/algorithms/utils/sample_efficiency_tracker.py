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
"""Sample efficiency tracker for OmniSafe algorithms."""

from __future__ import annotations

import os
from typing import Any
import numpy as np
from omnisafe.evaluator import Evaluator


class SampleEfficiencyTracker:
    """Manages periodic evaluations during training for sample efficiency tracking.

    Evaluates the policy at regular intervals and stores results for later analysis.
    """

    def __init__(
        self,
        eval_fraction: float = 0.05,
        eval_episodes: int = 50,
    ):
        """Initialize sample efficiency tracker.

        Args:
            eval_fraction: Fraction of total training steps between evaluations (default: 0.05 = 5%)
            eval_episodes: Episodes per evaluation (default: 50)
        """
        self.eval_fraction = eval_fraction
        self.eval_episodes = eval_episodes
        self._data: dict[str, Any] = {}
        self._eval_interval: int = 0
        self._epochs_per_eval: int = 0

    def setup_for_training(
        self,
        total_steps: int,
        steps_per_epoch: int,
        original_save_freq: int,
    ) -> tuple[int, int]:
        """Calculate evaluation intervals and adjust save frequency.

        Args:
            total_steps: Total training steps
            steps_per_epoch: Steps per epoch
            original_save_freq: Original model save frequency

        Returns:
            Tuple of (eval_interval, epochs_per_eval) for training loop
        """
        # Calculate dynamic interval based on total training steps
        self._eval_interval = int(total_steps * self.eval_fraction)

        # Calculate epochs per evaluation
        self._epochs_per_eval = int(self._eval_interval / steps_per_epoch)

        # Print tracking info
        print(f"\n{'='*80}")
        print(f"SAMPLE EFFICIENCY TRACKING ENABLED")
        print(f"{'='*80}")
        print(f"Total training steps: {total_steps:,}")
        print(f"Steps per epoch: {steps_per_epoch:,}")
        print(f"Evaluation interval: {self._eval_interval:,} steps ({self.eval_fraction*100:.0f}%)")
        print(f"Epochs per evaluation: {self._epochs_per_eval}")
        print(f"Episodes per evaluation: {self.eval_episodes}")
        print(f"Expected checkpoints: ~{int(1/self.eval_fraction)}")
        print(f"Checkpoint save frequency: every {self._epochs_per_eval} epochs (auto-adjusted)")
        print(f"{'='*80}\n")

        return self._eval_interval, self._epochs_per_eval

    def run_periodic_eval(
        self,
        log_dir: str,
        total_steps: int,
        sample_efc: bool = True,
    ) -> None:
        """Execute periodic evaluation and store in memory.

        Args:
            log_dir: Directory containing saved checkpoints
            total_steps: Current training step count
            sample_efc: Whether this is a sample efficiency evaluation
        """
        # Find the most recently saved checkpoint
        torch_save_dir = os.path.join(log_dir, 'torch_save')
        checkpoints = [f for f in os.listdir(torch_save_dir) if f.endswith('.pt')]
        if not checkpoints:
            print(f"[Warning] No checkpoints found for evaluation at {total_steps} steps")
            return

        # Use the most recent checkpoint (highest epoch number)
        checkpoint_name = sorted(checkpoints, key=lambda x: int(x.split('-')[1].split('.')[0]))[-1]

        # Load and evaluate the checkpoint
        evaluator = Evaluator()
        evaluator.load_saved(
            save_dir=log_dir,
            model_name=checkpoint_name,
            render_mode='rgb_array'
        )
        results = evaluator.evaluate(
            num_episodes=self.eval_episodes,
            cost_criteria=1.0,
            record_video=False,  # Skip video for intermediate evaluations
            sample_efc=sample_efc
        )

        # Store in running dictionary (will be saved to CSV at final evaluation)
        self._data[str(total_steps)] = {
            'reward_mean': float(np.mean(results['episode_rewards'])),
            'reward_std': float(np.std(results['episode_rewards'])),
            'cost_mean': float(np.mean(results['episode_costs'])),
            'cost_std': float(np.std(results['episode_costs'])),
            'episode_rewards': [float(r) for r in results['episode_rewards']],
            'episode_costs': [float(c) for c in results['episode_costs']],
        }

        print(f"[Periodic Eval @ {total_steps} steps] "
              f"Reward: {self._data[str(total_steps)]['reward_mean']:.2f} ± "
              f"{self._data[str(total_steps)]['reward_std']:.2f}, "
              f"Cost: {self._data[str(total_steps)]['cost_mean']:.2f} ± "
              f"{self._data[str(total_steps)]['cost_std']:.2f}")

        evaluator._env.close()

    def get_results(self) -> dict[str, Any]:
        """Get accumulated evaluation results.

        Returns:
            Dictionary mapping step counts to evaluation metrics
        """
        return self._data

    def has_data(self) -> bool:
        """Check if any evaluation data has been collected.

        Returns:
            True if data exists, False otherwise
        """
        return len(self._data) > 0
