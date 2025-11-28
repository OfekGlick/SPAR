"""
Highway-env specific configuration for BAFS experiments.

Contains:
- Algorithm lists
- Default training configuration
- Default launch script parameters
"""

import torch
from itertools import combinations


def generate_sensor_subsets():
    """Generate all non-empty subsets of the 4 highway modalities.

    Note: Use with budget-aware-highway-fast-v0 which has all 4 modalities.
    intersection-v1 only has 3 modalities (no TimeToCollision).

    Returns:
        List of 15 sensor configurations (each is a list of modality names)
    """
    # Highway modalities (matching BudgetAwareHighway.DEFAULT_TYPES)
    # All 4 are available in highway-fast-v0
    modalities = ['Kinematics', 'LidarObservation', 'OccupancyGrid', 'TimeToCollision']

    # Generate all combinations of length 1 to 4 (2^4 - 1 = 15 subsets)
    subsets = []
    for length in range(1, len(modalities) + 1):
        for combo in combinations(modalities, length):
            subsets.append(list(combo))
    return subsets

# ══════════════════════════════════════════════════════════════════════════════
# Algorithm Lists
# ══════════════════════════════════════════════════════════════════════════════

SAFE_ALGOS = ['PPOLag']
UNSAFE_ALGOS = ['PPO']

# ══════════════════════════════════════════════════════════════════════════════
# Training Configuration (for run_bafs_highway.py)
# ══════════════════════════════════════════════════════════════════════════════

CUSTOM_CFGS = {
    'train_cfgs': {
        'vector_env_nums': 1,
        'parallel': 1,
        'device': f'cuda:0' if torch.cuda.is_available() else 'cpu'
    },
    'algo_cfgs': {
        'steps_per_epoch': 8192,
        'update_iters': 40,
        'batch_size': 1024,
        'gamma': 0.9,
        'kl_early_stop': True,
    },
    'model_cfgs': {
        'actor_type': 'auto'  # Auto-detect based on action space
    },
    'logger_cfgs': {
        'wandb_project': 'SPAR Highway - Learning and Sample Efficiency',
        'use_wandb': True,
        'manifest_filename': 'run_manifest.csv',  # Can be overridden via --manifest-filename
    },
    'env_cfgs': {
        'render_mode': 'rgb_array',
        'config': {
            # Environment configuration matching original highway-env defaults
            "destination": None,  # Random turn each episode (adds task diversity)
            "simulation_frequency": 15,  # [Hz]
            "policy_frequency": 1,  # [Hz] - Original 15:1 ratio (reverted from 5)
        },
    },
}

# ══════════════════════════════════════════════════════════════════════════════
# Launch Script Defaults (for launch_highway.py)
# ══════════════════════════════════════════════════════════════════════════════

DEFAULT_LAUNCH_PARAMS = {
    'run_py': '/home/ofek.glick/SPAR/run_bafs_highway.py',
    'sbatch_template': 'sbatch_template_general.sh',
    'sbatch_dir': './sbatch_files_highway',
    'envs': [
        # "budget-aware-highway-fast-v0",
        # "budget-aware-roundabout-v0",
        "budget-aware-intersection-v1",
    ],
    'budget_ratios': [0.2, 0.5, 0.8],
    'cost_usage': [1, 0],
    'all_obs_usage': [1, 0],
    'random_obs_selection': [1, 0],
    'sd_regulizer': [1, 0],
    'penalty_coef': [0.0, 1.0],  # Penalty coefficient for PPO (0=no penalty, 1=full cost penalty)
    'seeds': list(range(10, 20)),
    'total_steps': 409_600,
    'eval_num_episodes': 250,
    'max_episode_steps': 250,
    'steps_per_epoch': 8192,

    # ── Sensor subset configurations ──────────────────────────────────────────
    # Each entry generates separate jobs × all seeds
    # None = all sensors (default: ['Kinematics', 'LidarObservation',
    #                               'OccupancyGrid', 'TimeToCollision'])
    # generate_sensor_subsets() = all 15 non-empty subsets
    'sensor_configs': [
        None,  # Default: all 4 sensors
        # Example custom configs (uncomment to use):
        # ['Kinematics', 'LidarObservation'],
        # ['Kinematics', 'OccupancyGrid'],
    ],
}
