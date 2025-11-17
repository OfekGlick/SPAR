"""
Highway-env specific configuration for BAFS experiments.

Contains:
- Algorithm lists
- Default training configuration
- Default launch script parameters
"""

import torch

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
    },
    'env_cfgs': {
        'render_mode': 'rgb_array',
        'config': {
            # Dense traffic configuration
            "initial_spacing": 2,  # Closer initial spacing (meters)
        },
    },
}

# ══════════════════════════════════════════════════════════════════════════════
# Launch Script Defaults (for launch_highway.py)
# ══════════════════════════════════════════════════════════════════════════════

DEFAULT_LAUNCH_PARAMS = {
    'run_py': '/home/ofek.glick/SPAR/run_bafs_highway.py',
    'sbatch_template': 'sbatch_template_clair.sh',
    'sbatch_dir': './sbatch_files_highway',
    'envs': [
        "budget-aware-roundabout-v0",
        "budget-aware-intersection-v1",
        "budget-aware-highway-fast-v0",
    ],
    'budget_ratios': [0.2, 0.5, 0.8],
    'cost_usage': [1, 0],
    'all_obs_usage': [1, 0],
    'random_obs_selection': [1, 0],
    'sd_regulizer': [1, 0],
    'penalty_coef': [0.0, 1.0],  # Penalty coefficient for PPO (0=no penalty, 1=full cost penalty)
    'seeds': list(range(0, 10)),
    'total_steps': 500_000,
    'eval_num_episodes': 250,
    'max_episode_steps': 250,
    'steps_per_epoch': 8192,
}
