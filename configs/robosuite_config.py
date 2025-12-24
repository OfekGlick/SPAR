"""
Robosuite-env specific configuration for SPAR experiments.

Contains:
- Algorithm lists
- Default training configuration
- Default launch script parameters
- Default modality costs
"""

import torch

# ══════════════════════════════════════════════════════════════════════════════
# Algorithm Lists
# ══════════════════════════════════════════════════════════════════════════════

SAFE_ALGOS = ['PPOLag']
UNSAFE_ALGOS = ['PPO']

# ══════════════════════════════════════════════════════════════════════════════
# Training Configuration (for run_spar_robosuite.py)
# ══════════════════════════════════════════════════════════════════════════════

CUSTOM_CFGS = {
    'train_cfgs': {
        'vector_env_nums': 1,
        'parallel': 1,
        'device': f'cuda:0' if torch.cuda.is_available() else 'cpu'
    },
    'algo_cfgs': {
        'update_iters': 40,
        'batch_size': 512,      # Typical for PPO with robosuite
        'gamma': 0.95,
        'kl_early_stop': True,
    },
    'model_cfgs': {
        'actor_type': 'auto',  # Auto-detect based on action space
    },
    'logger_cfgs': {
        'wandb_project': 'SPAR Robosuite - Learning and Sample Efficiency',
        'use_wandb': True,
    },
    'env_cfgs': {
        'robot': 'Panda',          # Recommended robot for benchmarking
        'control_freq': 20,
        'reward_scale': 1.0,
        'reward_shaping': True,    # Dense rewards for better learning

        # Modality costs (can be adjusted)
        'modality_costs': {
            'robot_proprioception': 1.0,  # Joint states, EEF pose, gripper
            'object_states': 1.0,          # Object positions/orientations
            'task_features': 1.0,          # Relative distances, task-specific
            'camera': 1.0,                 # Grayscale camera images (16x16)
        },
    },
}

# ══════════════════════════════════════════════════════════════════════════════
# Launch Script Defaults (for launch_robosuite.py)
# ══════════════════════════════════════════════════════════════════════════════

DEFAULT_LAUNCH_PARAMS = {
    'run_py': '/home/ofek.glick/SPAR/run_spar_robosuite.py',
    'sbatch_template': 'sbatch_template_clair.sh',
    'sbatch_dir': './sbatch_files_robosuite',
    'envs': [
        "budget-aware-Lift",
        "budget-aware-Door",
    ],
    'robot': 'Panda',
    'budget_ratios': [0.2, 0.5, 0.8],
    'cost_usage': [1, 0],
    'all_obs_usage': [0, 1],
    'random_obs_selection': [0, 1],
    'sd_regulizer': [1, 0],
    'penalty_coef': [0.0, 1.0],  # Penalty coefficient for PPO (0=no penalty, 1=full cost penalty)
    'seeds': list(range(0, 10)),
    'total_steps': 2_000_000,
    'eval_num_episodes': 50,
    'max_episode_steps': 500,
    'steps_per_epoch': 8192,
}

# ══════════════════════════════════════════════════════════════════════════════
# Default Modality Costs
# ══════════════════════════════════════════════════════════════════════════════

DEFAULT_MODALITY_COSTS = {
    'robot_proprioception': 1.0,
    'object_states': 1.0,
    'task_features': 1.0,
    'camera': 1.0,
}
