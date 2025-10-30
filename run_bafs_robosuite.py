import omnisafe
import argparse
from pprint import pprint
from args_utils import parse_arguments
import torch
import gymnasium as gym
from gymnasium import spaces
from omnisafe.envs.core import make as omnisafe_make
from bafs_envs import budget_aware_robosuite

custom_cfgs = {
    'train_cfgs': {
        'vector_env_nums': 1,
        'parallel': 1,
        'device': f'cuda:0' if torch.cuda.is_available() else 'cpu'
    },
    'algo_cfgs': {
        'update_iters': 40,
        'batch_size': 512,      # Typical for PPO with robosuite
        'gamma': 0.95,
        'zero_barrier_eps': 1.0e-8,  # numerical clamp inside log
        'zero_barrier_coef': 0.1,    # strength of the regularizer
        'kl_early_stop': True,
    },
    'model_cfgs': {
        'actor_type': 'auto',  # Auto-detect based on action space
    },
    'logger_cfgs': {
        'wandb_project': 'BAFS 2.5 - Robosuite',
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
        },
    },
}


def adjust_config(custom_cfgs, args):
    """
    Adjust the configuration according to the arguments passed.

    Args:
        custom_cfgs (dict): The configuration dictionary.
        args (argparse.Namespace): The arguments passed.

    Returns:
        None
    """
    # Lagrangian-based algorithms
    if args.algo in ['SACLag', 'SACPID', 'PPOLag', 'SPOLag', 'CPPOPID']:
        custom_cfgs['lagrange_cfgs'] = {
            'cost_limit': args.budget
        }

    # CPO uses cost_limit in algo_cfgs
    if args.algo in ['CPO']:
        custom_cfgs['algo_cfgs']['cost_limit'] = args.budget

    # Saute/Simmer-based algorithms
    elif args.algo in ['PPOSaute', 'TRPOSaute', 'PPOSimmer']:
        custom_cfgs['algo_cfgs']['safety_budget'] = args.budget
        custom_cfgs['algo_cfgs']['saute_gamma'] = 0.99999  # How much to discount the future safety budget
        custom_cfgs['algo_cfgs']['unsafe_reward'] = -1

    # Update training configurations
    custom_cfgs['train_cfgs']['total_steps'] = args.total_steps
    custom_cfgs['algo_cfgs']['use_cost'] = args.use_cost
    custom_cfgs['env_cfgs']['max_episode_steps'] = args.max_episode_steps
    custom_cfgs['algo_cfgs']['steps_per_epoch'] = args.steps_per_epoch
    custom_cfgs['env_cfgs']['use_all_obs'] = args.use_all_obs
    custom_cfgs['env_cfgs']['seed'] = args.seed
    custom_cfgs['algo_cfgs']['sd_regulizer'] = args.sd_regulizer
    custom_cfgs['algo_cfgs']['no_zero_act'] = args.no_zero_act
    custom_cfgs['algo_cfgs']['obs_modality_normalize'] = args.obs_modality_normalize
    # Random mask baseline: override actor type and ensure proper env config
    if args.random_obs_selection:
        custom_cfgs['model_cfgs']['actor_type'] = 'random_mask'
        custom_cfgs['env_cfgs']['use_all_obs'] = False  # Must use BAFS env with masks
        print("=== RANDOM MASK BASELINE MODE ===")
        print("Actor will learn environment actions but use random modality selection")


def detect_actor_type(env_id, env_cfgs):
    """
    Auto-detect the appropriate actor type based on the environment's action space.

    Args:
        env_id (str): Environment ID
        env_cfgs (dict): Environment configuration

    Returns:
        str: Appropriate actor type ('gaussian_learning', 'categorical_learning', or 'multihead')
    """
    # Create a temporary environment to inspect action space
    try:
        # Try creating with OmniSafe's make() for custom registered environments
        temp_env = omnisafe_make(env_id, num_envs=1, **env_cfgs)
    except:
        # Fallback for standard gymnasium environments
        try:
            temp_env = gym.make(env_id, **env_cfgs)
        except:
            temp_env = gym.make(env_id)

    action_space = temp_env.action_space
    temp_env.close()

    # Determine actor type based on action space
    if isinstance(action_space, spaces.Tuple):
        # Multi-head actor: check first element (env action)
        env_action_space = action_space[0]
        if isinstance(env_action_space, spaces.Box):
            print(f"Detected Tuple[Box, MultiBinary] action space → using 'multihead' (continuous)")
            return 'multihead_double'
        elif isinstance(env_action_space, spaces.Discrete):
            print(f"Detected Tuple[Discrete, MultiBinary] action space → using 'multihead' (discrete)")
            return 'multihead_double'
        else:
            raise ValueError(f"Unsupported action space in Tuple: {type(env_action_space)}")

    elif isinstance(action_space, spaces.Box):
        # Continuous action space (default for robosuite)
        print(f"Detected Box action space → using 'gaussian_learning'")
        return 'gaussian_learning'

    elif isinstance(action_space, spaces.Discrete):
        # Discrete action space
        print(f"Detected Discrete action space → using 'categorical_learning'")
        return 'categorical_learning'

    else:
        raise ValueError(f"Unsupported action space type: {type(action_space)}")


def train_agent(eval_num_episodes=50):
    """
    Train a safe agent on robosuite tasks.

    Args:
        eval_num_episodes (int): The number of episodes to evaluate the agent.

    Returns:
        None
    """
    # Auto-detect actor type if not explicitly set or if set to 'auto'
    if 'actor_type' not in custom_cfgs.get('model_cfgs', {}) or custom_cfgs['model_cfgs']['actor_type'] == 'auto':
        detected_actor_type = detect_actor_type(args.env_id, custom_cfgs.get('env_cfgs', {}))
        custom_cfgs.setdefault('model_cfgs', {})['actor_type'] = detected_actor_type

    agent = omnisafe.Agent(args.algo, args.env_id, custom_cfgs=custom_cfgs)
    # agent.learn()
    agent.learn_with_sample_efficiency()
    agent.evaluate(num_episodes=eval_num_episodes)


if __name__ == '__main__':
    args, unparsed_args = parse_arguments()
    adjust_config(custom_cfgs, args)

    # Validate configuration
    assert custom_cfgs['env_cfgs']['max_episode_steps'] <= custom_cfgs['algo_cfgs']['steps_per_epoch'], \
        'Max episode steps should be less than or equal to steps per epoch - otherwise you won\'t get any episodic data'

    print("\n" + "="*80)
    print("ROBOSUITE TRAINING CONFIGURATION")
    print("="*80)
    print(f"Environment: {args.env_id}")
    print(f"Algorithm: {args.algo}")
    print(f"Robot: {custom_cfgs['env_cfgs']['robot']}")
    print(f"Budget: {args.budget}")
    print(f"Use all observations: {args.use_all_obs}")
    print(f"Total steps: {custom_cfgs['train_cfgs']['total_steps']}")
    print(f"Episode length: {custom_cfgs['env_cfgs']['max_episode_steps']}")
    print("="*80 + "\n")

    pprint(custom_cfgs)
    print("\n")

    train_agent(eval_num_episodes=args.eval_num_episodes)
