import omnisafe
import argparse
from pprint import pprint
import omnisafe.algorithms.on_policy.second_order.cpo
from args_utils import parse_arguments
import torch
import gymnasium as gym
from gymnasium import spaces
from omnisafe.envs.core import make as omnisafe_make

from bafs_envs import budget_aware_highway
custom_cfgs = {
    'train_cfgs': {
        'total_steps': 409_600,
        'vector_env_nums': 1,
        'parallel': 1,
        'device': f'cuda:0' if torch.cuda.is_available() else 'cpu'
    },
    'algo_cfgs': {
        'steps_per_epoch': 8192,
        'update_iters': 40,
        'batch_size': 512,
        'gamma': 0.9,
        'zero_barrier_eps': 1.0e-8,  # numerical clamp inside log
        'zero_barrier_coef': 0.1,  # strength of the regularizer
        'kl_early_stop': False,
        'target_kl': 0.5,
        # 'obs_modality_normalize': True,
    },
    'model_cfgs': {
        'actor_type': 'auto'  # Auto-detect based on action space
    },
    'logger_cfgs': {
        'wandb_project': 'BAFS 2.2 - Highway',
        'use_wandb': True,
    },
    'env_cfgs': {
        'render_mode': 'rgb_array',
        'config': {
            # Dense traffic configuration
            # "vehicles_count": 100,  # Double the vehicles
            "initial_spacing": 2,  # Closer initial spacing (meters)
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
    if args.algo in ['SACLag', 'SACPID', 'PPOLag', 'SPOLag', 'SACLag', 'CPPOPID']:
        custom_cfgs['lagrange_cfgs'] = {
            'cost_limit': args.budget
        }
    if args.algo in ['CPO']:
        custom_cfgs['algo_cfgs']['cost_limit'] = args.budget
    elif args.algo in ['PPOSaute', 'TRPOSaute', 'PPOSimmer']:
        custom_cfgs['algo_cfgs']['safety_budget'] = args.budget
        custom_cfgs['algo_cfgs']['saute_gamma'] = 0.99999  # How much to discount the future safety budget
        custom_cfgs['algo_cfgs']['unsafe_reward'] = -1
        # custom_cfgs['train_cfgs']['device'] = 'cpu'
    custom_cfgs['train_cfgs']['total_steps'] = args.total_steps
    custom_cfgs['algo_cfgs']['use_cost'] = args.use_cost
    custom_cfgs['env_cfgs']['max_episode_steps'] = args.max_episode_steps
    custom_cfgs['algo_cfgs']['steps_per_epoch'] = args.steps_per_epoch
    custom_cfgs['env_cfgs']['use_all_obs'] = args.use_all_obs
    custom_cfgs['env_cfgs']['seed'] = args.seed
    custom_cfgs['algo_cfgs']['sd_regulizer'] = args.sd_regulizer
    custom_cfgs['algo_cfgs']['no_zero_act'] = args.no_zero_act

    # Random mask baseline: override actor type and ensure proper env config
    if args.random_obs_selection:
        custom_cfgs['model_cfgs']['actor_type'] = 'random_mask'
        custom_cfgs['env_cfgs']['use_all_obs'] = False  # Must use BAFS env with masks
        print("=== RANDOM MASK BASELINE MODE ===")
        print("Actor will learn environment actions but use random modality selection")
    # try:
    #     custom_cfgs['env_cfgs']['feature_costs'] = [float(val) for val in args.feature_cost]
    # except:
    #     custom_cfgs['env_cfgs']['feature_costs'] = None
    # custom_cfgs['env_cfgs']['features_to_use'] = [int(val) for val in
    #                                               args.features_to_use] if args.features_to_use else None


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
            return 'multihead'
        elif isinstance(env_action_space, spaces.Discrete):
            print(f"Detected Tuple[Discrete, MultiBinary] action space → using 'multihead' (discrete)")
            return 'multihead'
        else:
            raise ValueError(f"Unsupported action space in Tuple: {type(env_action_space)}")

    elif isinstance(action_space, spaces.Box):
        # Continuous action space
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
    Train a safe agent.
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
    # if args.use_all_obs:
    #     agent.sample_random_actions(num_episodes=1_000, max_episode_length=500)
    agent.learn()
    agent.evaluate(num_episodes=eval_num_episodes)


if __name__ == '__main__':
    args, unparsed_args = parse_arguments()
    adjust_config(custom_cfgs, args)
    # custom_cfgs['env_cfgs']['max_episode_steps'] = 20
    # custom_cfgs['env_cfgs']['render_mode'] = 'rgb_array'
    # custom_cfgs['train_cfgs']['total_steps'] = 80
    # custom_cfgs['algo_cfgs']['steps_per_epoch'] = 20
    # custom_cfgs['algo_cfgs']['sd_regulizer'] = True
    # custom_cfgs['env_cfgs']['use_all_obs'] = True
    # args.eval_num_episodes = 20
    assert custom_cfgs['env_cfgs']['max_episode_steps'] <= custom_cfgs['algo_cfgs']['steps_per_epoch'], \
        'Max episode steps should be less than or equal to steps per epoch - otherwise you wont get any episodic data'
    pprint(custom_cfgs)
    train_agent(eval_num_episodes=args.eval_num_episodes)
