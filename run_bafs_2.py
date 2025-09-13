import omnisafe
import argparse
from pprint import pprint
import omnisafe.algorithms.on_policy.second_order.cpo
from args_utils import parse_arguments
import torch
import budget_aware_highway

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
        'zero_barrier_coef': 0.1,     # strength of the regularizer
        'kl_early_stop': True,
        # 'obs_modality_normalize': True,
    },
    'model_cfgs': {
        'actor_type': 'multihead'
    },
    'logger_cfgs': {
        'wandb_project': 'BAFS 2.2',
        'use_wandb': True,
    },
    'env_cfgs': {
        'use_all_obs': False,
        'max_episode_steps': 250,
        "action": {
            "type": "ContinuousAction"
        },
        'render_mode': 'rgb_array',
        'config': {
            "observation": {
                "type": "Kinematics",
                "absolute": False,
                "features": ["x", "y", "vx", "vy", "heading", "lane_id"],
                "vehicles_count": 15,
            },
            "policy_frequency": 1,
            "simulation_frequency": 15,
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
    if args.algo in ['SACLag', 'SACPID', 'PPOLag', 'SPOLag', 'SACLag']:
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
    custom_cfgs['env_cfgs']['use_all_obs'] = args.use_all_obs
    custom_cfgs['env_cfgs']['seed'] = args.seed
    custom_cfgs['algo_cfgs']['sd_regulizer'] = args.sd_regulizer
    custom_cfgs['algo_cfgs']['no_zero_act'] = args.no_zero_act
    # try:
    #     custom_cfgs['env_cfgs']['feature_costs'] = [float(val) for val in args.feature_cost]
    # except:
    #     custom_cfgs['env_cfgs']['feature_costs'] = None
    # custom_cfgs['env_cfgs']['features_to_use'] = [int(val) for val in
    #                                               args.features_to_use] if args.features_to_use else None


def train_agent(eval_num_episodes=50):
    """
    Train a safe agent.
    Args:
        eval_num_episodes (int): The number of episodes to evaluate the agent.
    Returns:
        None
    """
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
    # custom_cfgs['train_cfgs']['total_steps'] = 100
    # custom_cfgs['algo_cfgs']['steps_per_epoch'] = 20
    # custom_cfgs['algo_cfgs']['sd_regulizer'] = True
    # custom_cfgs['env_cfgs']['use_all_obs'] = True
    # args.eval_num_episodes = 5
    assert custom_cfgs['env_cfgs']['max_episode_steps'] <= custom_cfgs['algo_cfgs'][
        'steps_per_epoch'], 'Max episode steps should be less than or equal to steps per epoch - otherwise you wont get any episodic data'
    pprint(custom_cfgs)
    train_agent(eval_num_episodes=args.eval_num_episodes)
