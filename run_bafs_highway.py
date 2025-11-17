"""
Training script for Highway-env BAFS experiments.

Trains safe RL agents with budget-aware observation masking on highway-env tasks.
"""

from pprint import pprint
from utils.args_utils import parse_arguments

# Import shared utilities
from utils.training_utils import adjust_config_base, train_agent_base

# Import highway-specific configuration
from configs.highway_config import CUSTOM_CFGS


def adjust_config(custom_cfgs: dict, args) -> None:
    """Adjust configuration for highway-specific settings.

    Args:
        custom_cfgs: Configuration dictionary to modify
        args: Parsed command-line arguments
    """
    # Apply base configuration (common across all environments)
    adjust_config_base(custom_cfgs, args)

    # Highway-specific adjustments (none currently needed)
    # Future: Add any highway-specific config overrides here


def main():
    """Main training function."""
    args, unparsed_args = parse_arguments()
    adjust_config(CUSTOM_CFGS, args)

    # Validate configuration
    assert CUSTOM_CFGS['env_cfgs']['max_episode_steps'] <= CUSTOM_CFGS['algo_cfgs']['steps_per_epoch'], \
        'Max episode steps should be less than or equal to steps per epoch - otherwise you won\'t get any episodic data'

    print("\n" + "="*80)
    print("HIGHWAY TRAINING CONFIGURATION")
    print("="*80)
    pprint(CUSTOM_CFGS)
    print("="*80 + "\n")

    # Train agent using shared utilities
    train_agent_base(
        algo=args.algo,
        env_id=args.env_id,
        custom_cfgs=CUSTOM_CFGS,
        eval_num_episodes=args.eval_num_episodes
    )


if __name__ == '__main__':
    main()
