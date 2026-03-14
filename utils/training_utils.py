"""
Shared utilities for training scripts (run_spar_highway.py, run_spar_robosuite.py).

Provides common functionality for:
- Actor type auto-detection
- Configuration adjustment for different algorithms
- Base training logic
"""

import gymnasium as gym
from gymnasium import spaces
from omnisafe.envs.core import make as omnisafe_make
import omnisafe
import spar_envs.budget_aware_highway
try:
    import spar_envs.budget_aware_robosuite
except ImportError:
    pass  # robosuite not installed (e.g., on Colab)


def detect_actor_type(env_id: str, env_cfgs: dict) -> str:
    """Auto-detect the appropriate actor type based on the environment's action space.

    Args:
        env_id: Environment ID
        env_cfgs: Environment configuration dictionary

    Returns:
        Appropriate actor type:
        - 'multihead_double' for Tuple[Box/Discrete, MultiBinary] (SPAR environments)
        - 'gaussian_learning' for continuous (Box) action spaces
        - 'categorical_learning' for discrete (Discrete) action spaces

    Raises:
        ValueError: If action space type is not supported
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
            print(f"Detected Tuple[Box, MultiBinary] action space → using 'multihead_double' (continuous)")
            return 'multihead_double'
        elif isinstance(env_action_space, spaces.Discrete):
            print(f"Detected Tuple[Discrete, MultiBinary] action space → using 'multihead_double' (discrete)")
            return 'multihead_double'
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


def adjust_config_base(custom_cfgs: dict, args) -> None:
    """Adjust configuration for different algorithm types (common logic).

    This function handles configuration for:
    - Lagrangian-based algorithms (SACLag, PPOLag, etc.)
    - CPO algorithms
    - Saute/Simmer algorithms
    - Training parameters
    - Random mask baseline mode

    Environment-specific settings should be applied separately after calling this.

    Args:
        custom_cfgs: Configuration dictionary to modify in-place
        args: Parsed command-line arguments

    Returns:
        None (modifies custom_cfgs in-place)
    """
    # ── Algorithm-specific configurations ─────────────────────────────────────

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
        custom_cfgs['algo_cfgs']['saute_gamma'] = 0.99999  # Discount future safety budget
        custom_cfgs['algo_cfgs']['unsafe_reward'] = -1

    # ── Training configurations ───────────────────────────────────────────────

    custom_cfgs['train_cfgs']['total_steps'] = args.total_steps
    custom_cfgs['algo_cfgs']['use_cost'] = args.use_cost
    custom_cfgs['env_cfgs']['max_episode_steps'] = args.max_episode_steps
    custom_cfgs['algo_cfgs']['steps_per_epoch'] = args.steps_per_epoch
    custom_cfgs['env_cfgs']['use_all_obs'] = args.use_all_obs
    custom_cfgs['env_cfgs']['seed'] = args.seed
    custom_cfgs['algo_cfgs']['penalty_coef'] = args.penalty_coef
    custom_cfgs['algo_cfgs']['sd_regulizer'] = args.sd_regulizer
    custom_cfgs['algo_cfgs']['obs_modality_normalize'] = args.obs_modality_normalize

    # ── Sensor subset restriction ─────────────────────────────────────────────
    if hasattr(args, 'available_sensors') and args.available_sensors is not None:
        custom_cfgs['env_cfgs']['available_sensors'] = args.available_sensors
        print(f"[Config] Restricting available sensors to: {args.available_sensors}")

    # ── Manifest filename ──────────────────────────────────────────────────────
    if hasattr(args, 'manifest_filename'):
        custom_cfgs.setdefault('logger_cfgs', {})['manifest_filename'] = args.manifest_filename
        print(f"[Config] Using manifest file: {args.manifest_filename}")

    # ── Wandb project override ─────────────────────────────────────────────────
    if hasattr(args, 'wandb_project') and args.wandb_project is not None:
        custom_cfgs.setdefault('logger_cfgs', {})['wandb_project'] = args.wandb_project
        print(f"[Config] Overriding wandb project: {args.wandb_project}")

    # ── Random mask baseline ──────────────────────────────────────────────────

    if args.random_obs_selection:
        custom_cfgs['model_cfgs']['actor_type'] = 'random_mask'
        custom_cfgs['env_cfgs']['use_all_obs'] = False  # Must use SPAR env with masks
        print("=== RANDOM MASK BASELINE MODE ===")
        print("Actor will learn environment actions but use random modality selection")


def train_agent_base(
    algo: str,
    env_id: str,
    custom_cfgs: dict,
    eval_num_episodes: int = 50
) -> None:
    """Core training loop using OmniSafe (common logic).

    Auto-detects actor type if needed, creates agent, and runs training
    with sample efficiency tracking.

    Args:
        algo: Algorithm name
        env_id: Environment ID
        custom_cfgs: Configuration dictionary
        eval_num_episodes: Number of episodes for evaluation

    Returns:
        None
    """
    # Auto-detect actor type if not explicitly set or if set to 'auto'
    if 'actor_type' not in custom_cfgs.get('model_cfgs', {}) or \
       custom_cfgs['model_cfgs']['actor_type'] == 'auto':
        detected_actor_type = detect_actor_type(env_id, custom_cfgs.get('env_cfgs', {}))
        custom_cfgs.setdefault('model_cfgs', {})['actor_type'] = detected_actor_type

    # Create and train agent
    agent = omnisafe.Agent(algo, env_id, custom_cfgs=custom_cfgs)
    agent.learn_with_sample_efficiency(eval_episodes=eval_num_episodes, eval_fraction=0.05)
