import omnisafe
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--algo',
        type=str,
        metavar='ALGO',
        default='PPO',
        help='algorithm to train',
        choices=omnisafe.ALGORITHMS['all'],
    )

    parser.add_argument(
        '--env-id',
        type=str,
        default="budget-aware-intersection-v1",
        help='environment to use',
    )
    parser.add_argument(
        '--use-cost',
        action='store_true',
        help='environment to use',
    )

    parser.add_argument(
        '--use-all-obs',
        action='store_true',
        help='whether to use all observations always or not',
    )
    parser.add_argument(
        '--random-obs-selection',
        action='store_true',
        help='use random modality selection baseline (learns env actions, random masks)',
    )
    parser.add_argument(
        '--eval-num-episodes',
        type=int,
        default=1_000,
        help='number of episodes to evaluate the agent',
    )
    parser.add_argument(
        '--total-steps',
        type=int,
        default=409_600,
        help='total number of steps to train the agent',
    )
    parser.add_argument(
        '--steps-per-epoch',
        type=int,
        default=8192,
        help='total number of steps to train the agent',
    )
    parser.add_argument(
        '--budget',
        type=float,
        default=1_000,
        help='the ratio of the budget to the maximum cost per episode',
    )
    parser.add_argument(
        '--max-episode-steps',
        type=float,
        default=250,
        help='the maximum number of steps per episode',
    )
    parser.add_argument(
        '--sd-regulizer',
        action='store_true',
        help='whether to use the sensor dropout regulizer or not',
    )
    parser.add_argument(
        '--no-zero-act',
        action='store_true',
        help='whether to use the zero-action regularizer or not',
    )
    parser.add_argument(
        '--obs-modality-normalize',
        action='store_true',
        help='whether to use the zero-action regularizer or not',
    )
    parser.add_argument(
        '--penalty-coef',
        type=float,
        default=0.0,
        help='penalty coefficient for cost (0=no penalty, 1=full cost penalty via OmniSafe)',
    )

    # parser.add_argument(
    #     '--feature-cost',
    #     type=float,
    #     nargs='+',
    #     help='list of feature costs',
    # )
    # parser.add_argument(
    #     '--features-to-use',
    #     type=int,
    #     nargs='+',
    #     help='list of feature costs',
    # )
    parser.add_argument(
        '--available-sensors',
        type=str,
        nargs='+',
        default=None,
        help='List of sensor/modality names to make available for selection '
             '(None = all sensors). Example: --available-sensors Kinematics LidarObservation',
    )
    parser.add_argument(
        '--seed',
        default=42,
        type=int,
        help='seed of the run',
    )
    parser.add_argument(
        '--manifest-filename',
        type=str,
        default='run_manifest.csv',
        help='CSV manifest filename for logging results (default: run_manifest.csv)',
    )
    parser.add_argument(
        '--wandb-project',
        type=str,
        default=None,
        help='Override wandb project name (e.g., "SPAR Highway - Sensor Ablation")',
    )
    return parser.parse_known_args()