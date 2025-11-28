"""
Launch script for Highway-env sensor subset baseline experiments.

Generates jobs for all possible sensor subset combinations with baseline configuration:
- use_all_obs=1 (no learned masking)
- penalty=0 (no cost penalty)
- PPO algorithm (unsafe baseline)
"""

import argparse
import time
from pathlib import Path
import numpy as np

# Import shared utilities
from utils.launch_utils import (
    read_template, format_sbatch, create_file, submit, build_python_command
)

# Import highway-specific configuration
from configs.highway_config import UNSAFE_ALGOS, DEFAULT_LAUNCH_PARAMS

# Import wrapper to read modality costs
from bafs_envs import budget_aware_highway


def get_feature_costs(env_id: str) -> tuple[np.ndarray, int]:
    """Instantiate wrapper to read per-modality costs and actual episode duration."""
    env = budget_aware_highway.BudgetAwareHighway(env_id, num_envs=1)
    try:
        costs = np.asarray(env.costs, dtype=np.float32)
        actual_max_steps = env.max_episode_steps
        return costs, actual_max_steps
    finally:
        try:
            env.close()
        except Exception:
            pass


def build_py_args_baseline(base_args: dict, costs: np.ndarray) -> dict:
    """Build arguments for baseline sensor subset experiments.

    Enforces baseline configuration:
    - use_all_obs=1 (all sensors in subset always active)
    - penalty_coef=0.0 (no cost penalty)
    """
    py_args = base_args.copy()

    # ── Enforce baseline configuration ────────────────────────────────────
    py_args['use_all_obs'] = True
    py_args['penalty_coef'] = 0.0
    py_args['use_cost'] = False
    py_args['sd_regulizer'] = False
    py_args['random_obs_selection'] = False
    py_args['manifest_filename'] = 'sensor_ablation_manifest.csv'

    print(f"[Sensor Subset Baseline] Active sensors: {base_args['available_sensors']}")

    # ── Filter costs based on available sensors ───────────────────────────
    if base_args.get('available_sensors') is not None:
        from bafs_envs.budget_aware_highway import BudgetAwareHighway
        all_sensors = BudgetAwareHighway.DEFAULT_TYPES
        available = base_args['available_sensors']

        # Filter costs array to match available sensors
        filtered_costs = []
        for sensor in available:
            if sensor in all_sensors:
                idx = all_sensors.index(sensor)
                filtered_costs.append(costs[idx])

        py_args['feature_cost'] = [f"{c:.4f}" for c in filtered_costs]
    else:
        py_args['feature_cost'] = [f"{c:.4f}" for c in costs.tolist()]

    return py_args


def build_filename(env_id: str, sensor_config: list, seed: int, tag: str = "") -> str:
    """Build job filename for sensor subset experiments."""
    env_short = env_id.replace("budget-aware-", "")

    # Build sensor tag with abbreviated names
    abbrev = {
        'Kinematics': 'Kin',
        'LidarObservation': 'Lid',
        'OccupancyGrid': 'Occ',
        'TimeToCollision': 'TTC'
    }
    sensor_abbrev = '_'.join(abbrev.get(s, s[:3]) for s in sorted(sensor_config))

    fname = f"PPO_{env_short}_baseline_Seed{seed}_sens{sensor_abbrev}"
    if tag:
        fname = f"{fname}_{tag}"
    return fname


def generate_baseline_jobs(
    env_id: str,
    sensor_configs: list,
    seeds: list,
    run_py: str,
    total_steps: int,
    eval_num_episodes: int,
    max_episode_steps: int,
    steps_per_epoch: int,
    tag: str = "",
    wandb_project: str = None
):
    """Generate baseline jobs for all sensor subsets × seeds."""

    # Get costs and actual max steps from environment
    costs, actual_max_steps = get_feature_costs(env_id)

    for sensor_config in sensor_configs:
        for seed in seeds:
            # Compute budget (full budget for baseline)
            budget_ratio = 1.0
            if sensor_config is not None:
                # Filter costs to active sensors
                from bafs_envs.budget_aware_highway import BudgetAwareHighway
                all_sensors = BudgetAwareHighway.DEFAULT_TYPES
                filtered_costs = [costs[all_sensors.index(s)] for s in sensor_config if s in all_sensors]
                total_cost = sum(filtered_costs)
            else:
                total_cost = sum(costs)

            budget = budget_ratio * actual_max_steps * total_cost

            # Build base arguments
            base_args = dict(
                algo='PPO',
                env_id=env_id,
                use_all_obs=True,
                penalty_coef=0.0,
                use_cost=False,
                sd_regulizer=False,
                random_obs_selection=False,
                available_sensors=sensor_config,
                eval_num_episodes=eval_num_episodes,
                total_steps=total_steps,
                budget=budget,
                max_episode_steps=actual_max_steps,
                steps_per_epoch=steps_per_epoch,
                seed=seed,
                obs_modality_normalize=True,
            )

            # Add wandb project if specified
            if wandb_project is not None:
                base_args['wandb_project'] = wandb_project

            # Build arguments with cost filtering
            py_args = build_py_args_baseline(base_args, costs)

            # Build command and filename
            python_cmd = build_python_command(run_py, py_args)
            filename = build_filename(env_id, sensor_config, seed, tag)

            yield (python_cmd, filename)


def main():
    """Main function to generate and submit Slurm jobs."""
    p = argparse.ArgumentParser(
        description="Generate sensor subset baseline jobs for Highway experiments"
    )

    # Use defaults from config
    p.add_argument("--run-py", type=str, default=DEFAULT_LAUNCH_PARAMS['run_py'])
    p.add_argument("--sbatch-template", type=str, default=DEFAULT_LAUNCH_PARAMS['sbatch_template'])
    p.add_argument("--sbatch-dir", type=str, default='./sbatch_files_highway_subsets')

    # Experiment parameters
    p.add_argument("--env", type=str, default="budget-aware-highway-fast-v0",
                   help="Single environment to test")
    p.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_LAUNCH_PARAMS['seeds'])
    p.add_argument("--total-steps", type=int, default=DEFAULT_LAUNCH_PARAMS['total_steps'])
    p.add_argument("--eval-num-episodes", type=int, default=DEFAULT_LAUNCH_PARAMS['eval_num_episodes'])
    p.add_argument("--max-episode-steps", type=int, default=DEFAULT_LAUNCH_PARAMS['max_episode_steps'])
    p.add_argument("--steps-per-epoch", type=int, default=DEFAULT_LAUNCH_PARAMS['steps_per_epoch'])

    # Execution options
    p.add_argument("--submit", action="store_true", help="Actually submit to Slurm")
    p.add_argument("--dry-run", action="store_true", help="Only print commands")
    p.add_argument("--tag", type=str, default="", help="Optional tag for filenames")
    p.add_argument("--wandb-project", type=str, default='"SPAR Highway - Sensor Ablation"',
                   help="Override wandb project name (e.g., 'SPAR Highway - Sensor Ablation')")

    args = p.parse_args()

    # Get sensor configs from config
    from configs.highway_config import generate_sensor_subsets
    sensor_configs = generate_sensor_subsets()

    print(f"Generating jobs for {len(sensor_configs)} sensor subsets × {len(args.seeds)} seeds")
    print(f"Total: {len(sensor_configs) * len(args.seeds)} jobs")

    sbatch_template = read_template(args.sbatch_template)
    out_dir = Path(args.sbatch_dir)

    created = []

    # Generate all job configurations
    for python_cmd, filename in generate_baseline_jobs(
        env_id=args.env,
        sensor_configs=sensor_configs,
        seeds=args.seeds,
        run_py=args.run_py,
        total_steps=args.total_steps,
        eval_num_episodes=args.eval_num_episodes,
        max_episode_steps=args.max_episode_steps,
        steps_per_epoch=args.steps_per_epoch,
        tag=args.tag,
        wandb_project=args.wandb_project,
    ):
        if args.dry_run:
            print(python_cmd)
        else:
            sbatch_text = sbatch_template.format(job=filename, python_cmd=python_cmd)
            sbatch_path = out_dir / f"{filename}.sh"
            create_file(sbatch_text, sbatch_path)
            created.append(sbatch_path)

    if args.dry_run:
        print("[dry-run] Done.")
        return

    if args.submit:
        # Submit jobs with small delay
        for pth in created:
            print(f"[sbatch] {pth.name}")
            from subprocess import run
            run(["sbatch", str(pth)])
            time.sleep(0.1)
    else:
        print(f"Wrote {len(created)} sbatch files to {out_dir}")
        for p in created[:5]:
            print("  ", p.name)
        if len(created) > 5:
            print("  ...")


if __name__ == "__main__":
    main()