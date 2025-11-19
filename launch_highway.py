"""
Launch script for Highway-env BAFS experiments.

Generates and optionally submits Slurm batch jobs for different algorithm/environment
combinations with budget-aware observation masking.
"""

import argparse
import time
from pathlib import Path
import numpy as np

# Import shared utilities
from utils.launch_utils import (
    read_template, format_sbatch, create_file, submit, generate_jobs
)

# Import highway-specific configuration
from configs.highway_config import SAFE_ALGOS, UNSAFE_ALGOS, DEFAULT_LAUNCH_PARAMS

# Import wrapper to read modality costs
from bafs_envs import budget_aware_highway


def get_feature_costs(env_id: str) -> tuple[np.ndarray, int]:
    """Instantiate wrapper to read per-modality costs and actual episode duration.

    Args:
        env_id: Environment ID

    Returns:
        Tuple of (per-modality costs, actual max_episode_steps from environment)
    """
    env = budget_aware_highway.BudgetAwareHighway(
        env_id,
        num_envs=1,
    )
    try:
        costs = np.asarray(env.costs, dtype=np.float32)
        actual_max_steps = env.max_episode_steps
        return costs, actual_max_steps
    finally:
        try:
            env.close()
        except Exception:
            pass


def build_py_args_highway(base_args: dict, costs: np.ndarray) -> dict:
    """Add highway-specific arguments to base args.

    Args:
        base_args: Base arguments dictionary
        costs: Per-modality costs

    Returns:
        Updated arguments dictionary with highway-specific fields
    """
    py_args = base_args.copy()
    # Highway includes feature costs as CLI argument
    py_args['feature_cost'] = [f"{c:.4f}" for c in costs.tolist()]
    return py_args


def build_filename_highway(job_info: dict) -> str:
    """Build job filename for highway experiments.

    Args:
        job_info: Dictionary with job information (algo, env_id, etc.)

    Returns:
        Filename string
    """
    env_short = job_info['env_id'].replace("budget-aware-", "")
    fname = (
        f"{job_info['algo']}_{env_short}_"
        f"cost{job_info['use_cost']}_all{job_info['use_all_obs']}_"
        f"Budget{job_info['budget']}_Seed{job_info['seed']}_"
        f"sd{job_info['sd_reg']}_random{job_info['random_obs_selection']}_"
        f"pen{job_info['penalty_coef']}"
    )
    if job_info['tag']:
        fname = f"{fname}_{job_info['tag']}"
    return fname


def main():
    """Main function to generate and submit Slurm jobs."""
    p = argparse.ArgumentParser(description="Generate & submit Slurm jobs for BAFS Highway experiments")

    # Use defaults from config
    p.add_argument("--run-py", type=str, default=DEFAULT_LAUNCH_PARAMS['run_py'],
                   help="Path to run_bafs_highway.py entrypoint")
    p.add_argument("--sbatch-template", type=str, default=DEFAULT_LAUNCH_PARAMS['sbatch_template'],
                   help="Path to sbatch template with {job} and {python_cmd} placeholders")
    p.add_argument("--sbatch-dir", type=str, default=DEFAULT_LAUNCH_PARAMS['sbatch_dir'],
                   help="Directory to write per-run sbatch files")

    # Experiment parameters
    p.add_argument("--envs", nargs="+", default=DEFAULT_LAUNCH_PARAMS['envs'])
    p.add_argument("--safe-algos", nargs="*", default=SAFE_ALGOS)
    p.add_argument("--unsafe-algos", nargs="*", default=UNSAFE_ALGOS)
    p.add_argument("--budget-ratios", nargs="+", type=float, default=DEFAULT_LAUNCH_PARAMS['budget_ratios'])
    p.add_argument("--cost-usage", nargs="+", type=int, default=DEFAULT_LAUNCH_PARAMS['cost_usage'],
                   help="0/1 for use_cost")
    p.add_argument("--all-obs-usage", nargs="+", type=int, default=DEFAULT_LAUNCH_PARAMS['all_obs_usage'],
                   help="0/1 for use_all_obs")
    p.add_argument("--random-obs-selection", nargs="+", type=int, default=DEFAULT_LAUNCH_PARAMS['random_obs_selection'],
                   help="0/1 for random_obs_selection")
    p.add_argument("--sd-regulizer", nargs="+", type=int, default=DEFAULT_LAUNCH_PARAMS['sd_regulizer'],
                   help="0/1 for sd_regulizer")
    p.add_argument("--penalty-coef", nargs="+", type=float, default=DEFAULT_LAUNCH_PARAMS['penalty_coef'],
                   help="Penalty coefficient values (0.0=no penalty, 1.0=full cost penalty)")
    p.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_LAUNCH_PARAMS['seeds'])
    p.add_argument("--total-steps", type=int, default=DEFAULT_LAUNCH_PARAMS['total_steps'])
    p.add_argument("--eval-num-episodes", type=int, default=DEFAULT_LAUNCH_PARAMS['eval_num_episodes'])
    p.add_argument("--max-episode-steps", type=int, default=DEFAULT_LAUNCH_PARAMS['max_episode_steps'])
    p.add_argument("--steps-per-epoch", type=int, default=DEFAULT_LAUNCH_PARAMS['steps_per_epoch'])

    # Execution options
    p.add_argument("--submit", action="store_true", help="Actually submit to Slurm")
    p.add_argument("--dry-run", action="store_true", help="Only print commands; do not write or submit")
    p.add_argument("--tag", type=str, default="", help="Optional tag added to sbatch filenames")

    args = p.parse_args()

    assert args.max_episode_steps <= args.steps_per_epoch, \
        "max_episode_steps must be <= steps_per_epoch (needed for episodic logging)"

    sbatch_template = read_template(args.sbatch_template)
    out_dir = Path(args.sbatch_dir)

    created = []

    # Generate all job configurations using shared utilities
    for python_cmd, filename in generate_jobs(
        envs=args.envs,
        safe_algos=args.safe_algos,
        unsafe_algos=args.unsafe_algos,
        budget_ratios=args.budget_ratios,
        cost_usage=args.cost_usage,
        all_obs_usage=args.all_obs_usage,
        random_obs_selection_opts=args.random_obs_selection,
        sd_regulizer_opts=args.sd_regulizer,
        penalty_coef_opts=args.penalty_coef,
        seeds=args.seeds,
        run_py=args.run_py,
        max_episode_steps=args.max_episode_steps,
        total_steps=args.total_steps,
        eval_num_episodes=args.eval_num_episodes,
        steps_per_epoch=args.steps_per_epoch,
        get_costs_callback=get_feature_costs,
        build_py_args_callback=build_py_args_highway,
        build_filename_callback=build_filename_highway,
        tag=args.tag,
    ):
        if args.dry_run:
            print(python_cmd)
        else:
            sbatch_text = format_sbatch(
                template=sbatch_template,
                job_name=filename,
                python_cmd=python_cmd
            )
            sbatch_path = out_dir / f"{filename}.sh"
            create_file(sbatch_text, sbatch_path)
            created.append(sbatch_path)

    if args.dry_run:
        print("[dry-run] Done.")
        return

    if args.submit:
        # Reorganize submissions to interleave seeds across job types
        # Group files by job type (everything except seed)
        from collections import defaultdict
        import re

        job_groups = defaultdict(list)
        for pth in created:
            # Extract job type by removing seed information
            # Pattern: _Seed<seed>_ in filename (highway uses "Seed" not "S")
            job_type = re.sub(r'_Seed\d+_', '_SEED_', pth.stem)
            job_groups[job_type].append(pth)

        # Sort each group by seed
        for job_type in job_groups:
            job_groups[job_type].sort(key=lambda p: int(re.search(r'_Seed(\d+)_', p.stem).group(1)))

        # Interleave in batches of 2 seeds
        batch_size = 2
        reordered = []
        job_types = list(job_groups.keys())
        max_seeds = max(len(files) for files in job_groups.values())

        for batch_start in range(0, max_seeds, batch_size):
            for job_type in job_types:
                batch = job_groups[job_type][batch_start:batch_start + batch_size]
                reordered.extend(batch)

        # Submit in the new order
        for pth in reordered:
            submit(pth)
            time.sleep(0.1)  # Small delay to avoid hammering scheduler
    else:
        print(f"Wrote {len(created)} sbatch files to {out_dir}")
        for p in created[:5]:
            print("  ", p.name)
        if len(created) > 5:
            print("  ...")


if __name__ == "__main__":
    main()
