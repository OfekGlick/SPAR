import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import List

# Import to get modality costs

# Algorithm configurations
SAFE_ALGOS = ['PPOLag', 'CPPOPID']
UNSAFE_ALGOS = ['PPO']


def read_template(template_path: str) -> str:
    with open(template_path, "r", encoding="utf-8") as f:
        return f.read()


def build_python_command(run_py: str, args: dict) -> str:
    """Build a CLI string. Lists are expanded as space-separated values.
    Booleans are included as flags when True.
    """
    parts = ['python', run_py]
    for k, v in args.items():
        flag = f"--{k.replace('_', '-')}"
        if isinstance(v, bool):
            if v:
                parts.append(flag)
        elif isinstance(v, (list, tuple)):
            if len(v) > 0:
                parts.append(flag)
                parts.extend([str(x) for x in v])
        else:
            parts.extend([flag, str(v)])
    return " ".join(parts)


def format_sbatch(template: str, job_name: str, python_cmd: str) -> str:
    """The template must contain placeholders for job name and python command."""
    try:
        return template.format(job=job_name, python_cmd=python_cmd)
    except Exception as e:
        raise RuntimeError(
            "Failed to format sbatch template. Ensure it contains {job} and {python_cmd} placeholders"
        ) from e


def create_file(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def submit(path: Path) -> int:
    print(f"[sbatch] {path}")
    proc = subprocess.run(["sbatch", str(path)], capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stdout.strip())
        print(proc.stderr.strip(), file=sys.stderr)
        raise RuntimeError(f"sbatch failed for {path.name}")
    print(proc.stdout.strip())
    return proc.returncode


def compute_budget(max_episode_steps: int, budget_ratio: float, modality_costs: dict) -> float:
    """Budget = ratio × (max_episode_steps × sum(costs_per_step))"""
    total_cost_per_step = sum(modality_costs.values())
    return float(budget_ratio * max_episode_steps * total_cost_per_step)


def get_modality_costs(env_id: str) -> dict:
    """Get default modality costs for robosuite environment."""
    # These are the default costs from budget_aware_robosuite.py
    # You can customize these per environment if needed
    return {
        'robot_proprioception': 1.0,
        'object_states': 1.0,
        'task_features': 1.0,
    }


def valid_combo(algo: str, use_cost: bool, use_all_obs: bool, sd_regulizer: bool, no_zero_act: bool, random_obs_selection: bool) -> bool:
    """Check if algorithm/configuration combination is valid."""
    if use_cost and use_all_obs:
        return False
    if use_cost and algo in UNSAFE_ALGOS:
        return False
    if (not use_cost) and (algo in SAFE_ALGOS):
        return False
    if use_all_obs and (algo in SAFE_ALGOS):
        return False
    if no_zero_act and use_all_obs:
        return False
    if sd_regulizer and use_all_obs:
        return False
    if not sd_regulizer and not use_all_obs:
        return False
    if random_obs_selection and use_all_obs:
        return False
    if random_obs_selection and algo in SAFE_ALGOS:
        return False
    return True


def main():
    p = argparse.ArgumentParser(description="Generate & submit Slurm jobs for BAFS Robosuite experiments")
    p.add_argument("--run-py", type=str, default="/home/ofek.glick/BAFS_2/run_bafs_robosuite.py",
                   help="Path to run_bafs_robosuite.py entrypoint")
    p.add_argument("--sbatch-template", type=str, default="sbatch_template_2.sh",
                   help="Path to sbatch template with {job} and {python_cmd} placeholders")
    p.add_argument("--sbatch-dir", type=str,
                   help="Directory to write per-run sbatch files", default='./sbatch_files_robosuite')

    # Robosuite environment parameters
    p.add_argument("--envs", nargs="+", default=[
        "budget-aware-Lift",
        "budget-aware-Door",
    ])
    p.add_argument("--robot", type=str, default="Panda",
                   help="Robot type (default: Panda)")

    # Algorithm configurations
    p.add_argument("--safe-algos", nargs="+", default=SAFE_ALGOS)
    p.add_argument("--unsafe-algos", nargs="+", default=UNSAFE_ALGOS)
    p.add_argument("--budget-ratios", nargs="+", type=float, default=[0.5, 0.8])
    p.add_argument("--cost-usage", nargs="+", type=int, default=[1, 0], help="0/1 for use_cost")
    p.add_argument("--all-obs-usage", nargs="+", type=int, default=[1, 0], help="0/1 for use_all_obs")
    p.add_argument("--random-obs-selection", nargs="+", type=int, default=[1, 0], help="0/1 for use_all_obs")
    p.add_argument("--sd-regulizer", nargs="+", type=int, default=[1, 0], help="0/1 for sd_regulizer")
    p.add_argument("--no-zero-act", nargs="+", type=int, default=[0], help="0/1 for no_zero_act")
    p.add_argument("--seeds", nargs="+", type=int, default=[
        31, 32, 33, 34, 35, 36, 37, 38,
        41, 42, 43, 44, 45, 46, 47, 48,
        51, 52, 53, 54, 55, 56, 57, 58,
    ])
    # Training parameters (robosuite benchmark defaults)
    p.add_argument("--total-steps", type=int, default=250_000,
                   help="Total training steps (500 epochs × 500 steps)")
    p.add_argument("--eval-num-episodes", type=int, default=50)
    p.add_argument("--max-episode-steps", type=int, default=500,
                   help="Episode horizon (robosuite benchmark standard)")
    p.add_argument("--steps-per-epoch", type=int, default=500,
                   help="Steps per epoch (should match max_episode_steps)")

    # Execution options
    p.add_argument("--submit", action="store_true", help="Actually submit to Slurm")
    p.add_argument("--dry-run", action="store_true", help="Only print commands; do not write or submit")
    p.add_argument("--tag", type=str, default="", help="Optional tag added to sbatch filenames")

    args = p.parse_args()

    assert args.max_episode_steps <= args.steps_per_epoch, \
        "max_episode_steps must be <= steps_per_epoch (needed for episodic logging)"

    sbatch_template = read_template(args.sbatch_template)
    out_dir = Path(args.sbatch_dir)

    created: List[Path] = []

    for env_id in args.envs:
        # Get modality costs for this environment
        modality_costs = get_modality_costs(env_id)

        # Unsafe baselines
        for algo in args.unsafe_algos:
            for use_all_obs in args.all_obs_usage:
                for sd_reg in args.sd_regulizer:
                    for no_zero_act in args.no_zero_act:
                        for random_obs_selection in args.random_obs_selection:
                            use_cost = False
                            if not valid_combo(algo, use_cost, bool(use_all_obs), bool(sd_reg), bool(no_zero_act), bool(random_obs_selection)):
                                continue
                            for seed in args.seeds:
                                # For unsafe baselines, use full budget
                                budget = compute_budget(args.max_episode_steps, 1.0, modality_costs)

                                py_args = dict(
                                    algo=algo,
                                    env_id=env_id,
                                    use_cost=bool(use_cost),
                                    use_all_obs=bool(use_all_obs),
                                    eval_num_episodes=args.eval_num_episodes,
                                    total_steps=args.total_steps,
                                    budget=budget,
                                    max_episode_steps=args.max_episode_steps,
                                    steps_per_epoch=args.steps_per_epoch,
                                    seed=seed,
                                    sd_regulizer=bool(sd_reg),
                                    no_zero_act=bool(no_zero_act),
                                    random_obs_selection=bool(random_obs_selection),
                                )

                                python_cmd = build_python_command(args.run_py, py_args)

                                # Clean environment name for filename
                                env_short = env_id.replace("budget-aware-", "")
                                fname = (f"{algo}_{env_short}_cost{int(use_cost)}_all{int(use_all_obs)}_"
                                         f"B{int(budget)}_S{seed}_sd{int(sd_reg)}_nz{int(no_zero_act)}_random{int(random_obs_selection)}")

                                if args.tag:
                                    fname = f"{fname}_{args.tag}"

                                sbatch_path = out_dir / f"{fname}.sh"
                                sbatch_text = format_sbatch(
                                    template=sbatch_template,
                                    job_name=fname,
                                    python_cmd=python_cmd
                                )

                                if args.dry_run:
                                    print(python_cmd)
                                else:
                                    create_file(sbatch_text, sbatch_path)
                                    created.append(sbatch_path)

        # Safe constrained algorithms
        for algo in args.safe_algos:
            for use_all_obs in args.all_obs_usage:
                for use_cost in args.cost_usage:
                    for br in args.budget_ratios:
                        for sd_reg in args.sd_regulizer:
                            for no_zero_act in args.no_zero_act:
                                for random_obs_selection in args.random_obs_selection:
                                    if not valid_combo(algo, bool(use_cost), bool(use_all_obs), bool(sd_reg),
                                                       bool(no_zero_act), bool(random_obs_selection)):
                                        continue

                                    budget = compute_budget(args.max_episode_steps, br, modality_costs)

                                    for seed in args.seeds:
                                        py_args = dict(
                                            algo=algo,
                                            env_id=env_id,
                                            use_cost=bool(use_cost),
                                            use_all_obs=bool(use_all_obs),
                                            eval_num_episodes=args.eval_num_episodes,
                                            total_steps=args.total_steps,
                                            budget=budget,
                                            max_episode_steps=args.max_episode_steps,
                                            steps_per_epoch=args.steps_per_epoch,
                                            seed=seed,
                                            sd_regulizer=bool(sd_reg),
                                            no_zero_act=bool(no_zero_act),
                                            random_obs_selection=bool(random_obs_selection),
                                        )

                                        python_cmd = build_python_command(args.run_py, py_args)

                                        # Clean environment name for filename
                                        env_short = env_id.replace("budget-aware-", "")
                                        fname = (f"{algo}_{env_short}_cost{int(use_cost)}_all{int(use_all_obs)}_"
                                                 f"B{int(budget)}_S{seed}_sd{int(sd_reg)}_nz{int(no_zero_act)}_random{int(random_obs_selection)}")

                                        if args.tag:
                                            fname = f"{fname}_{args.tag}"

                                        sbatch_path = out_dir / f"{fname}.sh"
                                        sbatch_text = format_sbatch(
                                            template=sbatch_template,
                                            job_name=fname,
                                            python_cmd=python_cmd
                                        )

                                        if args.dry_run:
                                            print(python_cmd)
                                        else:
                                            create_file(sbatch_text, sbatch_path)
                                            created.append(sbatch_path)

    if args.dry_run:
        print("[dry-run] Done.")
        return

    if args.submit:
        for pth in created:
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
