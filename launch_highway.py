import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import List

import numpy as np

# Local wrapper just to read modality costs; ensure it is importable in PYTHONPATH
from bafs_envs import budget_aware_highway

# SAFE_ALGOS = ['SACLag', 'SPOLag', 'PPOLag', 'CUP', 'CPPOPID']
SAFE_ALGOS = ['PPOLag', 'CPPOPID']
# UNSAFE_ALGOS = ['SAC', 'SPO', 'PPO']
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
    """The template must contain exactly one '{}' for the python command."""
    try:
        return template.format(job=job_name, python_cmd=python_cmd)
    except Exception as e:
        raise RuntimeError(
            "Failed to format sbatch template. Ensure it contains a single '{}' placeholder"
        ) from e


def create_file(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def submit(path: Path) -> int:
    print(f"[sbatch] {path}")
    proc = subprocess.run(["sbatch", str(path)], capture_output=True, text=True)
    # Slurm usually returns "Submitted batch job <id>"
    if proc.returncode != 0:
        print(proc.stdout.strip())
        print(proc.stderr.strip(), file=sys.stderr)
        raise RuntimeError(f"sbatch failed for {path.name}")
    print(proc.stdout.strip())
    return proc.returncode


def compute_budget(max_episode_steps: int, budget_ratio: float, feature_costs: np.ndarray) -> float:
    """Budget = ratio × (max_episode_steps × sum(costs_per_step))"""
    return float(budget_ratio * max_episode_steps * float(np.sum(feature_costs)))


def get_feature_costs(env_id: str, *, max_episode_steps: int) -> np.ndarray:
    """Instantiate wrapper just to read per-modality costs."""
    # Wrapper expects num_envs; pass a benign value
    env = budget_aware_highway.BudgetAwareHighway(
        env_id,
        num_envs=1,
        max_episode_steps=max_episode_steps,
    )
    try:
        return np.asarray(env.costs, dtype=np.float32)
    finally:
        # Make sure to close underlying env resources
        try:
            env.close()
        except Exception:
            pass


def valid_combo(algo: str, use_cost: bool, use_all_obs: bool, sd_regulizer: bool, no_zero_act: bool, random_obs_selection: bool) -> bool:
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
    p = argparse.ArgumentParser(description="Generate & submit Slurm jobs for BAFS experiments")
    p.add_argument("--run-py", type=str, default="/home/ofek.glick/BAFS_2/run_bafs_highway.py",
                   help="Path to run_bafs_highway.py entrypoint")
    p.add_argument("--sbatch-template", type=str, default="sbatch_template_2.sh",
                   help="Path to sbatch template with '{}' placeholder")
    p.add_argument("--sbatch-dir", type=str,
                   help="Directory to write per-run sbatch files", default='./sbatch_files_highway')
    # Experiment parameters
    p.add_argument("--envs", nargs="+", default=[
        "budget-aware-highway-fast-v0",
        # "budget-aware-merge-v0",
        "budget-aware-roundabout-v0",
        "budget-aware-parking-v0",
        # "budget-aware-intersection-v1",
    ])
    p.add_argument("--safe-algos", nargs="*", default=SAFE_ALGOS)
    p.add_argument("--unsafe-algos", nargs="*", default=UNSAFE_ALGOS)
    p.add_argument("--budget-ratios", nargs="+", type=float, default=[0.5, 0.8])
    p.add_argument("--cost-usage", nargs="+", type=int, default=[1, 0], help="0/1 for use_cost")
    p.add_argument("--all-obs-usage", nargs="+", type=int, default=[1, 0], help="0/1 for use_all_obs")
    p.add_argument("--random-obs-selection", nargs="+", type=int, default=[1, 0], help="0/1 for use_all_obs")
    p.add_argument("--sd-regulizer", nargs="+", type=int, default=[1, 0], help="0/1 for use_all_obs")
    p.add_argument("--no-zero-act", nargs="+", type=int, default=[0], help="0/1 for use_all_obs")
    p.add_argument("--seeds", nargs="+", type=int, default=[
        # 31, 32, 33, 34, 35, 36, 37, 38,
        # 41, 42, 43, 44, 45, 46, 47, 48,
        51, 52, 53, 54, 55, 56, 57, 58,
        61, 62, 63, 64, 65, 66, 67, 68
    ])
    p.add_argument("--total-steps", type=int, default=102_400)
    p.add_argument("--eval-num-episodes", type=int, default=250)
    p.add_argument("--max-episode-steps", type=int, default=250)
    p.add_argument("--steps-per-epoch", type=int, default=8192)
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
        # Read per-modality costs from the wrapper; used for budget + CLI
        feature_costs = get_feature_costs(env_id, max_episode_steps=args.max_episode_steps)

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
                                py_args = dict(
                                    algo=algo,
                                    env_id=env_id,
                                    use_cost=bool(use_cost),
                                    use_all_obs=bool(use_all_obs),
                                    eval_num_episodes=args.eval_num_episodes,
                                    total_steps=args.total_steps,
                                    budget=compute_budget(args.max_episode_steps, 1.0, feature_costs),
                                    feature_cost=[f"{c:.4f}" for c in feature_costs.tolist()],
                                    max_episode_steps=args.max_episode_steps,
                                    steps_per_epoch=args.steps_per_epoch,
                                    seed=seed,
                                    sd_regulizer=bool(sd_reg),
                                    no_zero_act=bool(no_zero_act),
                                    random_obs_selection=bool(random_obs_selection),
                                )
                                python_cmd = build_python_command(args.run_py, py_args)
                                env_short = env_id.replace("budget-aware-", "")
                                fname = f"{algo}_{env_short}_cost{int(use_cost)}_all{int(use_all_obs)}_Budget{int(py_args['budget'])}_Seed{seed}_sd{int(sd_reg)}_nozero{int(no_zero_act)}_random{int(random_obs_selection)}"
                                sbatch_text = format_sbatch(template=sbatch_template, job_name=fname, python_cmd=python_cmd)
                                if args.tag:
                                    fname = f"{fname}_{args.tag}"
                                sbatch_path = out_dir / f"{fname}.sh"

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
                                    if not valid_combo(algo, bool(use_cost), bool(use_all_obs), bool(sd_reg), bool(no_zero_act), bool(random_obs_selection)):
                                        continue
                                    budget = compute_budget(args.max_episode_steps, br, feature_costs)
                                    for seed in args.seeds:
                                        py_args = dict(
                                            algo=algo,
                                            env_id=env_id,
                                            use_cost=bool(use_cost),
                                            use_all_obs=bool(use_all_obs),
                                            eval_num_episodes=args.eval_num_episodes,
                                            total_steps=args.total_steps,
                                            budget=budget,
                                            feature_cost=[f"{c:.4f}" for c in feature_costs.tolist()],
                                            max_episode_steps=args.max_episode_steps,
                                            steps_per_epoch=args.steps_per_epoch,
                                            seed=seed,
                                            sd_regulizer=bool(sd_reg),
                                            no_zero_act=bool(no_zero_act),
                                            random_obs_selection=bool(random_obs_selection),
                                        )
                                        python_cmd = build_python_command(args.run_py, py_args)
                                        fname = f"{algo}_{env_id}_cost{int(use_cost)}_all{int(use_all_obs)}_B{int(budget)}_S{seed}_sd{int(sd_reg)}_nozero{int(no_zero_act)}_random{int(random_obs_selection)}"
                                        sbatch_text = format_sbatch(template=sbatch_template, job_name=fname, python_cmd=python_cmd)
                                        if args.tag:
                                            fname = f"{fname}_{args.tag}"
                                        sbatch_path = out_dir / f"{fname}.sh"

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
            # small delay to avoid hammering scheduler/logs
            submit(pth)
            time.sleep(0.1)
    else:
        print(f"Wrote {len(created)} sbatch files to {out_dir}")
        for p in created[:5]:
            print("  ", p.name)
        if len(created) > 5:
            print("  ...")


if __name__ == "__main__":
    main()
