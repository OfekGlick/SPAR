"""
Shared utilities for launch scripts (launch_highway.py, launch_robosuite.py).

Provides common functionality for:
- Reading sbatch templates
- Building CLI commands
- Generating Slurm job files
- Validating algorithm/configuration combinations
- Computing budgets
"""

import subprocess
import sys
from pathlib import Path
from typing import List, Callable, Optional, Union, Dict
import numpy as np


def read_template(template_path: str) -> str:
    """Read sbatch template file."""
    with open(template_path, "r", encoding="utf-8") as f:
        return f.read()


def build_python_command(run_py: str, args: dict) -> str:
    """Build a CLI string from arguments dictionary.

    Lists are expanded as space-separated values.
    Booleans are included as flags when True.

    Args:
        run_py: Path to Python script to run
        args: Dictionary of arguments

    Returns:
        Command string ready for execution
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
    """Format sbatch template with job name and python command.

    Template must contain placeholders for job name and python command.

    Args:
        template: Sbatch template string
        job_name: Name for the job
        python_cmd: Python command to execute

    Returns:
        Formatted sbatch script
    """
    try:
        return template.format(job=job_name, python_cmd=python_cmd)
    except Exception as e:
        raise RuntimeError(
            "Failed to format sbatch template. Ensure it contains {job} and {python_cmd} placeholders"
        ) from e


def create_file(text: str, path: Path) -> None:
    """Create a file with parent directories.

    Args:
        text: Content to write
        path: Path to file (parent directories created if needed)
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def submit(path: Path) -> int:
    """Submit sbatch file to Slurm scheduler.

    Args:
        path: Path to sbatch file

    Returns:
        Return code from sbatch command

    Raises:
        RuntimeError: If sbatch submission fails
    """
    print(f"[sbatch] {path}")
    proc = subprocess.run(["sbatch", str(path)], capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stdout.strip())
        print(proc.stderr.strip(), file=sys.stderr)
        raise RuntimeError(f"sbatch failed for {path.name}")
    print(proc.stdout.strip())
    return proc.returncode


def compute_budget(
    max_episode_steps: int,
    budget_ratio: float,
    costs: Union[np.ndarray, dict, list]
) -> float:
    """Compute budget from episode length, ratio, and per-modality costs.

    Budget = ratio × (max_episode_steps × sum(costs_per_step))

    Args:
        max_episode_steps: Maximum steps per episode
        budget_ratio: Fraction of total budget to use (0.0-1.0)
        costs: Per-modality costs (can be ndarray, dict values, or list)

    Returns:
        Budget value
    """
    if isinstance(costs, dict):
        total_cost_per_step = sum(costs.values())
    elif isinstance(costs, (list, tuple)):
        total_cost_per_step = sum(costs)
    else:  # numpy array
        total_cost_per_step = float(np.sum(costs))

    return float(budget_ratio * max_episode_steps * total_cost_per_step)


def valid_combo(
    algo: str,
    use_cost: bool,
    use_all_obs: bool,
    sd_regulizer: bool,
    random_obs_selection: bool,
    safe_algos: List[str],
    unsafe_algos: List[str],
    penalty_coef: float = 0.0
) -> bool:
    """Check if algorithm/configuration combination is valid.

    Args:
        algo: Algorithm name
        use_cost: Whether cost constraints are used
        use_all_obs: Whether all observations are used (no masking)
        sd_regulizer: Whether sensor dropout regularizer is used
        random_obs_selection: Whether random observation selection is used
        safe_algos: List of safe (constrained) algorithm names
        unsafe_algos: List of unsafe (unconstrained) algorithm names
        penalty_coef: Penalty coefficient for cost (default: 0.0)

    Returns:
        True if combination is valid, False otherwise
    """
    # Cost constraints incompatible with full observations
    if use_cost and use_all_obs:
        return False

    # Unsafe algorithms can't use cost constraints
    if use_cost and algo in unsafe_algos:
        return False

    # Safe algorithms require cost constraints
    if (not use_cost) and (algo in safe_algos):
        return False

    # Safe algorithms incompatible with full observations
    if use_all_obs and (algo in safe_algos):
        return False

    # SD regularizer incompatible with full observations
    if sd_regulizer and use_all_obs:
        return False

    # SD regularizer required when not using all observations
    if not sd_regulizer and not use_all_obs:
        return False

    # Random observation selection incompatible with full observations
    if random_obs_selection and use_all_obs:
        return False

    # Random observation selection incompatible with safe algorithms
    if random_obs_selection and algo in safe_algos:
        return False

    # Penalty coefficient > 0 only valid for unsafe algos with masking enabled
    if penalty_coef > 0.0:
        # Must be unsafe algorithm
        if algo not in unsafe_algos:
            return False
        # Must have masking enabled (use_all_obs=False)
        if use_all_obs:
            return False
        # Must not use random observation selection
        if random_obs_selection:
            return False

    return True


def generate_jobs(
    envs: List[str],
    safe_algos: List[str],
    unsafe_algos: List[str],
    budget_ratios: List[float],
    cost_usage: List[int],
    all_obs_usage: List[int],
    random_obs_selection_opts: List[int],
    sd_regulizer_opts: List[int],
    penalty_coef_opts: List[float],
    seeds: List[int],
    run_py: str,
    max_episode_steps: int,
    total_steps: int,
    eval_num_episodes: int,
    steps_per_epoch: int,
    get_costs_callback: Callable[[str], Union[np.ndarray, dict]],
    build_py_args_callback: Callable[[dict], dict],
    build_filename_callback: Callable[[dict], str],
    tag: str = "",
):
    """Generate job configurations for all experiment combinations.

    This is a generator that yields (py_args, filename) tuples for each valid
    experiment configuration. Environment-specific logic is handled via callbacks.

    Args:
        envs: List of environment IDs
        safe_algos: List of safe algorithm names
        unsafe_algos: List of unsafe algorithm names
        budget_ratios: List of budget ratios to test
        cost_usage: List of 0/1 for use_cost
        all_obs_usage: List of 0/1 for use_all_obs
        random_obs_selection_opts: List of 0/1 for random_obs_selection
        sd_regulizer_opts: List of 0/1 for sd_regulizer
        penalty_coef_opts: List of penalty coefficient values (e.g., [0.0, 1.0])
        seeds: List of random seeds
        run_py: Path to training script
        max_episode_steps: Maximum steps per episode
        total_steps: Total training steps
        eval_num_episodes: Number of evaluation episodes
        steps_per_epoch: Steps per training epoch
        get_costs_callback: Function(env_id) -> costs (env-specific)
        build_py_args_callback: Function(base_args) -> py_args (env-specific modifications)
        build_filename_callback: Function(job_info) -> filename (env-specific naming)
        tag: Optional tag for job files

    Yields:
        Tuple of (python_command, filename) for each job
    """
    for env_id in envs:
        # Get per-modality costs (environment-specific)
        costs = get_costs_callback(env_id)

        # ── Unsafe baselines ──────────────────────────────────────────────────
        for use_all_obs in all_obs_usage:
            for sd_reg in sd_regulizer_opts:
                for random_obs_selection in random_obs_selection_opts:
                    for penalty_coef in penalty_coef_opts:
                        for algo in unsafe_algos:
                            use_cost = False
                            if not valid_combo(algo, use_cost, bool(use_all_obs), bool(sd_reg),
                                             bool(random_obs_selection), safe_algos, unsafe_algos,
                                             penalty_coef=float(penalty_coef)):
                                continue

                            for seed in seeds:
                                # For unsafe baselines, use full budget
                                budget = compute_budget(max_episode_steps, 1.0, costs)

                                base_args = dict(
                                    algo=algo,
                                    env_id=env_id,
                                    use_cost=bool(use_cost),
                                    use_all_obs=bool(use_all_obs),
                                    eval_num_episodes=eval_num_episodes,
                                    total_steps=total_steps,
                                    budget=budget,
                                    max_episode_steps=max_episode_steps,
                                    steps_per_epoch=steps_per_epoch,
                                    seed=seed,
                                    sd_regulizer=bool(sd_reg),
                                    random_obs_selection=bool(random_obs_selection),
                                    penalty_coef=float(penalty_coef),
                                    obs_modality_normalize=True,
                                )

                                # Environment-specific modifications
                                py_args = build_py_args_callback(base_args, costs)

                                # Build command and filename
                                python_cmd = build_python_command(run_py, py_args)

                                job_info = dict(
                                    algo=algo,
                                    env_id=env_id,
                                    use_cost=int(use_cost),
                                    use_all_obs=int(use_all_obs),
                                    budget=int(budget),
                                    seed=seed,
                                    sd_reg=int(sd_reg),
                                    random_obs_selection=int(random_obs_selection),
                                    penalty_coef=float(penalty_coef),
                                    tag=tag,
                                )
                                filename = build_filename_callback(job_info)

                                yield (python_cmd, filename)

        # ── Safe constrained algorithms ───────────────────────────────────────
        for use_all_obs in all_obs_usage:
            for use_cost in cost_usage:
                for br in budget_ratios:
                    for sd_reg in sd_regulizer_opts:
                        for random_obs_selection in random_obs_selection_opts:
                            for algo in safe_algos:
                                if not valid_combo(algo, bool(use_cost), bool(use_all_obs),
                                                  bool(sd_reg), bool(random_obs_selection),
                                                  safe_algos, unsafe_algos, penalty_coef=0.0):
                                    continue

                                budget = compute_budget(max_episode_steps, br, costs)

                                for seed in seeds:
                                    base_args = dict(
                                        algo=algo,
                                        env_id=env_id,
                                        use_cost=bool(use_cost),
                                        use_all_obs=bool(use_all_obs),
                                        eval_num_episodes=eval_num_episodes,
                                        total_steps=total_steps,
                                        budget=budget,
                                        max_episode_steps=max_episode_steps,
                                        steps_per_epoch=steps_per_epoch,
                                        seed=seed,
                                        sd_regulizer=bool(sd_reg),
                                        random_obs_selection=bool(random_obs_selection),
                                        penalty_coef=0.0,  # Safe algos typically don't use penalty
                                        obs_modality_normalize=True,
                                    )

                                    # Environment-specific modifications
                                    py_args = build_py_args_callback(base_args, costs)

                                    # Build command and filename
                                    python_cmd = build_python_command(run_py, py_args)

                                    job_info = dict(
                                        algo=algo,
                                        env_id=env_id,
                                        use_cost=int(use_cost),
                                        use_all_obs=int(use_all_obs),
                                        budget=int(budget),
                                        seed=seed,
                                        sd_reg=int(sd_reg),
                                        random_obs_selection=int(random_obs_selection),
                                        penalty_coef=0.0,  # Safe algos typically don't use penalty
                                        tag=tag,
                                    )
                                    filename = build_filename_callback(job_info)

                                    yield (python_cmd, filename)
