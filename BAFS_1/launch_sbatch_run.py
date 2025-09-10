import argparse
import subprocess
import os
import numpy as np
import budget_aware_robot


def build_sbatch_from_argparse(args):
    with open("../sbatch_template_2.sh", "r") as f:
        sbatch_template = f.read()
    python_command = "python /home/ofek.glick/RLC_code/safe_robotics.py"
    for arg, value in vars(args).items():
        if isinstance(value, bool):
            if value:
                python_command += f" --{arg.replace('_', '-')}"
        else:
            python_command += f" --{arg.replace('_', '-')} {value}"
    print(python_command)
    formatted_sbatch = sbatch_template.format(python_command)
    return formatted_sbatch


def create_sbatch_file(sbatch_text: str, sbatch_path: str) -> None:
    """
    Create an sbatch file with the given text.
    Args:
        sbatch_text (str): The text to write to the sbatch file.
        sbatch_path (str): The path to the sbatch file.
    """
    os.makedirs(os.path.dirname(sbatch_path), exist_ok=True)
    with open(sbatch_path, "w") as f:
        f.write(sbatch_text)


def run_sbatch_files(sbatch_files: list) -> None:
    for sbatch_path in sbatch_files:
        print(f"Running sbatch file: {sbatch_path}")
        subprocess.run(["sbatch", sbatch_path])


def calculate_budget(
        max_episode_length: int = 1000,
        budget_ratio: float = 0.5,
        feature_cost: np.ndarray = None,
) -> tuple[float, np.ndarray]:
    """
        Calculate the budget for the given environment.
        Args:
            env_name (str): The name of the environment.
        Returns:
            float: The calculated budget.
    """

    max_cost_per_episode = max_episode_length * feature_cost.sum()
    budget = budget_ratio * max_cost_per_episode
    return budget


def configure_run_from_args(
        algo,
        env,
        use_cost,
        use_all_obs,
        num_eval_episodes,
        total_steps,
        budget_ratio,
        feature_cost,
        max_episode_steps,
        steps_per_epoch,
        seed,
        sbatch_path_dir,
        sbatch_files_created,
):
    budget = calculate_budget(
        max_episode_length=max_episode_steps,
        budget_ratio=budget_ratio,
        feature_cost=feature_cost,
    )
    if use_cost and use_all_obs:
        return
    if use_cost and algo in unsafe_algorithms:
        return
    if not use_cost and algo in safe_algorithms:
        return
    if use_all_obs and algo in safe_algorithms:
        return
    args = argparse.Namespace(
        algo=algo,
        env_id=env,
        use_cost=use_cost,
        use_all_obs=use_all_obs,
        eval_num_episodes=num_eval_episodes,
        total_steps=total_steps,
        budget=budget,
        feature_cost=" ".join([f"{cost:.2f}" for cost in feature_cost]),
        max_episode_steps=max_episode_steps,
        steps_per_epoch=steps_per_epoch,
        seed=seed,
    )
    sbatch = build_sbatch_from_argparse(args)
    sbatch_path = f"{sbatch_path_dir}/{algo}_{env}_{use_cost}_{use_all_obs}_{str(int(budget))}_{seed}.sh"
    create_sbatch_file(sbatch, sbatch_path)
    sbatch_files_created.append(sbatch_path)


if __name__ == '__main__':
    # To run from cli use like this:
    # python launch_sbatch_run.py --algo CPO --env-id BudgetAwareFetchReachDense-v3 --use-cost
    # Otherwise, define the arguments here:

    safe_algorithms = ['CPO']
    unsafe_algorithms = ['PPO']

    cost_usage = [True]
    all_obs_usage = [False, True]

    total_steps = 500_000
    num_eval_episodes = 20
    max_episode_steps = 250
    steps_per_epoch = 250

    assert max_episode_steps <= steps_per_epoch, 'Max episode steps should be less than or equal to steps per epoch - otherwise you wont get any episodic data'
    budget_ratios = [0.1, 0.35, 0.7]

    envs = [
        "BudgetAwareFetchReachDense-v3",
        # "BudgetAwareFetchSlideDense-v3",
        # "BudgetAwareFetchPushDense-v3",
    ]

    sbatch_path_dir = f"/home/ofek.glick/RLC_code/sbatch_runs_files"
    sbatch_files_created = []
    for seed in [31, 32 ,33 ,34, 35]:
        for env in envs:
            env_instance = budget_aware_robot.DiscreteActionWrapperCMDP(env)
            obs_dim = env_instance.observation_space.shape[0]
            feature_cost = 5 * np.random.random(size=obs_dim)
            feature_cost = np.array([float(cost) for cost in feature_cost])
            for algo1 in unsafe_algorithms:
                for use_all_obs in all_obs_usage:
                    configure_run_from_args(
                        algo=algo1,
                        env=env,
                        use_cost=False,
                        use_all_obs=use_all_obs,
                        num_eval_episodes=num_eval_episodes,
                        total_steps=total_steps,
                        budget_ratio=1,
                        feature_cost=feature_cost,
                        max_episode_steps=max_episode_steps,
                        steps_per_epoch=steps_per_epoch,
                        seed=seed,
                        sbatch_path_dir=sbatch_path_dir,
                        sbatch_files_created=sbatch_files_created,
                    )
                    for algo2 in safe_algorithms:
                        for use_cost in cost_usage:
                            for budget_ratio in budget_ratios:
                                configure_run_from_args(
                                    algo=algo2,
                                    env=env,
                                    use_cost=use_cost,
                                    use_all_obs=use_all_obs,
                                    num_eval_episodes=num_eval_episodes,
                                    total_steps=total_steps,
                                    budget_ratio=budget_ratio,
                                    feature_cost=feature_cost,
                                    max_episode_steps=max_episode_steps,
                                    steps_per_epoch=steps_per_epoch,
                                    seed=seed,
                                    sbatch_path_dir=sbatch_path_dir,
                                    sbatch_files_created=sbatch_files_created,
                                )

    run_sbatch_files(sbatch_files_created)
