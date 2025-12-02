"""
Generate per-task sensor activation heatmaps from analysis CSVs.

This script loads the per-algorithm CSVs produced by
`analyze_sensor_activations.py` (files named
`algorithm_comparison_<task>_<algorithm>.csv`) and creates a figure per task
showing side-by-side heatmaps (green=active, black=inactive) for each
algorithm's average sensor usage across timesteps.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap


def get_sensor_columns(task: str) -> List[str]:
    """Return the ordered list of sensor columns for a task."""
    task_lower = task.lower()
    if "highway" in task_lower or "intersection" in task_lower:
        return ["Kinematics", "LidarObservation", "OccupancyGrid", "TimeToCollision"]
    return ["robot_proprioception", "object_states", "task_features", "camera"]


def prettify_sensor_name(sensor: str) -> str:
    """
    Convert sensor names to pretty display format.
    Examples:
        TimeToCollision -> Time To Collision
        robot_proprioception -> Robot Proprioception
        LidarObservation -> Lidar Observation
    """
    import re

    # Replace underscores with spaces
    sensor = sensor.replace('_', ' ')

    # Add spaces before capital letters in camelCase (e.g., TimeToCollision -> Time To Collision)
    sensor = re.sub(r'([a-z])([A-Z])', r'\1 \2', sensor)

    # Capitalize each word
    sensor = ' '.join(word.capitalize() for word in sensor.split())

    return sensor


def load_algorithm_csvs(analysis_dir: Path) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Load all algorithm comparison CSVs into a nested dict:
    {task: {algorithm: dataframe}}.
    """
    results: Dict[str, Dict[str, pd.DataFrame]] = {}
    csv_paths = sorted(analysis_dir.glob("algorithm_comparison_*_*.csv"))

    if not csv_paths:
        raise SystemExit(f"No algorithm_comparison CSVs found in {analysis_dir}")

    for csv_path in csv_paths:
        try:
            df = pd.read_csv(csv_path)
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] Failed to read {csv_path}: {exc}")
            continue

        if df.empty or "task" not in df or "algorithm" not in df:
            print(f"[WARN] Skipping {csv_path} (missing task/algorithm columns).")
            continue

        task = df["task"].iloc[0]
        algorithm = df["algorithm"].iloc[0]
        results.setdefault(task, {})[algorithm] = df

    if not results:
        raise SystemExit("No valid CSVs loaded; cannot plot.")

    return results


def create_cmap() -> LinearSegmentedColormap:
    """Create a black-to-green colormap for visualizing activation."""
    cmap = LinearSegmentedColormap.from_list("black_green", ["black", "#00ff00"])
    cmap.set_bad(color="#444444")
    return cmap


def map_budgets_to_percentages(algo_frames: Dict[str, pd.DataFrame]) -> Dict[str, str]:
    """
    Map actual budget values to percentages (20%, 50%, 80%).
    Returns a dict mapping original algo_id to percentage-based label.
    """
    import re

    # Find all PPOLag budgets
    budgets = []
    for algo_id in algo_frames.keys():
        budget_match = re.search(r'Budget(\d+)', algo_id)
        if budget_match:
            budgets.append(int(budget_match.group(1)))

    # Map smallest=20%, middle=50%, largest=80%
    budget_to_pct = {}
    if len(budgets) >= 3:
        sorted_budgets = sorted(set(budgets))
        budget_to_pct[sorted_budgets[0]] = '20%'
        budget_to_pct[sorted_budgets[1]] = '50%'
        budget_to_pct[sorted_budgets[2]] = '80%'

    # Create display name mapping
    display_names = {}
    for algo_id in algo_frames.keys():
        budget_match = re.search(r'Budget(\d+)', algo_id)
        if budget_match:
            budget = int(budget_match.group(1))
            pct = budget_to_pct.get(budget, f'{budget}')
            display_names[algo_id] = f'PPO-Lag ({pct})'
        elif 'pen' in algo_id.lower():
            display_names[algo_id] = 'PPO (Penalty)'
        elif 'use_all_obs' in algo_id.lower():
            display_names[algo_id] = 'PPO (All Sensors)'
        elif 'random' in algo_id.lower() and 'mask' in algo_id.lower():
            display_names[algo_id] = 'PPO (Random Mask)'
        elif algo_id == 'PPO':
            display_names[algo_id] = 'PPO (Baseline)'
        else:
            display_names[algo_id] = algo_id

    return display_names


def get_algorithm_sort_order(algo_id: str) -> int:
    """
    Get sort order for algorithms to control display order in heatmaps.
    Lower numbers appear first (left to right in plots).

    Order: PPO-Lag (80%, 50%, 20%), PPO-Penalty, PPO-All, PPO-Random, PPO-Baseline
    """
    import re

    if 'Budget' in algo_id:
        budget_match = re.search(r'Budget(\d+)', algo_id)
        if budget_match:
            budget = int(budget_match.group(1))
            return -budget  # Negative so higher budgets sort first
        return 0

    if 'pen' in algo_id.lower():
        return 1000
    elif 'use_all_obs' in algo_id.lower():
        return 2000
    elif 'random' in algo_id.lower() and 'mask' in algo_id.lower():
        return 3000
    elif algo_id == 'PPO':
        return 4000

    return 9999


def plot_task_heatmaps(
    task: str,
    algo_frames: Dict[str, pd.DataFrame],
    output_dir: Path,
    cmap: LinearSegmentedColormap,
) -> Path:
    """Create and save the heatmap figure for a single task."""
    sensors = get_sensor_columns(task)

    # Filter out "random mask" and "use all obs" variants
    filtered_algo_frames = {
        algo_id: df for algo_id, df in algo_frames.items()
        if not ('use-all-obs' in algo_id.lower() or 'random-mask' in algo_id.lower())
    }

    # Get display names and sort algorithms
    display_names = map_budgets_to_percentages(filtered_algo_frames)
    algorithms = sorted(filtered_algo_frames.keys(), key=get_algorithm_sort_order)

    n_algos = len(algorithms)
    fig, axes = plt.subplots(
        1, n_algos, figsize=(4 * n_algos, 3), sharey=True, constrained_layout=True
    )
    if n_algos == 1:
        axes = [axes]  # type: ignore[list-item]

    for ax, algorithm in zip(axes, algorithms):
        df = algo_frames[algorithm]
        sensor_arrays = []
        for sensor in sensors:
            col = f"avg_{sensor}"
            if col not in df:
                sensor_arrays.append(np.full(df.shape[0], np.nan))
            else:
                sensor_arrays.append(df[col].to_numpy())

        data = np.vstack(sensor_arrays)
        im = ax.imshow(
            data,
            aspect="auto",
            origin="lower",
            cmap=cmap,
            vmin=0.0,
            vmax=1.0,
            interpolation="nearest",
        )

        # Use display name for title
        ax.set_title(display_names[algorithm], fontsize=14, fontweight='bold')
        ax.set_xlabel("Step", fontsize=12)
        n_steps = len(df)
        n_ticks = min(5, n_steps) if n_steps > 1 else 1
        tick_positions = np.linspace(0, max(n_steps - 1, 0), num=n_ticks)
        tick_indices = np.clip(np.round(tick_positions).astype(int), 0, n_steps - 1)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([int(df["step"].iloc[idx]) for idx in tick_indices], fontsize=10)

        ax.set_yticks(range(len(sensors)))
        ax.set_yticklabels([prettify_sensor_name(s) for s in sensors], fontsize=10)

    axes[0].set_ylabel("Sensor", fontsize=12)
    fig.suptitle(f"{task} sensor activation (avg)", fontsize=16, fontweight='bold')

    # Single colorbar for the entire row
    cbar = fig.colorbar(im, ax=axes, shrink=0.75, pad=0.02)
    cbar.set_label("Activation probability", fontsize=12)

    # Save to rliable/results/budget-aware-{task}/ directory
    budget_aware_dir = f"budget-aware-{task}"
    task_output_dir = output_dir / budget_aware_dir
    task_output_dir.mkdir(parents=True, exist_ok=True)
    safe_task = task.replace("/", "_")
    output_path = task_output_dir / f"{safe_task}_sensor_heatmaps.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot per-task sensor activation heatmaps from analysis CSVs."
    )
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=Path("../results/sensor_activation_data"),
        help="Directory containing algorithm_comparison_*.csv files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("../results"),
        help="Base directory to save heatmap images (plots saved to {output_dir}/{task}/)",
    )
    args = parser.parse_args()

    algo_data = load_algorithm_csvs(args.analysis_dir)
    cmap = create_cmap()

    saved_paths = []
    for task, frames in algo_data.items():
        print(f"Plotting task: {task} ({len(frames)} algorithms)")
        output_path = plot_task_heatmaps(task, frames, args.output_dir, cmap)
        saved_paths.append(output_path)
        print(f"  Saved: {output_path}")

    print("\nGenerated figures:")
    for path in saved_paths:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
