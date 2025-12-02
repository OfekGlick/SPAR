"""
Aggregate sensor activation masks across runs to produce per-step averages
for each (task, algorithm) pair.

The script scans run directories laid out as:
    runs/{run_config}/seed-XXX-*/evaluation_videos/*_masks.csv

For every CSV it sums sensor activations step-by-step, aggregates across
episodes/seeds/run variants, and writes one CSV per (task, algorithm) pair
listing the average activation of each sensor at every timestep.
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd


SEED_DIR_PATTERN = re.compile(r"seed.*")


def discover_mask_csvs(runs_dir: Path) -> List[Dict[str, Path]]:
    """Return metadata for every *_masks.csv file under runs/."""
    csv_entries: List[Dict[str, Path]] = []

    for run_config_dir in sorted(runs_dir.iterdir()):
        if not run_config_dir.is_dir():
            continue

        for seed_dir in sorted(run_config_dir.glob("seed*")):
            if not SEED_DIR_PATTERN.match(seed_dir.name):
                continue

            evaluation_dir = seed_dir / "evaluation_videos"
            if not evaluation_dir.exists():
                continue

            for csv_path in sorted(evaluation_dir.glob("*_masks.csv")):
                csv_entries.append(
                    {
                        "run_config": run_config_dir.name,
                        "seed": seed_dir.name,
                        "path": csv_path,
                    }
                )

    return csv_entries


def determine_algorithm_variant(run_config: str) -> str:
    """Return algorithm variant name for a given run configuration."""
    parts = run_config.split("-")
    base = parts[0] if parts else "unknown"
    run_lower = run_config.lower()

    if base == "PPO":
        if "use_all_obs" in run_lower:
            return "PPO-use-all-obs"
        if "random_mask" in run_lower:
            return "PPO-random-mask"
        pen_match = re.search(r"pen([0-9.]+)", run_config)
        if pen_match:
            return f"PPO-pen{pen_match.group(1)}"
        return "PPO"

    if base == "PPOLag":
        budget_match = re.search(r"Budget(\d+)", run_config)
        if budget_match:
            return f"PPOLag-Budget{budget_match.group(1)}"
        return "PPOLag"

    return base


def extract_task_algorithm(run_config: str) -> Tuple[str, str]:
    """Extract task and algorithm variant names from a run_config folder name."""
    parts = run_config.split("-")
    algorithm = determine_algorithm_variant(run_config)

    if "Door" in run_config:
        task = "Door"
    elif "Lift" in run_config:
        task = "Lift"
    elif "highway-fast-v0" in run_config:
        task = "highway-fast-v0"
    elif "intersection" in run_config:
        task = "intersection"
    else:
        task = parts[1] if len(parts) > 1 else "unknown"

    return task, algorithm


def aggregate_per_step(csv_entries: Iterable[Dict[str, Path]]) -> Tuple[
    Dict[Tuple[str, str, int], Dict[str, float]],
    Dict[Tuple[str, str, int], Dict[str, int]],
    Dict[Tuple[str, str, int], int],
    Dict[Tuple[str, str, int], Set[str]],
    Dict[Tuple[str, str], Set[str]],
]:
    """
    Aggregate sensor activations per (task, algorithm, step).

    Returns:
        sensor_sums: Sum of sensor activations for every key.
        sensor_counts: Count of samples contributing to each sensor sum.
        row_counts: Number of timesteps aggregated into each key.
        seed_trackers: Set of seed ids contributing to a key.
        sensors_per_pair: union of sensors seen for each (task, algorithm).
    """
    sensor_sums: Dict[Tuple[str, str, int], Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    sensor_counts: Dict[Tuple[str, str, int], Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    row_counts: Dict[Tuple[str, str, int], int] = defaultdict(int)
    seed_trackers: Dict[Tuple[str, str, int], Set[str]] = defaultdict(set)
    sensors_per_pair: Dict[Tuple[str, str], Set[str]] = defaultdict(set)

    for entry in csv_entries:
        csv_path: Path = entry["path"]
        run_config: str = entry["run_config"]
        seed_id: str = entry["seed"]
        task, algorithm = extract_task_algorithm(run_config)

        try:
            df = pd.read_csv(csv_path)
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] Failed to read {csv_path}: {exc}")
            continue

        sensor_cols = [
            col
            for col in df.columns
            if col
            not in {
                "step",
                "step_cost",
                "cumulative_cost",
                "episode",
            }
        ]
        if "step" not in df.columns:
            print(f"[WARN] Missing 'step' column in {csv_path}, skipping.")
            continue
        if not sensor_cols:
            print(f"[WARN] No sensor columns found in {csv_path}, skipping.")
            continue

        sensors_per_pair[(task, algorithm)].update(sensor_cols)

        for _, row in df.iterrows():
            step = int(row["step"])
            key = (task, algorithm, step)
            row_counts[key] += 1

            for sensor in sensor_cols:
                value = row.get(sensor)
                if pd.isna(value):
                    continue
                sensor_sums[key][sensor] += float(value)
                sensor_counts[key][sensor] += 1

            seed_trackers[key].add(seed_id)

    return sensor_sums, sensor_counts, row_counts, seed_trackers, sensors_per_pair


def build_output_frames(
    sensor_sums: Dict[Tuple[str, str, int], Dict[str, float]],
    sensor_counts: Dict[Tuple[str, str, int], Dict[str, int]],
    row_counts: Dict[Tuple[str, str, int], int],
    seed_trackers: Dict[Tuple[str, str, int], Set[str]],
    sensors_per_pair: Dict[Tuple[str, str], Set[str]],
) -> Dict[Tuple[str, str], pd.DataFrame]:
    """Construct per-(task,algorithm) DataFrames ready for saving."""
    frames: Dict[Tuple[str, str], pd.DataFrame] = {}

    for (task, algorithm), sensors in sensors_per_pair.items():
        rows: List[Dict[str, Optional[float]]] = []
        sensors_sorted = sorted(sensors)

        relevant_steps = sorted(
            step
            for (t, a, step) in sensor_sums.keys()
            if t == task and a == algorithm
        )

        for step in relevant_steps:
            key = (task, algorithm, step)
            row: Dict[str, Optional[float]] = {
                "task": task,
                "algorithm": algorithm,
                "step": step,
                "n_samples": row_counts.get(key, 0),
                "n_unique_seeds": len(seed_trackers[key]),
            }

            for sensor in sensors_sorted:
                count = sensor_counts[key].get(sensor, 0)
                if count == 0:
                    row[sensor] = None
                else:
                    row[sensor] = sensor_sums[key][sensor] / count

            rows.append(row)

        if not rows:
            continue

        frames[(task, algorithm)] = pd.DataFrame(rows).sort_values("step")

    return frames


def save_frames(frames: Dict[Tuple[str, str], pd.DataFrame], output_dir: Path) -> List[Path]:
    """Write DataFrames to CSV and return their paths."""
    saved_paths: List[Path] = []
    output_dir.mkdir(parents=True, exist_ok=True)

    for (task, algorithm), df in frames.items():
        safe_task = task.replace("/", "_")
        output_path = output_dir / f"{algorithm}_{safe_task}_per_step_activation.csv"
        df.to_csv(output_path, index=False)
        saved_paths.append(output_path)

    return saved_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create per-step average sensor activations per algorithm/task.",
    )
    parser.add_argument("--runs-dir", type=Path, default=Path("../../runs"), help="Path to runs directory")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("../results/sensor_activation_data/per_algo_sensor_usage"),
        help="Directory to store aggregated CSVs",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs_dir: Path = args.runs_dir
    output_dir: Path = args.output_dir

    if not runs_dir.exists():
        raise SystemExit(f"Runs directory not found: {runs_dir}")

    csv_entries = discover_mask_csvs(runs_dir)
    if not csv_entries:
        raise SystemExit("No *_masks.csv files found under runs/, nothing to aggregate.")

    print(f"Found {len(csv_entries)} mask CSVs. Aggregating...")
    sensor_sums, sensor_counts, row_counts, seed_trackers, sensors_per_pair = aggregate_per_step(csv_entries)

    frames = build_output_frames(sensor_sums, sensor_counts, row_counts, seed_trackers, sensors_per_pair)
    if not frames:
        raise SystemExit("No aggregated frames were produced. Check input files.")

    saved = save_frames(frames, output_dir)
    print("Saved CSVs:")
    for path in saved:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
