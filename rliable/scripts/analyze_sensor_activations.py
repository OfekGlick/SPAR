"""
Analyze Sensor Activation Patterns for Algorithm Comparison

This script processes evaluation CSV files from runs/ directory to analyze
how different algorithms (PPO, PPOLag, etc.) use sensors across tasks.

The CSVs contain binary sensor activation data (0/1) indicating which sensors
were active at each timestep during evaluation episodes.

Output:
- Per-seed averaged sensor activations (Level 1)
- Algorithm comparison data showing sensor usage patterns (Level 2)
"""

import pandas as pd
from pathlib import Path
from tqdm import tqdm
import re
from typing import Dict, List, Any, Tuple
import argparse


def find_evaluation_csvs(runs_dir: Path) -> List[Dict[str, Any]]:
    """
    Discover all evaluation CSV files in runs directory.

    Returns list of dicts with:
        - csv_path: Path to CSV file
        - run_config: Run configuration folder name
        - seed: Seed number (e.g., '000', '001')
    """
    csv_files = []

    # Pattern: runs/{run_config}/seed-{seed}-{timestamp}/evaluation_videos/*.csv
    for run_config_dir in runs_dir.iterdir():
        if not run_config_dir.is_dir():
            continue

        run_config = run_config_dir.name

        # Find all seed directories
        for seed_dir in run_config_dir.iterdir():
            if not seed_dir.is_dir() or not seed_dir.name.startswith('seed'):
                continue

            # Use the full seed directory name (seed-000-<timestamp>) to keep each run unique
            seed = seed_dir.name

            # Find evaluation_videos folder
            eval_videos_dir = seed_dir / 'evaluation_videos'
            if not eval_videos_dir.exists():
                continue

            # Find all CSV files (looking for *_masks.csv files)
            for csv_file in eval_videos_dir.glob('*.csv'):
                if '_masks.csv' in csv_file.name:
                    csv_files.append({
                        'csv_path': csv_file,
                        'run_config': run_config,
                        'seed': seed
                    })

    return csv_files


def determine_algorithm_variant(run_config: str) -> str:
    """Map run_config name to the correct algorithm variant label."""
    parts = run_config.split('-')
    base = parts[0] if parts else 'unknown'
    run_lower = run_config.lower()

    if base == 'PPO':
        if 'use_all_obs' in run_lower:
            return 'PPO-use-all-obs'
        if 'random_mask' in run_lower:
            return 'PPO-random-mask'
        pen_match = re.search(r'pen([0-9.]+)', run_config)
        if pen_match:
            return f"PPO-pen{pen_match.group(1)}"
        return 'PPO'

    if base == 'PPOLag':
        budget_match = re.search(r'Budget(\d+)', run_config)
        if budget_match:
            return f"PPOLag-Budget{budget_match.group(1)}"
        return 'PPOLag'

    return base


def extract_algorithm_and_task(run_config: str) -> Dict[str, str]:
    """
    Extract algorithm and task from run_config folder name.

    Examples:
        PPO-Door-random_mask-sd_reg-...
            → {'algorithm': 'PPO', 'task': 'Door'}
        PPOLag-Lift-use_cost_sd_reg-Budget400-...
            → {'algorithm': 'PPOLag', 'task': 'Lift'}
        PPO-highway-fast-v0-use_all_obs-...
            → {'algorithm': 'PPO', 'task': 'highway-fast-v0'}

    Args:
        run_config: Run configuration folder name

    Returns:
        Dict with 'algorithm' and 'task' keys
    """
    parts = run_config.split('-')

    if len(parts) < 2:
        return {'algorithm': 'unknown', 'task': 'unknown'}

    algorithm = determine_algorithm_variant(run_config)

    # Extract task (may be multi-part like highway-fast-v0)
    if 'Door' in run_config:
        task = 'Door'
    elif 'Lift' in run_config:
        task = 'Lift'
    elif 'highway-fast-v0' in run_config:
        task = 'highway-fast-v0'
    elif 'intersection' in run_config:
        task = 'intersection'
    else:
        task = parts[1]

    return {'algorithm': algorithm, 'task': task}


def parse_filename(filename: str) -> Dict[str, Any]:
    """
    Extract episode metadata from CSV filename.

    Example: eval_ep38_reward65.4_cost0.0_masks.csv
        → {'episode_num': 38, 'episode_reward': 65.4, 'episode_cost': 0.0}

    Args:
        filename: CSV filename

    Returns:
        Dict with episode_num, episode_reward, episode_cost
    """
    # Pattern: eval_ep{num}_reward{reward}_cost{cost}_masks.csv
    match = re.match(r'eval_ep(\d+)_reward([\d.]+)_cost([\d.]+)_masks\.csv', filename)

    if match:
        return {
            'episode_num': int(match.group(1)),
            'episode_reward': float(match.group(2)),
            'episode_cost': float(match.group(3))
        }
    else:
        return {
            'episode_num': -1,
            'episode_reward': 0.0,
            'episode_cost': 0.0
        }


def get_sensor_columns(task: str) -> List[str]:
    """
    Determine which columns are sensors based on task type.

    Highway tasks have different sensors than robosuite tasks.

    Args:
        task: Task name (Door, Lift, highway-fast-v0, etc.)

    Returns:
        List of sensor column names (excludes step, step_cost, cumulative_cost)
    """
    if 'highway' in task.lower():
        return ['Kinematics', 'LidarObservation', 'OccupancyGrid', 'TimeToCollision']
    else:  # Door, Lift (robosuite tasks)
        return ['robot_proprioception', 'object_states', 'task_features', 'camera']


def compute_per_seed_averages(runs_dir: Path, output_dir: Path) -> Dict[str, pd.DataFrame]:
    """
    Level 1 Aggregation: Average sensor activations per timestep for each seed.

    For each (task, algorithm, run_config, seed, step):
        - Average sensor values across all episodes in that seed

    Args:
        runs_dir: Path to runs directory
        output_dir: Path to output directory

    Returns:
        Dict mapping task name to per-seed averaged DataFrame
    """
    csv_files = find_evaluation_csvs(runs_dir)
    print(f"Found {len(csv_files)} CSV files")

    if len(csv_files) == 0:
        print("WARNING: No CSV files found!")
        return {}

    all_data = []

    for csv_info in tqdm(csv_files, desc="Processing CSVs"):
        try:
            df = pd.read_csv(csv_info['csv_path'])

            # Extract metadata
            algo_task = extract_algorithm_and_task(csv_info['run_config'])
            filename_info = parse_filename(csv_info['csv_path'].name)

            # Add metadata to each row
            df['task'] = algo_task['task']
            df['algorithm'] = algo_task['algorithm']
            df['run_config'] = csv_info['run_config']
            df['seed'] = csv_info['seed']
            df['episode_num'] = filename_info['episode_num']

            all_data.append(df)

        except Exception as e:
            print(f"\nError processing {csv_info['csv_path']}: {e}")
            continue

    if len(all_data) == 0:
        print("ERROR: No data loaded successfully!")
        return {}

    # Combine all episode data
    print("\nCombining all episode data...")
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"Total rows: {len(combined_df):,}")

    # Process each task separately (different sensor schemas)
    task_seed_dfs = {}

    for task in combined_df['task'].unique():
        print(f"\nProcessing task: {task}")
        task_df = combined_df[combined_df['task'] == task].copy()
        sensor_cols = get_sensor_columns(task)

        print(f"  Sensor columns: {sensor_cols}")

        # Verify all sensor columns exist
        missing_cols = [col for col in sensor_cols if col not in task_df.columns]
        if missing_cols:
            print(f"  WARNING: Missing columns {missing_cols} for task {task}")
            # Use only available sensor columns
            sensor_cols = [col for col in sensor_cols if col in task_df.columns]

        # Group by seed and step, average across episodes
        agg_dict = {sensor: 'mean' for sensor in sensor_cols}
        agg_dict.update({
            'step_cost': 'mean',
            'cumulative_cost': 'mean',
            'episode_num': 'count'  # Count episodes
        })

        grouped = task_df.groupby(
            ['task', 'algorithm', 'run_config', 'seed', 'step']
        ).agg(agg_dict).reset_index()

        # Rename count column
        grouped.rename(columns={'episode_num': 'n_episodes'}, inplace=True)

        # Rename sensor columns to avg_*
        for sensor in sensor_cols:
            grouped.rename(columns={sensor: f'avg_{sensor}'}, inplace=True)
        grouped.rename(columns={
            'step_cost': 'avg_step_cost',
            'cumulative_cost': 'avg_cumulative_cost'
        }, inplace=True)

        # Save per-seed averages
        output_file = output_dir / f"seed_averaged_{task}.csv"
        grouped.to_csv(output_file, index=False)
        task_seed_dfs[task] = grouped
        print(f"  Saved: {output_file} ({len(grouped):,} rows)")

    return task_seed_dfs


def compute_algorithm_comparison(task_seed_dfs: Dict[str, pd.DataFrame], output_dir: Path) -> Dict[Tuple[str, str], pd.DataFrame]:
    """
    Level 2 Aggregation: Average across seeds for algorithm comparison.

    For each (task, algorithm, step):
        - Average sensor activations across all seeds
        - Compute std across seeds to show variance

    Args:
        task_seed_dfs: Dict mapping task name to per-seed DataFrame
        output_dir: Path to output directory

    Returns:
        Dict mapping (task, algorithm) to DataFrame
    """
    comparison_dfs: Dict[Tuple[str, str], pd.DataFrame] = {}

    for task, seed_df in task_seed_dfs.items():
        print(f"\nComputing algorithm comparison for task: {task}")

        sensor_cols = get_sensor_columns(task)

        # Filter to only sensors that exist in the dataframe
        avg_sensor_cols = [f'avg_{sensor}' for sensor in sensor_cols if f'avg_{sensor}' in seed_df.columns]

        # Build aggregation dict dynamically based on available sensors
        agg_dict = {}
        for avg_col in avg_sensor_cols:
            agg_dict[avg_col] = ['mean', 'std']

        # Also aggregate costs
        agg_dict['avg_step_cost'] = ['mean', 'std']
        agg_dict['avg_cumulative_cost'] = ['mean', 'std']
        agg_dict['n_episodes'] = 'sum'  # Total episodes across seeds

        # Group by algorithm and step (across seeds)
        grouped = seed_df.groupby(['task', 'algorithm', 'step']).agg(agg_dict).reset_index()

        # Flatten multi-level columns
        new_cols = ['task', 'algorithm', 'step']
        for avg_col in avg_sensor_cols:
            sensor = avg_col.replace('avg_', '')
            new_cols.extend([avg_col, f'std_{sensor}'])

        new_cols.extend([
            'avg_step_cost', 'std_step_cost',
            'avg_cumulative_cost', 'std_cumulative_cost',
            'n_episodes_total'
        ])

        grouped.columns = new_cols

        # Count number of seeds
        seed_counts = seed_df.groupby(['task', 'algorithm', 'step'])['seed'].nunique().reset_index()
        seed_counts.rename(columns={'seed': 'n_seeds'}, inplace=True)
        grouped = grouped.merge(seed_counts, on=['task', 'algorithm', 'step'])

        algorithms = grouped['algorithm'].unique()
        print(f"  Algorithms: {', '.join(algorithms)}")
        print(f"  Seeds per algorithm: {grouped['n_seeds'].max()} (max)")

        for algorithm in algorithms:
            algo_df = grouped[grouped['algorithm'] == algorithm].copy()
            safe_algo = algorithm.replace('/', '_')
            output_file = output_dir / f"algorithm_comparison_{task}_{safe_algo}.csv"
            algo_df.to_csv(output_file, index=False)
            comparison_dfs[(task, algorithm)] = algo_df
            print(f"  Saved: {output_file} ({len(algo_df)} rows)")

    return comparison_dfs


def main():
    """Main entry point for sensor activation analysis."""
    parser = argparse.ArgumentParser(
        description="Analyze sensor activation patterns for algorithm comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # Basic usage
  python analyze_sensor_activations.py

  # Custom paths
  python analyze_sensor_activations.py --runs-dir ./runs --output-dir ./sensor_analysis

  # Skip saving per-seed averages (save space)
  python analyze_sensor_activations.py --skip-seed-avg
        """
    )
    parser.add_argument('--runs-dir', default='../../runs', help='Path to runs directory')
    parser.add_argument('--output-dir', default='../results/sensor_activation_data', help='Output directory')
    parser.add_argument('--skip-seed-avg', action='store_true',
                       help='Skip saving per-seed averages (saves space)')

    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    output_dir = Path(args.output_dir)

    # Validate inputs
    if not runs_dir.exists():
        print(f"ERROR: Runs directory not found: {runs_dir}")
        return

    output_dir.mkdir(exist_ok=True, parents=True)

    print("="*60)
    print("SENSOR ACTIVATION ANALYSIS")
    print("="*60)
    print(f"Runs directory: {runs_dir}")
    print(f"Output directory: {output_dir}")
    print()

    # Level 1: Per-seed averages
    print("="*60)
    print("LEVEL 1: Computing per-seed averages...")
    print("="*60)
    task_seed_dfs = compute_per_seed_averages(runs_dir, output_dir)

    if len(task_seed_dfs) == 0:
        print("\nERROR: No data processed. Exiting.")
        return

    # Level 2: Algorithm comparison
    print("\n" + "="*60)
    print("LEVEL 2: Computing algorithm comparison...")
    print("="*60)
    comparison_dfs = compute_algorithm_comparison(task_seed_dfs, output_dir)

    # Summary
    print("\n" + "="*60)
    print("COMPLETE!")
    print("="*60)
    print(f"Output directory: {output_dir}")
    unique_tasks = sorted({task for task, _ in comparison_dfs.keys()})
    print(f"Tasks processed: {len(unique_tasks)}")
    print("\nPrimary output files (for algorithm comparison):")
    for task, algorithm in comparison_dfs.keys():
        safe_algo = algorithm.replace('/', '_')
        print(f"  - algorithm_comparison_{task}_{safe_algo}.csv")

    if not args.skip_seed_avg:
        print("\nPer-seed files (for detailed analysis):")
        for task in task_seed_dfs.keys():
            print(f"  - seed_averaged_{task}.csv")

    print("\n" + "="*60)
    print("Next steps:")
    print("  1. Load algorithm_comparison_*.csv files")
    print("  2. Plot sensor activation rates over time")
    print("  3. Compare sensor usage across algorithms")
    print("  4. Analyze cost vs performance trade-offs")
    print("="*60)


if __name__ == '__main__':
    main()
