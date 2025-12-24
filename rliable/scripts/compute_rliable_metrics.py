"""
Compute rliable metrics for SPAR Robosuite experiments.

This script loads experimental results from the CSV manifest and computes:
1. Aggregate performance metrics (IQM, Mean, Median)
2. Performance profiles
3. Sample efficiency curves
4. Probability of improvement comparisons
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')

# Import rliable - try relative imports first, fall back to absolute
try:
    from .. import library as rly
    from .. import metrics as rly_metrics
    from .. import plot_utils as rly_plot
except ImportError:
    # If relative imports fail (when running script directly), use absolute imports
    import sys
    from pathlib import Path
    # Add parent directory to path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    import library as rly
    import metrics as rly_metrics
    import plot_utils as rly_plot


class RliableAnalyzer:
    """Analyzer for computing rliable metrics on experimental data."""

    def __init__(self, csv_path: str, output_dir: str = "../results"):
        """
        Initialize the analyzer.

        Args:
            csv_path: Path to the manifest CSV file
            output_dir: Directory to save results and plots
        """
        self.csv_path = csv_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        # Load data
        print(f"Loading data from {csv_path}...")

        # Define the columns we want (some may not exist in all CSV files)
        desired_cols = [
            'timestamp', 'algo', 'env', 'seed', 'budget', 'obs_mode', 'actor_type',
            'use_cost', 'use_all_obs', 'random_obs_selection',
            'reward_mean', 'reward_std', 'cost_mean', 'cost_std',
            'episode_rewards', 'episode_costs', 'sample_efficiency_curve',
            'reward_normalized', 'cost_normalized', 'obs_modality_normalize',
            'log_dir'  # Needed for penalty coefficient detection
        ]

        # Load CSV using csv module to properly handle complex quoted fields
        # pandas sometimes misparses CSVs with nested JSON
        import csv as csv_module
        import sys

        # Increase field size limit for large JSON fields
        # Use a large but reasonable limit (10MB)
        csv_module.field_size_limit(10 * 1024 * 1024)

        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv_module.DictReader(f)
            data = list(reader)

        # Convert to DataFrame
        self.df = pd.DataFrame(data)

        # Filter to only the columns we want that exist
        available_cols = self.df.columns.tolist()
        cols_to_keep = [col for col in desired_cols if col in available_cols]

        print(f"  Using {len(cols_to_keep)} of {len(available_cols)} available columns")

        # Keep only the desired columns
        self.df = self.df[cols_to_keep]

        # Convert boolean columns
        bool_cols = ['use_cost', 'use_all_obs', 'random_obs_selection', 'obs_modality_normalize']
        for col in bool_cols:
            if col in self.df.columns:
                self.df[col] = self.df[col].fillna(False).astype(bool)

        # Convert numeric columns
        numeric_cols = ['reward_mean', 'reward_std', 'cost_mean', 'cost_std',
                       'reward_normalized', 'cost_normalized']
        for col in numeric_cols:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')

        print(f"Loaded {len(self.df)} experimental runs")

        # Display names for plots (user-friendly)
        self.display_names = {
            'PPO': 'PPO (Baseline - No Cost)',
            'PPO-penalty': 'PPO (Penalty)',
            'PPO-use-all-obs': 'PPO (All Sensors)',
            'PPO-random-mask': 'PPO (Random Selection)',
            'PPOLag-20pct': 'PPO-Lag (20% Budget)',
            'PPOLag-50pct': 'PPO-Lag (50% Budget)',
            'PPOLag-80pct': 'PPO-Lag (80% Budget)'
        }

        # Parse JSON columns
        self._parse_json_columns()

    def _parse_json_columns(self):
        """Parse JSON string columns into Python objects."""
        print("Parsing JSON columns...")

        def safe_json_parse(x, default):
            """Safely parse JSON with error handling."""
            if not isinstance(x, str) or not x.strip():
                return default

            try:
                # Try parsing as-is
                return json.loads(x)
            except json.JSONDecodeError:
                try:
                    # Try with ast.literal_eval as fallback
                    import ast
                    return ast.literal_eval(x)
                except:
                    # If all else fails, return default
                    print(f"  Warning: Could not parse value (length {len(x)}), using default")
                    return default

        # Parse episode rewards/costs
        if 'episode_rewards' in self.df.columns:
            print("  Parsing episode_rewards...")
            self.df['episode_rewards_parsed'] = self.df['episode_rewards'].apply(
                lambda x: safe_json_parse(x, [])
            )

        if 'episode_costs' in self.df.columns:
            print("  Parsing episode_costs...")
            self.df['episode_costs_parsed'] = self.df['episode_costs'].apply(
                lambda x: safe_json_parse(x, [])
            )

        # Parse sample efficiency curves
        if 'sample_efficiency_curve' in self.df.columns:
            print("  Parsing sample_efficiency_curve...")
            self.df['sample_efficiency_parsed'] = self.df['sample_efficiency_curve'].apply(
                lambda x: safe_json_parse(x, {})
            )

    def get_algorithm_groups(self, df: Optional[pd.DataFrame] = None) -> List[str]:
        """
        Get unique algorithm configurations for grouping.

        Args:
            df: Optional dataframe to use (default: self.df)

        Returns:
            List of algorithm group identifiers
        """
        if df is None:
            df = self.df

        # Create budget mappings: smallest->20%, middle->50%, largest->80%
        budget_mappings = {}

        for algo_name in df['algo'].unique():
            algo_df = df[df['algo'] == algo_name]
            budgets = pd.to_numeric(algo_df['budget'], errors='coerce').dropna().unique()

            if len(budgets) >= 3:
                sorted_budgets = sorted(budgets)
                budget_mappings[algo_name] = {
                    sorted_budgets[0]: '20pct',
                    sorted_budgets[1]: '50pct',
                    sorted_budgets[2]: '80pct'
                }
                print(f"  Budget mapping for {algo_name}: {sorted_budgets[0]:.2f}->20%, {sorted_budgets[1]:.2f}->50%, {sorted_budgets[2]:.2f}->80%")
            elif len(budgets) == 2:
                # Fallback for environments with only 2 budgets
                sorted_budgets = sorted(budgets)
                budget_mappings[algo_name] = {
                    sorted_budgets[0]: '50pct',
                    sorted_budgets[1]: '80pct'
                }
                print(f"  Budget mapping for {algo_name}: {sorted_budgets[0]:.2f}->50%, {sorted_budgets[1]:.2f}->80%")

        # Create algorithm ID based on obs_mode column and penalty detection
        def create_algo_id(row):
            """Create algorithm identifier with budget information."""
            algo_name = row['algo']
            obs_mode = row.get('obs_mode', 'SelectedObs')  # Use obs_mode column

            # Check for penalty coefficient using cost_mean
            # PPO runs with penalty have cost_mean = 0.0
            # PPO runs without penalty have cost_mean > 0
            has_penalty = False
            if obs_mode == 'SelectedObs' and algo_name == 'PPO':
                try:
                    cost_mean = float(row.get('cost_mean', '0'))
                    has_penalty = (cost_mean == 0.0)
                except (ValueError, TypeError):
                    has_penalty = False

            # Handle budget-constrained algorithms (PPOLag)
            if obs_mode == 'SelectedObs' and algo_name in budget_mappings:
                if pd.notna(row.get('budget')) and row['budget'] != 'None':
                    try:
                        budget_val = float(row['budget'])
                        if budget_val in budget_mappings[algo_name]:
                            budget_pct = budget_mappings[algo_name][budget_val]
                            return f"{algo_name}-{budget_pct}"
                    except (ValueError, TypeError):
                        pass

            # Handle non-budget configs
            if obs_mode == 'AllObs':
                return "PPO-use-all-obs"
            elif obs_mode == 'RandomMask':
                return "PPO-random-mask"
            elif obs_mode == 'SelectedObs':
                # Baseline PPO: split by penalty coefficient
                if has_penalty:
                    return "PPO-penalty"
                else:
                    return "PPO"

            # Fallback
            return algo_name

        df['algo_id'] = df.apply(create_algo_id, axis=1)

        algo_groups = sorted(df['algo_id'].unique().tolist())
        print(f"\nFound {len(algo_groups)} algorithm configurations:")
        for ag in algo_groups:
            count = len(df[df['algo_id'] == ag])
            print(f"  {ag}: {count} runs")

        return algo_groups

    def get_unique_environments(self) -> List[str]:
        """
        Get unique environments from the dataframe.

        Returns:
            List of unique environment names sorted alphabetically
        """
        envs = self.df['env'].unique().tolist()
        envs.sort()
        print(f"\nFound {len(envs)} unique environments:")
        for env in envs:
            count = len(self.df[self.df['env'] == env])
            print(f"  {env}: {count} runs")
        return envs

    def prepare_score_dict(self, metric: str = 'reward_mean',
                          task_column: str = 'env',
                          env_filter: Optional[str] = None) -> Dict[str, np.ndarray]:
        """
        Prepare score dictionary in rliable format: {algo: (num_runs × num_tasks)}.

        Args:
            metric: Column name for the metric to analyze (e.g., 'reward_mean', 'cost_mean')
            task_column: Column name for task identification (e.g., 'env')
            env_filter: If provided, filter to single environment (each run becomes a task)

        Returns:
            Dictionary mapping algorithm to score array
        """
        print(f"\nPreparing score dictionary for metric: {metric}")
        if env_filter:
            print(f"  Filtering to environment: {env_filter}")

        # Filter by environment if specified
        df_to_use = self.df
        if env_filter:
            df_to_use = self.df[self.df['env'] == env_filter].copy()
            if len(df_to_use) == 0:
                print(f"  WARNING: No data found for environment {env_filter}")
                return {}
            print(f"  Found {len(df_to_use)} runs for {env_filter}")

        algo_groups = self.get_algorithm_groups(df=df_to_use)
        score_dict = {}

        for algo_id in algo_groups:
            algo_df = df_to_use[df_to_use['algo_id'] == algo_id]

            if len(algo_df) == 0:
                continue

            if env_filter:
                # Single environment: each run is a "task"
                # Shape: (num_runs, 1) - each row is a run, single column
                scores = algo_df[metric].values.reshape(-1, 1)
                score_array = scores
            else:
                # Multi-environment (original behavior): each environment is a task
                # Get unique tasks
                tasks = algo_df[task_column].unique()

                # For each task, collect scores across runs
                task_scores = []
                for task in tasks:
                    task_df = algo_df[algo_df[task_column] == task]
                    scores = task_df[metric].values
                    task_scores.append(scores)

                # Pad to same length (in case different numbers of runs per task)
                max_runs = max(len(s) for s in task_scores)
                padded_scores = []
                for scores in task_scores:
                    if len(scores) < max_runs:
                        # Pad with mean of existing scores
                        padding = np.full(max_runs - len(scores), np.mean(scores))
                        scores = np.concatenate([scores, padding])
                    padded_scores.append(scores)

                # Shape: (num_runs, num_tasks)
                score_array = np.array(padded_scores).T

            score_dict[algo_id] = score_array
            print(f"  {algo_id}: {score_array.shape} (runs × tasks)")

        return score_dict

    def compute_aggregate_metrics(self, score_dict: Dict[str, np.ndarray],
                                 reps: int = 50000) -> Tuple[Dict, Dict]:
        """
        Compute aggregate metrics with confidence intervals.

        Args:
            score_dict: Dictionary mapping algorithm to score array
            reps: Number of bootstrap replications

        Returns:
            Tuple of (point_estimates, interval_estimates)
        """
        print(f"\nComputing aggregate metrics with {reps} bootstrap replications...")

        # Define aggregate functions - only IQM
        aggregate_func = lambda x: np.array([
            rly_metrics.aggregate_iqm(x),
        ])

        # Compute point estimates and CIs
        point_estimates, interval_estimates = rly.get_interval_estimates(
            score_dict,
            func=aggregate_func,
            method='percentile',
            task_bootstrap=False,
            reps=reps,
            confidence_interval_size=0.95
        )

        print("Aggregate metrics computed successfully!")
        return point_estimates, interval_estimates

    def compute_performance_profiles(self, score_dict: Dict[str, np.ndarray],
                                    tau_list: np.ndarray = None,
                                    reps: int = 50000) -> Tuple[Dict, Dict]:
        """
        Compute performance profiles.

        Args:
            score_dict: Dictionary mapping algorithm to score array
            tau_list: Array of threshold values
            reps: Number of bootstrap replications

        Returns:
            Tuple of (profiles, profile_cis)
        """
        print(f"\nComputing performance profiles...")

        if tau_list is None:
            # Auto-generate threshold list based on data range
            all_scores = np.concatenate([scores.flatten() for scores in score_dict.values()])
            min_score = np.percentile(all_scores, 1)
            max_score = np.percentile(all_scores, 99)
            tau_list = np.linspace(min_score, max_score, 100)
            print(f"  Auto-generated {len(tau_list)} thresholds from {min_score:.2f} to {max_score:.2f}")

        profiles, profile_cis = rly.create_performance_profile(
            score_dict,
            tau_list,
            use_score_distribution=True,
            method='percentile',
            task_bootstrap=False,
            reps=reps,
            confidence_interval_size=0.95
        )

        print("Performance profiles computed successfully!")
        return profiles, profile_cis, tau_list

    def compute_sample_efficiency(self, reps: int = 50000,
                                  env_filter: Optional[str] = None) -> Tuple[Dict, Dict, List]:
        """
        Compute sample efficiency curves from training checkpoints.

        Args:
            reps: Number of bootstrap replications
            env_filter: If provided, filter to single environment

        Returns:
            Tuple of (point_estimates_dict, interval_estimates_dict, frames_list)
        """
        print(f"\nComputing sample efficiency curves...")
        if env_filter:
            print(f"  Filtering to environment: {env_filter}")

        # Filter by environment if specified
        df_to_use = self.df
        if env_filter:
            df_to_use = self.df[self.df['env'] == env_filter].copy()
            if len(df_to_use) == 0:
                print(f"  WARNING: No data found for environment {env_filter}")
                return {}, {}, []

        # Check if sample efficiency data exists
        if 'sample_efficiency_parsed' not in df_to_use.columns:
            print("  No sample efficiency column found in data!")
            return {}, {}, []

        # Get all unique checkpoints
        all_checkpoints = set()
        for _, row in df_to_use.iterrows():
            if 'sample_efficiency_parsed' in row and isinstance(row['sample_efficiency_parsed'], dict):
                all_checkpoints.update(row['sample_efficiency_parsed'].keys())

        if not all_checkpoints:
            print("  No sample efficiency data found!")
            return {}, {}, []

        checkpoints = sorted([int(c) for c in all_checkpoints])
        print(f"  Found {len(checkpoints)} checkpoints: {checkpoints[:5]}{'...' if len(checkpoints) > 5 else ''}")

        algo_groups = df_to_use['algo_id'].unique()

        # For each checkpoint, create score dict and compute IQM
        point_estimates_dict = {algo: [] for algo in algo_groups}
        interval_estimates_dict = {algo: [] for algo in algo_groups}

        for checkpoint in checkpoints:
            print(f"  Processing checkpoint {checkpoint}...")

            # Build score dict for this checkpoint
            checkpoint_scores = {}
            for algo_id in algo_groups:
                algo_df = df_to_use[df_to_use['algo_id'] == algo_id]

                scores_list = []
                for _, row in algo_df.iterrows():
                    curve_data = row['sample_efficiency_parsed']
                    if isinstance(curve_data, dict) and str(checkpoint) in curve_data:
                        checkpoint_data = curve_data[str(checkpoint)]
                        if isinstance(checkpoint_data, dict) and 'reward_mean' in checkpoint_data:
                            scores_list.append(checkpoint_data['reward_mean'])

                if scores_list:
                    # Each run is a "task" for rliable
                    checkpoint_scores[algo_id] = np.array(scores_list).reshape(-1, 1)

            if not checkpoint_scores:
                continue

            # Compute IQM for this checkpoint
            aggregate_func = lambda x: np.array([rly_metrics.aggregate_iqm(x)])

            point_est, interval_est = rly.get_interval_estimates(
                checkpoint_scores,
                func=aggregate_func,
                method='percentile',
                task_bootstrap=False,
                reps=reps,
                confidence_interval_size=0.95
            )

            for algo_id in algo_groups:
                if algo_id in point_est:
                    point_estimates_dict[algo_id].append(point_est[algo_id][0])
                    interval_estimates_dict[algo_id].append(interval_est[algo_id][:, 0])
                else:
                    point_estimates_dict[algo_id].append(np.nan)
                    interval_estimates_dict[algo_id].append(np.array([np.nan, np.nan]))

        print("Sample efficiency curves computed successfully!")
        return point_estimates_dict, interval_estimates_dict, checkpoints

    def compute_probability_of_improvement(self, score_dict: Dict[str, np.ndarray],
                                          pairs: List[Tuple[str, str]] = None,
                                          reps: int = 50000) -> Tuple[Dict, Dict]:
        """
        Compute probability of improvement for algorithm pairs.

        Args:
            score_dict: Dictionary mapping algorithm to score array
            pairs: List of (algo_x, algo_y) pairs to compare
            reps: Number of bootstrap replications

        Returns:
            Tuple of (probability_estimates, probability_cis)
        """
        print(f"\nComputing probability of improvement...")

        algos = list(score_dict.keys())

        if pairs is None:
            # Auto-generate interesting pairs
            # Compare SPAR vs AllObs, Learned vs Random
            pairs = []
            for i, algo1 in enumerate(algos):
                for algo2 in algos[i+1:]:
                    pairs.append((algo1, algo2))
            print(f"  Auto-generated {len(pairs)} pairs")

        probability_estimates = {}
        probability_cis = {}

        for algo_x, algo_y in pairs:
            if algo_x not in score_dict or algo_y not in score_dict:
                print(f"  Skipping {algo_x} vs {algo_y}: not in score_dict")
                continue

            pair_name = f"{algo_x},{algo_y}"
            print(f"  Computing P({algo_x} > {algo_y})...")

            # Use StratifiedIndependentBootstrap for independent samples
            try:
                from rliable.library import StratifiedIndependentBootstrap
                from numpy.random import RandomState

                rs = RandomState(42)
                bs = StratifiedIndependentBootstrap(
                    score_dict[algo_x],
                    score_dict[algo_y],
                    random_state=rs
                )

                prob_func = lambda x, y: np.array([rly_metrics.probability_of_improvement(x, y)])

                point_est = prob_func(score_dict[algo_x], score_dict[algo_y])
                ci = bs.conf_int(prob_func, reps=reps, size=0.95, method='percentile')

                probability_estimates[pair_name] = point_est[0]
                probability_cis[pair_name] = ci[:, 0]
            except ModuleNotFoundError:
                print(f"  WARNING: rliable module not found, skipping probability of improvement")
                return {}, {}

        print("Probability of improvement computed successfully!")
        return probability_estimates, probability_cis

    def _rename_dict_keys(self, data_dict: Dict) -> Dict:
        """Rename dictionary keys using display names mapping."""
        return {self.display_names.get(k, k): v for k, v in data_dict.items()}

    def _get_algorithm_sort_order(self, algo_id: str) -> int:
        """
        Get sort order for algorithms to control display order in plots.
        Lower numbers appear first.

        Desired order:
        1. PPO-Lag runs (80%, 50%, 20%)
        2. PPO (Penalty)
        3. PPO (All Sensors)
        4. PPO (Random Mask)
        5. PPO (Baseline)
        """
        order_map = {
            'PPOLag-80pct': 7,
            'PPOLag-50pct': 6,
            'PPOLag-20pct': 5,
            'PPO-penalty': 4,
            'PPO-use-all-obs': 3,
            'PPO-random-mask': 2,
            'PPO': 1
        }
        return order_map.get(algo_id, 999)  # Unknown algorithms go to the end

    def _sort_dict_by_algorithm_order(self, data_dict: Dict) -> Dict:
        """Sort dictionary by custom algorithm ordering."""
        sorted_items = sorted(data_dict.items(), key=lambda x: self._get_algorithm_sort_order(x[0]))
        return dict(sorted_items)

    def plot_aggregate_metrics(self, point_estimates: Dict, interval_estimates: Dict,
                              metric_names: List[str] = None, metric_suffix: str = ''):
        """Plot aggregate metrics with confidence intervals."""
        if metric_names is None:
            metric_names = ['IQM']  # Only IQM

        print(f"\nPlotting aggregate metrics...")

        # Sort by custom algorithm order first
        point_estimates_sorted = self._sort_dict_by_algorithm_order(point_estimates)
        interval_estimates_sorted = self._sort_dict_by_algorithm_order(interval_estimates)

        # Rename algorithm IDs to display names
        point_est_display = self._rename_dict_keys(point_estimates_sorted)
        interval_est_display = self._rename_dict_keys(interval_estimates_sorted)

        fig, axes = rly_plot.plot_interval_estimates(
            point_est_display,
            interval_est_display,
            metric_names=metric_names,
            xlabel='Aggregate Score (IQM)',
            subfigure_width=4.0,
            row_height=0.5
        )

        filename = f'aggregate_metrics{metric_suffix}.png' if metric_suffix else 'aggregate_metrics.png'
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Saved to {output_path}")
        plt.close()

    def plot_performance_profiles(self, profiles: Dict, profile_cis: Dict,
                                 tau_list: np.ndarray, metric_suffix: str = ''):
        """Plot performance profiles."""
        print(f"\nPlotting performance profiles...")

        # Sort by custom algorithm order first
        profiles_sorted = self._sort_dict_by_algorithm_order(profiles)
        profile_cis_sorted = self._sort_dict_by_algorithm_order(profile_cis)

        # Rename algorithm IDs to display names
        profiles_display = self._rename_dict_keys(profiles_sorted)
        profile_cis_display = self._rename_dict_keys(profile_cis_sorted)

        fig, ax = plt.subplots(figsize=(10, 6))

        rly_plot.plot_performance_profiles(
            profiles_display,
            tau_list,
            performance_profile_cis=profile_cis_display,
            ax=ax,
            xlabel=r'Score Threshold $\tau$',
            ylabel=r'Fraction of runs with score $> \tau$',
            legend=True
        )

        filename = f'performance_profiles{metric_suffix}.png' if metric_suffix else 'performance_profiles.png'
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Saved to {output_path}")
        plt.close()

    def plot_sample_efficiency_curves(self, point_estimates: Dict,
                                     interval_estimates: Dict,
                                     frames: List[int], metric_suffix: str = ''):
        """Plot sample efficiency curves."""
        print(f"\nPlotting sample efficiency curves...")

        # Convert frames to millions for better readability
        frames_millions = [f / 1e6 for f in frames]

        # Prepare data for plotting
        plot_point_est = {}
        plot_interval_est = {}

        for algo, values in point_estimates.items():
            if len(values) > 0 and not all(np.isnan(values)):
                plot_point_est[algo] = np.array(values)
                intervals = np.array([iv for iv in interval_estimates[algo]])
                plot_interval_est[algo] = intervals.T

        if not plot_point_est:
            print("  No valid data to plot!")
            return

        # Rename algorithm IDs to display names
        plot_point_est_display = self._rename_dict_keys(plot_point_est)
        plot_interval_est_display = self._rename_dict_keys(plot_interval_est)

        fig, ax = plt.subplots(figsize=(10, 6))

        rly_plot.plot_sample_efficiency_curve(
            frames_millions,
            plot_point_est_display,
            plot_interval_est_display,
            xlabel='Training Steps (millions)',
            ylabel='Aggregate IQM Score',
            ax=ax,
            legend=True
        )

        filename = f'sample_efficiency{metric_suffix}.png' if metric_suffix else 'sample_efficiency.png'
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Saved to {output_path}")
        plt.close()

    def plot_probability_of_improvement(self, prob_estimates: Dict, prob_cis: Dict,
                                       metric_suffix: str = ''):
        """Plot probability of improvement using rliable's built-in function."""
        print(f"\nPlotting probability of improvement...")

        if not prob_estimates:
            print("  No probability estimates to plot!")
            return

        # Rename algorithm IDs to display names in the tuples
        prob_est_display = {}
        prob_cis_display = {}
        for (algo1, algo2), value in prob_estimates.items():
            new_key = (self.display_names.get(algo1, algo1), self.display_names.get(algo2, algo2))
            prob_est_display[new_key] = value
            if (algo1, algo2) in prob_cis:
                prob_cis_display[new_key] = prob_cis[(algo1, algo2)]

        # Determine figure size based on number of pairs
        n_pairs = len(prob_est_display)
        figsize = (8, max(4, n_pairs * 0.3))

        fig, ax = plt.subplots(figsize=figsize)

        rly_plot.plot_probability_of_improvement(
            prob_est_display,
            prob_cis_display,
            pair_separator=',',
            ax=ax,
            figsize=figsize,
            xlabel='P(Algorithm X > Algorithm Y)',
            alpha=0.75
        )

        plt.title('Probability of Improvement', fontsize=12, pad=10)
        plt.tight_layout()

        filename = f'probability_of_improvement{metric_suffix}.png' if metric_suffix else 'probability_of_improvement.png'
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Saved to {output_path}")
        plt.close()

    def save_results(self, results: Dict, filename: str):
        """Save results to JSON file."""
        output_path = self.output_dir / filename

        # Convert numpy arrays to lists for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_serializable(item) for item in obj]
            else:
                return obj

        serializable_results = convert_to_serializable(results)

        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)

        print(f"\nResults saved to {output_path}")

    def generate_summary_report(self, aggregate_results: Dict, metric_suffix: str = ''):
        """Generate a markdown summary report."""
        print(f"\nGenerating summary report...")

        report_lines = [
            "# Rliable Metrics Summary Report",
            "",
            "## Aggregate Performance Metrics (IQM)",
            "",
            "| Algorithm | IQM |",
            "|-----------|-----|"
        ]

        for algo, point_est in aggregate_results['point_estimates'].items():
            interval_est = aggregate_results['interval_estimates'][algo]

            # Only IQM (metric 0)
            pe = point_est[0]
            ci_low, ci_high = interval_est[0, 0], interval_est[1, 0]
            row = f"| {algo} | {pe:.3f} [{ci_low:.3f}, {ci_high:.3f}] |"

            report_lines.append(row)

        report_lines.extend([
            "",
            "## Notes",
            "- Values shown as: point_estimate [CI_lower, CI_upper]",
            "- IQM = Interquartile Mean (mean of middle 50% of scores)",
            "- Confidence intervals computed using stratified bootstrap (95% CI, 50k replications)",
            f"- Analysis based on {len(self.df)} experimental runs",
            ""
        ])

        filename = f'summary_report{metric_suffix}.md' if metric_suffix else 'summary_report.md'
        report_path = self.output_dir / filename
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))

        print(f"  Summary report saved to {report_path}")

    def _identify_best_performers(self, env_results: Dict[str, Dict], metric: str = 'reward') -> Dict[str, Dict[str, int]]:
        """
        Identify the best and second-best performing algorithms for each environment.

        Args:
            env_results: Dictionary mapping environment names to their aggregate_results
            metric: Which metric to use ('reward' or 'cost')

        Returns:
            Dictionary mapping env_name to {'best': algo_id, 'second_best': algo_id}
        """
        best_performers = {}

        for env_name, aggregate_results in env_results.items():
            # Get IQM values for all algorithms (metric 0 is IQM)
            algo_scores = {}
            for algo_id, point_est in aggregate_results['point_estimates'].items():
                algo_scores[algo_id] = point_est[0]  # IQM is at index 0

            # Sort by score (descending for reward, ascending for cost)
            sorted_algos = sorted(algo_scores.items(), key=lambda x: x[1], reverse=(metric == 'reward'))

            best_performers[env_name] = {
                'best': sorted_algos[0][0] if len(sorted_algos) > 0 else None,
                'second_best': sorted_algos[1][0] if len(sorted_algos) > 1 else None
            }

        return best_performers

    def _prettify_env_name(self, env_name: str) -> str:
        """
        Convert environment name to display format by removing 'budget-aware-' prefix
        and capitalizing appropriately.

        Args:
            env_name: Raw environment name (e.g., 'budget-aware-Door', 'budget-aware-highway-fast-v0')

        Returns:
            Pretty environment name (e.g., 'Door', 'Highway', 'Intersection')
        """
        # Remove 'budget-aware-' prefix
        display_name = env_name.replace('budget-aware-', '')

        # Handle specific environment names
        if 'highway' in display_name.lower():
            return 'Highway'
        elif 'intersection' in display_name.lower():
            return 'Intersection'
        elif 'roundabout' in display_name.lower():
            return 'Roundabout'
        else:
            # For Door, Lift, etc., just capitalize first letter
            return display_name.capitalize()

    def _extract_budget_mappings_for_env(self, env_name: str) -> Dict[str, float]:
        """
        Extract budget percentage to actual value mappings for a given environment.

        Args:
            env_name: Environment name

        Returns:
            Dictionary mapping budget percentages ('20%', '50%', '80%') to actual budget values
        """
        env_df = self.df[self.df['env'] == env_name]

        # Get PPOLag budgets
        ppolag_df = env_df[env_df['algo'] == 'PPOLag']
        budgets = pd.to_numeric(ppolag_df['budget'], errors='coerce').dropna().unique()

        budget_mapping = {}
        if len(budgets) >= 3:
            sorted_budgets = sorted(budgets)
            budget_mapping['20%'] = sorted_budgets[0]
            budget_mapping['50%'] = sorted_budgets[1]
            budget_mapping['80%'] = sorted_budgets[2]
        elif len(budgets) == 2:
            sorted_budgets = sorted(budgets)
            budget_mapping['50%'] = sorted_budgets[0]
            budget_mapping['80%'] = sorted_budgets[1]

        return budget_mapping

    def export_latex_table(self, aggregate_results: Dict, env_name: str = '', metric_suffix: str = ''):
        """
        Export aggregate metrics to CSV format for LaTeX tables.

        Creates two CSV files:
        1. Point estimates for each metric
        2. Confidence intervals in LaTeX-friendly format

        Args:
            aggregate_results: Dictionary containing point_estimates and interval_estimates
            env_name: Environment name (for the filename)
            metric_suffix: Metric suffix (reward/cost)
        """
        print(f"\nExporting LaTeX-friendly tables...")

        # Prepare data for export - only IQM
        algorithms = []
        iqm_vals = []
        iqm_ci_lower = []
        iqm_ci_upper = []

        for algo_id, point_est in aggregate_results['point_estimates'].items():
            interval_est = aggregate_results['interval_estimates'][algo_id]

            # Use display name instead of algorithm ID
            display_name = self.display_names.get(algo_id, algo_id)
            algorithms.append(display_name)

            # IQM (metric 0 - now the only metric)
            iqm_vals.append(f"{point_est[0]:.3f}")
            iqm_ci_lower.append(f"{interval_est[0, 0]:.3f}")
            iqm_ci_upper.append(f"{interval_est[1, 0]:.3f}")

        # Create DataFrame for CSV export
        df_export = pd.DataFrame({
            'Algorithm': algorithms,
            'IQM': iqm_vals,
            'IQM_CI_Lower': iqm_ci_lower,
            'IQM_CI_Upper': iqm_ci_upper
        })

        # Save to CSV
        if env_name:
            csv_filename = f'latex_table_{env_name}{metric_suffix}.csv'
        else:
            csv_filename = f'latex_table{metric_suffix}.csv'

        csv_path = self.output_dir / csv_filename
        df_export.to_csv(csv_path, index=False)
        print(f"  LaTeX table CSV saved to {csv_path}")

        # Also create a LaTeX-formatted version - only IQM
        latex_lines = [
            "% LaTeX table - copy this into your document",
            "\\begin{table}[ht]",
            "\\centering",
            f"\\caption{{Aggregate Performance Metrics (IQM){' - ' + env_name if env_name else ''}{' - ' + metric_suffix.replace('_', ' ').title() if metric_suffix else ''}}}",
            "\\begin{tabular}{l|c}",
            "\\hline",
            "Algorithm & IQM \\\\",
            "\\hline"
        ]

        for idx, algo in enumerate(algorithms):
            # Escape underscores for LaTeX
            algo_latex = algo.replace('%', '\\%')

            iqm_str = f"${iqm_vals[idx]} \\pm [{iqm_ci_lower[idx]}, {iqm_ci_upper[idx]}]$"

            latex_lines.append(f"{algo_latex} & {iqm_str} \\\\")

        # Add budget mapping information for cost tables
        budget_note = ""
        if metric_suffix == "_cost" and env_name:
            budget_mapping = self._extract_budget_mappings_for_env(env_name)
            if budget_mapping:
                budget_strs = [f"{pct}={val:.0f}" for pct, val in sorted(budget_mapping.items())]
                budget_note = f"% Budget percentages: {', '.join(budget_strs)}\n"

        latex_lines.extend([
            "\\hline",
            "\\end{tabular}",
            "\\label{tab:results" + (f"_{env_name}" if env_name else "") + metric_suffix + "}",
            "\\end{table}",
            "",
            "% Note: Values shown as: point_estimate ± [CI_lower, CI_upper]",
            "% IQM = Interquartile Mean (mean of middle 50% of scores)",
            "% Confidence intervals computed using stratified bootstrap (95% CI, 50k replications)"
        ])

        # Add budget note if available
        if budget_note:
            latex_lines.append(budget_note.rstrip())

        # Save LaTeX table
        if env_name:
            latex_filename = f'latex_table_{env_name}{metric_suffix}.tex'
        else:
            latex_filename = f'latex_table{metric_suffix}.tex'

        latex_path = self.output_dir / latex_filename
        with open(latex_path, 'w') as f:
            f.write('\n'.join(latex_lines))

        print(f"  LaTeX table .tex saved to {latex_path}")

    def export_combined_latex_table(self, env_results: Dict[str, Dict], metric_suffix: str = ''):
        """
        Export a single combined LaTeX table for all environments with separating lines.

        Args:
            env_results: Dictionary mapping environment names to their aggregate_results
            metric_suffix: Metric suffix (reward/cost)
        """
        print(f"\nExporting combined LaTeX table for all environments...")

        # Collect budget mappings for all environments (for both reward and cost tables)
        env_budget_mappings = {}
        for env_name in env_results.keys():
            budget_mapping = self._extract_budget_mappings_for_env(env_name)
            if budget_mapping:
                env_budget_mappings[env_name] = budget_mapping

        # Identify best and second-best performers for each environment
        metric_type = 'reward' if 'reward' in metric_suffix else 'cost'
        best_performers = self._identify_best_performers(env_results, metric=metric_type)

        # Collect all data organized by environment
        all_data = []

        for env_name, aggregate_results in env_results.items():
            for algo_id, point_est in aggregate_results['point_estimates'].items():
                interval_est = aggregate_results['interval_estimates'][algo_id]
                display_name = self.display_names.get(algo_id, algo_id)

                all_data.append({
                    'Environment': env_name,
                    'Algorithm': display_name,
                    'AlgoID': algo_id,  # Keep algo_id for best/second-best identification
                    'IQM': f"{point_est[0]:.3f}",
                    'IQM_CI_Lower': f"{interval_est[0, 0]:.3f}",
                    'IQM_CI_Upper': f"{interval_est[1, 0]:.3f}"
                })

        # Create DataFrame for CSV export
        df_combined = pd.DataFrame(all_data)
        csv_filename = f'latex_table_combined{metric_suffix}.csv'
        csv_path = self.output_dir / csv_filename
        df_combined.to_csv(csv_path, index=False)
        print(f"  Combined CSV saved to {csv_path}")

        # Create LaTeX-formatted version
        latex_lines = [
            "% LaTeX table - copy this into your document",
            "\\begin{table}[ht]",
            "\\centering",
            f"\\caption{{Aggregate Performance Metrics (IQM) - All Environments{' - ' + metric_suffix.replace('_', ' ').title() if metric_suffix else ''}}}",
            "\\begin{tabular}{l|c}",
            "\\hline",
            "Algorithm & IQM \\\\",
            "\\hline"
        ]

        # Group by environment
        current_env = None
        for idx, row in df_combined.iterrows():
            env_name = row['Environment']

            # Add environment separator if new environment
            if env_name != current_env:
                if current_env is not None:
                    latex_lines.append("\\hline")
                pretty_env_name = self._prettify_env_name(env_name)
                latex_lines.append(f"\\multicolumn{{2}}{{c}}{{\\textbf{{{pretty_env_name}}}}} \\\\")
                latex_lines.append("\\hline")
                current_env = env_name

            # Escape underscores for LaTeX
            algo_latex = row['Algorithm'].replace('%', '\\%')

            # Format IQM value based on performance ranking
            algo_id = row['AlgoID']
            iqm_value = row['IQM']
            ci_lower = row['IQM_CI_Lower']
            ci_upper = row['IQM_CI_Upper']

            if best_performers.get(env_name, {}).get('best') == algo_id:
                # Best performance: bold
                iqm_str = f"$\\mathbf{{{iqm_value}}} \\pm [{ci_lower}, {ci_upper}]$"
            elif best_performers.get(env_name, {}).get('second_best') == algo_id:
                # Second-best performance: underline
                iqm_str = f"$\\underline{{{iqm_value}}} \\pm [{ci_lower}, {ci_upper}]$"
            else:
                # Regular formatting
                iqm_str = f"${iqm_value} \\pm [{ci_lower}, {ci_upper}]$"

            latex_lines.append(f"{algo_latex} & {iqm_str} \\\\")

        latex_lines.extend([
            "\\hline",
            "\\end{tabular}",
            "\\label{tab:results_combined" + metric_suffix + "}",
            "\\end{table}",
            "",
            "% Note: Values shown as: point_estimate ± [CI_lower, CI_upper]",
            "% IQM = Interquartile Mean (mean of middle 50% of scores)",
            "% Confidence intervals computed using stratified bootstrap (95% CI, 50k replications)"
        ])

        # Add budget mapping information for cost tables
        if env_budget_mappings:
            latex_lines.append("%")
            latex_lines.append("% Budget percentages per environment:")
            for env_name, budget_mapping in sorted(env_budget_mappings.items()):
                pretty_env_name = self._prettify_env_name(env_name)
                budget_strs = [f"{pct}={val:.0f}" for pct, val in sorted(budget_mapping.items())]
                latex_lines.append(f"%   {pretty_env_name}: {', '.join(budget_strs)}")

        # Save LaTeX table
        latex_filename = f'latex_table_combined{metric_suffix}.tex'
        latex_path = self.output_dir / latex_filename
        with open(latex_path, 'w') as f:
            f.write('\n'.join(latex_lines))

        print(f"  Combined LaTeX table saved to {latex_path}")

    def export_combined_reward_cost_table(self, reward_results: Dict, cost_results: Dict, env_name: str = ''):
        """
        Export a single LaTeX table with both reward and cost IQMs.

        Args:
            reward_results: Dictionary containing reward aggregate_results
            cost_results: Dictionary containing cost aggregate_results
            env_name: Optional environment name for file naming
        """
        print(f"\nExporting combined reward+cost LaTeX table for {env_name or 'aggregate'}...")

        # Collect algorithms and their metrics
        algorithms = []
        reward_iqm_vals = []
        reward_iqm_ci_lower = []
        reward_iqm_ci_upper = []
        cost_iqm_vals = []
        cost_iqm_ci_lower = []
        cost_iqm_ci_upper = []

        # Get all algorithms from reward results (assuming same algorithms in both)
        for algo_id in reward_results['point_estimates'].keys():
            display_name = self.display_names.get(algo_id, algo_id)
            algorithms.append(display_name)

            # Reward IQM
            reward_point_est = reward_results['point_estimates'][algo_id]
            reward_interval_est = reward_results['interval_estimates'][algo_id]
            reward_iqm_vals.append(f"{reward_point_est[0]:.3f}")
            reward_iqm_ci_lower.append(f"{reward_interval_est[0, 0]:.3f}")
            reward_iqm_ci_upper.append(f"{reward_interval_est[1, 0]:.3f}")

            # Cost IQM
            cost_point_est = cost_results['point_estimates'][algo_id]
            cost_interval_est = cost_results['interval_estimates'][algo_id]
            cost_iqm_vals.append(f"{cost_point_est[0]:.3f}")
            cost_iqm_ci_lower.append(f"{cost_interval_est[0, 0]:.3f}")
            cost_iqm_ci_upper.append(f"{cost_interval_est[1, 0]:.3f}")

        # Create DataFrame for CSV export
        df_export = pd.DataFrame({
            'Algorithm': algorithms,
            'Reward_IQM': reward_iqm_vals,
            'Reward_CI_Lower': reward_iqm_ci_lower,
            'Reward_CI_Upper': reward_iqm_ci_upper,
            'Cost_IQM': cost_iqm_vals,
            'Cost_CI_Lower': cost_iqm_ci_lower,
            'Cost_CI_Upper': cost_iqm_ci_upper
        })

        # Save to CSV
        if env_name:
            csv_filename = f'latex_table_{env_name}_reward_cost.csv'
        else:
            csv_filename = 'latex_table_reward_cost.csv'

        csv_path = self.output_dir / csv_filename
        df_export.to_csv(csv_path, index=False)
        print(f"  Combined reward+cost CSV saved to {csv_path}")

        # Create LaTeX-formatted version
        latex_lines = [
            "% LaTeX table - copy this into your document",
            "\\begin{table}[ht]",
            "\\centering",
            f"\\caption{{Aggregate Performance Metrics (IQM){' - ' + env_name if env_name else ''}}}",
            "\\begin{tabular}{l|c|c}",
            "\\hline",
            "Algorithm & Reward IQM & Cost IQM \\\\",
            "\\hline"
        ]

        for idx, algo in enumerate(algorithms):
            # Escape underscores for LaTeX
            algo_latex = algo.replace('%', '\\%')

            reward_str = f"${reward_iqm_vals[idx]} \\pm [{reward_iqm_ci_lower[idx]}, {reward_iqm_ci_upper[idx]}]$"
            cost_str = f"${cost_iqm_vals[idx]} \\pm [{cost_iqm_ci_lower[idx]}, {cost_iqm_ci_upper[idx]}]$"

            latex_lines.append(f"{algo_latex} & {reward_str} & {cost_str} \\\\")

        # Add budget mapping information for cost column
        budget_note = ""
        if env_name:
            budget_mapping = self._extract_budget_mappings_for_env(env_name)
            if budget_mapping:
                budget_strs = [f"{pct}={val:.0f}" for pct, val in sorted(budget_mapping.items())]
                budget_note = f"% Budget percentages: {', '.join(budget_strs)}\n"

        latex_lines.extend([
            "\\hline",
            "\\end{tabular}",
            "\\label{tab:results" + (f"_{env_name}" if env_name else "") + "_reward_cost}",
            "\\end{table}",
            "",
            "% Note: Values shown as: point_estimate ± [CI_lower, CI_upper]",
            "% IQM = Interquartile Mean (mean of middle 50% of scores)",
            "% Confidence intervals computed using stratified bootstrap (95% CI, 50k replications)"
        ])

        # Add budget note if available
        if budget_note:
            latex_lines.append(budget_note.rstrip())

        # Save LaTeX table
        if env_name:
            latex_filename = f'latex_table_{env_name}_reward_cost.tex'
        else:
            latex_filename = 'latex_table_reward_cost.tex'

        latex_path = self.output_dir / latex_filename
        with open(latex_path, 'w') as f:
            f.write('\n'.join(latex_lines))

        print(f"  Combined reward+cost LaTeX table saved to {latex_path}")

    def export_multi_env_reward_cost_table(self, env_reward_results: Dict[str, Dict], env_cost_results: Dict[str, Dict]):
        """
        Export a single combined LaTeX table for all environments with both reward and cost IQMs.

        Args:
            env_reward_results: Dictionary mapping environment names to their reward aggregate_results
            env_cost_results: Dictionary mapping environment names to their cost aggregate_results
        """
        print(f"\nExporting combined multi-environment reward+cost LaTeX table...")

        # Collect budget mappings for all environments
        env_budget_mappings = {}
        for env_name in env_reward_results.keys():
            budget_mapping = self._extract_budget_mappings_for_env(env_name)
            if budget_mapping:
                env_budget_mappings[env_name] = budget_mapping

        # Identify best and second-best performers for reward only
        reward_best_performers = self._identify_best_performers(env_reward_results, metric='reward')

        # Collect all data organized by environment
        all_data = []

        for env_name in env_reward_results.keys():
            reward_results = env_reward_results[env_name]
            cost_results = env_cost_results[env_name]

            for algo_id in reward_results['point_estimates'].keys():
                display_name = self.display_names.get(algo_id, algo_id)

                # Reward IQM
                reward_point_est = reward_results['point_estimates'][algo_id]
                reward_interval_est = reward_results['interval_estimates'][algo_id]

                # Cost IQM
                cost_point_est = cost_results['point_estimates'][algo_id]
                cost_interval_est = cost_results['interval_estimates'][algo_id]

                all_data.append({
                    'Environment': env_name,
                    'Algorithm': display_name,
                    'AlgoID': algo_id,
                    'Reward_IQM': f"{reward_point_est[0]:.3f}",
                    'Reward_CI_Lower': f"{reward_interval_est[0, 0]:.3f}",
                    'Reward_CI_Upper': f"{reward_interval_est[1, 0]:.3f}",
                    'Cost_IQM': f"{cost_point_est[0]:.3f}",
                    'Cost_CI_Lower': f"{cost_interval_est[0, 0]:.3f}",
                    'Cost_CI_Upper': f"{cost_interval_est[1, 0]:.3f}"
                })

        # Create DataFrame for CSV export
        df_combined = pd.DataFrame(all_data)
        csv_filename = 'latex_table_combined_reward_cost.csv'
        csv_path = self.output_dir / csv_filename
        df_combined.to_csv(csv_path, index=False)
        print(f"  Combined CSV saved to {csv_path}")

        # Create LaTeX-formatted version
        latex_lines = [
            "% LaTeX table - copy this into your document",
            "\\begin{table}[ht]",
            "\\centering",
            "\\caption{Aggregate Performance Metrics (IQM) - All Environments}",
            "\\begin{tabular}{l|c|c}",
            "\\hline",
            "Algorithm & Reward IQM & Cost IQM \\\\",
            "\\hline"
        ]

        # Group by environment
        current_env = None
        for idx, row in df_combined.iterrows():
            env_name = row['Environment']

            # Add environment separator if new environment
            if env_name != current_env:
                if current_env is not None:
                    latex_lines.append("\\hline")
                pretty_env_name = self._prettify_env_name(env_name)
                latex_lines.append(f"\\multicolumn{{3}}{{c}}{{\\textbf{{{pretty_env_name}}}}} \\\\")
                latex_lines.append("\\hline")
                current_env = env_name

            # Escape underscores for LaTeX
            algo_latex = row['Algorithm'].replace('%', '\\%')
            algo_id = row['AlgoID']

            # Format Reward IQM based on performance ranking
            reward_iqm = row['Reward_IQM']
            reward_ci_lower = row['Reward_CI_Lower']
            reward_ci_upper = row['Reward_CI_Upper']

            if reward_best_performers.get(env_name, {}).get('best') == algo_id:
                reward_str = f"$\\mathbf{{{reward_iqm}}} \\pm [{reward_ci_lower}, {reward_ci_upper}]$"
            elif reward_best_performers.get(env_name, {}).get('second_best') == algo_id:
                reward_str = f"$\\underline{{{reward_iqm}}} \\pm [{reward_ci_lower}, {reward_ci_upper}]$"
            else:
                reward_str = f"${reward_iqm} \\pm [{reward_ci_lower}, {reward_ci_upper}]$"

            # Cost IQM - no performance formatting, just regular display
            cost_iqm = row['Cost_IQM']
            cost_ci_lower = row['Cost_CI_Lower']
            cost_ci_upper = row['Cost_CI_Upper']
            cost_str = f"${cost_iqm} \\pm [{cost_ci_lower}, {cost_ci_upper}]$"

            latex_lines.append(f"{algo_latex} & {reward_str} & {cost_str} \\\\")

        latex_lines.extend([
            "\\hline",
            "\\end{tabular}",
            "\\label{tab:results_combined_reward_cost}",
            "\\end{table}",
            "",
            "% Note: Values shown as: point_estimate ± [CI_lower, CI_upper]",
            "% IQM = Interquartile Mean (mean of middle 50% of scores)",
            "% Confidence intervals computed using stratified bootstrap (95% CI, 50k replications)"
        ])

        # Add budget mapping information
        if env_budget_mappings:
            latex_lines.append("%")
            latex_lines.append("% Budget percentages per environment:")
            for env_name, budget_mapping in sorted(env_budget_mappings.items()):
                pretty_env_name = self._prettify_env_name(env_name)
                budget_strs = [f"{pct}={val:.0f}" for pct, val in sorted(budget_mapping.items())]
                latex_lines.append(f"%   {pretty_env_name}: {', '.join(budget_strs)}")

        # Save LaTeX table
        latex_filename = 'latex_table_combined_reward_cost.tex'
        latex_path = self.output_dir / latex_filename
        with open(latex_path, 'w') as f:
            f.write('\n'.join(latex_lines))

        print(f"  Combined multi-environment reward+cost LaTeX table saved to {latex_path}")

    def generate_combined_summary_report(self, env_results: Dict[str, Dict], metric_suffix: str = ''):
        """
        Generate a markdown summary report combining all environments.

        Args:
            env_results: Dictionary mapping environment names to their aggregate_results
            metric_suffix: Metric suffix (reward/cost)
        """
        print(f"\nGenerating combined summary report...")

        report_lines = [
            "# Rliable Metrics Summary Report - All Environments",
            "",
            f"## Aggregate Performance Metrics (IQM){' - ' + metric_suffix.replace('_', ' ').title() if metric_suffix else ''}",
            ""
        ]

        # Process each environment
        for env_idx, (env_name, aggregate_results) in enumerate(env_results.items()):
            if env_idx > 0:
                report_lines.append("")

            report_lines.extend([
                f"### {env_name}",
                "",
                "| Algorithm | IQM |",
                "|-----------|-----|"
            ])

            for algo, point_est in aggregate_results['point_estimates'].items():
                interval_est = aggregate_results['interval_estimates'][algo]

                # Only IQM (metric 0)
                pe = point_est[0]
                ci_low, ci_high = interval_est[0, 0], interval_est[1, 0]
                display_name = self.display_names.get(algo, algo)
                row = f"| {display_name} | {pe:.3f} [{ci_low:.3f}, {ci_high:.3f}] |"

                report_lines.append(row)

        report_lines.extend([
            "",
            "## Notes",
            "- Values shown as: point_estimate [CI_lower, CI_upper]",
            "- IQM = Interquartile Mean (mean of middle 50% of scores)",
            "- Confidence intervals computed using stratified bootstrap (95% CI, 50k replications)",
            f"- Analysis based on {len(self.df)} experimental runs",
            ""
        ])

        filename = f'summary_report_combined{metric_suffix}.md'
        report_path = self.output_dir / filename
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))

        print(f"  Combined summary report saved to {report_path}")

    def run_full_analysis(self, metric: str = 'reward_mean'):
        """Run complete rliable analysis pipeline."""
        print("=" * 80)
        print(f"RLIABLE METRICS ANALYSIS - {metric.upper()}")
        print("=" * 80)

        # Create metric suffix for file naming
        metric_suffix = f"_{metric.replace('_mean', '').replace('_normalized', '_norm')}"

        # 1. Prepare data
        score_dict = self.prepare_score_dict(metric=metric)

        # 2. Compute aggregate metrics
        point_est, interval_est = self.compute_aggregate_metrics(score_dict, reps=50_000)
        self.plot_aggregate_metrics(point_est, interval_est, metric_suffix=metric_suffix)

        aggregate_results = {
            'point_estimates': point_est,
            'interval_estimates': interval_est
        }
        self.save_results(aggregate_results, f'aggregate_metrics{metric_suffix}.json')
        self.generate_summary_report(aggregate_results, metric_suffix=metric_suffix)

        # Export LaTeX tables
        self.export_latex_table(aggregate_results, metric_suffix=metric_suffix)

        # 3. Compute performance profiles
        profiles, profile_cis, tau_list = self.compute_performance_profiles(score_dict)
        self.plot_performance_profiles(profiles, profile_cis, tau_list, metric_suffix=metric_suffix)

        profile_results = {
            'profiles': profiles,
            'profile_cis': profile_cis,
            'tau_list': tau_list
        }
        self.save_results(profile_results, f'performance_profiles{metric_suffix}.json')

        # 4. Compute sample efficiency
        se_point_est, se_interval_est, frames = self.compute_sample_efficiency()
        if se_point_est:
            self.plot_sample_efficiency_curves(se_point_est, se_interval_est, frames, metric_suffix=metric_suffix)

            se_results = {
                'point_estimates': se_point_est,
                'interval_estimates': se_interval_est,
                'frames': frames
            }
            self.save_results(se_results, f'sample_efficiency{metric_suffix}.json')

        # 5. Compute probability of improvement
        prob_est, prob_cis = self.compute_probability_of_improvement(score_dict, reps=50_000)
        self.plot_probability_of_improvement(prob_est, prob_cis, metric_suffix=metric_suffix)

        prob_results = {
            'probability_estimates': prob_est,
            'probability_cis': prob_cis
        }
        self.save_results(prob_results, f'probability_of_improvement{metric_suffix}.json')

        print("\n" + "=" * 80)
        print(f"ANALYSIS COMPLETE for {metric}!")
        print(f"Results saved to: {self.output_dir}")
        print("=" * 80)

    def run_per_environment_analysis(self, env_name: str, metric: str = 'reward_mean'):
        """
        Run complete rliable analysis pipeline for a single environment.

        Args:
            env_name: Name of the environment to analyze
            metric: Metric to analyze (reward_mean, cost_mean, etc.)
        """
        print("=" * 80)
        print(f"RLIABLE METRICS ANALYSIS - {env_name} - {metric.upper()}")
        print("=" * 80)

        # Create environment-specific output directory
        env_output_dir = self.output_dir / env_name
        env_output_dir.mkdir(exist_ok=True, parents=True)

        # Temporarily change output directory
        original_output_dir = self.output_dir
        self.output_dir = env_output_dir

        try:
            metric_suffix = f"_{metric.replace('_mean', '').replace('_normalized', '_norm')}"

            # 1. Prepare data (filtered to single environment)
            score_dict = self.prepare_score_dict(metric=metric, env_filter=env_name)

            if not score_dict:
                print(f"  WARNING: No data to analyze for {env_name}")
                return

            # 2. Compute aggregate metrics
            point_est, interval_est = self.compute_aggregate_metrics(score_dict, reps=50_000)
            self.plot_aggregate_metrics(point_est, interval_est, metric_suffix=metric_suffix)
            aggregate_results = {'point_estimates': point_est, 'interval_estimates': interval_est}
            self.generate_summary_report(aggregate_results, metric_suffix=metric_suffix)

            # Export LaTeX tables
            self.export_latex_table(aggregate_results, env_name=env_name, metric_suffix=metric_suffix)

            # # 3. Compute performance profiles
            # profiles, profile_cis, tau_list = self.compute_performance_profiles(score_dict)
            # self.plot_performance_profiles(profiles, profile_cis, tau_list, metric_suffix=metric_suffix)
            #
            # # 4. Compute sample efficiency
            # se_point_est, se_interval_est, frames = self.compute_sample_efficiency(env_filter=env_name)
            # if se_point_est:
            #     self.plot_sample_efficiency_curves(se_point_est, se_interval_est, frames, metric_suffix=metric_suffix)

            # # 5. Compute probability of improvement
            # prob_est, prob_cis = self.compute_probability_of_improvement(score_dict, reps=50_000)
            # self.plot_probability_of_improvement(prob_est, prob_cis, metric_suffix=metric_suffix)

            print(f"\nANALYSIS COMPLETE for {env_name} - {metric}!")
            print(f"Results saved to: {self.output_dir}")

        finally:
            # Restore original output directory
            self.output_dir = original_output_dir


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Compute rliable metrics for SPAR experiments')
    parser.add_argument('--csv', type=str,
                       default='../data/processed/manifest_for_rliable_deduplicated.csv',
                       help='Path to manifest CSV file')
    parser.add_argument('--output-dir', type=str,
                       default='../results',
                       help='Output directory for results')
    parser.add_argument('--metric', type=str, default=None,
                       choices=['reward_mean', 'cost_mean', 'reward_normalized', 'cost_normalized'],
                       help='Single metric to analyze')
    parser.add_argument('--metrics', type=str, nargs='+',
                       choices=['reward_mean', 'cost_mean', 'reward_normalized', 'cost_normalized'],
                       help='Multiple metrics to analyze')
    parser.add_argument('--per-env', action='store_true',
                       help='Generate separate plots for each environment')
    parser.add_argument('--envs', type=str, nargs='+',
                       help='Specific environments to analyze (default: all)')

    args = parser.parse_args()

    # Determine which metrics to compute
    if args.metric:
        metrics_to_compute = [args.metric]
    elif args.metrics:
        metrics_to_compute = args.metrics
    else:
        metrics_to_compute = ['reward_mean', 'cost_mean']

    print("\n" + "=" * 80)
    print("RLIABLE ANALYSIS - STARTING")
    print("=" * 80)
    print(f"Metrics to compute: {', '.join(metrics_to_compute)}")
    print(f"CSV file: {args.csv}")
    print(f"Output directory: {args.output_dir}")
    if args.per_env:
        print(f"Per-environment analysis: enabled")
        if args.envs:
            print(f"Environments filter: {', '.join(args.envs)}")
    print("=" * 80 + "\n")

    # Initialize analyzer
    analyzer = RliableAnalyzer(args.csv, args.output_dir)

    if args.per_env:
        # Per-environment analysis (NEW BEHAVIOR)
        print("\nMODE: PER-ENVIRONMENT ANALYSIS\n")

        # Get environments to analyze
        all_envs = analyzer.get_unique_environments()
        envs_to_analyze = args.envs if args.envs else all_envs

        # Validate requested environments
        envs_to_analyze = [e for e in envs_to_analyze if e in all_envs]

        if not envs_to_analyze:
            print("ERROR: No valid environments to analyze!")
            return

        # Run analysis for each environment and metric
        total_tasks = len(envs_to_analyze) * len(metrics_to_compute)
        current_task = 0

        # Store results for combined table generation
        env_results_by_metric = {metric: {} for metric in metrics_to_compute}

        for env_name in envs_to_analyze:
            for metric in metrics_to_compute:
                current_task += 1
                print(f"\n[{current_task}/{total_tasks}] Processing {env_name} - {metric}")

                # Get aggregate results for this environment
                metric_suffix = f"_{metric.replace('_mean', '').replace('_normalized', '_norm')}"
                score_dict = analyzer.prepare_score_dict(metric=metric, env_filter=env_name)

                if score_dict:
                    point_est, interval_est = analyzer.compute_aggregate_metrics(score_dict, reps=50_000)
                    aggregate_results = {'point_estimates': point_est, 'interval_estimates': interval_est}
                    env_results_by_metric[metric][env_name] = aggregate_results

                # Run the full per-environment analysis
                analyzer.run_per_environment_analysis(env_name, metric=metric)

        # Generate combined tables and reports for each metric
        print("\n" + "=" * 80)
        print("GENERATING COMBINED TABLES FOR ALL ENVIRONMENTS")
        print("=" * 80)

        for metric in metrics_to_compute:
            if env_results_by_metric[metric]:
                metric_suffix = f"_{metric.replace('_mean', '').replace('_normalized', '_norm')}"
                analyzer.export_combined_latex_table(env_results_by_metric[metric], metric_suffix=metric_suffix)
                analyzer.generate_combined_summary_report(env_results_by_metric[metric], metric_suffix=metric_suffix)

        # Generate combined reward+cost tables if both metrics were computed
        if 'reward_mean' in env_results_by_metric and 'cost_mean' in env_results_by_metric:
            if env_results_by_metric['reward_mean'] and env_results_by_metric['cost_mean']:
                print("\n" + "=" * 80)
                print("GENERATING COMBINED REWARD+COST TABLES")
                print("=" * 80)

                # Generate per-environment reward+cost tables
                for env_name in envs_to_analyze:
                    if env_name in env_results_by_metric['reward_mean'] and env_name in env_results_by_metric['cost_mean']:
                        print(f"\nGenerating reward+cost table for {env_name}...")
                        analyzer.export_combined_reward_cost_table(
                            env_results_by_metric['reward_mean'][env_name],
                            env_results_by_metric['cost_mean'][env_name],
                            env_name=env_name
                        )

                # Generate multi-environment reward+cost table
                print(f"\nGenerating combined multi-environment reward+cost table...")
                analyzer.export_multi_env_reward_cost_table(
                    env_results_by_metric['reward_mean'],
                    env_results_by_metric['cost_mean']
                )

    else:
        # Combined analysis (LEGACY BEHAVIOR)
        print("\nMODE: COMBINED ANALYSIS (LEGACY)\n")
        for metric in metrics_to_compute:
            analyzer.run_full_analysis(metric=metric)
            print("\n")

    print("\n" + "=" * 80)
    print("ALL ANALYSES COMPLETE!")
    print(f"Results saved to: {args.output_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()
