"""
Compute rliable metrics for BAFS Robosuite experiments.

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
from typing import Dict, List, Tuple
import warnings

warnings.filterwarnings('ignore')

# Import rliable using relative imports
from .. import library as rly
from .. import metrics as rly_metrics
from .. import plot_utils as rly_plot


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

        # Define the columns we actually need (CSV has many trailing commas)
        usecols = [
            'timestamp', 'algo', 'env', 'seed', 'budget', 'obs_mode', 'actor_type',
            'use_cost', 'use_all_obs', 'random_obs_selection',
            'reward_mean', 'reward_std', 'cost_mean', 'cost_std',
            'episode_rewards', 'episode_costs', 'sample_efficiency_curve',
            'reward_normalized', 'cost_normalized', 'obs_modality_normalize'
        ]

        self.df = pd.read_csv(csv_path, usecols=usecols, low_memory=False)

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

    def get_algorithm_groups(self) -> List[str]:
        """
        Get unique algorithm configurations for grouping.

        Returns:
            List of algorithm group identifiers
        """
        # First, identify budget mappings for algorithms with budgets
        # For each algorithm, find unique budget values and map smaller->50%, larger->80%
        budget_mappings = {}

        for algo_name in self.df['algo'].unique():
            algo_df = self.df[self.df['algo'] == algo_name]
            budgets = algo_df['budget'].dropna()

            # Filter out 'None' string values
            budgets = budgets[budgets != 'None']

            if len(budgets) > 0:
                try:
                    # Convert to numeric and get unique values
                    numeric_budgets = pd.to_numeric(budgets, errors='coerce').dropna().unique()

                    if len(numeric_budgets) >= 2:
                        # Sort budgets: smaller = 50%, larger = 80%
                        sorted_budgets = sorted(numeric_budgets)
                        budget_mappings[algo_name] = {
                            sorted_budgets[0]: '50pct',
                            sorted_budgets[-1]: '80pct'  # Use last in case there are more than 2
                        }
                        print(f"  Budget mapping for {algo_name}: {sorted_budgets[0]:.2f}->50%, {sorted_budgets[-1]:.2f}->80%")
                except:
                    pass

        # Create algorithm identifier based on key configuration parameters
        def create_algo_id(row):
            """Create algorithm identifier with budget information."""
            algo_name = row['algo']
            obs_mode = 'AllObs' if row['use_all_obs'] else 'BAFS'
            mask_type = 'Random' if row.get('random_obs_selection', False) else 'Learned'

            # Add budget percentage if this algorithm has budget mappings
            budget_str = ''
            if algo_name in budget_mappings and pd.notna(row.get('budget', None)) and row.get('budget') != 'None':
                try:
                    budget_val = float(row['budget'])
                    if budget_val in budget_mappings[algo_name]:
                        budget_str = f"_Budget{budget_mappings[algo_name][budget_val]}"
                except (ValueError, TypeError):
                    pass

            return f"{algo_name}_{obs_mode}_{mask_type}{budget_str}"

        self.df['algo_id'] = self.df.apply(create_algo_id, axis=1)

        algo_groups = self.df['algo_id'].unique().tolist()
        print(f"\nFound {len(algo_groups)} algorithm configurations:")
        for ag in algo_groups:
            count = len(self.df[self.df['algo_id'] == ag])
            print(f"  {ag}: {count} runs")

        return algo_groups

    def prepare_score_dict(self, metric: str = 'reward_mean',
                          task_column: str = 'env') -> Dict[str, np.ndarray]:
        """
        Prepare score dictionary in rliable format: {algo: (num_runs × num_tasks)}.

        Args:
            metric: Column name for the metric to analyze (e.g., 'reward_mean', 'cost_mean')
            task_column: Column name for task identification (e.g., 'env')

        Returns:
            Dictionary mapping algorithm to score array
        """
        print(f"\nPreparing score dictionary for metric: {metric}")

        algo_groups = self.get_algorithm_groups()
        score_dict = {}

        for algo_id in algo_groups:
            algo_df = self.df[self.df['algo_id'] == algo_id]

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

        # Define aggregate functions
        aggregate_func = lambda x: np.array([
            rly_metrics.aggregate_mean(x),
            rly_metrics.aggregate_median(x),
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
                                    reps: int = 2000) -> Tuple[Dict, Dict]:
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

    def compute_sample_efficiency(self, reps: int = 50000) -> Tuple[Dict, Dict, List]:
        """
        Compute sample efficiency curves from training checkpoints.

        Args:
            reps: Number of bootstrap replications

        Returns:
            Tuple of (point_estimates_dict, interval_estimates_dict, frames_list)
        """
        print(f"\nComputing sample efficiency curves...")

        # Get all unique checkpoints
        all_checkpoints = set()
        for _, row in self.df.iterrows():
            if isinstance(row['sample_efficiency_parsed'], dict):
                all_checkpoints.update(row['sample_efficiency_parsed'].keys())

        if not all_checkpoints:
            print("  No sample efficiency data found!")
            return {}, {}, []

        checkpoints = sorted([int(c) for c in all_checkpoints])
        print(f"  Found {len(checkpoints)} checkpoints: {checkpoints[:5]}{'...' if len(checkpoints) > 5 else ''}")

        algo_groups = self.df['algo_id'].unique()

        # For each checkpoint, create score dict and compute IQM
        point_estimates_dict = {algo: [] for algo in algo_groups}
        interval_estimates_dict = {algo: [] for algo in algo_groups}

        for checkpoint in checkpoints:
            print(f"  Processing checkpoint {checkpoint}...")

            # Build score dict for this checkpoint
            checkpoint_scores = {}
            for algo_id in algo_groups:
                algo_df = self.df[self.df['algo_id'] == algo_id]

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
            # Compare BAFS vs AllObs, Learned vs Random
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

        print("Probability of improvement computed successfully!")
        return probability_estimates, probability_cis

    def plot_aggregate_metrics(self, point_estimates: Dict, interval_estimates: Dict,
                              metric_names: List[str] = None, metric_suffix: str = ''):
        """Plot aggregate metrics with confidence intervals."""
        if metric_names is None:
            metric_names = ['Mean', 'Median', 'IQM']

        print(f"\nPlotting aggregate metrics...")

        fig, axes = rly_plot.plot_interval_estimates(
            point_estimates,
            interval_estimates,
            metric_names=metric_names,
            xlabel='Aggregate Score',
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

        fig, ax = plt.subplots(figsize=(10, 6))

        rly_plot.plot_performance_profiles(
            profiles,
            tau_list,
            performance_profile_cis=profile_cis,
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

        fig, ax = plt.subplots(figsize=(10, 6))

        rly_plot.plot_sample_efficiency_curve(
            frames_millions,
            plot_point_est,
            plot_interval_est,
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

        # Determine figure size based on number of pairs
        n_pairs = len(prob_estimates)
        figsize = (8, max(4, n_pairs * 0.3))

        fig, ax = plt.subplots(figsize=figsize)

        rly_plot.plot_probability_of_improvement(
            prob_estimates,
            prob_cis,
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

        metric_names = ['Mean', 'Median', 'IQM']

        report_lines = [
            "# Rliable Metrics Summary Report",
            "",
            "## Aggregate Performance Metrics",
            "",
            "| Algorithm | Mean | Median | IQM |",
            "|-----------|------|--------|-----|"
        ]

        for algo, point_est in aggregate_results['point_estimates'].items():
            interval_est = aggregate_results['interval_estimates'][algo]

            row = f"| {algo} |"
            for i in range(len(metric_names)):
                pe = point_est[i]
                ci_low, ci_high = interval_est[0, i], interval_est[1, i]
                row += f" {pe:.3f} [{ci_low:.3f}, {ci_high:.3f}] |"

            report_lines.append(row)

        report_lines.extend([
            "",
            "## Notes",
            "- Values shown as: point_estimate [CI_lower, CI_upper]",
            "- Confidence intervals computed using stratified bootstrap (95% CI)",
            f"- Analysis based on {len(self.df)} experimental runs",
            ""
        ])

        filename = f'summary_report{metric_suffix}.md' if metric_suffix else 'summary_report.md'
        report_path = self.output_dir / filename
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))

        print(f"  Summary report saved to {report_path}")

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
        point_est, interval_est = self.compute_aggregate_metrics(score_dict, reps=2_000)
        self.plot_aggregate_metrics(point_est, interval_est, metric_suffix=metric_suffix)

        aggregate_results = {
            'point_estimates': point_est,
            'interval_estimates': interval_est
        }
        self.save_results(aggregate_results, f'aggregate_metrics{metric_suffix}.json')
        self.generate_summary_report(aggregate_results, metric_suffix=metric_suffix)

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
        prob_est, prob_cis = self.compute_probability_of_improvement(score_dict, reps=500)
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


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Compute rliable metrics for BAFS experiments')
    parser.add_argument('--csv', type=str,
                       default='../data/processed/manifest_for_rliable_deduplicated.csv',
                       help='Path to manifest CSV file')
    parser.add_argument('--output-dir', type=str,
                       default='../results',
                       help='Output directory for results')
    parser.add_argument('--metric', type=str,
                       default=None,
                       choices=['reward_mean', 'cost_mean', 'reward_normalized', 'cost_normalized'],
                       help='Single metric to analyze (for backward compatibility). '
                            'If not specified, both reward_mean and cost_mean will be analyzed.')
    parser.add_argument('--metrics', type=str, nargs='+',
                       choices=['reward_mean', 'cost_mean', 'reward_normalized', 'cost_normalized'],
                       help='Multiple metrics to analyze (e.g., --metrics reward_mean cost_mean)')

    args = parser.parse_args()

    # Determine which metrics to compute
    if args.metric:
        # Single metric specified (backward compatibility)
        metrics_to_compute = [args.metric]
    elif args.metrics:
        # Multiple metrics specified
        metrics_to_compute = args.metrics
    else:
        # Default: compute both reward and cost
        metrics_to_compute = ['reward_mean', 'cost_mean']

    print("\n" + "=" * 80)
    print("RLIABLE ANALYSIS - STARTING")
    print("=" * 80)
    print(f"Metrics to compute: {', '.join(metrics_to_compute)}")
    print(f"CSV file: {args.csv}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 80 + "\n")

    # Initialize analyzer
    analyzer = RliableAnalyzer(args.csv, args.output_dir)

    # Run analysis for each metric
    for metric in metrics_to_compute:
        analyzer.run_full_analysis(metric=metric)
        print("\n")

    print("=" * 80)
    print("ALL ANALYSES COMPLETE!")
    print(f"Results saved to: {args.output_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()
