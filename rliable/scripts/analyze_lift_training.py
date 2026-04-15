"""
In-depth training analysis for the Lift task.

Queries W&B for all budgeted Lift runs (20%, 50%, 80%), fetches per-epoch
training metrics, aggregates across seeds, and produces a multi-panel figure
covering performance, value estimates, policy entropy, and sensor activations.

Usage:
    python -m rliable.scripts.analyze_lift_training
    python -m rliable.scripts.analyze_lift_training --entity my-team
    python -m rliable.scripts.analyze_lift_training --dry-run
"""

import argparse
import re
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import wandb

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════════════════════

ROBOSUITE_PROJECT = 'SPAR Robosuite - Learning and Sample Efficiency'

# All metrics to fetch from W&B history
FETCH_METRICS = [
    'TotalEnvSteps',
    'Metrics/EpRet',
    'Metrics/EpCost',
    'Metrics/LagrangeMultiplier',
    'Value/reward',
    'Value/cost',
    'Train/Entropy',
    'Train/MaskEntropy',
    'Train/ContinuousEntropy',
    'Metrics/EpActivationSensor_0',
    'Metrics/EpActivationSensor_1',
    'Metrics/EpActivationSensor_2',
    'Metrics/EpActivationSensor_3',
]

# Robosuite sensor index -> name
SENSOR_NAMES = {
    0: 'Proprioception',
    1: 'Object States',
    2: 'Task Features',
    3: 'Camera',
}

# Display names for each metric
METRIC_DISPLAY = {
    'Metrics/EpRet':                    'Episode Return',
    'Metrics/EpCost':                   'Episode Cost',
    'Metrics/LagrangeMultiplier':       'Lagrange Multiplier',
    'Value/reward':                     'Reward Value Estimate',
    'Value/cost':                       'Cost Value Estimate',
    'Train/Entropy':                    'Policy Entropy',
    'Train/MaskEntropy':                'Mask Entropy',
    'Train/ContinuousEntropy':          'Continuous Entropy',
    'Metrics/EpActivationSensor_0':     f'Sensor 0: {SENSOR_NAMES[0]}',
    'Metrics/EpActivationSensor_1':     f'Sensor 1: {SENSOR_NAMES[1]}',
    'Metrics/EpActivationSensor_2':     f'Sensor 2: {SENSOR_NAMES[2]}',
    'Metrics/EpActivationSensor_3':     f'Sensor 3: {SENSOR_NAMES[3]}',
}

# Subplot layout: each entry is (row, col, metric_key)
# 4 rows x 4 cols; sensors fill all 4 cols in the last row
SUBPLOT_LAYOUT = [
    (0, 0, 'Metrics/EpRet'),
    (0, 1, 'Metrics/EpCost'),
    (0, 2, 'Metrics/LagrangeMultiplier'),
    (1, 0, 'Value/reward'),
    (1, 1, 'Value/cost'),
    (1, 2, 'Train/Entropy'),
    (2, 0, 'Train/MaskEntropy'),
    (2, 1, 'Train/ContinuousEntropy'),
    (3, 0, 'Metrics/EpActivationSensor_0'),
    (3, 1, 'Metrics/EpActivationSensor_1'),
    (3, 2, 'Metrics/EpActivationSensor_2'),
    (3, 3, 'Metrics/EpActivationSensor_3'),
]

# Budget colours matching the rest of the project
BUDGET_COLORS = {'20%': '#F18F01', '50%': '#A23B72', '80%': '#17B6B5'}
BUDGET_ORDER  = ['20%', '50%', '80%']
BUDGET_DISPLAY_MAP = {400: '20%', 1000: '50%', 1600: '80%'}

# Common interpolation grid size
N_GRID = 300


# ══════════════════════════════════════════════════════════════════════════════
# Analyser
# ══════════════════════════════════════════════════════════════════════════════

class LiftTrainingAnalyzer:
    """Fetch and plot Lift training dynamics from W&B."""

    def __init__(
        self,
        entity: Optional[str] = None,
        output_dir: Optional[Path] = None,
    ):
        self.entity  = entity
        self.output_dir = output_dir or (
            Path(__file__).parent.parent / 'results' / 'lift_analysis'
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.api = wandb.Api(timeout=60)

    # ── Run filtering ─────────────────────────────────────────────────────

    @staticmethod
    def _budget_level(run) -> Optional[str]:
        """Return budget percentage label for a run, or None if not budgeted."""
        # From config
        cost_limit = (
            run.config.get('lagrange_cfgs', {}).get('cost_limit')
            or run.config.get('algo_cfgs', {}).get('cost_limit')
            or run.config.get('algo_cfgs', {}).get('safety_budget')
        )
        if cost_limit is not None:
            rounded = int(round(float(cost_limit)))
            if rounded in BUDGET_DISPLAY_MAP:
                return BUDGET_DISPLAY_MAP[rounded]
        # Fallback: parse from name
        m = re.search(r'Budget(\d+)', run.name)
        if m:
            rounded = int(m.group(1))
            if rounded in BUDGET_DISPLAY_MAP:
                return BUDGET_DISPLAY_MAP[rounded]
        return None

    def fetch_lift_runs(self) -> Dict[str, list]:
        """Fetch all budgeted Lift runs, grouped by budget level."""
        path = f'{self.entity}/{ROBOSUITE_PROJECT}' if self.entity else ROBOSUITE_PROJECT
        try:
            all_runs = list(self.api.runs(path))
        except Exception as e:
            print(f'Error fetching runs: {e}', file=sys.stderr)
            return {}

        grouped: Dict[str, list] = {b: [] for b in BUDGET_ORDER}
        for run in all_runs:
            if 'lift' not in run.name.lower():
                continue
            level = self._budget_level(run)
            if level is None:
                continue
            grouped[level].append(run)

        for level, runs in grouped.items():
            print(f'  {level}: {len(runs)} runs')
        return grouped

    # ── History fetching & aggregation ───────────────────────────────────

    @staticmethod
    def _fetch_history(run) -> Optional[pd.DataFrame]:
        """Fetch training history for a single run."""
        df = pd.DataFrame(run.history(keys=FETCH_METRICS, samples=N_GRID))
        if df.empty or 'TotalEnvSteps' not in df.columns:
            return None
        df = df.dropna(subset=['TotalEnvSteps']).sort_values('TotalEnvSteps')
        return df

    @staticmethod
    def _interpolate_to_grid(
        df: pd.DataFrame,
        step_grid: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Interpolate each metric column onto a common step grid."""
        steps = df['TotalEnvSteps'].values
        result = {}
        for metric in FETCH_METRICS:
            if metric == 'TotalEnvSteps' or metric not in df.columns:
                continue
            values = df[metric].values
            valid  = ~np.isnan(values)
            if valid.sum() < 2:
                result[metric] = np.full(len(step_grid), np.nan)
            else:
                result[metric] = np.interp(
                    step_grid, steps[valid], values[valid],
                    left=np.nan, right=np.nan,
                )
        return result

    def aggregate_runs(
        self,
        runs: list,
        verbose: bool = False,
    ) -> Optional[Dict]:
        """Aggregate histories across seeds: returns mean, std, and step grid."""
        histories = []
        for run in tqdm(runs, desc='    Fetching history', leave=False):
            df = self._fetch_history(run)
            if df is None:
                if verbose:
                    print(f'    Skipped {run.name} (no TotalEnvSteps data)')
                continue
            histories.append(df)

        if not histories:
            return None

        max_steps = max(df['TotalEnvSteps'].max() for df in histories)
        step_grid = np.linspace(0, max_steps, N_GRID)

        interpolated: Dict[str, List[np.ndarray]] = {
            m: [] for m in FETCH_METRICS if m != 'TotalEnvSteps'
        }
        for df in histories:
            row = self._interpolate_to_grid(df, step_grid)
            for metric, values in row.items():
                interpolated[metric].append(values)

        aggregated = {'step_grid': step_grid}
        for metric, arrays in interpolated.items():
            stacked = np.array(arrays)          # (n_seeds, N_GRID)
            aggregated[metric] = {
                'mean': np.nanmean(stacked, axis=0),
                'std':  np.nanstd(stacked,  axis=0),
            }
        return aggregated

    # ── Plotting ──────────────────────────────────────────────────────────

    def plot(self, data_by_budget: Dict[str, Optional[Dict]]) -> None:
        """Create the multi-panel analysis figure."""
        sns.set_style('whitegrid')
        plt.rcParams.update({
            'font.size': 10,
            'axes.labelsize': 10,
            'axes.titlesize': 11,
            'legend.fontsize': 9,
            'figure.dpi': 150,
            'font.family': 'sans-serif',
        })

        n_rows, n_cols = 4, 4
        fig = plt.figure(figsize=(5.5 * n_cols, 4.2 * n_rows))
        gs  = gridspec.GridSpec(
            n_rows, n_cols,
            figure=fig,
            hspace=0.45,
            wspace=0.35,
        )

        # Build axes dict and hide unused cells
        axes: Dict[Tuple[int, int], plt.Axes] = {}
        used_cells = {(r, c) for r, c, _ in SUBPLOT_LAYOUT}
        for r in range(n_rows):
            for c in range(n_cols):
                ax = fig.add_subplot(gs[r, c])
                if (r, c) in used_cells:
                    axes[(r, c)] = ax
                else:
                    ax.set_visible(False)

        # Plot each metric
        for row, col, metric in SUBPLOT_LAYOUT:
            ax = axes[(row, col)]
            plotted_any = False

            for budget in BUDGET_ORDER:
                agg = data_by_budget.get(budget)
                if agg is None or metric not in agg:
                    continue
                metric_data = agg[metric]
                steps = agg['step_grid'] / 1e6      # display in millions
                mean  = metric_data['mean']
                std   = metric_data['std']
                color = BUDGET_COLORS[budget]

                ax.plot(steps, mean, label=f'{budget} Budget',
                        color=color, linewidth=1.8)
                ax.fill_between(
                    steps, mean - std, mean + std,
                    color=color, alpha=0.18,
                )
                plotted_any = True

            ax.set_title(METRIC_DISPLAY.get(metric, metric),
                         fontsize=11, fontweight='bold', pad=6)
            ax.set_xlabel('Environment Steps (M)', fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_axisbelow(True)

            # Cost limit reference line for EpCost
            if metric == 'Metrics/EpCost':
                for budget, agg in data_by_budget.items():
                    if agg is None:
                        continue
                    # derive cost_limit from budget label
                    limit_map = {'20%': 400, '50%': 1000, '80%': 1600}
                    limit = limit_map.get(budget)
                    if limit:
                        ax.axhline(limit, color=BUDGET_COLORS[budget],
                                   linestyle='--', linewidth=1,
                                   alpha=0.6)

        # Shared legend above the figure
        handles = [
            plt.Line2D([0], [0], color=BUDGET_COLORS[b], linewidth=2, label=f'{b} Budget')
            for b in BUDGET_ORDER
        ]
        fig.legend(
            handles=handles,
            loc='upper center',
            bbox_to_anchor=(0.5, 1.015),
            ncol=len(BUDGET_ORDER),
            fontsize=11,
            framealpha=0.9,
            edgecolor='#cccccc',
        )

        fig.suptitle(
            'Lift Task — Training Dynamics by Budget Level',
            fontsize=15, fontweight='bold', y=1.03,
        )

        for ext in ('png', 'pdf'):
            path = self.output_dir / f'lift_training_analysis.{ext}'
            fig.savefig(path, dpi=300, bbox_inches='tight')
            print(f'Saved: {path}')
        plt.close(fig)

    # ── Entry point ───────────────────────────────────────────────────────

    def run(self, dry_run: bool = False, verbose: bool = False) -> None:
        print(f'\nFetching Lift runs from: {ROBOSUITE_PROJECT}')
        grouped = self.fetch_lift_runs()

        if dry_run:
            return

        data_by_budget: Dict[str, Optional[Dict]] = {}
        for budget in BUDGET_ORDER:
            runs = grouped.get(budget, [])
            if not runs:
                print(f'\n  {budget}: no runs found, skipping.')
                data_by_budget[budget] = None
                continue
            print(f'\n  Aggregating {budget} ({len(runs)} seeds)...')
            data_by_budget[budget] = self.aggregate_runs(runs, verbose=verbose)

        print('\nGenerating plot...')
        self.plot(data_by_budget)


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Analyse Lift task training dynamics from W&B.'
    )
    parser.add_argument(
        '--entity', type=str, default=None,
        help='W&B entity/username (default: auto-detect)',
    )
    parser.add_argument(
        '--output-dir', type=Path,
        default=Path(__file__).parent.parent / 'results' / 'lift_analysis',
        help='Output directory for plots',
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='List matching runs without fetching history',
    )
    parser.add_argument(
        '--verbose', action='store_true',
        help='Print per-run details',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    analyzer = LiftTrainingAnalyzer(
        entity=args.entity,
        output_dir=args.output_dir,
    )
    analyzer.run(dry_run=args.dry_run, verbose=args.verbose)


if __name__ == '__main__':
    main()
