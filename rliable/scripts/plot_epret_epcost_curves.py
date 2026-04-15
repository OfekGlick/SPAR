"""
Plot EpRet (Episode Return) and EpCost (Episode Cost) training curves.

Creates publication-quality training curve plots for Highway, Door, and Lift environments.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re
import warnings

warnings.filterwarnings('ignore')

# Use non-interactive backend for saving
import matplotlib
matplotlib.use('Agg')

# Set style
sns.set_style("whitegrid")
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'font.family': 'sans-serif',
})


class TrainingCurvePlotter:
    """Plot EpRet and EpCost training curves."""

    # Algorithm display names
    ALGO_DISPLAY_NAMES = {
        'use_all_obs': 'Full Observation',
        'random_mask': 'Random Selection',
        'sd_reg_pen1.0': 'Penalty ($\\lambda=1$)',
        'sd_reg': 'SPAR (Unconstrained)',
        'Budget24': 'SPAR 20%',
        'Budget60': 'SPAR 50%',
        'Budget96': 'SPAR 80%',
        'Budget400': 'SPAR 20%',
        'Budget1000': 'SPAR 50%',
        'Budget1600': 'SPAR 80%',
    }

    # Color palette - consistent across all plots
    ALGO_COLORS = {
        'use_all_obs': '#2E86AB',       # Blue - baseline
        'random_mask': '#95B8D1',       # Light blue
        'sd_reg_pen1.0': '#C73E1D',     # Red - penalty
        'sd_reg': '#6A994E',            # Green - unconstrained
        'Budget24': '#F18F01',          # Orange - 20%
        'Budget60': '#A23B72',          # Purple - 50%
        'Budget96': '#17B6B5',          # Teal - 80%
        'Budget400': '#F18F01',         # Orange - 20%
        'Budget1000': '#A23B72',        # Purple - 50%
        'Budget1600': '#17B6B5',        # Teal - 80%
    }

    # Line styles
    ALGO_LINESTYLES = {
        'use_all_obs': '-',
        'random_mask': '--',
        'sd_reg_pen1.0': '-.',
        'sd_reg': ':',
        'Budget24': '-',
        'Budget60': '-',
        'Budget96': '-',
        'Budget400': '-',
        'Budget1000': '-',
        'Budget1600': '-',
    }

    # Plot order (for legend consistency)
    ALGO_ORDER = [
        'use_all_obs', 'random_mask', 'sd_reg_pen1.0', 'sd_reg',
        'Budget96', 'Budget60', 'Budget24',  # Highway
        'Budget1600', 'Budget1000', 'Budget400',  # Door/Lift
    ]

    def __init__(self, data_dir: str, output_dir: str):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

    def _parse_algo_key(self, col_name: str) -> str:
        """Extract algorithm key from column name."""
        # Check for budget-based PPOLag
        budget_match = re.search(r'Budget(\d+)', col_name)
        if budget_match:
            return f"Budget{budget_match.group(1)}"

        # Check for other algorithm types
        if 'use_all_obs' in col_name:
            return 'use_all_obs'
        elif 'random_mask' in col_name:
            return 'random_mask'
        elif 'sd_reg_pen1.0' in col_name or 'pen1.0' in col_name:
            return 'sd_reg_pen1.0'
        elif 'sd_reg' in col_name and 'pen' not in col_name:
            return 'sd_reg'

        return None

    def _load_csv(self, filepath: Path) -> dict:
        """Load CSV and organize by algorithm."""
        df = pd.read_csv(filepath)

        step_col = df.columns[0]
        steps = df[step_col].values

        # Find value columns (not MIN/MAX)
        value_cols = [c for c in df.columns
                      if not c.endswith('__MIN')
                      and not c.endswith('__MAX')
                      and c != step_col]

        data = {'steps': steps}

        for col in value_cols:
            algo_key = self._parse_algo_key(col)
            if algo_key is None:
                continue

            min_col = f"{col}__MIN"
            max_col = f"{col}__MAX"

            data[algo_key] = {
                'mean': df[col].values,
                'min': df[min_col].values if min_col in df.columns else None,
                'max': df[max_col].values if max_col in df.columns else None,
            }

        return data

    def plot_training_curve(self, env_name: str, metric: str = 'EpRet',
                           ax=None, show_legend: bool = True):
        """Plot a single training curve."""
        filepath = self.data_dir / f"{env_name} {metric}.csv"
        if not filepath.exists():
            print(f"File not found: {filepath}")
            return None

        data = self._load_csv(filepath)
        steps = data['steps']

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))

        # Plot in consistent order
        plotted_algos = []
        for algo_key in self.ALGO_ORDER:
            if algo_key not in data:
                continue

            algo_data = data[algo_key]
            color = self.ALGO_COLORS.get(algo_key, '#808080')
            label = self.ALGO_DISPLAY_NAMES.get(algo_key, algo_key)
            linestyle = self.ALGO_LINESTYLES.get(algo_key, '-')

            # Plot mean line
            ax.plot(steps, algo_data['mean'],
                   label=label, color=color,
                   linestyle=linestyle, linewidth=2)

            # Plot confidence band
            if algo_data['min'] is not None and algo_data['max'] is not None:
                ax.fill_between(steps, algo_data['min'], algo_data['max'],
                               color=color, alpha=0.15)

            plotted_algos.append(algo_key)

        # Formatting
        ax.set_xlabel('Training Step', fontsize=14)
        ylabel = 'Episode Return' if metric == 'EpRet' else 'Episode Cost'
        ax.set_ylabel(ylabel, fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', which='major', labelsize=12)

        if show_legend:
            ax.legend(loc='best', framealpha=0.9, fontsize=10)

        return ax

    def plot_environment(self, env_name: str, save: bool = True):
        """Plot both EpRet and EpCost for an environment."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Plot EpRet
        self.plot_training_curve(env_name, 'EpRet', ax=axes[0], show_legend=True)
        axes[0].set_title(f'{env_name} - Episode Return', fontsize=16, fontweight='bold')

        # Plot EpCost
        self.plot_training_curve(env_name, 'EpCost', ax=axes[1], show_legend=False)
        axes[1].set_title(f'{env_name} - Episode Cost', fontsize=16, fontweight='bold')

        plt.tight_layout()

        if save:
            output_path = self.output_dir / f"{env_name.lower()}_training_curves.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {output_path}")

            # Also save PDF for publication
            pdf_path = self.output_dir / f"{env_name.lower()}_training_curves.pdf"
            plt.savefig(pdf_path, bbox_inches='tight')
            print(f"Saved: {pdf_path}")

            plt.close()

        return fig

    def plot_all_environments(self, save: bool = True):
        """Plot training curves for all environments in a grid."""
        envs = ['highway', 'door', 'lift']
        env_display = {'highway': 'Highway', 'door': 'Door', 'lift': 'Lift'}

        fig, axes = plt.subplots(3, 2, figsize=(14, 12))

        for row, env in enumerate(envs):
            # EpRet
            self.plot_training_curve(env, 'EpRet', ax=axes[row, 0],
                                    show_legend=(row == 0))
            axes[row, 0].set_title(f'{env_display[env]} - Episode Return',
                                   fontsize=14, fontweight='bold')

            # EpCost
            self.plot_training_curve(env, 'EpCost', ax=axes[row, 1],
                                    show_legend=False)
            axes[row, 1].set_title(f'{env_display[env]} - Episode Cost',
                                   fontsize=14, fontweight='bold')

        plt.tight_layout()

        if save:
            output_path = self.output_dir / "all_training_curves.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {output_path}")

            pdf_path = self.output_dir / "all_training_curves.pdf"
            plt.savefig(pdf_path, bbox_inches='tight')
            print(f"Saved: {pdf_path}")

            plt.close()

        return fig

    def plot_single_metric_all_envs(self, metric: str = 'EpRet', save: bool = True):
        """Plot a single metric across all environments side by side."""
        envs = ['highway', 'door', 'lift']
        env_display = {'highway': 'Highway', 'door': 'Door', 'lift': 'Lift'}

        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

        for idx, env in enumerate(envs):
            self.plot_training_curve(env, metric, ax=axes[idx],
                                    show_legend=(idx == 0))
            axes[idx].set_title(f'{env_display[env]}', fontsize=16, fontweight='bold')

        metric_name = 'Episode Return' if metric == 'EpRet' else 'Episode Cost'
        fig.suptitle(f'{metric_name} Training Curves', fontsize=18, fontweight='bold', y=1.02)
        plt.tight_layout()

        if save:
            output_path = self.output_dir / f"all_envs_{metric.lower()}.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {output_path}")

            pdf_path = self.output_dir / f"all_envs_{metric.lower()}.pdf"
            plt.savefig(pdf_path, bbox_inches='tight')
            print(f"Saved: {pdf_path}")

            plt.close()

        return fig


def main():
    """Main execution function."""
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent  # CSV files are in rliable/
    output_dir = script_dir.parent / "results" / "training_curves"

    plotter = TrainingCurvePlotter(data_dir=str(data_dir), output_dir=str(output_dir))

    print("="*70)
    print("Generating Training Curve Plots")
    print("="*70)

    # Plot individual environments
    for env in ['highway', 'door', 'lift']:
        print(f"\n--- {env.capitalize()} ---")
        plotter.plot_environment(env, save=True)

    # Plot all environments together
    print("\n--- All Environments Grid ---")
    plotter.plot_all_environments(save=True)

    # Plot single metrics across all environments
    print("\n--- EpRet All Environments ---")
    plotter.plot_single_metric_all_envs('EpRet', save=True)

    print("\n--- EpCost All Environments ---")
    plotter.plot_single_metric_all_envs('EpCost', save=True)

    print("\n" + "="*70)
    print(f"All plots saved to: {output_dir}")
    print("="*70)


if __name__ == "__main__":
    main()
