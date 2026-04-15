"""
Plot Reward Value Estimates for all Lift task configurations.

Queries W&B for all Lift runs across every configuration (Full Observation,
Random Selection, Penalty, SPAR Unconstrained, SPAR 20/50/80%), aggregates
Value/reward across seeds, and produces a single publication-quality figure.

Usage:
    python -m rliable.scripts.plot_lift_value_estimates
    python -m rliable.scripts.plot_lift_value_estimates --entity my-team
"""

import argparse
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use('Agg')
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
# Constants  (colours and names match the rest of the project)
# ══════════════════════════════════════════════════════════════════════════════

ROBOSUITE_PROJECT = 'SPAR Robosuite - Learning and Sample Efficiency'

CONFIGS = {
    'use_all_obs':   {'label': 'Full Observation',         'color': '#2E86AB', 'ls': '-'},
    'random_mask':   {'label': 'Random Selection',         'color': '#95B8D1', 'ls': '--'},
    'penalty':       {'label': r'Penalty ($\lambda=1$)',   'color': '#C73E1D', 'ls': '-.'},
    'unconstrained': {'label': 'SPAR (Unconstrained)',     'color': '#6A994E', 'ls': ':'},
    '20%':           {'label': 'SPAR 20\%',                'color': '#F18F01', 'ls': '-'},
    '50%':           {'label': 'SPAR 50\%',                'color': '#A23B72', 'ls': '-'},
    '80%':           {'label': 'SPAR 80\%',                'color': '#17B6B5', 'ls': '-'},
}

CONFIG_ORDER = ['use_all_obs', 'random_mask', 'penalty', 'unconstrained', '20%', '50%', '80%']

N_GRID = 300


# ══════════════════════════════════════════════════════════════════════════════
# Run classification
# ══════════════════════════════════════════════════════════════════════════════

def classify_run(run) -> Optional[str]:
    """Return the configuration key for a run, or None if unrecognised."""
    name = run.name.lower()
    cfg  = run.config

    if 'lift' not in name:
        return None

    # Budget-level runs (check before unconstrained to avoid mis-classification)
    for budget_val, key in [(400, '20%'), (1000, '50%'), (1600, '80%')]:
        if f'budget{budget_val}' in name:
            return key
        cost_limit = (
            cfg.get('lagrange_cfgs', {}).get('cost_limit')
            or cfg.get('algo_cfgs', {}).get('cost_limit')
        )
        if cost_limit is not None and int(round(float(cost_limit))) == budget_val:
            return key

    if 'use_all_obs' in name or cfg.get('env_cfgs', {}).get('use_all_obs'):
        return 'use_all_obs'
    if 'random_mask' in name:
        return 'random_mask'
    if 'pen1.0' in name or 'pen1' in name:
        return 'penalty'
    if 'sd_reg' in name or cfg.get('algo_cfgs', {}).get('sd_regulizer'):
        return 'unconstrained'

    return None


# ══════════════════════════════════════════════════════════════════════════════
# Data fetching & aggregation
# ══════════════════════════════════════════════════════════════════════════════

def fetch_grouped_runs(api: wandb.Api, entity: Optional[str]) -> Dict[str, list]:
    path = f'{entity}/{ROBOSUITE_PROJECT}' if entity else ROBOSUITE_PROJECT
    try:
        all_runs = list(api.runs(path))
    except Exception as e:
        print(f'Error fetching runs: {e}', file=sys.stderr)
        return {}

    grouped: Dict[str, list] = {k: [] for k in CONFIG_ORDER}
    for run in all_runs:
        key = classify_run(run)
        if key:
            grouped[key].append(run)

    for key, runs in grouped.items():
        print(f'  {CONFIGS[key]["label"]}: {len(runs)} runs')
    return grouped


def aggregate_metric(runs: list, metric: str = 'Value/reward') -> Optional[Dict]:
    """Fetch metric history for each run and return mean ± std on a common grid."""
    arrays = []
    max_steps = 0

    histories = []
    for run in tqdm(runs, desc='    fetching', leave=False):
        df = pd.DataFrame(run.history(keys=['TotalEnvSteps', metric], samples=N_GRID))
        if df.empty or 'TotalEnvSteps' not in df.columns or metric not in df.columns:
            continue
        df = df.dropna(subset=['TotalEnvSteps', metric]).sort_values('TotalEnvSteps')
        if len(df) < 2:
            continue
        histories.append(df)
        max_steps = max(max_steps, df['TotalEnvSteps'].max())

    if not histories:
        return None

    step_grid = np.linspace(0, max_steps, N_GRID)
    for df in histories:
        vals = np.interp(
            step_grid,
            df['TotalEnvSteps'].values,
            df[metric].values,
            left=np.nan, right=np.nan,
        )
        arrays.append(vals)

    stacked = np.array(arrays)
    return {
        'steps': step_grid,
        'mean':  np.nanmean(stacked, axis=0),
        'std':   np.nanstd(stacked,  axis=0),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Plotting
# ══════════════════════════════════════════════════════════════════════════════

def plot(data_by_config: Dict[str, Optional[Dict]], output_dir: Path) -> None:
    sns.set_style('whitegrid')
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 13,
        'axes.titlesize': 14,
        'legend.fontsize': 11,
        'figure.dpi': 150,
        'font.family': 'sans-serif',
    })

    fig, ax = plt.subplots(figsize=(9, 5))

    for key in CONFIG_ORDER:
        agg = data_by_config.get(key)
        if agg is None:
            continue

        cfg   = CONFIGS[key]
        steps = agg['steps'] / 1e6
        mean  = agg['mean']
        std   = agg['std']

        ax.plot(steps, mean,
                label=cfg['label'],
                color=cfg['color'],
                linestyle=cfg['ls'],
                linewidth=2.0)
        ax.fill_between(steps, mean - std, mean + std,
                        color=cfg['color'], alpha=0.15)

    ax.set_xlabel('Environment Steps (M)', fontsize=13)
    ax.set_ylabel('Reward Value Estimate', fontsize=13)
    ax.set_title('Lift Task — Reward Value Estimates', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', framealpha=0.9, edgecolor='#cccccc')
    ax.grid(True, alpha=0.35)
    ax.set_axisbelow(True)

    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    for ext in ('png', 'pdf'):
        path = output_dir / f'lift_value_estimates.{ext}'
        fig.savefig(path, dpi=300, bbox_inches='tight')
        print(f'Saved: {path}')
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Plot Reward Value Estimates for all Lift configurations.'
    )
    parser.add_argument('--entity', type=str, default=None,
                        help='W&B entity/username (default: auto-detect)')
    parser.add_argument('--output-dir', type=Path,
                        default=Path(__file__).parent.parent / 'results' / 'lift_analysis',
                        help='Output directory for the plot')
    parser.add_argument('--dry-run', action='store_true',
                        help='List matching runs without fetching history')
    return parser.parse_args()


def main():
    args = parse_args()
    api  = wandb.Api(timeout=60)

    print(f'\nFetching Lift runs from: {ROBOSUITE_PROJECT}')
    grouped = fetch_grouped_runs(api, args.entity)

    if args.dry_run:
        return

    data_by_config: Dict[str, Optional[Dict]] = {}
    for key in CONFIG_ORDER:
        runs = grouped.get(key, [])
        if not runs:
            print(f'\n  {CONFIGS[key]["label"]}: no runs, skipping.')
            data_by_config[key] = None
            continue
        print(f'\n  Aggregating {CONFIGS[key]["label"]} ({len(runs)} seeds)...')
        data_by_config[key] = aggregate_metric(runs, metric='Value/reward')

    print('\nGenerating plot...')
    plot(data_by_config, args.output_dir)


if __name__ == '__main__':
    main()
