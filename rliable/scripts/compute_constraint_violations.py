"""
Compute constraint violation ratios from W&B evaluation data.

Queries W&B for all budgeted SPAR runs, retrieves per-episode evaluation costs,
and calculates the fraction of episodes where EpCost >= cost_limit.

Usage:
    python -m rliable.scripts.compute_constraint_violations
    python -m rliable.scripts.compute_constraint_violations --entity my-team --env-filter Highway
    python -m rliable.scripts.compute_constraint_violations --dry-run --verbose
"""

import argparse
import re
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
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

WANDB_PROJECTS = {
    "Highway": "SPAR Highway - Learning and Sample Efficiency",
    "Robosuite": "SPAR Robosuite - Learning and Sample Efficiency",
}

ENV_DISPLAY_NAMES = {
    "budget-aware-highway-fast-v0": "Highway",
    "budget-aware-Lift": "Lift",
    "budget-aware-Door": "Door",
    "highway-fast-v0": "Highway",
    "Lift": "Lift",
    "Door": "Door",
}

# Known budget -> display level mappings (from plot_epret_epcost_curves.py)
BUDGET_DISPLAY = {
    24: "20%", 60: "50%", 96: "80%",       # Highway
    400: "20%", 1000: "50%", 1600: "80%",   # Robosuite
}


# ══════════════════════════════════════════════════════════════════════════════
# Analyzer
# ══════════════════════════════════════════════════════════════════════════════

class ConstraintViolationAnalyzer:
    """Query W&B evaluation data and compute constraint violation ratios."""

    def __init__(
        self,
        entity: Optional[str] = None,
        projects: Optional[Dict[str, str]] = None,
        output_dir: Optional[Path] = None,
    ):
        self.entity = entity
        self.projects = projects or WANDB_PROJECTS
        self.output_dir = output_dir or Path(__file__).parent.parent / "results" / "constraint_violations"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.api = wandb.Api(timeout=60)

    # ── Config extraction helpers ─────────────────────────────────────────

    @staticmethod
    def _extract_cost_limit(config: Dict[str, Any]) -> Optional[float]:
        """Extract cost_limit from run config (checks all algorithm-specific paths)."""
        # lagrange_cfgs.cost_limit (PPOLag, SACLag, SACPID, SPOLag, CPPOPID)
        val = config.get("lagrange_cfgs", {}).get("cost_limit")
        if val is not None:
            return float(val)

        # algo_cfgs.cost_limit (CPO)
        val = config.get("algo_cfgs", {}).get("cost_limit")
        if val is not None and float(val) > 0:
            return float(val)

        # algo_cfgs.safety_budget (PPOSaute, TRPOSaute, PPOSimmer)
        val = config.get("algo_cfgs", {}).get("safety_budget")
        if val is not None and float(val) > 0:
            return float(val)

        return None

    @staticmethod
    def _cost_limit_from_name(run_name: str) -> Optional[float]:
        """Fallback: parse Budget{N} from run name."""
        m = re.search(r'Budget(\d+)', run_name)
        return float(m.group(1)) if m else None

    @staticmethod
    def _classify_budget_level(cost_limit: float) -> str:
        """Map a cost_limit value to its budget percentage label."""
        rounded = int(round(cost_limit))
        if rounded in BUDGET_DISPLAY:
            return BUDGET_DISPLAY[rounded]
        # Fallback: estimate from known max costs
        if cost_limit < 200:  # Highway range
            pct = cost_limit / (250 * 4) * 100  # max_steps * num_sensors
        else:  # Robosuite range
            pct = cost_limit / (500 * 4) * 100
        return f"~{int(round(pct))}%"

    @staticmethod
    def _parse_environment(run) -> str:
        """Extract display environment name from run config or name."""
        env_id = run.config.get("env_id", "")
        if env_id in ENV_DISPLAY_NAMES:
            return ENV_DISPLAY_NAMES[env_id]
        # Try partial match
        for key, display in ENV_DISPLAY_NAMES.items():
            if key in env_id or key in run.name:
                return display
        # Last resort: extract from name
        for env_name in ["Highway", "Lift", "Door", "Intersection", "Roundabout"]:
            if env_name.lower() in run.name.lower():
                return env_name
        return env_id or "Unknown"

    @staticmethod
    def _parse_algorithm(run) -> str:
        """Extract algorithm name from run config or name."""
        algo = run.config.get("algo", "")
        if algo:
            return algo
        # Fallback: first component of run name
        return run.name.split("-")[0] if run.name else "Unknown"

    @staticmethod
    def _parse_seed(run) -> Optional[int]:
        """Extract seed from run config."""
        seed = run.config.get("env_cfgs", {}).get("seed")
        if seed is not None:
            return int(seed)
        seed = run.config.get("seed")
        if seed is not None:
            return int(seed)
        return None

    def _is_budgeted_run(self, run) -> bool:
        """Check if this run uses a budget constraint."""
        if self._extract_cost_limit(run.config) is not None:
            return True
        if self._cost_limit_from_name(run.name) is not None:
            return True
        return False

    # ── Core computation ──────────────────────────────────────────────────

    def fetch_runs(self, project_name: str) -> list:
        """Fetch all runs from a W&B project."""
        path = f"{self.entity}/{project_name}" if self.entity else project_name
        try:
            runs = self.api.runs(path)
            return list(runs)
        except Exception as e:
            print(f"  Error fetching runs from '{path}': {e}", file=sys.stderr)
            return []

    def compute_violations_for_run(self, run) -> Optional[Dict[str, Any]]:
        """Compute constraint violation ratio for a single run."""
        cost_limit = self._extract_cost_limit(run.config)
        if cost_limit is None:
            cost_limit = self._cost_limit_from_name(run.name)
        if cost_limit is None:
            return None

        # Fetch evaluation episode costs
        ep_costs = []
        for row in run.scan_history(keys=["Evaluation/EpCost"], page_size=1000):
            val = row.get("Evaluation/EpCost")
            if val is not None and not (isinstance(val, float) and np.isnan(val)):
                ep_costs.append(float(val))

        if not ep_costs:
            return None

        total = len(ep_costs)
        violations = sum(1 for c in ep_costs if c >= cost_limit)

        return {
            "run_name": run.name,
            "run_id": run.id,
            "algorithm": self._parse_algorithm(run),
            "environment": self._parse_environment(run),
            "seed": self._parse_seed(run),
            "cost_limit": cost_limit,
            "budget_level": self._classify_budget_level(cost_limit),
            "total_eval_episodes": total,
            "violation_count": violations,
            "violation_ratio": violations / total,
            "mean_ep_cost": float(np.mean(ep_costs)),
            "std_ep_cost": float(np.std(ep_costs)),
            "episode_costs": ep_costs,
        }

    def analyze_all_projects(
        self,
        env_filter: Optional[str] = None,
        algo_filter: Optional[str] = None,
        dry_run: bool = False,
        verbose: bool = False,
    ) -> pd.DataFrame:
        """Query all projects and compute violation ratios."""
        all_results: List[Dict[str, Any]] = []

        for display_name, project_name in self.projects.items():
            print(f"\n{'='*70}")
            print(f"Project: {project_name}")
            print(f"{'='*70}")

            runs = self.fetch_runs(project_name)
            # For the Highway project, only include runs with "highway" in the name
            if display_name == "Highway":
                runs = [r for r in runs if "highway" in r.name.lower()]
            budgeted = [r for r in runs if self._is_budgeted_run(r)]
            print(f"  Total runs: {len(runs)}, Budgeted runs: {len(budgeted)}")

            if dry_run:
                for r in budgeted:
                    cl = self._extract_cost_limit(r.config) or self._cost_limit_from_name(r.name)
                    print(f"    {r.name}  (cost_limit={cl})")
                continue

            for run in tqdm(budgeted, desc=f"  Processing {display_name}"):
                # Optional filters
                if env_filter and env_filter.lower() not in self._parse_environment(run).lower():
                    continue
                if algo_filter and algo_filter.lower() not in self._parse_algorithm(run).lower():
                    continue

                result = self.compute_violations_for_run(run)
                if result is None:
                    if verbose:
                        print(f"    Skipped {run.name} (no eval data or cost_limit)")
                    continue

                all_results.append(result)
                if verbose:
                    print(
                        f"    {result['run_name']}: "
                        f"violations={result['violation_count']}/{result['total_eval_episodes']} "
                        f"({result['violation_ratio']:.1%}), "
                        f"mean_cost={result['mean_ep_cost']:.1f}, "
                        f"limit={result['cost_limit']:.0f}"
                    )

        if not all_results:
            print("\nNo results found.")
            return pd.DataFrame()

        df = pd.DataFrame(all_results)
        print(f"\nTotal runs analyzed: {len(df)}")
        return df

    # ── Aggregation & output ──────────────────────────────────────────────

    @staticmethod
    def aggregate_results(df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate per-run results by environment, algorithm, and budget level."""
        if df.empty:
            return df

        grouped = df.groupby(["environment", "algorithm", "budget_level"]).agg(
            mean_violation_ratio=("violation_ratio", "mean"),
            std_violation_ratio=("violation_ratio", "std"),
            num_runs=("violation_ratio", "count"),
            total_episodes=("total_eval_episodes", "sum"),
            total_violations=("violation_count", "sum"),
            mean_ep_cost=("mean_ep_cost", "mean"),
            mean_cost_limit=("cost_limit", "mean"),
        ).reset_index()

        grouped["overall_violation_ratio"] = grouped["total_violations"] / grouped["total_episodes"]
        grouped["std_violation_ratio"] = grouped["std_violation_ratio"].fillna(0.0)

        return grouped.sort_values(["environment", "budget_level", "algorithm"]).reset_index(drop=True)

    def plot_violations(self, df_raw: pd.DataFrame) -> None:
        """Create a thesis-quality violin plot of normalised episode costs per budget.

        Each episode cost is divided by its run's cost_limit so all budget levels
        share a common y-axis. The dashed line at y=1.0 is the constraint boundary —
        any episode above it is a violation.
        """
        if df_raw.empty:
            return

        # ── Style ─────────────────────────────────────────────────────────────
        sns.set_style("whitegrid")
        plt.rcParams.update({
            'font.size': 12,
            'axes.labelsize': 13,
            'axes.titlesize': 14,
            'legend.fontsize': 11,
            'figure.dpi': 150,
            'font.family': 'sans-serif',
        })

        BUDGET_COLORS = {
            '20%': '#F18F01',
            '50%': '#A23B72',
            '80%': '#17B6B5',
        }
        BUDGET_ORDER = ['20%', '50%', '80%']
        ENV_ORDER    = ['Highway', 'Lift', 'Door']

        # ── Build long-format dataframe of normalised costs ────────────────────
        records = []
        for _, row in df_raw.iterrows():
            for cost in row['episode_costs']:
                records.append({
                    'environment':       row['environment'],
                    'budget_level':      row['budget_level'],
                    'normalised_cost':   cost / row['cost_limit'],
                })
        df_long = pd.DataFrame(records)

        envs_present = [e for e in ENV_ORDER if e in df_long['environment'].unique()]
        n_envs = len(envs_present)

        fig, axes = plt.subplots(
            1, n_envs,
            figsize=(4.4 * n_envs, 5.2),
            sharey=True,
        )
        if n_envs == 1:
            axes = [axes]

        for ax, env in zip(axes, envs_present):
            env_df = df_long[df_long['environment'] == env]

            # Violin
            sns.violinplot(
                data=env_df,
                x='budget_level',
                y='normalised_cost',
                order=BUDGET_ORDER,
                palette=BUDGET_COLORS,
                inner='box',       # shows median + IQR inside each violin
                cut=0,             # don't extrapolate beyond observed range
                linewidth=1.1,
                ax=ax,
            )

            # Constraint boundary
            ax.axhline(
                1.0,
                color='#C73E1D',
                linestyle='--',
                linewidth=1.8,
                zorder=5,
                label='Budget limit',
            )


            ax.set_title(env, fontsize=14, fontweight='bold', pad=8)
            ax.set_xlabel('Budget Level', fontsize=12)
            ax.set_xticklabels(BUDGET_ORDER, fontsize=12)
            ax.grid(True, axis='y', alpha=0.35, zorder=0)
            ax.set_axisbelow(True)

        axes[0].set_ylabel('Episode Cost / Budget Limit', fontsize=13)

        # Shared legend for the constraint line
        from matplotlib.lines import Line2D
        legend_handles = [
            Line2D([0], [0], color='#C73E1D', linestyle='--',
                   linewidth=1.8, label='Budget limit'),
        ]
        # Add budget-level colour patches
        from matplotlib.patches import Patch
        for budget in BUDGET_ORDER:
            legend_handles.append(
                Patch(facecolor=BUDGET_COLORS[budget], label=f'{budget} Budget')
            )

        fig.legend(
            handles=legend_handles,
            loc='upper center',
            bbox_to_anchor=(0.5, 1.04),
            ncol=len(legend_handles),
            fontsize=10,
            framealpha=0.9,
            edgecolor='#cccccc',
        )

        fig.suptitle(
            'Episode Cost Distribution Relative to Budget Constraint',
            fontsize=14, fontweight='bold', y=1.11,
        )

        plt.tight_layout()

        for ext in ('png', 'pdf'):
            path = self.output_dir / f'constraint_violations.{ext}'
            fig.savefig(path, dpi=300, bbox_inches='tight')
            print(f'Saved: {path}')
        plt.close(fig)

    def save_results(self, df_raw: pd.DataFrame, df_agg: pd.DataFrame) -> None:
        """Save raw and aggregated results to CSV."""
        if df_raw.empty:
            return

        raw_path = self.output_dir / "constraint_violations_per_run.csv"
        agg_path = self.output_dir / "constraint_violations_aggregated.csv"

        df_raw.drop(columns=['episode_costs'], errors='ignore').to_csv(raw_path, index=False)
        df_agg.to_csv(agg_path, index=False)

        print(f"\nResults saved:")
        print(f"  Per-run:     {raw_path}")
        print(f"  Aggregated:  {agg_path}")

    def generate_latex_table(self, df_agg: pd.DataFrame) -> None:
        """Generate a booktabs LaTeX table of constraint violation statistics."""
        if df_agg.empty:
            return

        BUDGET_ORDER = ['20%', '50%', '80%']
        ENV_ORDER    = ['Highway', 'Lift', 'Door']

        # Pool across algorithms: compute overall violation rate
        pooled = (
            df_agg.groupby(['environment', 'budget_level'])
            .agg(
                total_episodes  =('total_episodes',     'sum'),
                total_violations=('total_violations',   'sum'),
                std_violation   =('std_violation_ratio','mean'),
            )
            .reset_index()
        )
        pooled['violation_pct'] = (
            pooled['total_violations'] / pooled['total_episodes'] * 100
        )

        envs_present = [e for e in ENV_ORDER if e in pooled['environment'].unique()]

        lines = []
        lines.append(r'\begin{table}[h]')
        lines.append(r'    \centering')
        lines.append(
            r'    \caption{Constraint violation rates at evaluation. '
            r'Violation rate is the fraction of episodes where the '
            r'cumulative sensor cost exceeded the budget limit. '
            r'Values shown as mean\,$\pm$\,std across seeds.}'
        )
        lines.append(r'    \label{tab:constraint_violations}')
        lines.append(r'    \begin{tabular}{llc}')
        lines.append(r'        \toprule')
        lines.append(
            r'        \textbf{Environment} & \textbf{Budget} '
            r'& \textbf{Violation Rate} \\'
        )
        lines.append(r'        \midrule')

        for i, env in enumerate(envs_present):
            env_df = pooled[pooled['environment'] == env]
            budgets_present = [b for b in BUDGET_ORDER if b in env_df['budget_level'].values]
            n = len(budgets_present)

            for j, budget in enumerate(budgets_present):
                row = env_df[env_df['budget_level'] == budget].iloc[0]

                viol_str = (
                    f"{row['violation_pct']:.1f}\\%"
                    f" $\\pm$ {row['std_violation'] * 100:.1f}\\%"
                )

                # Print environment name only on the first row of its block
                if j == 0:
                    env_cell = f'\\multirow{{{n}}}{{*}}{{{env}}}'
                else:
                    env_cell = ''

                lines.append(
                    f'        {env_cell} & {budget} & {viol_str} \\\\'
                )

            # Separator between environments (midrule) except after the last one
            if i < len(envs_present) - 1:
                lines.append(r'        \midrule')

        lines.append(r'        \bottomrule')
        lines.append(r'    \end{tabular}')
        lines.append(r'\end{table}')

        tex = '\n'.join(lines) + '\n'

        tex_path = self.output_dir / 'constraint_violations.tex'
        tex_path.write_text(tex, encoding='utf-8')
        print(f'  LaTeX table: {tex_path}')


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute constraint violation ratios from W&B evaluation data."
    )
    parser.add_argument(
        "--entity", type=str, default=None,
        help="W&B entity/username (default: auto-detect from logged-in user)",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(__file__).parent.parent / "results" / "constraint_violations",
        help="Output directory for CSV files",
    )
    parser.add_argument(
        "--env-filter", type=str, default=None,
        help="Filter to a specific environment (e.g., 'Highway', 'Lift', 'Door')",
    )
    parser.add_argument(
        "--algo-filter", type=str, default=None,
        help="Filter to a specific algorithm (e.g., 'PPOLag')",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print per-run details during processing",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="List matching budgeted runs without fetching evaluation data",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    analyzer = ConstraintViolationAnalyzer(
        entity=args.entity,
        output_dir=args.output_dir,
    )

    df_raw = analyzer.analyze_all_projects(
        env_filter=args.env_filter,
        algo_filter=args.algo_filter,
        dry_run=args.dry_run,
        verbose=args.verbose,
    )

    if args.dry_run or df_raw.empty:
        return

    df_agg = analyzer.aggregate_results(df_raw)
    analyzer.plot_violations(df_raw)
    analyzer.save_results(df_raw, df_agg)
    analyzer.generate_latex_table(df_agg)


if __name__ == "__main__":
    main()
