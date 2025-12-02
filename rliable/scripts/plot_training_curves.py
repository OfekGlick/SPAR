"""
Plot sensor activation and Lagrangian multiplier curves over training.

This script reads CSV files exported from WandB and creates line plots showing:
1. Sensor activation percentages over training steps
2. Lagrangian multiplier values over training steps

The plots use the standardized algorithm naming convention from the rliable analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['figure.dpi'] = 150

# Use non-interactive backend
import matplotlib
matplotlib.use('Agg')


class TrainingCurvePlotter:
    """Plot training curves for sensor activations and Lagrangian multipliers."""

    # Algorithm display names (matching rliable analysis)
    DISPLAY_NAMES = {
        'PPO': 'PPO (Baseline - No Cost)',
        'PPO-penalty': 'PPO (Penalty)',
        'PPO-use-all-obs': 'PPO (Baseline - No Cost)',
        'PPO-random-mask': 'PPO (Random Selection)',
        'PPOLag-20pct': 'PPO-Lag 20%',
        'PPOLag-50pct': 'PPO-Lag 50%',
        'PPOLag-80pct': 'PPO-Lag 80%'
    }

    # Color palette for algorithms (consistent across plots)
    COLORS = {
        'PPOLag-80pct': '#2E86AB',      # Blue
        'PPOLag-50pct': '#A23B72',      # Purple
        'PPOLag-20pct': '#F18F01',      # Orange
        'PPO-penalty': '#C73E1D',       # Red
        'PPO-use-all-obs': '#6A994E',   # Green
        'PPO-random-mask': '#17B6B5',   # Teal/Cyan
        'PPO': '#95B8D1',               # Light blue
    }

    # Sensor names for each environment
    SENSOR_NAMES = {
        'Highway': {
            0: 'Kinematics',
            1: 'Lidar Observation',
            2: 'Occupancy Grid',
            3: 'Time To Collision'
        },
        'Door': {
            0: 'Proprio State',
            1: 'Object State',
            2: 'Task Features',
            3: 'Camera'
        },
        'Lift': {
            0: 'Proprio State',
            1: 'Object State',
            2: 'Task Features',
            3: 'Camera'
        }
    }

    def __init__(self, data_dir: str = "../data/wandb_files", output_dir: str = "../results"):
        """
        Initialize the plotter.

        Args:
            data_dir: Directory containing CSV files from WandB
            output_dir: Directory to save plots
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        # Storage for loaded data
        self.sensor_data = {}  # {env: {sensor_idx: DataFrame}}
        self.lagrangian_data = {}  # {env: DataFrame}

    def load_data(self):
        """Load all CSV files from the data directory."""
        print("Loading CSV files...")

        for csv_file in self.data_dir.glob("*.csv"):
            filename = csv_file.stem

            # Parse filename
            if " - Lagrangian" in filename:
                env_name = filename.split(" - ")[0]
                print(f"  Loading Lagrangian data for {env_name}")
                df = pd.read_csv(csv_file)
                self.lagrangian_data[env_name] = df

            elif " - activation sensor " in filename:
                # Extract environment and sensor index
                parts = filename.split(" - activation sensor ")
                env_name = parts[0]
                sensor_info = parts[1]
                sensor_idx = int(sensor_info.split()[0])

                print(f"  Loading {env_name} sensor {sensor_idx}")
                df = pd.read_csv(csv_file)

                if env_name not in self.sensor_data:
                    self.sensor_data[env_name] = {}
                self.sensor_data[env_name][sensor_idx] = df

        print(f"\nLoaded data for {len(self.sensor_data)} environments")
        print(f"  Sensor activation files: {sum(len(v) for v in self.sensor_data.values())}")
        print(f"  Lagrangian files: {len(self.lagrangian_data)}")

    def _parse_algorithm_from_column(self, column_name: str) -> str:
        """
        Parse algorithm identifier from WandB column name.

        Args:
            column_name: Column name from WandB export

        Returns:
            Algorithm key for DISPLAY_NAMES and COLORS dicts
        """
        import re

        # Column format: "Name: PPO-env-config... - Metrics/..."
        if 'random_mask' in column_name or 'RandomMask' in column_name:
            return 'PPO-random-mask'
        elif 'pen1.0' in column_name or 'penalty' in column_name:
            return 'PPO-penalty'
        elif 'AllObs' in column_name or ('PPO-' in column_name and 'use_cost' not in column_name and 'random' not in column_name and 'pen' not in column_name and 'Budget' not in column_name):
            return 'PPO-use-all-obs'
        elif 'Budget' in column_name:
            # Extract budget value and map to percentage
            budget_match = re.search(r'Budget(\d+)', column_name)
            if budget_match:
                budget_val = int(budget_match.group(1))

                # Map budget values to percentages
                # Highway uses ~120 total (24=20%, 60=50%, 96=80%)
                # Door/Lift use ~2000 total (400=20%, 1000=50%, 1600=80%)
                if budget_val in [24, 400]:
                    return 'PPOLag-20pct'
                elif budget_val in [60, 1000]:
                    return 'PPOLag-50pct'
                elif budget_val in [96, 1600]:
                    return 'PPOLag-80pct'
                else:
                    # Calculate approximate percentage for unknown budgets
                    # Try to infer from the value range
                    if budget_val < 100:  # Highway range
                        pct = (budget_val / 120) * 100
                    else:  # Door/Lift range
                        pct = (budget_val / 2000) * 100

                    if pct < 35:
                        return 'PPOLag-20pct'
                    elif pct < 65:
                        return 'PPOLag-50pct'
                    else:
                        return 'PPOLag-80pct'
            return 'Unknown'
        elif 'PPO-' in column_name:
            return 'PPO'  # Baseline PPO
        else:
            return 'Unknown'

    def plot_sensor_activations(self, env_name: str, save: bool = True):
        """
        Plot sensor activation curves for a given environment.

        Args:
            env_name: Environment name (e.g., 'Highway', 'Door', 'Lift')
            save: Whether to save the plot
        """
        if env_name not in self.sensor_data:
            print(f"Warning: No sensor data found for {env_name}")
            return

        sensor_dict = self.sensor_data[env_name]
        n_sensors = len(sensor_dict)

        # Create subplots (2 rows x 2 cols for 4 sensors)
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        for sensor_idx in range(n_sensors):
            if sensor_idx not in sensor_dict:
                continue

            df = sensor_dict[sensor_idx]
            ax = axes[sensor_idx]

            # Get sensor name
            sensor_name = self.SENSOR_NAMES.get(env_name, {}).get(sensor_idx, f"Sensor {sensor_idx}")

            # Extract step and value columns (assumes first col is Step, rest are metrics)
            step_col = df.columns[0]

            # Find all algorithm columns (those without __MIN or __MAX suffix)
            value_cols = [col for col in df.columns if not col.endswith('__MIN') and not col.endswith('__MAX') and col != step_col]

            for value_col in value_cols:
                # Parse algorithm from column name
                algo_key = self._parse_algorithm_from_column(value_col)

                # Skip unknown algorithms
                if algo_key == 'Unknown':
                    print(f"Warning: Could not parse algorithm from column: {value_col}")
                    continue

                steps = df[step_col].values
                values = df[value_col].values

                # Get MIN/MAX columns if they exist
                min_col = f"{value_col}__MIN"
                max_col = f"{value_col}__MAX"

                # Get color and label from predefined dicts
                color = self.COLORS.get(algo_key, '#808080')  # Default gray if not found
                label = self.DISPLAY_NAMES.get(algo_key, algo_key)

                # Plot mean line
                ax.plot(steps, values, label=label, color=color, linewidth=2)

                # Plot confidence interval if available
                if min_col in df.columns and max_col in df.columns:
                    min_values = df[min_col].values
                    max_values = df[max_col].values
                    ax.fill_between(steps, min_values, max_values,
                                   color=color, alpha=0.2)

            # Formatting
            ax.set_xlabel('Training Step', fontsize=18)
            ax.set_ylabel('Activation Percentage', fontsize=18)
            ax.set_title(f'{sensor_name}', fontweight='bold', fontsize=20)
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 1.0])
            ax.tick_params(axis='both', which='major', labelsize=14)

            # Only show legend for first plot
            if sensor_idx == 0:
                ax.legend(loc='best', framealpha=0.9, fontsize=12)

        # Hide unused subplots
        for idx in range(n_sensors, len(axes)):
            axes[idx].axis('off')

        plt.suptitle(f'{env_name} - Sensor Activation Over Training',
                    fontsize=24, fontweight='bold', y=0.995)
        plt.tight_layout()

        if save:
            output_path = self.output_dir / f"{env_name}_sensor_activations.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {output_path}")
            plt.close()

    def plot_lagrangian_multiplier(self, env_name: str, save: bool = True):
        """
        Plot Lagrangian multiplier curve for a given environment.

        Args:
            env_name: Environment name (e.g., 'Highway', 'Door', 'Lift')
            save: Whether to save the plot
        """
        if env_name not in self.lagrangian_data:
            print(f"Warning: No Lagrangian data found for {env_name}")
            return

        df = self.lagrangian_data[env_name]

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))

        # Extract columns
        step_col = df.columns[0]
        value_cols = [col for col in df.columns if not col.endswith('__MIN') and not col.endswith('__MAX') and col != step_col]

        for value_col in value_cols:
            # Parse algorithm from column name
            algo_key = self._parse_algorithm_from_column(value_col)

            # Only plot PPOLag algorithms (they have Lagrangian multipliers)
            if 'PPOLag' not in algo_key:
                continue

            steps = df[step_col].values
            values = df[value_col].values

            # Get MIN/MAX columns if they exist
            min_col = f"{value_col}__MIN"
            max_col = f"{value_col}__MAX"

            # Get color and label from predefined dicts
            color = self.COLORS.get(algo_key, '#A23B72')
            label = self.DISPLAY_NAMES.get(algo_key, algo_key)

            # Plot mean line
            ax.plot(steps, values, label=label, color=color, linewidth=2.5)

            # Plot confidence interval if available
            if min_col in df.columns and max_col in df.columns:
                min_values = df[min_col].values
                max_values = df[max_col].values
                ax.fill_between(steps, min_values, max_values,
                               color=color, alpha=0.2)

        # Formatting
        ax.set_xlabel('Training Step', fontsize=18)
        ax.set_ylabel('Lagrange Multiplier Value', fontsize=18)
        ax.set_title(f'{env_name} - Lagrange Multiplier Over Training',
                    fontsize=22, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.legend(loc='best', framealpha=0.9, fontsize=12)

        plt.tight_layout()

        if save:
            output_path = self.output_dir / f"{env_name}_lagrangian.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {output_path}")
            plt.close()

    def plot_all_environments(self):
        """Plot all sensor activations and Lagrangian curves for all environments."""
        print("\n" + "="*70)
        print("Generating all plots...")
        print("="*70 + "\n")

        # Plot sensor activations
        for env_name in self.sensor_data.keys():
            print(f"\n--- {env_name} Sensor Activations ---")
            self.plot_sensor_activations(env_name, save=True)

        # Plot Lagrangian multipliers
        for env_name in self.lagrangian_data.keys():
            print(f"\n--- {env_name} Lagrangian Multiplier ---")
            self.plot_lagrangian_multiplier(env_name, save=True)

        print("\n" + "="*70)
        print("All plots generated successfully!")
        print(f"Saved to: {self.output_dir.absolute()}")
        print("="*70)

    def plot_combined_sensors(self, env_name: str, algo_key: str = 'PPOLag-50pct', save: bool = True):
        """
        Plot all sensors on a single plot for comparison (for a specific algorithm).

        Args:
            env_name: Environment name
            algo_key: Algorithm to plot (default: PPOLag-50pct)
            save: Whether to save the plot
        """
        if env_name not in self.sensor_data:
            print(f"Warning: No sensor data found for {env_name}")
            return

        sensor_dict = self.sensor_data[env_name]

        fig, ax = plt.subplots(figsize=(12, 7))

        # Color palette for sensors
        sensor_colors = ['#2E86AB', '#A23B72', '#F18F01', '#6A994E']

        for sensor_idx, df in sensor_dict.items():
            sensor_name = self.SENSOR_NAMES.get(env_name, {}).get(sensor_idx, f"Sensor {sensor_idx}")

            # Extract data
            step_col = df.columns[0]

            # Find the column for the specified algorithm
            value_cols = [col for col in df.columns if not col.endswith('__MIN') and not col.endswith('__MAX') and col != step_col]

            # Find matching algorithm column
            algo_col = None
            for col in value_cols:
                if self._parse_algorithm_from_column(col) == algo_key:
                    algo_col = col
                    break

            if algo_col is None:
                print(f"Warning: Algorithm {algo_key} not found for {env_name} sensor {sensor_idx}")
                continue

            steps = df[step_col].values
            values = df[algo_col].values

            # Get confidence intervals
            min_col = f"{algo_col}__MIN"
            max_col = f"{algo_col}__MAX"

            color = sensor_colors[sensor_idx % len(sensor_colors)]

            # Plot
            ax.plot(steps, values, label=sensor_name, color=color, linewidth=2.5)

            if min_col in df.columns and max_col in df.columns:
                min_values = df[min_col].values
                max_values = df[max_col].values
                ax.fill_between(steps, min_values, max_values,
                               color=color, alpha=0.15)

        # Formatting
        algo_display_name = self.DISPLAY_NAMES.get(algo_key, algo_key)
        ax.set_xlabel('Training Step', fontsize=14)
        ax.set_ylabel('Activation Percentage', fontsize=14)
        ax.set_title(f'{env_name} - All Sensors Over Training ({algo_display_name})',
                    fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.0])
        ax.legend(loc='best', framealpha=0.9, ncol=2)

        plt.tight_layout()

        if save:
            output_path = self.output_dir / f"{env_name}_all_sensors_combined_{algo_key}.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {output_path}")
            plt.close()


def main():
    """Main execution function."""
    # Set paths relative to script location
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent / "data" / "wandb_files"
    output_dir = script_dir.parent / "results" / "training_curves"

    # Create plotter
    plotter = TrainingCurvePlotter(data_dir=str(data_dir), output_dir=str(output_dir))

    # Load data
    plotter.load_data()

    # Generate all plots
    plotter.plot_all_environments()

    # Also create combined sensor plots for different algorithms
    print("\n" + "="*70)
    print("Generating combined sensor plots for key algorithms...")
    print("="*70 + "\n")

    # Plot for PPOLag variants
    key_algorithms = ['PPOLag-20pct', 'PPOLag-50pct', 'PPOLag-80pct']

    for env_name in plotter.sensor_data.keys():
        for algo_key in key_algorithms:
            print(f"\n--- {env_name} Combined Sensors ({algo_key}) ---")
            plotter.plot_combined_sensors(env_name, algo_key=algo_key, save=True)


if __name__ == "__main__":
    main()
