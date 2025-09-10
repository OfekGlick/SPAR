import numpy as np
import rliable
from rliable import library as rlib
from rliable import metrics
from rliable import plot_utils
import matplotlib.pyplot as plt


# Example: Function to load OmniSafe results into a dictionary
def load_omnisafe_results(log_dir):
    """
    Load OmniSafe results from a specified directory.
    Assumes the directory contains numpy arrays or logs with episodic returns, costs, etc.
    """
    results = {}
    # Replace with your actual loading logic (e.g., np.load, pandas, etc.)
    results['episodic_returns'] = np.load(f"{log_dir}/episodic_returns.npy")
    results['episodic_costs'] = np.load(f"{log_dir}/episodic_costs.npy")
    return results


# Process OmniSafe results to compute metrics with rliable
def compute_metrics_with_rliable(results):
    """
    Compute reliability metrics using rliable.
    Args:
        results: Dictionary with keys like 'episodic_returns' as numpy arrays.
    Returns:
        metrics_dict: Dictionary of computed metrics.
    """
    episodic_returns = results['episodic_returns']

    # Prepare the input for rliable: shape must be (N, M), where:
    # N = number of algorithms or runs
    # M = number of seeds (e.g., repetitions of the experiment)
    # We assume episodic_returns is (N, M) for simplicity.
    if episodic_returns.ndim != 2:
        raise ValueError("Episodic returns must have 2 dimensions (N, M).")

    # Compute aggregate metrics using rliable
    metrics_dict = {
        "IQM": metrics.aggregate_iqm(episodic_returns),
        "Median": metrics.aggregate_median(episodic_returns),
        "Mean": metrics.aggregate_mean(episodic_returns),
        "Optimality Gap": metrics.aggregate_optimality_gap(episodic_returns),
    }

    # Compute bootstrapped confidence intervals for IQM and Median
    metrics_dict["IQM_CI"] = metrics.bootstrap_confidence_interval(
        episodic_returns, metrics.aggregate_iqm
    )
    metrics_dict["Median_CI"] = metrics.bootstrap_confidence_interval(
        episodic_returns, metrics.aggregate_median
    )

    return metrics_dict


# Visualization with rliable
def plot_metrics(metrics_dict, output_path="reliability_metrics.png"):
    """
    Visualize the computed metrics using rliable's plot utilities.
    """
    algorithms = list(metrics_dict.keys())
    iqm_scores = [metrics_dict[algo]["IQM"] for algo in algorithms]
    iqm_cis = [metrics_dict[algo]["IQM_CI"] for algo in algorithms]

    # Plot IQM with confidence intervals
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_utils.plot_interval_estimates(
        iqm_scores, iqm_cis, algorithms, xlabel="IQM", ax=ax
    )
    plt.title("Interquartile Mean (IQM) with Confidence Intervals")
    plt.savefig(output_path)
    plt.show()


if __name__ == "__main__":
    # Example: Load OmniSafe results
    log_directory = "path/to/omnisafe/logs"
    omnisafe_results = load_omnisafe_results(log_directory)

    # Compute reliability metrics
    reliability_metrics = compute_metrics_with_rliable(omnisafe_results)

    # Print metrics
    for metric, value in reliability_metrics.items():
        print(f"{metric}: {value}")

    # Plot and save the metrics visualization
    plot_metrics(reliability_metrics)
