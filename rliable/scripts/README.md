# RLiable Analysis Scripts

This directory contains scripts for analyzing BAFS experiment results using the RLiable library.

## Scripts

### compute_rliable_metrics.py
Main analysis script that computes statistical metrics and generates visualizations for BAFS experiments.

**Features:**
- Computes aggregate metrics (mean, median, IQM, optimality gap)
- Generates performance profiles
- Calculates probability of improvement
- Analyzes sample efficiency curves
- Creates comprehensive plots and reports

**Usage:**
```bash
# Run from project root directory
python -m rliable.scripts.compute_rliable_metrics [--csv CSV_PATH] [--output-dir OUTPUT_DIR] [--metric METRIC]
```

**Arguments:**
- `--csv`: Path to manifest CSV file (default: `rliable/data/processed/manifest_for_rliable_deduplicated.csv`)
- `--output-dir`: Directory for results (default: `rliable/results`)
- `--metric`: Specific metric to analyze (optional)

**Example:**
```bash
# Run full analysis on deduplicated data (from project root)
python -m rliable.scripts.compute_rliable_metrics

# Analyze specific metric
python -m rliable.scripts.compute_rliable_metrics --metric reward_mean

# Use custom data file
python -m rliable.scripts.compute_rliable_metrics --csv path/to/custom_manifest.csv
```

### deduplicate_manifest.py
Utility script to remove duplicate experiment runs from manifest files.

**Features:**
- Identifies duplicate runs based on configuration parameters
- Keeps only the earliest evaluation for each unique configuration
- Creates backup of original file
- Outputs deduplicated CSV for analysis

**Usage:**
```bash
# Run from project root directory
python -m rliable.scripts.deduplicate_manifest
```

**Note:** The script uses hardcoded paths relative to the scripts directory:
- Input: `rliable/data/raw/manifest_for_rliable.csv`
- Output: `rliable/data/processed/manifest_for_rliable_deduplicated.csv`
- Backup: `rliable/data/raw/manifest_for_rliable_backup.csv`

## Directory Structure

The scripts expect the following directory structure (relative paths from this directory):

```
rliable/
├── scripts/                    (you are here)
│   ├── compute_rliable_metrics.py
│   └── deduplicate_manifest.py
├── data/
│   ├── raw/                   (original manifest files)
│   └── processed/             (deduplicated manifests)
└── results/
    ├── plots/                 (PNG visualizations)
    ├── metrics/               (JSON metric files)
    └── reports/               (Markdown reports)
```

## Requirements

These scripts require the following Python packages:
- pandas
- numpy
- matplotlib
- seaborn
- rliable (custom library in parent directory)

## Notes

- Always run scripts from the `rliable/scripts/` directory to ensure relative paths work correctly
- Deduplicate your manifest before running analysis to avoid skewed statistics
- Results are organized by metric type (reward vs cost) in the output directory
