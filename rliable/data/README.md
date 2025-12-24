# RLiable Data Directory

This directory contains manifest CSV files for SPAR experiment analysis.

## Directory Structure

```
data/
├── raw/                        # Original, unprocessed manifest files
│   ├── manifest_for_rliable.csv
│   ├── manifest_for_rliable_backup.csv
│   ├── run_manifest.csv
│   └── run_manifest2.csv
└── processed/                  # Processed and deduplicated manifests
    ├── manifest_for_rliable_deduplicated.csv
    └── results.json
```

## File Descriptions

### raw/
Contains original manifest files generated from experiment evaluations.

- **manifest_for_rliable.csv**: Original manifest with all experiment runs
- **manifest_for_rliable_backup.csv**: Backup copy created by deduplication script
- **run_manifest.csv** / **run_manifest2.csv**: Additional manifest files from different experiment batches

### processed/
Contains cleaned and processed data ready for analysis.

- **manifest_for_rliable_deduplicated.csv**: Main manifest file with duplicates removed
  - This is the default input for `compute_rliable_metrics.py`
  - Duplicates are identified by: algo, env, seed, budget, obs_mode, actor_type, use_cost, use_all_obs, sd_regulizer, random_obs_selection
  - For duplicates, only the earliest evaluation is kept

- **results.json**: Processed experiment results in JSON format

## Manifest CSV Schema

Each manifest CSV contains the following columns:

| Column | Description |
|--------|-------------|
| timestamp | Evaluation timestamp (MM/DD/YYYY HH:MM) |
| algo | Algorithm name (PPO, TRPO, etc.) |
| env | Environment ID |
| seed | Random seed |
| budget | Sensor budget constraint |
| obs_mode | Observation mode |
| actor_type | Actor network type |
| use_cost | Whether cost penalties are used |
| use_all_obs | Whether all observations are used |
| sd_regulizer | Standard deviation regularizer value |
| random_obs_selection | Whether random observation selection is used |
| reward_mean | Mean episode reward |
| reward_std | Standard deviation of episode reward |
| cost_mean | Mean episode cost |
| cost_std | Standard deviation of episode cost |
| episode_rewards | JSON array of all episode rewards |
| episode_costs | JSON array of all episode costs |
| sample_efficiency_curve | JSON array of learning curve data |
| reward_normalized | Normalized reward metric |
| cost_normalized | Normalized cost metric |
| obs_modality_normalize | Observation modality normalization flag |
| total_steps | Total training steps |
| num_eval_episodes | Number of evaluation episodes |
| status | Evaluation status |
| log_dir | Path to experiment logs |
| action_space_type | Type of action space (continuous/discrete) |
| obs_space_shape | Shape of observation space |

## Usage

### Processing Workflow

1. **Raw data collection**: Experiment evaluations generate manifest files in `raw/`
2. **Deduplication**: Run `deduplicate_manifest.py` to create processed manifests
3. **Analysis**: Use `compute_rliable_metrics.py` with processed manifests

### Adding New Data

To add new experiment data:

1. Place your manifest CSV in the `raw/` directory
2. Run the deduplication script (or skip if no duplicates expected)
3. Update the `--csv` argument when running analysis scripts

## Notes

- Always keep backups of original raw data
- The processed directory is regenerated from raw data, so modifications should be made to raw files
- Large manifest files (>1GB) may require additional memory when loading with pandas
