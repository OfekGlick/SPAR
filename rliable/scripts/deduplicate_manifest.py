"""
Deduplicate the manifest CSV by keeping only the earlier evaluation for duplicate runs.

For runs with the same configuration (algo, env, seed, budget, obs_mode, actor_type,
use_cost, use_all_obs, sd_regulizer, random_obs_selection), keep only the earlier one.
"""

import pandas as pd
from datetime import datetime

# Load the CSV
csv_path = '../data/raw/manifest_for_rliable.csv'
print(f"Loading {csv_path}...")

# Define columns to use (to avoid issues with trailing commas)
usecols = [
    'timestamp', 'algo', 'env', 'seed', 'budget', 'obs_mode', 'actor_type',
    'use_cost', 'use_all_obs', 'sd_regulizer', 'random_obs_selection',
    'reward_mean', 'reward_std', 'cost_mean', 'cost_std',
    'episode_rewards', 'episode_costs', 'sample_efficiency_curve',
    'reward_normalized', 'cost_normalized', 'obs_modality_normalize',
    'total_steps', 'num_eval_episodes', 'status', 'log_dir',
    'action_space_type', 'obs_space_shape'
]

df = pd.read_csv(csv_path, usecols=usecols, low_memory=False)
print(f"Loaded {len(df)} rows")

# Define the columns that identify a unique run
run_identifier_cols = [
    'algo', 'env', 'seed', 'budget', 'obs_mode', 'actor_type',
    'use_cost', 'use_all_obs', 'sd_regulizer', 'random_obs_selection'
]

print(f"\nChecking for duplicates based on: {run_identifier_cols}")

# Convert timestamp to datetime for sorting
df['timestamp_dt'] = pd.to_datetime(df['timestamp'], format='%m/%d/%Y %H:%M', errors='coerce')

# Sort by the identifier columns and timestamp (earlier first)
df_sorted = df.sort_values(by=run_identifier_cols + ['timestamp_dt'])

# Count duplicates before
duplicate_count = df_sorted.duplicated(subset=run_identifier_cols, keep=False).sum()
print(f"Found {duplicate_count} rows that are duplicates (including originals)")

# Show some example duplicates
if duplicate_count > 0:
    print("\nExample duplicate groups:")
    duplicates = df_sorted[df_sorted.duplicated(subset=run_identifier_cols, keep=False)]
    for i, (name, group) in enumerate(duplicates.groupby(run_identifier_cols)):
        if i >= 3:  # Show only first 3 examples
            break
        print(f"\nGroup {i+1}: {len(group)} duplicates")
        print(f"  Config: algo={group.iloc[0]['algo']}, env={group.iloc[0]['env']}, "
              f"seed={group.iloc[0]['seed']}, budget={group.iloc[0]['budget']}")
        for idx, row in group.iterrows():
            print(f"    Timestamp: {row['timestamp']}, Reward: {row['reward_mean']:.3f}")

# Keep only the first occurrence (earliest timestamp) for each unique run
df_deduplicated = df_sorted.drop_duplicates(subset=run_identifier_cols, keep='first')

# Drop the helper timestamp column
df_deduplicated = df_deduplicated.drop(columns=['timestamp_dt'])

removed_count = len(df) - len(df_deduplicated)
print(f"\n{'='*80}")
print(f"DEDUPLICATION SUMMARY")
print(f"{'='*80}")
print(f"Original rows: {len(df)}")
print(f"Deduplicated rows: {len(df_deduplicated)}")
print(f"Removed duplicates: {removed_count}")
print(f"{'='*80}\n")

# Save the deduplicated CSV
output_path = '../data/processed/manifest_for_rliable_deduplicated.csv'
df_deduplicated.to_csv(output_path, index=False)
print(f"Saved deduplicated CSV to: {output_path}")

# Also create a backup of the original
import shutil
backup_path = '../data/raw/manifest_for_rliable_backup.csv'
shutil.copy2(csv_path, backup_path)
print(f"Created backup of original at: {backup_path}")

print("\nTo use the deduplicated version, you can either:")
print("  1. Run analysis on 'manifest_for_rliable_deduplicated.csv'")
print("  2. Or replace the original (after verifying):")
print(f"     mv {output_path} {csv_path}")
