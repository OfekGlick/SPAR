# SPAR: Selective Perception under Active Resource management for Reinforcement Learning

A comprehensive framework for training safe reinforcement learning agents that learn to dynamically select which sensor modalities to use under cost constraints.

## Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Repository Structure](#repository-structure)
- [Training](#training)
- [Evaluation and Analysis](#evaluation-and-analysis)
- [Environments](#environments)
- [Algorithms](#algorithms)
- [Citation](#citation)

## Overview

**SPAR (Selective Perception under Active Resource management)** extends standard reinforcement learning by enabling agents to learn:
1. **What action to take** in the environment (e.g., steering, acceleration)
2. **Which sensors to use** at each timestep (e.g., camera, lidar, proprioception)

This is formulated as a **Constrained Markov Decision Process (CMDP)** where:
- Each sensor modality has an associated **cost**
- The agent must satisfy a **budget constraint** on total sensor usage
- The agent learns to trade off task performance vs. sensor cost

### Why SPAR?

In real-world robotics and autonomous systems:
- **Sensors are expensive** (computational cost, energy, latency)
- **Different sensors provide different information** (camera vs. lidar vs. proprioception)
- **Optimal sensor usage varies by situation** (use camera when far, use lidar when close)

SPAR learns **context-dependent sensor selection policies** that maximize task performance while respecting sensor budgets.

## Key Features

- ✅ **Multiple Environments**: Highway driving (highway-env) and robotic manipulation (robosuite)
- ✅ **Modular Design**: Easy to add new environments and sensors
- ✅ **Comprehensive Analysis**: Built-in rliable metrics and visualization
- ✅ **Scalable Training**: Slurm job generation for cluster experiments
- ✅ **Publication-Ready Results**: Automatic plot generation and LaTeX tables

## Installation

### Prerequisites
- Python 3.8+
- CUDA (optional, for GPU training)

### Setup

1. **Clone the repository**:
```bash
git clone https://github.com/yourusername/SPAR.git
cd SPAR
```

2. **Create conda environment**:
```bash
conda create -n spar python=3.10
conda activate spar
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Install highway-env**:
```bash
cd base_envs/highway_env
pip install -e .
cd ../..
```

5. **Install robosuite**:
```bash
cd base_envs/robosuite
pip install -e .
cd ../..
```

6. **Install SPAR environments**:
```bash
pip install -e .
```

## Quick Start

### Single Training Run

**Highway Environment:**
```bash
python run_spar_highway.py \
  --algo PPOLag \
  --env-id budget-aware-highway-fast-v0 \
  --budget 1000 \
  --total-steps 409600 \
  --seed 42
```

**Robosuite Environment:**
```bash
python run_spar_robosuite.py \
  --algo PPOLag \
  --env-id budget-aware-Lift \
  --budget 500 \
  --total-steps 2000000 \
  --seed 0
```

### Baseline (All Sensors)

```bash
python run_spar_highway.py \
  --algo PPO \
  --env-id budget-aware-highway-fast-v0 \
  --use-all-obs \
  --total-steps 409600 \
  --seed 42
```

### Generate Slurm Jobs

For cluster-scale experiments:

```bash
# Highway experiments
python launch_highway.py --submit

# Robosuite experiments
python launch_robosuite.py --submit
```

## Repository Structure

```
SPAR_2/
├── spar_envs/              # Budget-aware environment wrappers
│   ├── budget_aware_base.py       # Base SPAR wrapper
│   ├── budget_aware_highway.py    # Highway-env SPAR wrapper
│   └── budget_aware_robosuite.py  # Robosuite SPAR wrapper
├── base_envs/              # Third-party base environments
│   ├── highway_env/       # Highway driving simulator
│   └── robosuite/         # Robotic manipulation simulator
├── omnisafe/              # Safe RL framework (customized)
│   ├── algorithms/        # RL algorithms (PPO, PPOLag, CPO, etc.)
│   ├── models/           # Actor and critic architectures
│   └── envs/             # Environment wrappers
├── configs/               # Training configurations
│   ├── highway_config.py
│   └── robosuite_config.py
├── utils/                 # Shared utilities
│   ├── args_utils.py     # Argument parsing
│   ├── training_utils.py # Training core functions
│   └── launch_utils.py   # Slurm job generation
├── rliable/              # Results analysis
│   ├── scripts/          # Analysis scripts
│   ├── data/            # Experiment manifests
│   └── results/         # Generated plots and tables
├── runs/                 # Training outputs and logs
├── tests/                # Unit tests
├── run_spar_highway.py   # Highway training entry point
├── run_spar_robosuite.py # Robosuite training entry point
├── launch_highway.py     # Highway job generator
└── launch_robosuite.py   # Robosuite job generator
```

See individual package READMEs for detailed documentation:
- [spar_envs/README.md](spar_envs/README.md) - Environment wrappers
- [rliable/README.md](rliable/README.md) - Analysis pipeline
- [configs/README.md](configs/README.md) - Configuration system

## Training

### Command-Line Arguments

Common arguments for `run_spar_highway.py` and `run_spar_robosuite.py`:

| Argument | Type | Description |
|----------|------|-------------|
| `--algo` | str | Algorithm name (PPO, PPOLag, CPO, etc.) |
| `--env-id` | str | Environment ID (e.g., budget-aware-highway-fast-v0) |
| `--budget` | float | Cost constraint budget |
| `--total-steps` | int | Total training steps |
| `--seed` | int | Random seed |
| `--use-all-obs` | flag | Disable masking (use all sensors) |
| `--random-obs-selection` | flag | Random mask baseline |
| `--sd-regulizer` | flag | Enable sensor dropout regularization |
| `--obs-modality-normalize` | flag | Enable observation scaling |
| `--penalty-coef` | float | Cost penalty coefficient (0-1) |
| `--available-sensors` | list | Restrict to specific modalities |
| `--wandb-project` | str | Weights & Biases project name |

### Training Modes

**1. Budget-Constrained Learning (SPAR)**
```bash
python run_spar_highway.py \
  --algo PPOLag \
  --env-id budget-aware-highway-fast-v0 \
  --budget 1000 \
  --sd-regulizer \
  --obs-modality-normalize
```

**2. Baseline (All Sensors)**
```bash
python run_spar_highway.py \
  --algo PPO \
  --env-id budget-aware-highway-fast-v0 \
  --use-all-obs
```

**3. Random Sensor Selection**
```bash
python run_spar_highway.py \
  --algo PPO \
  --env-id budget-aware-highway-fast-v0 \
  --random-obs-selection
```

**4. Penalty-Based (No Hard Constraint)**
```bash
python run_spar_highway.py \
  --algo PPO \
  --env-id budget-aware-highway-fast-v0 \
  --penalty-coef 0.1
```

### Monitoring Training

Training progress is logged to:
- **Weights & Biases** (if configured)
- **CSV manifest**: `rliable/data/raw/run_manifest.csv`
- **Local logs**: `runs/{run_config}/seed-{seed}-{timestamp}/`

Monitor key metrics:
- `Metrics/EpRet` - Episode return (reward)
- `Metrics/EpCost` - Episode cost (sensor usage)
- `Train/Entropy` - Policy entropy
- `Loss/Loss_pi` - Policy loss

## Evaluation and Analysis

### Running Analysis

**Compute aggregate metrics (all environments combined):**
```bash
cd rliable/scripts
python compute_rliable_metrics.py \
  --csv ../data/raw/run_manifest.csv \
  --output-dir ../results
```

**Per-environment analysis:**
```bash
python compute_rliable_metrics.py \
  --csv ../data/raw/run_manifest.csv \
  --output-dir ../results \
  --per-env
```

### Generated Outputs

Analysis produces:
- **Plots**: IQM performance, sample efficiency curves, performance profiles
- **Tables**: LaTeX-formatted tables with confidence intervals
- **Reports**: Markdown summaries with statistical tests

Output locations:
- `rliable/results/aggregate_metrics_reward.png` - Overall performance
- `rliable/results/budget-aware-{env}/` - Per-environment results
- `rliable/results/latex_table_combined_reward.tex` - Combined LaTeX table

### Sensor Activation Analysis

**Analyze sensor usage patterns:**
```bash
cd rliable/scripts
python analyze_sensor_activations.py
```

**Generate heatmaps:**
```bash
python plot_sensor_activation_heatmaps.py
```

Output: `rliable/results/budget-aware-{env}/{env}_sensor_heatmaps.png`

## Environments

### Highway-Env (Autonomous Driving)

**Available Environments:**
- `budget-aware-highway-fast-v0` (4 modalities)
- `budget-aware-intersection-v1` (3 modalities)
- `budget-aware-roundabout-v0` (4 modalities)

**Sensor Modalities:**
1. **Kinematics** - Vehicle state (position, velocity, heading)
2. **LidarObservation** - Distance to nearby vehicles
3. **OccupancyGrid** - Spatial occupancy map
4. **TimeToCollision** - Collision risk estimates

**Action Space:** `Tuple(Discrete(5), MultiBinary(M))`
- Environment action: {IDLE, LEFT, RIGHT, FASTER, SLOWER}
- Sensor mask: Binary vector (1 = use sensor, 0 = mask)

### Robosuite (Robotic Manipulation)

**Available Environments:**
- `budget-aware-Lift` (4 modalities)
- `budget-aware-Door` (4 modalities)

**Sensor Modalities:**
1. **Robot Proprioception** - Joint positions, velocities, end-effector pose
2. **Object States** - Object positions, orientations, velocities
3. **Task Features** - Goal-relative distances and metrics
4. **Camera** - 16×16 grayscale image

**Action Space:** `Tuple(Box(7), MultiBinary(4))`
- Environment action: 7-DOF end-effector control
- Sensor mask: Binary vector for 4 modalities

See [spar_envs/README.md](spar_envs/README.md) for detailed environment documentation.

## Algorithms

### Safe RL Algorithms

**Lagrangian Methods:**
- `PPOLag` - PPO with Lagrangian constraint handling

**Penalty Methods:**
- `PPO` with penalty coefficient

**PID-Based:**
- `CPPOPID` - PPO with PID Lagrangian controller

See [omnisafe/README.md](omnisafe/README.md) for algorithm details.

## Configuration

Training configurations are defined in `configs/`:

**Highway Config** (`configs/highway_config.py`):
- Default steps: 409,600
- Episode length: 250 steps
- Budget ratios: 20%, 50%, 80%
- Sensor costs: Uniform (1.0 each)

**Robosuite Config** (`configs/robosuite_config.py`):
- Default steps: 2,000,000
- Episode length: 500 steps
- Budget ratios: 20%, 50%, 80%
- Sensor costs: Camera=10, others=1

See [configs/README.md](configs/README.md) for configuration details.

## Advanced Usage

### Custom Sensor Costs

```python
python run_spar_highway.py \
  --env-id budget-aware-highway-fast-v0 \
  --budget 1000 \
  --custom-modality-costs 1.0 2.0 1.5 3.0  # Per-modality costs
```

### Sensor Ablation Studies

Test specific sensor combinations:
```bash
python launch_highway_sensor_subsets.py --generate --submit
```

## Project Structure Overview

```
Training Pipeline:
  launch_*.py → Slurm Jobs → run_spar_*.py → OmniSafe → SPAR Env → Base Env

Analysis Pipeline:
  runs/ → CSV Manifest → compute_rliable_metrics.py → Plots & Tables

Environment Stack:
  Base Env (Highway/Robosuite)
    ↓
  SPAR Wrapper (Modality masking + cost tracking)
    ↓
  OmniSafe Wrappers (Normalization, time limits)
    ↓
  Safe RL Algorithm (PPOLag, CPO, etc.)
```

## Troubleshooting

**Import Errors:**
```bash
# Ensure all packages are installed in editable mode
pip install -e .
cd base_envs/highway_env && pip install -e .
cd ../robosuite && pip install -e .
```

**CUDA Out of Memory:**
```bash
# Reduce batch size in config
python run_spar_highway.py --algo PPOLag --env-id budget-aware-highway-fast-v0
# Edit configs/highway_config.py: 'steps_per_epoch': 4096
```

**Wandb Login:**
```bash
wandb login
# Enter your API key from https://wandb.ai/authorize
```

## Citation

If you use this code in your research, please cite:

```bibtex
need to add
```

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## Contact

For questions or issues, please:
- Open an issue on GitHub
- Email: ofek.gluck@gmail.com

## Acknowledgments

This project builds upon:
- [OmniSafe](https://github.com/PKU-Alignment/omnisafe) - Safe RL framework
- [highway-env](https://github.com/eleurent/highway-env) - Highway driving simulator
- [robosuite](https://github.com/ARISE-Initiative/robosuite) - Robotic manipulation framework
- [rliable](https://github.com/google-research/rliable) - RL evaluation metrics
