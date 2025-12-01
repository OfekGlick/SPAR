# BAFS Environments

Budget-Aware Feature Selection environment wrappers that add modality-level observation masking and cost tracking to standard RL environments.

## Overview

BAFS environments extend standard Gym/Gymnasium environments by:
1. **Decomposing observations** into semantic modalities (e.g., camera, lidar, proprioception)
2. **Adding sensor selection actions** to the action space
3. **Tracking per-modality costs** as auxiliary signals
4. **Implementing CMDP interface** for safe RL algorithms

## Architecture

```
Standard Environment (Gym/Gymnasium)
  ↓
BAFS Wrapper (this package)
  - Modality registration
  - Observation masking
  - Cost computation
  - CMDP interface
  ↓
Modified Environment with:
  - Action: Tuple(env_action, sensor_mask)
  - Observation: Masked based on sensor_mask
  - Cost: Per-step sensor usage cost
```

## Base Class: `BudgetAwareBase`

### Purpose
Abstract base class that implements core BAFS functionality. All environment-specific wrappers inherit from this class.

### Key Methods

**`__init__(env, **kwargs)`**
- `modality_costs` (dict): Per-modality costs (default: 1.0 for all)
- `use_all_obs` (bool): If True, disable masking (baseline mode)
- `sensor_dropout_rescale` (bool): Rescale observations when modalities masked
- `cost_penalty_coef` (float): Coefficient for cost penalty (0-1)
- `available_sensors` (list): Restrict to specific modalities (optional)

**`register_modality(name: str, indices: Union[int, List[int]])`**
Register a new observation modality.

**`step(action: Tuple) -> Tuple[obs, reward, terminated, truncated, info]`**
Execute environment step with sensor masking:
1. Extract environment action and sensor mask from compound action
2. Step base environment with environment action
3. Apply sensor mask to observation
4. Compute per-step cost based on active sensors
5. Return masked observation, reward, cost, termination flags, info

**`reset(**kwargs) -> Tuple[obs, info]`**
Reset environment and return initial masked observation.

### CMDP Interface

BAFS environments implement the Constrained MDP interface required by safe RL algorithms:

```python
obs, info = env.reset()
done = False
total_cost = 0

while not done:
    # Action = (environment_action, sensor_mask)
    action = agent.get_action(obs)

    obs, reward, terminated, truncated, info = env.step(action)

    # Cost is returned in info dict
    cost = info['cost']  # Per-step sensor usage cost
    total_cost += cost

    done = terminated or truncated

print(f"Total cost: {total_cost}, Budget: {env.cost_limit}")
```

## Highway Wrapper: `BudgetAwareHighway`

### Supported Environments

**`budget-aware-highway-fast-v0`** (4 modalities):
- Kinematics
- LidarObservation
- OccupancyGrid
- TimeToCollision

**`budget-aware-intersection-v1`** (3 modalities):
- Kinematics
- LidarObservation
- OccupancyGrid

**`budget-aware-roundabout-v0`** (4 modalities):
- Kinematics
- LidarObservation
- OccupancyGrid
- TimeToCollision

### Observation Space

Original highway-env observations are flattened 2D arrays (vehicles × features). BAFS decomposes them into:

| Modality | Indices | Description |
|----------|---------|-------------|
| Kinematics | 0-1 | Position (x, y) for each vehicle |
| LidarObservation | 2-3 | Lidar measurements (distance, angle) |
| OccupancyGrid | 4-5 | Grid cell occupancy |
| TimeToCollision | 6 | Collision time estimates |

### Action Space

```python
action_space = Tuple(
    Discrete(5),        # {IDLE, LEFT, RIGHT, FASTER, SLOWER}
    MultiBinary(M)      # M = number of modalities
)
```

### Usage Example

```python
import gymnasium as gym
from bafs_envs import BudgetAwareHighway

# Create environment
env = gym.make('budget-aware-highway-fast-v0')

# Configure modality costs
env = BudgetAwareHighway(
    env,
    modality_costs={
        'Kinematics': 1.0,
        'LidarObservation': 2.0,
        'OccupancyGrid': 1.5,
        'TimeToCollision': 3.0
    }
)

# Training loop
obs, info = env.reset()
for _ in range(1000):
    # Agent selects both driving action and sensors
    env_action = env.action_space[0].sample()  # Driving action
    sensor_mask = env.action_space[1].sample() # Sensor selection
    action = (env_action, sensor_mask)

    obs, reward, terminated, truncated, info = env.step(action)

    print(f"Active sensors: {sensor_mask}")
    print(f"Cost this step: {info['cost']}")

    if terminated or truncated:
        obs, info = env.reset()
```

## Robosuite Wrapper: `BudgetAwareRobosuite`

### Supported Environments

**`budget-aware-Lift`** - Pick and lift a cube
**`budget-aware-Door`** - Open a door with handle

Both environments have 4 modalities:
1. **Robot Proprioception** - Joint states, end-effector pose
2. **Object States** - Object positions, orientations, velocities
3. **Task Features** - Goal-relative metrics
4. **Camera** - 16×16 grayscale images

### Observation Space

Robosuite observations are dictionaries with multiple keys. BAFS flattens and groups them:

| Modality | Keys | Description |
|----------|------|-------------|
| robot_proprioception | `robot0_joint_pos`, `robot0_joint_vel`, `robot0_eef_pos`, `robot0_eef_quat`, `robot0_gripper_qpos` | Robot state |
| object_states | `cube_pos`, `cube_quat`, `gripper_to_cube_pos` (Lift only) | Object state |
| task_features | Task-specific features (distances, completion indicators) | Goal metrics |
| camera | `frontview_image`, `agentview_image`, `robot0_eye_in_hand_image` | Visual observations |

### Action Space

```python
action_space = Tuple(
    Box(-1, 1, shape=(7,)),  # End-effector control (position + gripper)
    MultiBinary(4)            # Sensor mask
)
```

### Configuration

```python
modality_costs = {
    'robot_proprioception': 1.0,  # Low cost (always available)
    'object_states': 1.0,          # Low cost (state estimation)
    'task_features': 1.0,          # Low cost (computed features)
    'camera': 10.0                 # High cost (image processing)
}
```

### Usage Example

```python
import gymnasium as gym
from bafs_envs import BudgetAwareRobosuite

# Create environment
env = gym.make('budget-aware-Lift')

# Configure with custom costs
env = BudgetAwareRobosuite(
    env,
    modality_costs={
        'robot_proprioception': 1.0,
        'object_states': 1.0,
        'task_features': 1.0,
        'camera': 10.0  # Camera is expensive
    },
    robot_type='Panda',
    use_camera=True
)

# Training loop
obs, info = env.reset()
for _ in range(1000):
    # Agent learns to use camera only when needed
    env_action = env.action_space[0].sample()  # Continuous control
    sensor_mask = agent.select_sensors(obs)    # Learned sensor policy
    action = (env_action, sensor_mask)

    obs, reward, terminated, truncated, info = env.step(action)

    if info['cost'] > 5:  # High cost (camera used)
        print("Using camera this step")
```

## Advanced Features

### Sensor Dropout Regularization

Train a "teacher" policy on full observations to guide learning:

```python
env = BudgetAwareHighway(
    env,
    use_all_obs=False,  # Student sees masked obs
    sd_regulizer=True   # Enable teacher-student distillation
)
```

The algorithm (e.g., PPO) will maintain two versions of the policy:
- **Student**: Trained on masked observations (used for rollouts)
- **Teacher**: Trained on full observations (used for auxiliary loss)

### Observation Normalization

Normalize each modality independently:

```python
env = BudgetAwareHighway(
    env,
    obs_modality_normalize=True  # Per-modality normalization
)
```

This is applied via OmniSafe wrappers (`ModalityObsNormalize`, `ModalityObsScale`).

### Fixed Sensor Subsets

Test performance with specific sensor combinations:

```python
# Only use Kinematics and Lidar
env = BudgetAwareHighway(
    env,
    available_sensors=['Kinematics', 'LidarObservation']
)
```

### Penalty-Based Training

Instead of hard budget constraint, use soft penalty:

```python
env = BudgetAwareHighway(
    env,
    cost_penalty_coef=0.1  # Add 0.1 * cost to reward
)

# Effective reward = env_reward - 0.1 * sensor_cost
```

#
## API Reference

### BudgetAwareBase

**Properties:**
- `modalities: Dict[str, List[int]]` - Modality name to observation indices
- `modality_costs: Dict[str, float]` - Per-modality costs
- `observation_space: Space` - Gym observation space (with mask appended)
- `action_space: Tuple[Space, MultiBinary]` - Compound action space

**Methods:**
- `register_modality(name, indices)` - Add a modality
- `step(action)` - Execute environment step with masking
- `reset()` - Reset environment
- `_apply_mask(obs, mask)` - Mask observation based on sensor selection
- `_compute_cost(mask)` - Compute per-step cost from sensor mask

### BudgetAwareHighway

Inherits from `BudgetAwareBase` with highway-env specific modality registration.

### BudgetAwareRobosuite

Inherits from `BudgetAwareBase` with robosuite-specific modality registration and dictionary observation handling.


## See Also

- [Main README](../README.md) - Repository overview
