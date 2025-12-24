"""
SPAR (Selective Perception under Active Resource Management) Environment Wrappers.

This package provides environment wrappers that add modality-level observation masking
to various RL environments, compatible with OmniSafe's safe RL algorithms.
"""

# Import environment classes with error handling to avoid blocking on circular imports
# Each environment is imported independently so failures in one don't block the other

__all__ = []

try:
    from spar_envs.budget_aware_highway import BudgetAwareHighway
    __all__.append('BudgetAwareHighway')
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import BudgetAwareHighway: {e}")
    BudgetAwareHighway = None

try:
    from spar_envs.budget_aware_robosuite import BudgetAwareRobosuite
    __all__.append('BudgetAwareRobosuite')
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import BudgetAwareRobosuite: {e}")
    BudgetAwareRobosuite = None
