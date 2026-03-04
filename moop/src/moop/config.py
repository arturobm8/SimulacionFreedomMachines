"""Re-export optimization config types from the central config package."""

from config.settings import OptimizationConfig, OptObjective, OptVarBounds

# Alias used throughout moop internals
OptConfig = OptimizationConfig

__all__ = ["OptConfig", "OptimizationConfig", "OptObjective", "OptVarBounds"]
