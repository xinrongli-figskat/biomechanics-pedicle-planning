"""Synthetic-data-compatible entry-point planning algorithms.

Public exports are loaded lazily so importing :mod:`pedicle_planning.ai` does
not eagerly import optional imaging and mesh dependencies.
"""

from importlib import import_module


__all__ = [
    "predict_entry_point",
    "get_last_prediction",
    "get_last_score",
    "EntryPointPredictor",
    "predictor",
    "config",
    "ScrewPlan",
    "LinearCalibration",
    "TrajectorySamplingConfig",
    "compute_trajectory_features",
    "detect_pedicle_isthmus",
    "AnatomicalFrame",
    "generate_candidates",
    "optimize_screw_plan",
]

_EXPORT_MODULES = {
    "predict_entry_point": ".inference",
    "get_last_prediction": ".inference",
    "get_last_score": ".inference",
    "EntryPointPredictor": ".predictor",
    "predictor": ".predictor",
    "config": ".config",
    "ScrewPlan": ".trajectory",
    "LinearCalibration": ".trajectory",
    "TrajectorySamplingConfig": ".trajectory",
    "compute_trajectory_features": ".trajectory",
    "detect_pedicle_isthmus": ".pedicle",
    "AnatomicalFrame": ".candidates",
    "generate_candidates": ".candidates",
    "optimize_screw_plan": ".candidates",
}


def __getattr__(name: str):
    """Load a public symbol only when it is first requested."""
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value
