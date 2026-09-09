"""Leakage-safe Random Forest Robust (RFR) gap filling for eddy-covariance data.

Reproduces the method of Zhu et al. (2022), *Stable gap-filling for longer eddy
covariance data gaps*, Agricultural and Forest Meteorology 314, 108777,
https://doi.org/10.1016/j.agrformet.2021.108777

The frozen scientific specification this package is checked against lives in
``docs/method_spec.md`` (prose) and ``docs/method_spec.yaml`` (machine readable).
Numerical reproduction targets and their supplementary sources are recorded in
``docs/supplement_benchmarks.md``.

The public API is assembled incrementally as the implementation steps land; see
``docs/method_spec.md`` for the contract each module must satisfy.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

from rfrgapfill.config import (
    DEFAULT_HYPERPARAMETER_GRID,
    AllocationBasis,
    BoundaryConvention,
    ColumnMap,
    ColumnMapError,
    ConfigError,
    CVStrategy,
    DailyStatisticStrategy,
    FeatureConfig,
    FeatureMode,
    GapClass,
    GapScenarioConfig,
    Hemisphere,
    MetricSubset,
    Mode,
    RFRConfig,
    ValidationConfig,
    orf_pairing_differences,
    require_orf_pairing,
)
from rfrgapfill.features import (
    RADIATION_CATEGORY,
    SEASON,
    FeatureError,
    RadiationClass,
    Season,
    build_feature_matrix,
    daily_flux_statistics,
    feature_names,
    radiation_tag,
    receptive_limiter_features,
    season_tag,
    time_distance_hours,
)
from rfrgapfill.fill import (
    FILL_COLUMN_SUFFIXES,
    FillError,
    FillMethod,
    FillReport,
    FillResult,
    RFRGapFiller,
    fill_column_names,
    method_label,
)
from rfrgapfill.leakage import (
    LeakageError,
    ValidationFeatureSet,
    available_target_mask,
    build_validation_features,
    detect_target_leakage,
    hide_target,
    holdout_mask_from_intervals,
    observed_target_mask,
    require_no_target_leakage,
)
from rfrgapfill.model import (
    FitReport,
    IncompletePolicy,
    InsufficientTrainingDataError,
    ModelError,
    NotFittedError,
    RFRModel,
)
from rfrgapfill.time import (
    TIME_DISTANCE_HOURS,
    DuplicatePolicy,
    TimeAxis,
    TimestampError,
    elapsed_hours,
    prepare_time_index,
)

try:  # pragma: no cover - trivial packaging fallback
    __version__ = _version("rfr-gapfill")
except PackageNotFoundError:  # pragma: no cover - source tree without install
    __version__ = "0.0.0.dev0"

#: Path-independent identifier recorded in run manifests and fill provenance.
PAPER_DOI = "10.1016/j.agrformet.2021.108777"

__all__ = [
    "DEFAULT_HYPERPARAMETER_GRID",
    "FILL_COLUMN_SUFFIXES",
    "PAPER_DOI",
    "RADIATION_CATEGORY",
    "SEASON",
    "TIME_DISTANCE_HOURS",
    "AllocationBasis",
    "BoundaryConvention",
    "CVStrategy",
    "ColumnMap",
    "ColumnMapError",
    "ConfigError",
    "DailyStatisticStrategy",
    "DuplicatePolicy",
    "FeatureConfig",
    "FeatureError",
    "FeatureMode",
    "FillError",
    "FillMethod",
    "FillReport",
    "FillResult",
    "FitReport",
    "GapClass",
    "GapScenarioConfig",
    "Hemisphere",
    "IncompletePolicy",
    "InsufficientTrainingDataError",
    "LeakageError",
    "MetricSubset",
    "Mode",
    "ModelError",
    "NotFittedError",
    "RFRConfig",
    "RFRGapFiller",
    "RFRModel",
    "RadiationClass",
    "Season",
    "TimeAxis",
    "TimestampError",
    "ValidationConfig",
    "ValidationFeatureSet",
    "__version__",
    "available_target_mask",
    "build_feature_matrix",
    "build_validation_features",
    "daily_flux_statistics",
    "detect_target_leakage",
    "elapsed_hours",
    "feature_names",
    "fill_column_names",
    "hide_target",
    "holdout_mask_from_intervals",
    "method_label",
    "observed_target_mask",
    "orf_pairing_differences",
    "prepare_time_index",
    "radiation_tag",
    "receptive_limiter_features",
    "require_no_target_leakage",
    "require_orf_pairing",
    "season_tag",
    "time_distance_hours",
]
