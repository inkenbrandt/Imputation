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
    R2Definition,
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
from rfrgapfill.gaps import (
    GAP_MANIFEST_COLUMNS,
    ArtificialGap,
    GapAllocation,
    GapError,
    GapManifest,
    GapScenarioGenerator,
    GapScenarioWarning,
    allocate_gaps,
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
from rfrgapfill.metrics import (
    CoreMetrics,
    EnergyBalanceComparison,
    MetricError,
    bias,
    compare_energy_balance,
    core_metrics,
    energy_balance_ratio,
    r2,
    regression_slope,
    rmse,
)
from rfrgapfill.model import (
    FitReport,
    IncompletePolicy,
    InsufficientTrainingDataError,
    ModelError,
    NotFittedError,
    RFRModel,
)
from rfrgapfill.provenance import (
    MANIFEST_FORMAT,
    MANIFEST_VERSION,
    REQUIRED_FIELDS,
    ProvenanceError,
    RowCounts,
    RunKind,
    RunManifest,
    ambiguity_choices,
    environment_versions,
    load_manifest,
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
    "GAP_MANIFEST_COLUMNS",
    "MANIFEST_FORMAT",
    "MANIFEST_VERSION",
    "PAPER_DOI",
    "RADIATION_CATEGORY",
    "REQUIRED_FIELDS",
    "SEASON",
    "TIME_DISTANCE_HOURS",
    "AllocationBasis",
    "ArtificialGap",
    "BoundaryConvention",
    "CVStrategy",
    "ColumnMap",
    "ColumnMapError",
    "ConfigError",
    "CoreMetrics",
    "DailyStatisticStrategy",
    "DuplicatePolicy",
    "EnergyBalanceComparison",
    "FeatureConfig",
    "FeatureError",
    "FeatureMode",
    "FillError",
    "FillMethod",
    "FillReport",
    "FillResult",
    "FitReport",
    "GapAllocation",
    "GapClass",
    "GapError",
    "GapManifest",
    "GapScenarioConfig",
    "GapScenarioGenerator",
    "GapScenarioWarning",
    "Hemisphere",
    "IncompletePolicy",
    "InsufficientTrainingDataError",
    "LeakageError",
    "MetricError",
    "MetricSubset",
    "Mode",
    "ModelError",
    "NotFittedError",
    "ProvenanceError",
    "R2Definition",
    "RFRConfig",
    "RFRGapFiller",
    "RFRModel",
    "RadiationClass",
    "RowCounts",
    "RunKind",
    "RunManifest",
    "Season",
    "TimeAxis",
    "TimestampError",
    "ValidationConfig",
    "ValidationFeatureSet",
    "__version__",
    "allocate_gaps",
    "ambiguity_choices",
    "available_target_mask",
    "bias",
    "build_feature_matrix",
    "build_validation_features",
    "compare_energy_balance",
    "core_metrics",
    "daily_flux_statistics",
    "detect_target_leakage",
    "elapsed_hours",
    "energy_balance_ratio",
    "environment_versions",
    "feature_names",
    "fill_column_names",
    "hide_target",
    "holdout_mask_from_intervals",
    "load_manifest",
    "method_label",
    "observed_target_mask",
    "orf_pairing_differences",
    "prepare_time_index",
    "r2",
    "radiation_tag",
    "receptive_limiter_features",
    "regression_slope",
    "require_no_target_leakage",
    "require_orf_pairing",
    "rmse",
    "season_tag",
    "time_distance_hours",
]
