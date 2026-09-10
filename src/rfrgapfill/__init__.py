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
)
from rfrgapfill.features import (
    DailyStatistics,
    FeatureError,
    FeatureMatrix,
    build_feature_matrix,
    daily_flux_statistics,
    feature_names,
    radiation_tag,
    season_tag,
    time_distance_hours,
)
from rfrgapfill.fill import FillResult, RFRGapFiller
from rfrgapfill.gaps import (
    ArtificialGaps,
    GapError,
    GapScenarioGenerator,
    generate_artificial_gaps,
)
from rfrgapfill.metrics import (
    CoreMetrics,
    MetricError,
    SubsetMetrics,
    bias,
    core_metrics,
    daytime_mask,
    energy_balance_ratio,
    metrics_by_subset,
    metrics_from_config,
    nighttime_mask,
    r2,
    regression_slope,
    rmse,
)
from rfrgapfill.model import ModelError, RFRModel, TrainingReport, model_version
from rfrgapfill.provenance import environment, observed_mask, run_manifest
from rfrgapfill.time import (
    TIME_DISTANCE_HOURS,
    DuplicatePolicy,
    TimeAxis,
    TimestampError,
    elapsed_hours,
    prepare_time_index,
)
from rfrgapfill.validation import (
    SCENARIOS,
    EnergyBalance,
    GapClassMetrics,
    TargetValidation,
    ValidationError,
    ValidationReport,
    compare_receptive_limiter,
    validate_rfr,
)

try:  # pragma: no cover - trivial packaging fallback
    __version__ = _version("rfr-gapfill")
except PackageNotFoundError:  # pragma: no cover - source tree without install
    __version__ = "0.0.0.dev0"

#: Path-independent identifier recorded in run manifests and fill provenance.
PAPER_DOI = "10.1016/j.agrformet.2021.108777"

__all__ = [
    "DEFAULT_HYPERPARAMETER_GRID",
    "PAPER_DOI",
    "SCENARIOS",
    "TIME_DISTANCE_HOURS",
    "AllocationBasis",
    "ArtificialGaps",
    "BoundaryConvention",
    "CVStrategy",
    "ColumnMap",
    "ColumnMapError",
    "ConfigError",
    "CoreMetrics",
    "DailyStatisticStrategy",
    "DailyStatistics",
    "DuplicatePolicy",
    "EnergyBalance",
    "FeatureConfig",
    "FeatureError",
    "FeatureMatrix",
    "FeatureMode",
    "FillResult",
    "GapClass",
    "GapClassMetrics",
    "GapError",
    "GapScenarioConfig",
    "GapScenarioGenerator",
    "Hemisphere",
    "MetricError",
    "MetricSubset",
    "Mode",
    "ModelError",
    "RFRConfig",
    "RFRGapFiller",
    "RFRModel",
    "SubsetMetrics",
    "TargetValidation",
    "TimeAxis",
    "TimestampError",
    "TrainingReport",
    "ValidationConfig",
    "ValidationError",
    "ValidationReport",
    "__version__",
    "bias",
    "build_feature_matrix",
    "compare_receptive_limiter",
    "core_metrics",
    "daily_flux_statistics",
    "daytime_mask",
    "elapsed_hours",
    "energy_balance_ratio",
    "environment",
    "feature_names",
    "generate_artificial_gaps",
    "metrics_by_subset",
    "metrics_from_config",
    "model_version",
    "nighttime_mask",
    "observed_mask",
    "prepare_time_index",
    "r2",
    "radiation_tag",
    "regression_slope",
    "rmse",
    "run_manifest",
    "season_tag",
    "time_distance_hours",
    "validate_rfr",
]
