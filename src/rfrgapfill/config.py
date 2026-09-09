"""Validated configuration objects.

Implements :class:`RFRConfig`, :class:`FeatureConfig`, :class:`GapScenarioConfig`
and :class:`ValidationConfig`. Every ambiguity recorded in ``docs/method_spec.md``
(A1-A10) is reachable through a configuration field here, and incompatible options
are rejected at construction rather than in the middle of a fit.

All configuration objects are frozen dataclasses. They validate and normalise in
``__post_init__``, expose ``to_dict()`` for the run manifest, and are the only
place scientific constants live: no science module may define its own thresholds,
durations, or driver lists.

Provenance is marked on every default. ``paper`` means the article or its
supplements state it; ``default`` means the paper is silent and this is our
documented choice, cross-referenced to the ambiguity table in
``docs/method_spec.md``.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import timedelta
from enum import Enum
from types import MappingProxyType
from typing import Any, Final, cast

import pandas as pd

from rfrgapfill.schema import (
    CANONICAL_VARIABLES,
    EBR_VARIABLES,
    ColumnMap,
    ColumnMapError,
    ConfigError,
    GapClass,
    Hemisphere,
    Mode,
    coerce_enum,
)

__all__ = [
    "DEFAULT_DAYTIME_THRESHOLD",
    "DEFAULT_GAP_DURATIONS",
    "DEFAULT_GAP_MIX",
    "DEFAULT_HYPERPARAMETER_GRID",
    "DEFAULT_RADIATION_THRESHOLDS",
    "AllocationBasis",
    "BoundaryConvention",
    "CVStrategy",
    "ColumnMap",
    "ColumnMapError",
    "ConfigError",
    "FeatureConfig",
    "FeatureMode",
    "GapClass",
    "GapScenarioConfig",
    "Hemisphere",
    "MetricSubset",
    "Mode",
    "RFRConfig",
    "ValidationConfig",
]


# ---------------------------------------------------------------------------
# Enumerated choices
# ---------------------------------------------------------------------------


class FeatureMode(str, Enum):
    """How target-derived daily statistics are computed (method_spec.md 3.5, A6)."""

    #: Leakage-safe default: daily statistics see only observations visible to the
    #: model, and the artificial-gap mask is built before features are computed.
    PAPER_SAFE = "paper_safe"
    #: Reserved compatibility mode. Enabled only if implementation evidence for a
    #: different derivation in the historical ``fluxlib`` code is found.
    LEGACY_FLUXLIB = "legacy_fluxlib"

    @classmethod
    def coerce(cls, value: object) -> FeatureMode:
        """Return ``value`` as a :class:`FeatureMode`."""
        return coerce_enum(cls, value, field_name="feature_mode")


class BoundaryConvention(str, Enum):
    """Radiation-category boundary handling at exactly 10 and 100 W m-2 (A2)."""

    #: ``x < lo`` weak, ``lo <= x <= hi`` medium, ``x > hi`` strong. The default.
    MEDIUM_INCLUSIVE = "medium_inclusive"
    #: ``x <= lo`` weak, ``lo < x < hi`` medium, ``x >= hi`` strong.
    MEDIUM_EXCLUSIVE = "medium_exclusive"

    @classmethod
    def coerce(cls, value: object) -> BoundaryConvention:
        """Return ``value`` as a :class:`BoundaryConvention`."""
        return coerce_enum(cls, value, field_name="boundary_convention")


class AllocationBasis(str, Enum):
    """Interpretation of the 20/30/50 gap mix (method_spec.md 4.3, A3)."""

    #: The shares apply to the number of withheld records. The default.
    MISSING_RECORDS = "missing_records"
    #: The shares apply to the number of gap events.
    GAP_EVENTS = "gap_events"

    @classmethod
    def coerce(cls, value: object) -> AllocationBasis:
        """Return ``value`` as an :class:`AllocationBasis`."""
        return coerce_enum(cls, value, field_name="allocation_basis")


class CVStrategy(str, Enum):
    """Cross-validation strategy inside ``GridSearchCV`` (method_spec.md 5, A5)."""

    #: Conventional ``GridSearchCV`` folds on the training portion. The default.
    KFOLD = "kfold"
    #: Time-aware blocked folds. A labelled enhancement, not paper reproduction.
    TIME_SERIES_SPLIT = "time_series_split"

    @property
    def is_paper_default(self) -> bool:
        """Whether this strategy is the paper-faithful default rather than an enhancement."""
        return self is CVStrategy.KFOLD

    @classmethod
    def coerce(cls, value: object) -> CVStrategy:
        """Return ``value`` as a :class:`CVStrategy`."""
        return coerce_enum(cls, value, field_name="cv_strategy")


class MetricSubset(str, Enum):
    """Observation subsets metrics are reported over (method_spec.md 6.2)."""

    ALL = "all"
    DAYTIME = "daytime"
    NIGHTTIME = "nighttime"

    @classmethod
    def coerce(cls, value: object) -> MetricSubset:
        """Return ``value`` as a :class:`MetricSubset`."""
        return coerce_enum(cls, value, field_name="metric_subset")


# ---------------------------------------------------------------------------
# Documented defaults
# ---------------------------------------------------------------------------

#: Radiation-category thresholds in W m-2 (paper: 10 and 100; boundaries are A2).
DEFAULT_RADIATION_THRESHOLDS: Final[tuple[float, float]] = (10.0, 100.0)

#: Daytime threshold on downward shortwave radiation, W m-2 (paper).
DEFAULT_DAYTIME_THRESHOLD: Final[float] = 20.0

#: Nominal artificial-gap durations by class (paper).
DEFAULT_GAP_DURATIONS: Final[Mapping[GapClass, timedelta]] = MappingProxyType(
    {
        GapClass.SHORT: timedelta(hours=24),
        GapClass.LONG: timedelta(days=7),
        GapClass.VERY_LONG: timedelta(days=30),
    }
)

#: Nominal 20/30/50 gap mix (paper; the basis it applies to is A3).
DEFAULT_GAP_MIX: Final[Mapping[GapClass, float]] = MappingProxyType(
    {GapClass.SHORT: 0.20, GapClass.LONG: 0.30, GapClass.VERY_LONG: 0.50}
)

#: Documented default ``GridSearchCV`` grid (A1).
#:
#: The article states that hyperparameters were optimised with ``GridSearchCV`` but
#: does not enumerate the grid, so this is **our** default and must never be
#: described as paper exact. It is deliberately small enough to run on a decade of
#: half-hourly data. An archived ``fluxlib`` grid may be added later as a named
#: preset alongside it.
DEFAULT_HYPERPARAMETER_GRID: Final[Mapping[str, tuple[Any, ...]]] = MappingProxyType(
    {
        "max_features": (1.0, "sqrt"),
        "min_samples_leaf": (1, 5),
        "n_estimators": (100, 300),
    }
)

#: Estimator parameters the package controls itself; they may not appear in a grid.
_RESERVED_GRID_KEYS: Final[frozenset[str]] = frozenset({"random_state", "n_jobs", "verbose"})

#: A bare ``d`` day unit, which pandas >= 3 deprecates in favour of ``D``.
_DAY_UNIT_RE: Final[re.Pattern[str]] = re.compile(r"(?<=[0-9 ])d(?![a-zA-Z])")

#: numpy's seed range; ``random_state`` must fit in it to stay portable.
_MAX_SEED: Final[int] = 2**32 - 1


# ---------------------------------------------------------------------------
# Shared validation helpers
# ---------------------------------------------------------------------------


def _check_finite(value: object, *, field_name: str) -> float:
    """Return ``value`` as a finite float or raise :class:`ConfigError`."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{field_name} must be a number, got {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise ConfigError(f"{field_name} must be finite, got {number!r}")
    return number


def _check_fraction(value: object, *, field_name: str, exclusive: bool = False) -> float:
    """Return ``value`` as a fraction in [0, 1] (or (0, 1) when ``exclusive``)."""
    number = _check_finite(value, field_name=field_name)
    low_ok = number > 0.0 if exclusive else number >= 0.0
    high_ok = number < 1.0 if exclusive else number <= 1.0
    if not (low_ok and high_ok):
        bounds = "(0, 1)" if exclusive else "[0, 1]"
        raise ConfigError(f"{field_name} must lie in {bounds}, got {number!r}")
    return number


def _check_positive_int(value: object, *, field_name: str, minimum: int = 1) -> int:
    """Return ``value`` as an int at or above ``minimum``."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{field_name} must be an integer, got {value!r}")
    if value < minimum:
        raise ConfigError(f"{field_name} must be >= {minimum}, got {value}")
    return value


def _to_timedelta(value: object, *, field_name: str) -> timedelta:
    """Return ``value`` as a positive :class:`datetime.timedelta`.

    Accepts anything ``pandas.Timedelta`` understands (``"30min"``, ``"7d"``,
    ``timedelta(hours=24)``), so durations and cadences can be written the way users
    already write them for pandas.
    """
    if isinstance(value, (bool, int, float)):
        raise ConfigError(
            f"{field_name} must be a duration string or timedelta (e.g. '30min'), "
            f"not a bare number: {value!r}"
        )
    if isinstance(value, str):
        # pandas >= 3 deprecates the lowercase day unit, but '7d'/'30d' is how the
        # paper's gap classes are written; canonicalise rather than warn the user.
        candidate: str | timedelta = _DAY_UNIT_RE.sub("D", value)
    elif isinstance(value, timedelta):
        candidate = value
    else:
        raise ConfigError(
            f"{field_name} must be a duration string or timedelta (e.g. '30min'), "
            f"got {type(value).__name__}"
        )
    try:
        delta = pd.Timedelta(candidate)
    except (ValueError, TypeError) as exc:
        raise ConfigError(f"{field_name}={value!r} is not a valid duration: {exc}") from exc
    if delta != delta:  # NaT
        raise ConfigError(f"{field_name}={value!r} is not a valid duration")
    if delta <= pd.Timedelta(0):
        raise ConfigError(f"{field_name} must be a positive duration, got {value!r}")
    result: timedelta = delta.to_pytimedelta()
    return result


def _iso(delta: timedelta) -> str:
    """Return an ISO-8601 duration string for manifests."""
    return str(pd.Timedelta(delta).isoformat())


def _normalise_gap_class_mapping(
    value: object,
    *,
    field_name: str,
    default: Mapping[GapClass, Any],
) -> dict[GapClass, Any]:
    """Return ``value`` keyed by :class:`GapClass`, filling absent classes from ``default``."""
    if not isinstance(value, Mapping):
        raise ConfigError(
            f"{field_name} must be a mapping keyed by gap class, got {type(value).__name__}"
        )
    normalised: dict[GapClass, Any] = {}
    for key, entry in value.items():
        gap_class = GapClass.coerce(key)
        if gap_class in normalised:
            raise ConfigError(f"{field_name} names gap class {gap_class.value!r} more than once")
        normalised[gap_class] = entry
    return {gap_class: normalised.get(gap_class, default[gap_class]) for gap_class in GapClass}


# ---------------------------------------------------------------------------
# FeatureConfig
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FeatureConfig:
    """Receptive-limiter feature configuration (method_spec.md section 3).

    ``use_receptive_limiter=False`` yields the supplement's ORF benchmark: the same
    estimator family, grid, training rows and drivers, with sections 3.1-3.4
    omitted.
    """

    #: RFR when true; ORF benchmark when false (Supplementary Figure S1).
    use_receptive_limiter: bool = True
    #: Leakage policy for target-derived daily statistics (A6).
    feature_mode: FeatureMode | str = FeatureMode.PAPER_SAFE
    #: ``(weak/medium, medium/strong)`` shortwave thresholds in W m-2.
    radiation_thresholds: tuple[float, float] = DEFAULT_RADIATION_THRESHOLDS
    #: How values exactly on a radiation threshold are binned (A2).
    boundary_convention: BoundaryConvention | str = BoundaryConvention.MEDIUM_INCLUSIVE
    #: Minimum visible target observations for a day's statistics to be computed (A4).
    #: Days below this leave the statistics missing; they are never imputed.
    min_daily_observations: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.use_receptive_limiter, bool):
            raise ConfigError(
                f"use_receptive_limiter must be a bool, got {self.use_receptive_limiter!r}"
            )
        mode = FeatureMode.coerce(self.feature_mode)
        if mode is FeatureMode.LEGACY_FLUXLIB:
            raise ConfigError(
                "feature_mode='legacy_fluxlib' is reserved and not implemented: it may be "
                "enabled only once implementation evidence for a different daily-statistic "
                "derivation is found (docs/method_spec.md, ambiguity A6). Use 'paper_safe'."
            )
        object.__setattr__(self, "feature_mode", mode)
        object.__setattr__(
            self, "boundary_convention", BoundaryConvention.coerce(self.boundary_convention)
        )

        thresholds = self.radiation_thresholds
        if not isinstance(thresholds, Sequence) or isinstance(thresholds, str):
            raise ConfigError(f"radiation_thresholds must be a pair of numbers, got {thresholds!r}")
        if len(thresholds) != 2:
            raise ConfigError(
                f"radiation_thresholds must contain exactly 2 values, got {len(thresholds)}"
            )
        low = _check_finite(thresholds[0], field_name="radiation_thresholds[0]")
        high = _check_finite(thresholds[1], field_name="radiation_thresholds[1]")
        if not low < high:
            raise ConfigError(
                f"radiation_thresholds must be strictly increasing, got ({low}, {high})"
            )
        object.__setattr__(self, "radiation_thresholds", (low, high))
        object.__setattr__(
            self,
            "min_daily_observations",
            _check_positive_int(self.min_daily_observations, field_name="min_daily_observations"),
        )

    @property
    def mode(self) -> FeatureMode:
        """The validated :class:`FeatureMode` (narrowed from the input union)."""
        assert isinstance(self.feature_mode, FeatureMode)
        return self.feature_mode

    @property
    def convention(self) -> BoundaryConvention:
        """The validated :class:`BoundaryConvention` (narrowed from the input union)."""
        assert isinstance(self.boundary_convention, BoundaryConvention)
        return self.boundary_convention

    @property
    def requires_hemisphere(self) -> bool:
        """Whether a hemisphere is needed, i.e. whether the season feature is built."""
        return self.use_receptive_limiter

    def replace(self, **changes: Any) -> FeatureConfig:
        """Return a revalidated copy with ``changes`` applied."""
        return replace(self, **changes)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation for the run manifest."""
        return {
            "use_receptive_limiter": self.use_receptive_limiter,
            "feature_mode": self.mode.value,
            "radiation_thresholds": list(self.radiation_thresholds),
            "boundary_convention": self.convention.value,
            "min_daily_observations": self.min_daily_observations,
        }


# ---------------------------------------------------------------------------
# GapScenarioConfig
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GapScenarioConfig:
    """Artificial-gap scenario configuration (method_spec.md section 4).

    Defaults reproduce the paper's scenario: 25% of available observations withheld
    as 24-hour, 7-day and 30-day gaps mixed 20/30/50, each interval requiring at
    least 50% genuinely observed target values.
    """

    #: Fraction of available observations withheld in total (paper: 0.25).
    missing_fraction: float = 0.25
    #: Share of the withheld total per gap class (paper: 20/30/50).
    gap_mix: Mapping[GapClass | str, float] = cast(
        "Mapping[GapClass | str, float]", DEFAULT_GAP_MIX
    )
    #: Duration of each gap class (paper: 24 h, 7 d, 30 d).
    durations: Mapping[GapClass | str, timedelta | str] = cast(
        "Mapping[GapClass | str, timedelta | str]", DEFAULT_GAP_DURATIONS
    )
    #: Whether the mix applies to withheld records or to gap events (A3).
    allocation_basis: AllocationBasis | str = AllocationBasis.MISSING_RECORDS
    #: Minimum genuinely observed fraction inside a proposed interval (paper: 0.50).
    min_observed_fraction: float = 0.50
    #: Whether artificial intervals may overlap each other.
    allow_overlap: bool = False
    #: Bounded retries per gap before the scenario is reported unsatisfiable.
    max_attempts_per_gap: int = 100
    #: Whether NEE/H/LE share identical gap locations in joint validation (paper: yes).
    shared_gaps_across_targets: bool = True
    #: Tolerance on the achieved withheld fraction when reporting (A7).
    fraction_tolerance: float = 0.05
    #: Tolerance on the achieved per-class allocation when reporting (A7).
    mix_tolerance: float = 0.05

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "missing_fraction",
            _check_fraction(self.missing_fraction, field_name="missing_fraction", exclusive=True),
        )
        object.__setattr__(
            self,
            "min_observed_fraction",
            _check_fraction(self.min_observed_fraction, field_name="min_observed_fraction"),
        )
        object.__setattr__(
            self,
            "fraction_tolerance",
            _check_fraction(self.fraction_tolerance, field_name="fraction_tolerance"),
        )
        object.__setattr__(
            self,
            "mix_tolerance",
            _check_fraction(self.mix_tolerance, field_name="mix_tolerance"),
        )
        object.__setattr__(self, "allocation_basis", AllocationBasis.coerce(self.allocation_basis))
        for name in ("allow_overlap", "shared_gaps_across_targets"):
            if not isinstance(getattr(self, name), bool):
                raise ConfigError(f"{name} must be a bool, got {getattr(self, name)!r}")
        object.__setattr__(
            self,
            "max_attempts_per_gap",
            _check_positive_int(self.max_attempts_per_gap, field_name="max_attempts_per_gap"),
        )

        mix_input = _normalise_gap_class_mapping(
            self.gap_mix, field_name="gap_mix", default={gap: 0.0 for gap in GapClass}
        )
        mix = {
            gap_class: _check_fraction(share, field_name=f"gap_mix[{gap_class.value!r}]")
            for gap_class, share in mix_input.items()
        }
        total = math.fsum(mix.values())
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-6):
            raise ConfigError(
                "gap_mix shares must sum to 1.0 "
                f"(got {total!r} from "
                + ", ".join(f"{cls.value}={share}" for cls, share in mix.items())
                + ")"
            )
        object.__setattr__(self, "gap_mix", MappingProxyType(mix))

        duration_input = _normalise_gap_class_mapping(
            self.durations, field_name="durations", default=DEFAULT_GAP_DURATIONS
        )
        durations = {
            gap_class: _to_timedelta(value, field_name=f"durations[{gap_class.value!r}]")
            for gap_class, value in duration_input.items()
        }
        object.__setattr__(self, "durations", MappingProxyType(durations))

    # -- accessors -----------------------------------------------------------

    @property
    def basis(self) -> AllocationBasis:
        """The validated :class:`AllocationBasis` (narrowed from the input union)."""
        assert isinstance(self.allocation_basis, AllocationBasis)
        return self.allocation_basis

    @property
    def active_classes(self) -> tuple[GapClass, ...]:
        """Gap classes with a non-zero share, in class order."""
        return tuple(gap for gap in GapClass if float(self.gap_mix[gap]) > 0.0)

    def share(self, gap_class: GapClass | str) -> float:
        """Return the configured share of ``gap_class``."""
        return float(self.gap_mix[GapClass.coerce(gap_class)])

    def duration(self, gap_class: GapClass | str) -> timedelta:
        """Return the configured duration of ``gap_class``."""
        value = self.durations[GapClass.coerce(gap_class)]
        assert isinstance(value, timedelta)
        return value

    def replace(self, **changes: Any) -> GapScenarioConfig:
        """Return a revalidated copy with ``changes`` applied."""
        return replace(self, **changes)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation for the run manifest."""
        return {
            "missing_fraction": self.missing_fraction,
            "gap_mix": {gap.value: self.share(gap) for gap in GapClass},
            "durations": {gap.value: _iso(self.duration(gap)) for gap in GapClass},
            "allocation_basis": self.basis.value,
            "min_observed_fraction": self.min_observed_fraction,
            "allow_overlap": self.allow_overlap,
            "max_attempts_per_gap": self.max_attempts_per_gap,
            "shared_gaps_across_targets": self.shared_gaps_across_targets,
            "fraction_tolerance": self.fraction_tolerance,
            "mix_tolerance": self.mix_tolerance,
        }


# ---------------------------------------------------------------------------
# ValidationConfig
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ValidationConfig:
    """Artificial-gap validation and metric reporting (method_spec.md sections 4, 6)."""

    #: The artificial-gap scenario the test set is drawn from.
    gaps: GapScenarioConfig = field(default_factory=GapScenarioConfig)
    #: Daytime is ``shortwave > daytime_threshold`` W m-2 (paper: 20).
    daytime_threshold: float = DEFAULT_DAYTIME_THRESHOLD
    #: Subsets metrics are reported over. Aggregate-only reporting hides the weak
    #: nighttime skill documented in ``docs/supplement_benchmarks.md``.
    subsets: Sequence[MetricSubset | str] = (
        MetricSubset.ALL,
        MetricSubset.DAYTIME,
        MetricSubset.NIGHTTIME,
    )
    #: Whether core metrics are also reported per gap class.
    report_by_gap_class: bool = True
    #: Whether bias IQR is reported per gap class (method_spec.md 6.3).
    bias_iqr_by_gap_class: bool = True
    #: Whether the energy-balance ratio is computed for H and LE (method_spec.md 6.4).
    compute_energy_balance_ratio: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.gaps, GapScenarioConfig):
            raise ConfigError(f"gaps must be a GapScenarioConfig, got {type(self.gaps).__name__}")
        object.__setattr__(
            self,
            "daytime_threshold",
            _check_finite(self.daytime_threshold, field_name="daytime_threshold"),
        )
        if isinstance(self.subsets, str) or not isinstance(self.subsets, Sequence):
            raise ConfigError(f"subsets must be a sequence of metric subsets, got {self.subsets!r}")
        if not self.subsets:
            raise ConfigError("subsets must name at least one metric subset")
        seen: list[MetricSubset] = []
        for entry in self.subsets:
            subset = MetricSubset.coerce(entry)
            if subset not in seen:
                seen.append(subset)
        object.__setattr__(self, "subsets", tuple(seen))
        for name in (
            "report_by_gap_class",
            "bias_iqr_by_gap_class",
            "compute_energy_balance_ratio",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ConfigError(f"{name} must be a bool, got {getattr(self, name)!r}")

    @property
    def metric_subsets(self) -> tuple[MetricSubset, ...]:
        """The validated subsets (narrowed from the input union)."""
        subsets = self.subsets
        assert isinstance(subsets, tuple)
        return subsets

    def replace(self, **changes: Any) -> ValidationConfig:
        """Return a revalidated copy with ``changes`` applied."""
        return replace(self, **changes)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation for the run manifest."""
        return {
            "gaps": self.gaps.to_dict(),
            "daytime_threshold": self.daytime_threshold,
            "subsets": [subset.value for subset in self.metric_subsets],
            "report_by_gap_class": self.report_by_gap_class,
            "bias_iqr_by_gap_class": self.bias_iqr_by_gap_class,
            "compute_energy_balance_ratio": self.compute_energy_balance_ratio,
        }


# ---------------------------------------------------------------------------
# RFRConfig
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RFRConfig:
    """Top-level run configuration for one site.

    Holds the driver mode, the site's time and location metadata, the estimator
    settings, and the nested feature, validation and column-mapping configuration.
    Models are fitted per site and per target, so one instance describes one site's
    run; the target itself is passed to ``fit``.
    """

    #: ``RFR3`` or ``RFR10``. Required: there is no silent default driver set.
    mode: Mode | str
    #: Time step of the input series. ``None`` infers it from the timestamps.
    frequency: timedelta | str | None = None
    #: Hemisphere for the season feature. Overrides ``latitude`` when both are given.
    hemisphere: Hemisphere | str | None = None
    #: Site latitude in degrees, used to infer the hemisphere when it is not given.
    latitude: float | None = None
    #: Site identifier recorded in run manifests and multi-site reports.
    site_id: str | None = None
    #: Seed for every stochastic component (gap sampling, forest, CV).
    random_state: int = 42
    #: ``n_jobs`` passed to the estimator and the grid search. ``None`` means 1.
    n_jobs: int | None = None
    #: ``GridSearchCV`` grid. Our documented default, never "paper exact" (A1).
    hyperparameter_grid: Mapping[str, Sequence[Any]] = DEFAULT_HYPERPARAMETER_GRID
    #: Cross-validation strategy inside the grid search (A5).
    cv_strategy: CVStrategy | str = CVStrategy.KFOLD
    #: Number of cross-validation folds.
    cv_folds: int = 5
    #: Whether folds are shuffled. Incompatible with ``time_series_split``.
    cv_shuffle: bool = False
    #: QC flag values that count as genuinely observed target measurements. FLUXNET
    #: uses 0 for measured; anything else was already gap-filled before ingestion.
    observed_qc_values: Sequence[int] = (0,)
    #: Receptive-limiter configuration.
    features: FeatureConfig = field(default_factory=FeatureConfig)
    #: Artificial-gap validation configuration.
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    #: Canonical-name to input-column mapping. May also be supplied to ``fit``.
    column_map: ColumnMap | Mapping[str, str] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", Mode.coerce(self.mode))

        if self.frequency is not None:
            object.__setattr__(
                self, "frequency", _to_timedelta(self.frequency, field_name="frequency")
            )

        if self.hemisphere is not None:
            object.__setattr__(self, "hemisphere", Hemisphere.coerce(self.hemisphere))
        if self.latitude is not None:
            # Validates range and type; the result is recomputed on demand.
            Hemisphere.from_latitude(self.latitude)
            object.__setattr__(self, "latitude", float(self.latitude))

        if self.site_id is not None and (
            not isinstance(self.site_id, str) or not self.site_id.strip()
        ):
            raise ConfigError(f"site_id must be a non-empty string or None, got {self.site_id!r}")

        if isinstance(self.random_state, bool) or not isinstance(self.random_state, int):
            raise ConfigError(
                "random_state must be an integer: reproducibility is a requirement of this "
                f"package, so an unseeded run is not offered. Got {self.random_state!r}"
            )
        if not 0 <= self.random_state <= _MAX_SEED:
            raise ConfigError(f"random_state must lie in [0, {_MAX_SEED}], got {self.random_state}")

        if self.n_jobs is not None:
            if isinstance(self.n_jobs, bool) or not isinstance(self.n_jobs, int):
                raise ConfigError(f"n_jobs must be an integer or None, got {self.n_jobs!r}")
            if self.n_jobs == 0:
                raise ConfigError("n_jobs must be non-zero (-1 uses all processors)")

        if not isinstance(self.features, FeatureConfig):
            raise ConfigError(
                f"features must be a FeatureConfig, got {type(self.features).__name__}"
            )
        if not isinstance(self.validation, ValidationConfig):
            raise ConfigError(
                f"validation must be a ValidationConfig, got {type(self.validation).__name__}"
            )

        object.__setattr__(self, "cv_strategy", CVStrategy.coerce(self.cv_strategy))
        object.__setattr__(
            self, "cv_folds", _check_positive_int(self.cv_folds, field_name="cv_folds", minimum=2)
        )
        if not isinstance(self.cv_shuffle, bool):
            raise ConfigError(f"cv_shuffle must be a bool, got {self.cv_shuffle!r}")
        if self.cv_shuffle and self.cv_strategy is CVStrategy.TIME_SERIES_SPLIT:
            raise ConfigError(
                "cv_shuffle=True is incompatible with cv_strategy='time_series_split': "
                "shuffling destroys the temporal blocking the strategy exists to preserve"
            )

        object.__setattr__(
            self, "hyperparameter_grid", _validate_hyperparameter_grid(self.hyperparameter_grid)
        )

        qc_values = self.observed_qc_values
        if isinstance(qc_values, (str, bytes)) or not isinstance(qc_values, Sequence):
            raise ConfigError(
                f"observed_qc_values must be a sequence of integers, got {qc_values!r}"
            )
        cleaned_qc: list[int] = []
        for value in qc_values:
            if isinstance(value, bool) or not isinstance(value, int):
                raise ConfigError(f"observed_qc_values must contain integers, got {value!r}")
            if value not in cleaned_qc:
                cleaned_qc.append(value)
        if not cleaned_qc:
            raise ConfigError(
                "observed_qc_values must name at least one flag value; an empty set would "
                "treat every measurement as pre-filled"
            )
        object.__setattr__(self, "observed_qc_values", tuple(cleaned_qc))

        object.__setattr__(self, "column_map", ColumnMap.coerce(self.column_map))
        assert isinstance(self.column_map, ColumnMap)
        if len(self.column_map) > 0:
            self.require_column_map(self.column_map)

        if self.features.requires_hemisphere and self.hemisphere is None and self.latitude is None:
            raise ConfigError(
                "the season feature needs a hemisphere: supply hemisphere='north'|'south' or "
                "latitude. There is no silent default (docs/method_spec.md 3.3, A9). Set "
                "features=FeatureConfig(use_receptive_limiter=False) for the ORF benchmark, "
                "which builds no season feature."
            )

    # -- accessors -----------------------------------------------------------

    @property
    def rfr_mode(self) -> Mode:
        """The validated :class:`Mode` (narrowed from the input union)."""
        assert isinstance(self.mode, Mode)
        return self.mode

    @property
    def cv(self) -> CVStrategy:
        """The validated :class:`CVStrategy` (narrowed from the input union)."""
        assert isinstance(self.cv_strategy, CVStrategy)
        return self.cv_strategy

    @property
    def columns(self) -> ColumnMap:
        """The validated :class:`ColumnMap` (empty when none was supplied)."""
        assert isinstance(self.column_map, ColumnMap)
        return self.column_map

    @property
    def time_step(self) -> timedelta | None:
        """The configured time step, or ``None`` when it is inferred from timestamps."""
        frequency = self.frequency
        assert frequency is None or isinstance(frequency, timedelta)
        return frequency

    @property
    def drivers(self) -> tuple[str, ...]:
        """Canonical driver variables required by the selected mode."""
        return self.rfr_mode.drivers

    @property
    def is_paper_faithful(self) -> bool:
        """Whether every option is the paper-faithful default rather than an enhancement.

        False as soon as a labelled enhancement is enabled (time-aware CV, shuffled
        folds) or the receptive limiter is switched off for the ORF benchmark.
        """
        return (
            self.features.use_receptive_limiter
            and self.features.mode is FeatureMode.PAPER_SAFE
            and self.cv.is_paper_default
            and not self.cv_shuffle
        )

    def resolve_hemisphere(self) -> Hemisphere:
        """Return the hemisphere, preferring an explicit value over ``latitude`` (A9)."""
        if self.hemisphere is not None:
            assert isinstance(self.hemisphere, Hemisphere)
            return self.hemisphere
        if self.latitude is not None:
            return Hemisphere.from_latitude(self.latitude)
        raise ConfigError("no hemisphere available: supply hemisphere='north'|'south' or latitude")

    @property
    def hemisphere_source(self) -> str | None:
        """Where the hemisphere came from: ``"explicit"``, ``"latitude"`` or ``None``."""
        if self.hemisphere is not None:
            return "explicit"
        if self.latitude is not None:
            return "latitude"
        return None

    # -- requirements --------------------------------------------------------

    def required_variables(self, *, include_ebr: bool = False) -> tuple[str, ...]:
        """Return the canonical variables this run needs, in specification order.

        ``include_ebr`` adds net radiation and soil heat flux, which the
        energy-balance ratio needs for H and LE even under RFR3.
        """
        required = list(self.drivers)
        if include_ebr:
            required += [name for name in EBR_VARIABLES if name not in required]
        return tuple(name for name in CANONICAL_VARIABLES if name in required)

    def require_column_map(
        self,
        column_map: ColumnMap | Mapping[str, str] | None = None,
        *,
        include_ebr: bool = False,
    ) -> ColumnMap:
        """Return ``column_map`` after checking it maps every variable this run needs.

        Falls back to the configured mapping when called without an argument. Raises
        :class:`~rfrgapfill.schema.ColumnMapError` naming every unmapped variable.
        """
        resolved = self.columns if column_map is None else ColumnMap.coerce(column_map)
        resolved.require(
            self.required_variables(include_ebr=include_ebr),
            context=f"mode={self.rfr_mode.value}",
        )
        return resolved

    def replace(self, **changes: Any) -> RFRConfig:
        """Return a revalidated copy with ``changes`` applied."""
        return replace(self, **changes)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation for the run manifest."""
        hemisphere = self.hemisphere
        return {
            "mode": self.rfr_mode.value,
            "frequency": None if self.time_step is None else _iso(self.time_step),
            "hemisphere": None if hemisphere is None else Hemisphere.coerce(hemisphere).value,
            "latitude": self.latitude,
            "resolved_hemisphere": (
                self.resolve_hemisphere().value if self.hemisphere_source else None
            ),
            "hemisphere_source": self.hemisphere_source,
            "site_id": self.site_id,
            "random_state": self.random_state,
            "n_jobs": self.n_jobs,
            "hyperparameter_grid": {
                key: list(values) for key, values in self.hyperparameter_grid.items()
            },
            "cv_strategy": self.cv.value,
            "cv_folds": self.cv_folds,
            "cv_shuffle": self.cv_shuffle,
            "observed_qc_values": list(self.observed_qc_values),
            "features": self.features.to_dict(),
            "validation": self.validation.to_dict(),
            "column_map": self.columns.to_dict(),
            "is_paper_faithful": self.is_paper_faithful,
        }


def _validate_hyperparameter_grid(grid: object) -> Mapping[str, tuple[Any, ...]]:
    """Return ``grid`` normalised, or raise :class:`ConfigError`.

    Keys are checked against ``RandomForestRegressor``'s parameters so a typo fails
    here rather than after the first fold. Parameters the package sets itself
    (``random_state``, ``n_jobs``, ``verbose``) are rejected outright.
    """
    if not isinstance(grid, Mapping):
        raise ConfigError(
            f"hyperparameter_grid must be a mapping of parameter -> values, "
            f"got {type(grid).__name__}"
        )
    if not grid:
        raise ConfigError("hyperparameter_grid must not be empty")

    normalised: dict[str, tuple[Any, ...]] = {}
    for key, values in grid.items():
        if not isinstance(key, str) or not key.isidentifier():
            raise ConfigError(f"hyperparameter_grid keys must be parameter names, got {key!r}")
        if key in _RESERVED_GRID_KEYS:
            raise ConfigError(
                f"hyperparameter_grid must not contain {key!r}: the package sets it from "
                "the run configuration"
            )
        if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
            raise ConfigError(
                f"hyperparameter_grid[{key!r}] must be a sequence of candidate values, "
                f"got {values!r}"
            )
        if len(values) == 0:
            raise ConfigError(f"hyperparameter_grid[{key!r}] must offer at least one value")
        normalised[key] = tuple(values)

    known = _estimator_parameters()
    unknown = sorted(set(normalised) - known) if known else []
    if unknown:
        raise ConfigError(
            f"hyperparameter_grid names parameter(s) RandomForestRegressor does not accept: "
            f"{', '.join(unknown)}"
        )
    return MappingProxyType(dict(sorted(normalised.items())))


def _estimator_parameters() -> frozenset[str]:
    """Return the parameter names of ``RandomForestRegressor``.

    Imported lazily so ``import rfrgapfill`` does not pay for scikit-learn, and
    tolerant of an unavailable estimator so configuration still validates.
    """
    try:
        from sklearn.ensemble import RandomForestRegressor
    except ImportError:  # pragma: no cover - scikit-learn is a hard dependency
        return frozenset()
    return frozenset(RandomForestRegressor().get_params())
