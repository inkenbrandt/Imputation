"""End-to-end artificial-gap validation (method_spec.md sections 4-6; Step 13).

The layer that turns the pieces below it into the paper's experiment. One call
places the artificial gaps, hides their truth, builds leakage-safe features,
fits, predicts the withheld intervals and scores them::

    report = validate_rfr(df, config=config, targets=["NEE", "H", "LE"])
    report.to_frame()          # target x gap class x subset, tidy
    report["LE"].manifest()    # the run manifest of that arm

and performs the steps in the order ``docs/method_spec.md`` section 3.5
requires, because the order *is* the protection:

1. the gap scenario places its intervals over the genuinely observed values;
2. the holdout mask is derived from those intervals - before any target-derived
   feature exists;
3. :func:`~rfrgapfill.leakage.build_validation_features` hides the held-out
   values and computes the daily statistics from what remains visible;
4. the model is fitted on the remaining eligible observations - the ~75% of
   Figure 2, never a random row-wise split;
5. the withheld rows are predicted and scored against the truth that was kept
   back for exactly that purpose, and for nothing else.

What this module adds to those pieces is the *bookkeeping* the paper's tables
need: the same predictions scored over all rows, daytime rows and nighttime rows
(section 6.2), again per gap class (section 6.3), the spread of bias across the
gaps of a class, and the energy-balance comparison of section 6.4 when H and LE
were validated together.

Three properties are worth stating outright, since a validation number is only
as good as what produced it:

* **the caller's frame is never modified.** Every stage works on a copy on a
  validated time axis, and the observed values that come back out are the ones
  that went in.
* **the gap locations are shared.** One manifest, one mask, every target - the
  paper's joint NEE/H/LE rule (section 4.4) - so the arms are comparable to each
  other and to the ORF benchmark scored on the same mask (section 3.6).
* **a run blocked by ambiguity A4 says so.** Under the default
  ``daily_statistic_strategy="missing"`` a gap covering a whole calendar day
  leaves every row of that day without daily statistics, so the 7-day and 30-day
  classes yield no complete feature rows and receive no predictions at all. That
  is reported as a :class:`ValidationWarning` and visible in the row accounting
  rather than arriving as a mysteriously empty gap class.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd

from rfrgapfill.config import DEFAULT_DAYTIME_THRESHOLD, MetricSubset, Mode, RFRConfig
from rfrgapfill.fill import method_label
from rfrgapfill.gaps import GapManifest, GapScenarioGenerator, Shortfall
from rfrgapfill.leakage import (
    ValidationFeatureSet,
    build_validation_features,
    require_no_target_leakage,
)
from rfrgapfill.metrics import (
    CoreMetrics,
    EnergyBalanceComparison,
    compare_energy_balance,
    core_metrics,
)
from rfrgapfill.model import IncompletePolicy, RFRModel
from rfrgapfill.provenance import RunManifest
from rfrgapfill.schema import (
    EBR_VARIABLES,
    NET_RADIATION,
    SHORTWAVE,
    SOIL_HEAT_FLUX,
    ColumnMap,
    ConfigError,
    FrozenRecord,
    GapClass,
)
from rfrgapfill.time import DuplicatePolicy, TimeAxis, as_datetime_index, prepare_time_index

__all__ = [
    "ENERGY_BALANCE_TABLE_COLUMNS",
    "BiasSpread",
    "EnergyBalanceCheck",
    "TargetValidation",
    "ValidationError",
    "ValidationReport",
    "ValidationWarning",
    "daytime_mask",
    "gap_class_labels",
    "subset_masks",
    "validate_rfr",
]

#: The two fluxes the energy-balance ratio is defined for (method_spec.md 6.4).
SENSIBLE_HEAT: str = "H"
LATENT_HEAT: str = "LE"


class ValidationError(RuntimeError):
    """Raised when an artificial-gap validation run cannot be assembled."""


class ValidationWarning(UserWarning):
    """Warns that a run completed but scored nothing it withheld.

    Not an error: a run whose gaps produced no complete feature rows is a
    legitimate consequence of ambiguity A4's default, and it must not pass
    quietly, because "no metrics for the 30-day class" reads like a defect and is
    actually a configuration choice (method_spec.md section 3.4).
    """


# ---------------------------------------------------------------------------
# Row selections: subsets and gap classes
# ---------------------------------------------------------------------------


def daytime_mask(
    shortwave: pd.Series,
    *,
    threshold: float = DEFAULT_DAYTIME_THRESHOLD,
) -> pd.Series:
    """Return the daytime rows: downward shortwave radiation above ``threshold``.

    The paper's definition (method_spec.md section 6.2) is ``SW_IN > 20 W m-2``,
    so the boundary value itself is night. A row whose radiation is missing is
    neither day nor night - see :func:`subset_masks`.
    """
    values = np.asarray(pd.to_numeric(shortwave, errors="coerce").to_numpy(), dtype=float)
    day: pd.Series = pd.Series(values > float(threshold), index=shortwave.index, name="daytime")
    return day


def subset_masks(
    shortwave: pd.Series,
    *,
    threshold: float = DEFAULT_DAYTIME_THRESHOLD,
) -> Mapping[MetricSubset, pd.Series]:
    """Return the ``all``/``daytime``/``nighttime`` row masks (method_spec.md 6.2).

    ``daytime`` is ``SW_IN > threshold`` and ``nighttime`` is
    ``SW_IN <= threshold``, so the two partition every row whose radiation is
    known. A row with **missing** radiation belongs to ``all`` and to neither of
    the others: the paper's split is defined by a measurement, and putting an
    unknown row on one side of it would silently invent that measurement.
    """
    values = np.asarray(pd.to_numeric(shortwave, errors="coerce").to_numpy(), dtype=float)
    index = shortwave.index
    return MappingProxyType(
        {
            MetricSubset.ALL: pd.Series(True, index=index, name="all"),
            MetricSubset.DAYTIME: pd.Series(values > float(threshold), index=index, name="daytime"),
            MetricSubset.NIGHTTIME: pd.Series(
                values <= float(threshold), index=index, name="nighttime"
            ),
        }
    )


def gap_class_labels(gaps: GapManifest, values: object) -> pd.Series:
    """Return the gap class each timestamp falls in, or ``None`` outside every gap.

    The label metrics are grouped by for section 6.3's per-class reporting, and a
    tidy column in its own right: joined onto a prediction frame it says which
    duration class produced each scored row.
    """
    if not isinstance(gaps, GapManifest):
        raise ValidationError(f"gaps must be a GapManifest, got {type(gaps).__name__}")
    index = as_datetime_index(values, field_name="timestamps")
    # `np.full(..., None)` rather than a scalar `None`, which pandas would turn
    # into NaN: an unlabelled row is *absent from every gap*, not a missing
    # number, and `.isna()` recognises both.
    labels: pd.Series = pd.Series(
        np.full(len(index), None, dtype=object), index=index, dtype=object, name="gap_class"
    )
    for gap in gaps:
        # Half-open [start, end), the convention every other interval in this
        # package uses, so adjacent gaps never claim the same row.
        labels.loc[(index >= gap.start) & (index < gap.end)] = gap.gap_class.value
    return labels


# ---------------------------------------------------------------------------
# Bias spread within a gap class
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BiasSpread(FrozenRecord):
    """How bias is distributed across the individual gaps of one duration class.

    One bias per placed interval - not per row - because that is the quantity
    Supplementary Table S8 spreads: a single gap the model drifts through is one
    bad fill, however many half hours it covers. This is the single-site analogue
    of the published cross-site IQR; the cross-site version belongs to the
    multi-site reproduction layer and is not this.

    Every field is ``None`` where the quantity is undefined, matching
    :mod:`rfrgapfill.metrics`: one scored gap has a bias but no spread, and none
    at all has neither.
    """

    #: Gaps of this class that had at least one scored row.
    n_gaps: int
    #: Gaps of this class the scenario placed, scored or not.
    n_gaps_offered: int
    #: 25th percentile of the per-gap biases.
    q1: float | None
    #: Median per-gap bias.
    median: float | None
    #: 75th percentile of the per-gap biases.
    q3: float | None
    #: ``q3 - q1``. Needs at least two scored gaps to exist.
    iqr: float | None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary for the run manifest."""
        return {
            "n_gaps": self.n_gaps,
            "n_gaps_offered": self.n_gaps_offered,
            "q1": self.q1,
            "median": self.median,
            "q3": self.q3,
            "iqr": self.iqr,
        }


def _bias_spread(
    gaps: GapManifest,
    gap_class: GapClass,
    *,
    measured: pd.Series,
    predicted: pd.Series,
    scored: np.ndarray,
) -> BiasSpread:
    """Return the spread of per-gap bias within ``gap_class``."""
    placed = gaps.by_class(gap_class)
    index = measured.index
    truth = measured.to_numpy(dtype=float)
    filled = predicted.to_numpy(dtype=float)
    biases: list[float] = []
    for gap in placed:
        inside = (index >= gap.start) & (index < gap.end) & scored
        if not inside.any():
            continue
        error = filled[inside] - truth[inside]
        error = error[np.isfinite(error)]
        if error.size:
            biases.append(float(error.mean()))
    if not biases:
        return BiasSpread(
            n_gaps=0, n_gaps_offered=len(placed), q1=None, median=None, q3=None, iqr=None
        )
    values = np.asarray(biases, dtype=float)
    # Linear interpolation between order statistics, the convention ambiguity
    # A11 settles for every quantile this package reports.
    q1, median, q3 = (float(value) for value in np.percentile(values, (25.0, 50.0, 75.0)))
    return BiasSpread(
        n_gaps=int(values.size),
        n_gaps_offered=len(placed),
        q1=q1,
        median=median,
        q3=q3,
        iqr=q3 - q1 if values.size > 1 else None,
    )


# ---------------------------------------------------------------------------
# The energy-balance check (method_spec.md 6.4; Step 19)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EnergyBalanceCheck(FrozenRecord):
    """The paper's independent energy-balance check, over the artificial gaps.

    ``EBR = sum(H + LE) / sum(NETRAD - G)`` computed twice on the withheld rows -
    once from the measured H and LE, once from the values the model put there -
    over every withheld row and again within each gap class. A fill that
    reproduces the fluxes should leave closure where the measurements had it, and
    this is the one check that does not score the model against its own target.

    Every withheld row is *offered*; a row enters both ratios only where the
    measured H and LE, both predictions, NETRAD and G are all present, and each
    :class:`~rfrgapfill.metrics.EnergyBalanceComparison` counts what each missing
    component cost. A measured flux that arrived already gap-filled counts as
    missing: it was never scored against, and it is not closure evidence either.
    """

    #: The target column playing H.
    sensible_heat: str
    #: The target column playing LE.
    latent_heat: str
    #: The comparison over every withheld row.
    overall: EnergyBalanceComparison
    #: The comparison within each gap class, when the run reports by gap class.
    by_gap_class: Mapping[GapClass, EnergyBalanceComparison]

    def __post_init__(self) -> None:
        object.__setattr__(self, "by_gap_class", MappingProxyType(dict(self.by_gap_class)))

    @property
    def targets(self) -> tuple[str, str]:
        """The ``(H, LE)`` target columns the check was computed from."""
        return (self.sensible_heat, self.latent_heat)

    def comparison(self, gap_class: GapClass | str | None = None) -> EnergyBalanceComparison:
        """Return the comparison over every withheld row, or within one gap class."""
        if gap_class is None:
            return self.overall
        wanted = GapClass.coerce(gap_class)
        if wanted not in self.by_gap_class:
            raise ValidationError(
                f"no energy-balance check for gap class {wanted.value!r}: this run reported "
                + (
                    ", ".join(cls.value for cls in self.by_gap_class)
                    or "no gap classes at all (report_by_gap_class=False)"
                )
            )
        return self.by_gap_class[wanted]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary for the run manifest.

        The overall comparison's fields sit at the top level, so a reader after
        only ``measured_ebr`` and ``filled_ebr`` need not know about gap classes.
        """
        return {
            "sensible_heat": self.sensible_heat,
            "latent_heat": self.latent_heat,
            **self.overall.to_dict(),
            "by_gap_class": {
                gap_class.value: comparison.to_dict()
                for gap_class, comparison in self.by_gap_class.items()
            },
        }


#: Column order of every ``energy_balance_frame``.
ENERGY_BALANCE_TABLE_COLUMNS: tuple[str, ...] = (
    "method",
    "mode",
    "sensible_heat",
    "latent_heat",
    "gap_class",
    "n",
    "n_offered",
    "n_missing_measured",
    "n_missing_filled",
    "n_missing_available_energy",
    "available_energy",
    "measured_ebr",
    "filled_ebr",
    "difference",
)


# ---------------------------------------------------------------------------
# One target's result
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class TargetValidation:
    """What one arm produced for one target: predictions, metrics and provenance.

    Equality is identity-based (``eq=False``): the record holds pandas objects,
    whose ``==`` is element-wise.

    :attr:`metrics` answers "how did this arm do"; :attr:`metrics_by_gap_class`
    answers the question the paper is actually about - whether it stays stable as
    the gap grows - and :attr:`features` carries the row accounting behind both.
    """

    #: The target flux column this result is for.
    target: str
    #: The arm that produced it: ``RFR3``, ``RFR10``, ``ORF3`` or ``ORF10``.
    method: str
    #: The configuration the run used.
    config: RFRConfig
    #: The fitted model, carrying its grid, best parameters and row accounting.
    model: RFRModel
    #: The leakage-safe feature set, carrying the masks and the untouched truth.
    features: ValidationFeatureSet
    #: Predictions on the run's full time axis; missing outside the artificial gaps.
    predictions: pd.Series
    #: Core metrics per subset, over every scored row (method_spec.md 6.1-6.2).
    metrics: Mapping[MetricSubset, CoreMetrics]
    #: Core metrics per gap class and subset (method_spec.md 6.3).
    metrics_by_gap_class: Mapping[GapClass, Mapping[MetricSubset, CoreMetrics]]
    #: Spread of per-gap bias within each class, when it was asked for.
    bias_spread: Mapping[GapClass, BiasSpread]
    #: The intervals that were withheld.
    gaps: GapManifest
    #: The validated time axis the run ran on.
    time_axis: TimeAxis
    #: The mapping from canonical variables to this frame's columns.
    column_map: ColumnMap
    #: The QC column separating measured values from pre-filled ones.
    qc_column: str | None = None
    #: The run's energy-balance check, carried by its H and LE results only.
    energy_balance_check: EnergyBalanceCheck | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "metrics", MappingProxyType(dict(self.metrics)))
        object.__setattr__(
            self,
            "metrics_by_gap_class",
            MappingProxyType(
                {
                    gap_class: MappingProxyType(dict(by_subset))
                    for gap_class, by_subset in self.metrics_by_gap_class.items()
                }
            ),
        )
        object.__setattr__(self, "bias_spread", MappingProxyType(dict(self.bias_spread)))

    # -- accessors -----------------------------------------------------------

    @property
    def overall(self) -> CoreMetrics:
        """The four core metrics over every scored row, of any gap class."""
        return self.metrics[MetricSubset.ALL]

    @property
    def scored_rows(self) -> int:
        """Rows that carried both a withheld measurement and a prediction."""
        return self.overall.n

    @property
    def withheld_rows(self) -> int:
        """Rows the scenario withheld, whether or not they could be scored."""
        return int(self.features.holdout_mask.sum())

    def metric(
        self,
        *,
        subset: MetricSubset | str = MetricSubset.ALL,
        gap_class: GapClass | str | None = None,
    ) -> CoreMetrics:
        """Return one cell of the metric table.

        ``gap_class=None`` is every scored row; a class name restricts to the rows
        that class's intervals withheld.
        """
        chosen = MetricSubset.coerce(subset)
        if gap_class is None:
            return self.metrics[chosen]
        wanted = GapClass.coerce(gap_class)
        if wanted not in self.metrics_by_gap_class:
            available = ", ".join(sorted(cls.value for cls in self.metrics_by_gap_class))
            raise ValidationError(
                f"no metrics for gap class {wanted.value!r}: this run reported "
                + (available or "no gap classes at all (report_by_gap_class=False)")
            )
        return self.metrics_by_gap_class[wanted][chosen]

    def predicted_holdout(self) -> pd.Series:
        """The predictions inside the artificial gaps, unpredicted rows included."""
        values: pd.Series = self.predictions.loc[self.features.holdout_mask.to_numpy()]
        return values

    # -- tables and export ---------------------------------------------------

    def to_frame(self) -> pd.DataFrame:
        """Return the tidy metric table for this target.

        One row per gap class and subset, with ``gap_class="all"`` for the rows
        that pool every class - the shape Step 18's gap-length comparison and any
        multi-site aggregation both consume.
        """
        groups: list[tuple[str, Mapping[MetricSubset, CoreMetrics]]] = [("all", self.metrics)]
        groups += [
            (gap_class.value, by_subset)
            for gap_class, by_subset in self.metrics_by_gap_class.items()
        ]
        rows = [
            {
                "target": self.target,
                "method": self.method,
                "mode": self.config.rfr_mode.value,
                "gap_class": label,
                "subset": subset.value,
                "n": scores.n,
                "n_offered": scores.n_offered,
                "r2": scores.r2,
                "slope": scores.slope,
                "rmse": scores.rmse,
                "bias": scores.bias,
            }
            for label, by_subset in groups
            for subset, scores in by_subset.items()
        ]
        table: pd.DataFrame = pd.DataFrame(rows, columns=list(METRIC_TABLE_COLUMNS))
        return table

    def bias_spread_frame(self) -> pd.DataFrame:
        """Return the spread of per-gap bias for this target, one row per gap class.

        The within-site bias IQR of section 6.3: over the individual intervals of
        one class at this site, not over sites - that is
        :func:`rfrgapfill.uncertainty.bias_iqr`. Empty when the run was configured
        with ``bias_iqr_by_gap_class=False``.
        """
        return _bias_spread_frame(_bias_spread_rows(self))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary for the run manifest."""
        return {
            "target": self.target,
            "method": self.method,
            "r2_definition": self.config.validation.r2.value,
            "daytime_threshold": self.config.validation.daytime_threshold,
            "metrics": {subset.value: scores.to_dict() for subset, scores in self.metrics.items()},
            "metrics_by_gap_class": {
                gap_class.value: {
                    subset.value: scores.to_dict() for subset, scores in by_subset.items()
                }
                for gap_class, by_subset in self.metrics_by_gap_class.items()
            },
            "bias_spread_by_gap_class": {
                gap_class.value: spread.to_dict() for gap_class, spread in self.bias_spread.items()
            },
            "energy_balance": (
                None if self.energy_balance_check is None else self.energy_balance_check.to_dict()
            ),
            "rows": self.features.to_dict(),
        }

    def manifest(
        self,
        *,
        extra: Mapping[str, Any] | None = None,
        created_at: str | None = None,
    ) -> RunManifest:
        """Return the run manifest of this arm (method_spec.md section 7).

        Assembled by :meth:`~rfrgapfill.provenance.RunManifest.from_validation`
        from the objects that own each section, so the manifest describes the run
        that happened rather than the configuration it was asked for.
        """
        return RunManifest.from_validation(
            config=self.config,
            model=self.model,
            gaps=self.gaps,
            features=self.features,
            target=self.target,
            column_map=self.column_map,
            qc_column=self.qc_column,
            time_axis=self.time_axis,
            metrics=self.to_dict(),
            extra=dict(extra) if extra else None,
            created_at=created_at,
        )

    def summary(self) -> str:
        """Return a short human-readable account of this target's result."""
        lines = [
            f"{self.method} {self.target}: {self.scored_rows} of {self.withheld_rows} "
            "withheld row(s) scored"
        ]
        lines += [
            f"  {subset.value:<10} {_format_metrics(scores)}"
            for subset, scores in self.metrics.items()
        ]
        lines += [
            f"  {gap_class.value:<10} {_format_metrics(by_subset[MetricSubset.ALL])}"
            for gap_class, by_subset in self.metrics_by_gap_class.items()
        ]
        return "\n".join(lines)


#: Column order of the tidy metric table, shared by every ``to_frame``.
METRIC_TABLE_COLUMNS: tuple[str, ...] = (
    "target",
    "method",
    "mode",
    "gap_class",
    "subset",
    "n",
    "n_offered",
    "r2",
    "slope",
    "rmse",
    "bias",
)

#: Column order of every ``bias_spread_frame``.
BIAS_SPREAD_TABLE_COLUMNS: tuple[str, ...] = (
    "target",
    "method",
    "mode",
    "gap_class",
    "n_gaps",
    "n_gaps_offered",
    "bias_q1",
    "bias_median",
    "bias_q3",
    "bias_iqr",
)


def _bias_spread_rows(result: TargetValidation) -> list[dict[str, Any]]:
    """Return one tidy row per gap class of ``result``'s bias spread."""
    return [
        {
            "target": result.target,
            "method": result.method,
            "mode": result.config.rfr_mode.value,
            "gap_class": gap_class.value,
            "n_gaps": spread.n_gaps,
            "n_gaps_offered": spread.n_gaps_offered,
            "bias_q1": spread.q1,
            "bias_median": spread.median,
            "bias_q3": spread.q3,
            "bias_iqr": spread.iqr,
        }
        for gap_class, spread in result.bias_spread.items()
    ]


def _bias_spread_frame(rows: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """Return bias-spread rows as a frame whose quartile columns are always float."""
    table: pd.DataFrame = pd.DataFrame(list(rows), columns=list(BIAS_SPREAD_TABLE_COLUMNS))
    for column in ("bias_q1", "bias_median", "bias_q3", "bias_iqr"):
        # A class with no scored gap gives a column of `None`, which would
        # otherwise arrive as object dtype.
        table[column] = pd.to_numeric(table[column], errors="coerce").astype("float64")
    for column in ("n_gaps", "n_gaps_offered"):
        table[column] = pd.to_numeric(table[column], errors="coerce").astype("Int64")
    return table


def _format_metrics(scores: CoreMetrics) -> str:
    """Render one :class:`CoreMetrics` as a fixed-width line."""

    def show(value: float | None) -> str:
        return "       -" if value is None else f"{value:8.3f}"

    return (
        f"n={scores.n:<6d} R2={show(scores.r2)} slope={show(scores.slope)} "
        f"RMSE={show(scores.rmse)} bias={show(scores.bias)}"
    )


# ---------------------------------------------------------------------------
# The whole run
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class ValidationReport:
    """One artificial-gap experiment: every target, scored on one shared gap mask.

    Equality is identity-based (``eq=False``): the record holds pandas objects.

    The report is a type rather than a dictionary because the targets of a joint
    run are not independent results - they share :attr:`gaps`, which is what makes
    them comparable and what the energy-balance comparison of section 6.4 needs.
    """

    #: The configuration every target was validated under.
    config: RFRConfig
    #: The artificial intervals, shared by every target (method_spec.md 4.4).
    gaps: GapManifest
    #: The validated time axis the run ran on.
    time_axis: TimeAxis
    #: One :class:`TargetValidation` per target, in the order they were requested.
    results: Mapping[str, TargetValidation]
    #: Canonical-to-column mapping the run resolved.
    column_map: ColumnMap
    #: QC column per target, where one was supplied.
    qc_columns: Mapping[str, str]
    #: The energy-balance check of section 6.4, when H and LE were both run.
    energy_balance_check: EnergyBalanceCheck | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "results", MappingProxyType(dict(self.results)))
        object.__setattr__(self, "qc_columns", MappingProxyType(dict(self.qc_columns)))

    @property
    def energy_balance(self) -> EnergyBalanceComparison | None:
        """The measured-against-filled comparison over every withheld row, if checked."""
        return None if self.energy_balance_check is None else self.energy_balance_check.overall

    # -- access --------------------------------------------------------------

    def __getitem__(self, target: str) -> TargetValidation:
        try:
            return self.results[target]
        except KeyError:
            raise ValidationError(
                f"no result for target {target!r}; this run validated "
                f"{', '.join(self.results) or 'nothing'}"
            ) from None

    def __iter__(self) -> Iterator[TargetValidation]:
        return iter(self.results.values())

    def __len__(self) -> int:
        return len(self.results)

    @property
    def targets(self) -> tuple[str, ...]:
        """The validated targets, in the order they were requested."""
        return tuple(self.results)

    @property
    def method(self) -> str:
        """The arm this run is: ``RFR3``, ``RFR10``, ``ORF3`` or ``ORF10``."""
        return method_label(self.config)

    def predictions(self) -> pd.DataFrame:
        """Return every target's predictions on the run's time axis, side by side."""
        frame: pd.DataFrame = pd.DataFrame(
            {str(result.predictions.name): result.predictions for result in self.results.values()},
            index=self.time_axis.index,
        )
        return frame

    # -- tables and export ---------------------------------------------------

    def to_frame(self) -> pd.DataFrame:
        """Return the tidy metric table for every target of this run."""
        frames = [result.to_frame() for result in self.results.values()]
        if not frames:
            empty: pd.DataFrame = pd.DataFrame(columns=list(METRIC_TABLE_COLUMNS))
            return empty
        table: pd.DataFrame = pd.concat(frames, ignore_index=True)
        return table

    def bias_spread_frame(self) -> pd.DataFrame:
        """Return the per-gap bias spread of every target of this run."""
        return _bias_spread_frame(
            [row for result in self.results.values() for row in _bias_spread_rows(result)]
        )

    def energy_balance_frame(self) -> pd.DataFrame:
        """Return the energy-balance check as a table: ``all``, then each gap class.

        Empty, with the same columns, when the run had no H/LE pair to check.
        """
        check = self.energy_balance_check
        rows: list[dict[str, Any]] = []
        if check is not None:
            scopes = [("all", check.overall)] + [
                (gap_class.value, comparison)
                for gap_class, comparison in check.by_gap_class.items()
            ]
            rows = [
                {
                    "method": self.method,
                    "mode": self.config.rfr_mode.value,
                    "sensible_heat": check.sensible_heat,
                    "latent_heat": check.latent_heat,
                    "gap_class": label,
                    **comparison.to_dict(),
                }
                for label, comparison in scopes
            ]
        table: pd.DataFrame = pd.DataFrame(rows, columns=list(ENERGY_BALANCE_TABLE_COLUMNS))
        for column in ("available_energy", "measured_ebr", "filled_ebr", "difference"):
            # An undefined ratio is `None`, which would otherwise arrive as object dtype.
            table[column] = pd.to_numeric(table[column], errors="coerce").astype("float64")
        for column in ("n", "n_offered", *(c for c in table.columns if c.startswith("n_missing"))):
            table[column] = pd.to_numeric(table[column], errors="coerce").astype("Int64")
        return table

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary of the whole run."""
        return {
            "method": self.method,
            "mode": self.config.rfr_mode.value,
            "targets": list(self.targets),
            "qc_columns": dict(self.qc_columns),
            "gaps": self.gaps.to_dict(),
            "time_axis": self.time_axis.to_dict(),
            "results": {target: result.to_dict() for target, result in self.results.items()},
            "energy_balance": (
                None if self.energy_balance_check is None else self.energy_balance_check.to_dict()
            ),
            "config": self.config.to_dict(),
        }

    def manifest(
        self,
        target: str,
        *,
        extra: Mapping[str, Any] | None = None,
        created_at: str | None = None,
    ) -> RunManifest:
        """Return the run manifest of one target of this run.

        One manifest per target, because a manifest describes one model: the
        features, the grid's best parameters and the row accounting are all
        target-specific, and a document claiming to describe three would have to
        drop two of them.
        """
        return self[target].manifest(extra=extra, created_at=created_at)

    def summary(self) -> str:
        """Return a human-readable account of the scenario and every target."""
        lines = [
            f"{self.method} validation of {', '.join(self.targets) or 'nothing'}",
            self.gaps.summary(),
        ]
        lines += [result.summary() for result in self.results.values()]
        check = self.energy_balance_check
        if check is not None:
            scopes = [("all", check.overall)] + [
                (gap_class.value, comparison)
                for gap_class, comparison in check.by_gap_class.items()
            ]
            lines.append(
                f"Energy balance of {check.sensible_heat} + {check.latent_heat} "
                "over the withheld rows (method_spec.md 6.4):"
            )
            lines += [
                f"  EBR {label:<10} {balance.n} of {balance.n_offered} row(s): "
                f"measured={_show(balance.measured)} filled={_show(balance.filled)} "
                f"difference={_show(balance.difference)}"
                for label, balance in scopes
            ]
        return "\n".join(lines)


def _show(value: float | None) -> str:
    """Render an optionally undefined metric for :meth:`ValidationReport.summary`."""
    return "undefined" if value is None else f"{value:.4f}"


# ---------------------------------------------------------------------------
# The orchestration
# ---------------------------------------------------------------------------


def validate_rfr(
    data: pd.DataFrame,
    *,
    config: RFRConfig,
    targets: str | Sequence[str],
    mode: Mode | str | None = None,
    column_map: ColumnMap | Mapping[str, str] | None = None,
    qc_columns: str | Mapping[str, str] | None = None,
    gaps: GapManifest | None = None,
    timestamp: str | None = None,
    on_duplicates: DuplicatePolicy | str = DuplicatePolicy.ERROR,
    on_shortfall: Shortfall = "raise",
    on_incomplete: IncompletePolicy | str = IncompletePolicy.MISSING,
    min_training_rows: int | None = None,
    check_leakage: bool = False,
    energy_balance_targets: tuple[str, str] | None = None,
) -> ValidationReport:
    """Run the paper's artificial-gap experiment and return the scored report.

    The whole of Step 13 in one call: place the gaps, hide the truth, build
    leakage-safe features, fit on what remains, predict the withheld intervals and
    score them by subset and by gap class. The withheld observations **are** the
    test set - there is no random row-wise split anywhere in this module, because
    that would destroy the temporal gap structure the paper is about.

    ``data`` is not modified. It is copied onto a validated time axis first, and
    its observed values are read but never written.

    :param config: the run's settings. Required rather than derived from a mode
        string, because a validation run needs what a mode alone cannot supply -
        the cadence, the hemisphere or latitude behind the season feature, the
        column mapping, the scenario, the seed - and each of those is validated
        as the configuration is built.
    :param targets: the target flux column, or several for the paper's joint
        NEE/H/LE run. Several targets share **one** set of gap locations
        (method_spec.md section 4.4).
    :param mode: overrides ``config.mode`` for this run, for the common case of
        scoring RFR3 and RFR10 against one another. Everything else - seed,
        scenario, grid - stays as configured, so the two arms differ only in
        their driver set.
    :param qc_columns: the QC/provenance column of each target, as one name for a
        single target or a mapping. Strongly recommended: without it every present
        value counts as a genuine measurement, so values that arrived already
        gap-filled would be trained on and scored against.
    :param gaps: a :class:`~rfrgapfill.gaps.GapManifest` to reuse instead of
        generating one - the way to score two arms, or an RFR/ORF pair, on exactly
        the same intervals.
    :param on_shortfall: what the generator does when it cannot place the
        requested design: ``"raise"`` (the default) or ``"warn"``.
    :param on_incomplete: ``"missing"`` (the default) leaves a withheld row whose
        predictors are incomplete unpredicted and therefore unscored; ``"raise"``
        fails instead. Nothing is ever imputed (method_spec.md section 7).
    :param check_leakage: run
        :func:`~rfrgapfill.leakage.require_no_target_leakage` on this
        configuration before fitting. Off by default because it rebuilds the
        feature matrix four times per target; worth paying for on a configuration
        that has not been checked before.
    :param energy_balance_targets: the ``(H, LE)`` target columns for the
        energy-balance comparison, when they are not called ``H`` and ``LE``.

    :raises ValidationError: when the request itself cannot be honoured - an
        unknown target column, a missing QC column, or an energy-balance
        comparison asked for without the mapping it needs.
    """
    if not isinstance(config, RFRConfig):
        raise ConfigError(f"config must be an RFRConfig, got {type(config).__name__}")
    if not isinstance(data, pd.DataFrame):
        raise ValidationError(f"data must be a pandas DataFrame, got {type(data).__name__}")
    if mode is not None:
        config = config.replace(mode=Mode.coerce(mode))

    names = _resolve_targets(targets)
    qc_map = _resolve_qc_columns(qc_columns, names)
    columns = config.require_column_map(column_map)
    settings = config.validation

    frame, axis = prepare_time_index(
        data,
        timestamp=timestamp if timestamp is not None else columns.timestamp,
        frequency=config.time_step,
        on_duplicates=on_duplicates,
    )
    _require_columns(frame, targets=names, qc_columns=qc_map, columns=columns)
    balance_pair = _resolve_energy_balance_targets(
        energy_balance_targets, targets=names, config=config, columns=columns, frame=frame
    )

    # Step 1: the intervals, and the mask they imply, before any target-derived
    # feature exists. One manifest for every target (method_spec.md 4.4).
    manifest = (
        GapScenarioGenerator(config).generate(
            frame,
            target=names,
            qc_column=dict(qc_map) if qc_map else None,
            on_shortfall=on_shortfall,
        )
        if gaps is None
        else _check_manifest(gaps)
    )
    holdout = manifest.mask(frame.index)
    if gaps is not None and not bool(holdout.any()):
        # A generated scenario that withholds nothing has already been reported by
        # the generator (or deliberately allowed through on_shortfall="warn"); a
        # handed-in manifest that withholds nothing is almost always one built for
        # another record, and would otherwise "validate" against an empty test set.
        where = (
            f"its intervals run {min(gap.start for gap in manifest)} to "
            f"{max(gap.end for gap in manifest)} and the data runs "
            f"{frame.index.min()} to {frame.index.max()}"
            if len(manifest)
            else "it places no interval at all"
        )
        raise ValidationError(f"the gap manifest withholds no row of this data: {where}")
    origin = frame.index.min()
    policy = IncompletePolicy.coerce(on_incomplete)
    subsets = subset_masks(frame[columns.column(SHORTWAVE)], threshold=settings.daytime_threshold)
    labels = gap_class_labels(manifest, frame.index)

    results: dict[str, TargetValidation] = {}
    for target in names:
        qc_column = qc_map.get(target)
        if check_leakage:
            require_no_target_leakage(
                frame,
                config=config,
                target=target,
                holdout=holdout,
                qc_column=qc_column,
                column_map=columns,
                origin=origin,
            )

        # Steps 2-5: hide the withheld values, compute the daily statistics from
        # what is still visible, assemble the matrix, keep the truth aside.
        features = build_validation_features(
            frame,
            config=config,
            target=target,
            holdout=holdout,
            qc_column=qc_column,
            column_map=columns,
            origin=origin,
        )
        model = RFRModel(config, target=target).fit(
            features.training_features(),
            features.training_target(),
            min_training_rows=min_training_rows,
        )
        predictions = _predict_holdout(model, features, policy=policy, index=axis.index)
        result = _score_target(
            target=target,
            config=config,
            model=model,
            features=features,
            predictions=predictions,
            gaps=manifest,
            axis=axis,
            columns=columns,
            qc_column=qc_column,
            subsets=subsets,
            labels=labels,
        )
        _warn_on_unscored(result)
        results[target] = result

    check: EnergyBalanceCheck | None = None
    if balance_pair is not None:
        check = _energy_balance(
            results,
            pair=balance_pair,
            frame=frame,
            columns=columns,
            labels=labels,
            by_gap_class=settings.report_by_gap_class,
        )
        # A manifest is written per target, so the one result H and LE share is
        # attached to both - otherwise it would be in neither manifest.
        for name in balance_pair:
            results[name] = replace(results[name], energy_balance_check=check)
    return ValidationReport(
        config=config,
        gaps=manifest,
        time_axis=axis,
        results=results,
        column_map=columns,
        qc_columns=qc_map,
        energy_balance_check=check,
    )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _resolve_targets(targets: str | Sequence[str]) -> tuple[str, ...]:
    """Return the requested targets, in order, rejecting duplicates and blanks."""
    if isinstance(targets, str):
        names: tuple[str, ...] = (targets,)
    elif isinstance(targets, Sequence):
        names = tuple(targets)
    else:
        raise ValidationError(
            f"targets must be a column name or a sequence of them, got {type(targets).__name__}"
        )
    if not names:
        raise ValidationError("name at least one target to validate")
    for name in names:
        if not isinstance(name, str) or not name.strip():
            raise ValidationError(f"every target must be a non-empty string, got {name!r}")
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValidationError(f"target(s) requested more than once: {', '.join(duplicates)}")
    return names


def _resolve_qc_columns(
    qc_columns: str | Mapping[str, str] | None,
    targets: tuple[str, ...],
) -> Mapping[str, str]:
    """Return the QC column of each target that has one."""
    if qc_columns is None:
        return MappingProxyType({})
    if isinstance(qc_columns, str):
        if len(targets) != 1:
            raise ValidationError(
                f"a single QC column {qc_columns!r} is ambiguous for {len(targets)} targets; "
                "pass a mapping from target to QC column"
            )
        return MappingProxyType({targets[0]: qc_columns})
    if not isinstance(qc_columns, Mapping):
        raise ValidationError(
            f"qc_columns must be a column name or a mapping, got {type(qc_columns).__name__}"
        )
    unknown = sorted(set(qc_columns) - set(targets))
    if unknown:
        raise ValidationError(
            f"qc_columns names target(s) this run does not validate: {', '.join(unknown)}"
        )
    return MappingProxyType(
        {target: qc_columns[target] for target in targets if target in qc_columns}
    )


def _require_columns(
    frame: pd.DataFrame,
    *,
    targets: tuple[str, ...],
    qc_columns: Mapping[str, str],
    columns: ColumnMap,
) -> None:
    """Raise unless every column this run reads is present in the frame.

    Checked here rather than left to the feature layer because the day/night
    split reads the radiation column before any feature is built, and a mistyped
    driver name should fail as a named column rather than as a ``KeyError``.
    """
    missing = [name for name in targets if name not in frame.columns]
    if missing:
        raise ValidationError(f"target column(s) not in the data: {', '.join(missing)}")
    absent = sorted({column for column in qc_columns.values() if column not in frame.columns})
    if absent:
        raise ValidationError(f"QC column(s) not in the data: {', '.join(absent)}")
    unmapped = columns.missing_columns(frame.columns)
    if unmapped:
        raise ValidationError(
            f"mapped driver column(s) not in the data: {', '.join(sorted(unmapped))}"
        )


def _check_manifest(gaps: GapManifest) -> GapManifest:
    """Return ``gaps`` after checking it is a manifest at all."""
    if not isinstance(gaps, GapManifest):
        raise ValidationError(
            f"gaps must be a GapManifest from GapScenarioGenerator, got {type(gaps).__name__}"
        )
    return gaps


def _resolve_energy_balance_targets(
    requested: tuple[str, str] | None,
    *,
    targets: tuple[str, ...],
    config: RFRConfig,
    columns: ColumnMap,
    frame: pd.DataFrame,
) -> tuple[str, str] | None:
    """Return the ``(H, LE)`` targets the EBR comparison will use, or ``None``.

    ``None`` whenever the comparison is switched off or the run simply has no H/LE
    pair to compare - a NEE-only run is not an incomplete energy-balance run.
    Asking for the comparison without the radiation and soil-heat-flux mapping it
    needs *is* an error: section 6.4 requires those two variables regardless of
    the driver mode, so skipping quietly would drop a required metric.
    """
    if requested is None:
        if not config.validation.compute_energy_balance_ratio:
            return None
        if not {SENSIBLE_HEAT, LATENT_HEAT} <= set(targets):
            return None
        pair = (SENSIBLE_HEAT, LATENT_HEAT)
    else:
        pair = (str(requested[0]), str(requested[1]))
        unknown = [name for name in pair if name not in targets]
        if unknown:
            raise ValidationError(
                "energy_balance_targets names target(s) this run does not validate: "
                f"{', '.join(unknown)}"
            )
    unmapped = columns.missing(EBR_VARIABLES)
    if unmapped:
        raise ValidationError(
            f"the energy-balance ratio needs {' and '.join(EBR_VARIABLES)} mapped regardless "
            f"of the driver mode (docs/method_spec.md 6.4), and {', '.join(unmapped)} is not. "
            "Map it, or set ValidationConfig(compute_energy_balance_ratio=False)."
        )
    absent = sorted(
        columns.column(name) for name in EBR_VARIABLES if columns.column(name) not in frame.columns
    )
    if absent:
        raise ValidationError(
            f"the energy-balance ratio needs column(s) absent from the data: {', '.join(absent)}"
        )
    return pair


def _predict_holdout(
    model: RFRModel,
    features: ValidationFeatureSet,
    *,
    policy: IncompletePolicy,
    index: pd.DatetimeIndex,
) -> pd.Series:
    """Predict the withheld rows and return them on the run's full time axis.

    Rows outside the artificial gaps are left missing rather than predicted:
    nothing the model produces goes anywhere near a value that was never
    withheld, which is what makes "the observed values are unchanged" checkable
    rather than merely intended.
    """
    predicted = model.predict(features.holdout_features(), on_incomplete=policy)
    series: pd.Series = pd.Series(np.nan, index=index, name=f"{features.target}_predicted")
    if len(predicted):
        series.loc[predicted.index] = predicted.to_numpy(dtype=float)
    return series


def _score_target(
    *,
    target: str,
    config: RFRConfig,
    model: RFRModel,
    features: ValidationFeatureSet,
    predictions: pd.Series,
    gaps: GapManifest,
    axis: TimeAxis,
    columns: ColumnMap,
    qc_column: str | None,
    subsets: Mapping[MetricSubset, pd.Series],
    labels: pd.Series,
) -> TargetValidation:
    """Score one target's predictions by subset and by gap class."""
    settings = config.validation
    scored = features.scoring_mask
    measured = features.truth

    def score(rows: np.ndarray) -> CoreMetrics:
        return core_metrics(measured.loc[rows], predictions.loc[rows], definition=settings.r2)

    metrics = {
        subset: score((scored & subsets[subset]).to_numpy()) for subset in settings.metric_subsets
    }
    by_class: dict[GapClass, Mapping[MetricSubset, CoreMetrics]] = {}
    spread: dict[GapClass, BiasSpread] = {}
    if settings.report_by_gap_class or settings.bias_iqr_by_gap_class:
        for gap_class in GapClass:
            in_class = scored.to_numpy() & (labels == gap_class.value).to_numpy()
            if settings.report_by_gap_class:
                by_class[gap_class] = MappingProxyType(
                    {
                        subset: score(in_class & subsets[subset].to_numpy())
                        for subset in settings.metric_subsets
                    }
                )
            if settings.bias_iqr_by_gap_class:
                spread[gap_class] = _bias_spread(
                    gaps,
                    gap_class,
                    measured=measured,
                    predicted=predictions,
                    scored=in_class,
                )
    return TargetValidation(
        target=target,
        method=method_label(config),
        config=config,
        model=model,
        features=features,
        predictions=predictions,
        metrics=metrics,
        metrics_by_gap_class=by_class,
        bias_spread=spread,
        gaps=gaps,
        time_axis=axis,
        column_map=columns,
        qc_column=qc_column,
    )


def _warn_on_unscored(result: TargetValidation) -> None:
    """Warn when a run withheld rows but scored none of them (ambiguity A4)."""
    withheld = result.withheld_rows
    if not withheld or result.scored_rows:
        return
    strategy = result.config.features.statistic_strategy.value
    warnings.warn(
        f"{result.method} {result.target}: none of the {withheld} withheld row(s) could be "
        f"scored. Under daily_statistic_strategy={strategy!r} a gap covering a whole calendar "
        "day has no daily target statistics, so no row of it has a complete feature vector "
        "(docs/method_spec.md 3.4, ambiguity A4); a long-gap run needs a reaching strategy - "
        "'rolling_available' or 'neighbor_day_fallback' - chosen deliberately and recorded in "
        "the manifest.",
        ValidationWarning,
        stacklevel=3,
    )


def _energy_balance(
    results: Mapping[str, TargetValidation],
    *,
    pair: tuple[str, str],
    frame: pd.DataFrame,
    columns: ColumnMap,
    labels: pd.Series,
    by_gap_class: bool,
) -> EnergyBalanceCheck:
    """Return the measured-against-filled energy-balance check (section 6.4; Step 19).

    Every withheld row is offered, and
    :func:`~rfrgapfill.metrics.compare_energy_balance` keeps only the rows where
    all six components are present, so the two ratios describe the same half
    hours and their difference is the effect of the fill rather than of a
    different row set. What it drops it counts, by component.

    A measured value must be a *genuine* one: a withheld value that arrived
    pre-filled is never scored against, and is no more closure evidence than it
    is scoring truth, so it enters as missing.
    """
    heat, latent = (results[name] for name in pair)
    # One mask for every target (section 4.4); intersected only so a mismatch
    # could never offer a row one of the fluxes was not withheld at.
    withheld = (heat.features.holdout_mask & latent.features.holdout_mask).to_numpy()
    measured_heat = heat.features.truth.where(heat.features.scoring_mask)
    measured_latent = latent.features.truth.where(latent.features.scoring_mask)
    radiation = frame[columns.column(NET_RADIATION)]
    soil = frame[columns.column(SOIL_HEAT_FLUX)]

    def compare(rows: np.ndarray) -> EnergyBalanceComparison:
        return compare_energy_balance(
            measured_sensible_heat=measured_heat.loc[rows],
            measured_latent_heat=measured_latent.loc[rows],
            filled_sensible_heat=heat.predictions.loc[rows],
            filled_latent_heat=latent.predictions.loc[rows],
            net_radiation=radiation.loc[rows],
            soil_heat_flux=soil.loc[rows],
        )

    classes = (
        {
            gap_class: compare(withheld & (labels == gap_class.value).to_numpy())
            for gap_class in GapClass
        }
        if by_gap_class
        else {}
    )
    return EnergyBalanceCheck(
        sensible_heat=pair[0],
        latent_heat=pair[1],
        overall=compare(withheld),
        by_gap_class=classes,
    )
