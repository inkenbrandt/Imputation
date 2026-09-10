"""Artificial-gap validation: the paper's experiment as one call.

:func:`validate_rfr` runs the whole of Zhu et al.'s validation design end to end
and returns everything needed to read the result: predictions, metrics, the gap
manifest and the configuration that produced them.

The order of operations is the method, not an implementation detail::

    identify genuinely measured target rows      provenance.observed_mask
        v
    generate the artificial-gap manifest         gaps.GapScenarioGenerator
        v
    keep the hidden truth aside                  never passed to features
        v
    mask the artificial gaps                     a masked copy of the frame
        v
    build leakage-safe features                  features.build_feature_matrix
        v
    fit and tune on what is left                 model.RFRModel
        v
    predict inside the gaps                      the withheld 25%
        v
    score against the hidden truth               metrics + EBR

Two of those steps are the ones that make or break the experiment.

**The gap mask is built before the features.** The daily target statistics of
method_spec.md 3.4 are derived from the target itself, so building them first
would let each held-out value help predict itself and inflate every metric
reported here (3.5, ambiguity A6). The masking is belt and braces: the withheld
values are removed from the frame the features are built from *and* excluded by
the ``available`` mask, so no future feature can reintroduce the leak quietly.

**The withheld observations are the test set.** Not a random 25% of rows - a set
of contiguous 24-hour, 7-day and 30-day intervals (method_spec.md 4.5). A
row-wise split would leave every held-out half-hour surrounded by its own
neighbours and would measure a problem the paper is not about; this package does
not offer one.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Final, cast

import numpy as np
import pandas as pd

from rfrgapfill.config import (
    ColumnMap,
    ConfigError,
    FeatureConfig,
    GapClass,
    GapScenarioConfig,
    Mode,
    RFRConfig,
    ValidationConfig,
)
from rfrgapfill.features import FeatureMatrix, build_feature_matrix
from rfrgapfill.gaps import ArtificialGaps, GapScenarioGenerator
from rfrgapfill.metrics import (
    SubsetMetrics,
    bias,
    energy_balance_ratio,
    metrics_from_config,
)
from rfrgapfill.model import RFRModel, TrainingReport
from rfrgapfill.provenance import observed_mask, run_manifest
from rfrgapfill.schema import NET_RADIATION, SHORTWAVE, SOIL_HEAT_FLUX
from rfrgapfill.time import DuplicatePolicy, TimeAxis, prepare_time_index

__all__ = [
    "SCENARIOS",
    "EnergyBalance",
    "GapClassMetrics",
    "TargetValidation",
    "ValidationError",
    "ValidationReport",
    "compare_receptive_limiter",
    "validate_rfr",
]


class ValidationError(ValueError):
    """Raised when a validation run cannot be carried out as requested.

    Reports a run that cannot proceed - a target the frame does not carry, no
    genuinely observed values to withhold, a scenario name that is not
    registered. A run that *completes* but falls short of the requested gap
    design is not an error: it is reported on
    :attr:`ArtificialGaps.satisfied <rfrgapfill.gaps.ArtificialGaps.satisfied>`.
    """


#: Named artificial-gap scenarios. ``zhu2022`` is the published design: 25%
#: withheld as 24-hour, 7-day and 30-day gaps mixed 20/30/50, each interval
#: needing 50% genuine measurements (method_spec.md section 4).
SCENARIOS: Final[Mapping[str, GapScenarioConfig]] = MappingProxyType(
    {"zhu2022": GapScenarioConfig()}
)

#: Default target names the energy-balance ratio is computed for (6.4).
_DEFAULT_HEAT_TARGETS: Final[tuple[str, str]] = ("H", "LE")


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GapClassMetrics:
    """Metrics for one gap-duration class (method_spec.md 6.3).

    Reported per class because that is where the paper's claim lives: RFR is
    offered as a method that stays stable as gaps get longer, and an aggregate
    number cannot show whether it did.
    """

    #: The duration class these metrics cover.
    gap_class: GapClass
    #: All/daytime/nighttime metrics over the withheld rows of this class.
    metrics: SubsetMetrics
    #: Gap events of this class that contributed scored rows.
    n_gaps: int
    #: Bias of each individual gap event, in manifest order.
    bias_by_gap: tuple[float, ...]

    @property
    def bias_iqr(self) -> float:
        """Interquartile range of the per-gap biases (method_spec.md 6.3).

        Spread *between gap events*, not between half-hours: it answers how much
        the bias of a filled gap varies from one gap to the next, which is the
        quantity the supplement's uncertainty discussion is built on. NaN with
        fewer than two events, where a range would be meaningless.

        The normalized joint-uncertainty ratios of Supplementary Table S8 are a
        different quantity and are **not** computed here: their denominator has
        not been reconstructed from the supplementary methods (ambiguity A8).
        """
        values = np.asarray([value for value in self.bias_by_gap if np.isfinite(value)])
        if values.size < 2:
            return float("nan")
        return float(np.percentile(values, 75) - np.percentile(values, 25))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary for the run report."""
        return {
            "gap_class": self.gap_class.value,
            "n_gaps": self.n_gaps,
            "bias_iqr": self.bias_iqr,
            "bias_by_gap": list(self.bias_by_gap),
            **self.metrics.to_dict(),
        }


@dataclass(frozen=True)
class EnergyBalance:
    """Measured and filled energy-balance ratios over the artificial gaps (6.4).

    ``EBR = sum(H + LE) / sum(NETRAD - G)``, computed twice over the same rows:
    once from the measurements that were hidden, once from what the model put in
    their place. The difference says whether gap filling distorted the site's
    energy balance - a check on the two heat fluxes together that neither one's
    RMSE can make on its own.
    """

    #: EBR from the hidden measurements.
    measured: float
    #: EBR from the model's predictions.
    filled: float
    #: Rows both fluxes were withheld, predicted and energy-closed on.
    n_rows: int
    #: Target names the two heat fluxes were read from.
    sensible_heat: str
    latent_heat: str

    @property
    def difference(self) -> float:
        """``filled - measured``. Positive means filling raised the ratio."""
        return self.filled - self.measured

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary for the run report."""
        return {
            "measured": self.measured,
            "filled": self.filled,
            "difference": self.difference,
            "n_rows": self.n_rows,
            "sensible_heat": self.sensible_heat,
            "latent_heat": self.latent_heat,
        }


@dataclass(frozen=True)
class TargetValidation:
    """One target's validation result: what was withheld, predicted and scored."""

    #: The target column.
    target: str
    #: All/daytime/nighttime metrics over every withheld observation.
    metrics: SubsetMetrics
    #: The same, per gap-duration class.
    by_gap_class: Mapping[GapClass, GapClassMetrics]
    #: One row per withheld observation: measured, predicted, gap id and class,
    #: and the shortwave radiation that split day from night.
    predictions: pd.DataFrame
    #: The artificial gaps this target was scored on.
    gaps: ArtificialGaps
    #: The fit that produced the predictions, including what it dropped.
    training: TrainingReport
    #: The design matrix's provenance, including daily-statistic fallbacks.
    features: FeatureMatrix
    #: The fitted model, kept so a caller can save it or inspect importances.
    model: RFRModel
    #: Genuine observations withheld by the artificial gaps.
    n_withheld: int
    #: Withheld observations the model could predict.
    n_predicted: int

    @property
    def n_unpredicted(self) -> int:
        """Withheld observations left unpredicted for want of a driver."""
        return self.n_withheld - self.n_predicted

    @property
    def coverage(self) -> float:
        """Share of the withheld observations that were predicted, in [0, 1].

        Well below 1 means the metrics describe an easier subset than the
        scenario asked for - usually a driver missing exactly where the gaps
        fell - and should be read before the metrics themselves.
        """
        if self.n_withheld == 0:
            return 0.0
        return self.n_predicted / self.n_withheld

    def metrics_frame(self) -> pd.DataFrame:
        """Return one row per subset and gap class, for tables and reports."""
        rows = self.metrics.to_frame().assign(gap_class="all")
        for gap_class, result in self.by_gap_class.items():
            rows = pd.concat([rows, result.metrics.to_frame().assign(gap_class=gap_class.value)])
        return cast(
            "pd.DataFrame",
            rows.reset_index()
            .assign(target=self.target)
            .loc[:, ["target", "gap_class", "subset", "n", "r2", "slope", "rmse", "bias"]],
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary for the run report."""
        return {
            "target": self.target,
            "n_withheld": self.n_withheld,
            "n_predicted": self.n_predicted,
            "n_unpredicted": self.n_unpredicted,
            "coverage": self.coverage,
            "metrics": self.metrics.to_dict(),
            "by_gap_class": {
                gap_class.value: result.to_dict() for gap_class, result in self.by_gap_class.items()
            },
            "training": self.training.to_dict(),
            "features": self.features.to_dict(),
        }


@dataclass(frozen=True)
class ValidationReport:
    """The result of one artificial-gap validation run.

    Everything Step 13 asks a run to return - predictions, metrics, the gap
    manifest and the configuration - reachable from one object, and serialisable
    in full through :meth:`to_dict` so a run can be archived beside the numbers
    it produced.
    """

    #: Per-target results, in the order the targets were requested.
    results: Mapping[str, TargetValidation]
    #: The artificial gaps used for each target. Identical objects across targets
    #: when the scenario shares gap locations, as the paper's joint validation does.
    gaps: Mapping[str, ArtificialGaps]
    #: Measured and filled energy-balance ratios, when H and LE were both validated.
    energy_balance: EnergyBalance | None
    #: The configuration the run used.
    config: RFRConfig
    #: The validated time axis of the input series.
    time_axis: TimeAxis
    #: Whether every target was scored on the same gap locations (4.4).
    shared_gaps: bool

    # -- access --------------------------------------------------------------

    def __getitem__(self, target: str) -> TargetValidation:
        try:
            return self.results[target]
        except KeyError:
            raise KeyError(
                f"{target!r} was not validated; this run scored: {', '.join(self.targets)}"
            ) from None

    def __iter__(self) -> Iterable[str]:
        return iter(self.results)

    @property
    def targets(self) -> tuple[str, ...]:
        """The validated targets, in request order."""
        return tuple(self.results)

    @property
    def gap_manifest(self) -> pd.DataFrame:
        """The artificial-gap manifest.

        One manifest when the gaps are shared, otherwise the per-target manifests
        stacked with a ``target`` column.
        """
        if self.shared_gaps:
            return next(iter(self.gaps.values())).manifest
        stacked: pd.DataFrame = pd.concat(
            [gaps.manifest.assign(target=target) for target, gaps in self.gaps.items()]
        )
        return stacked

    @property
    def satisfied(self) -> bool:
        """Whether every target's gap scenario met its requested design (A7)."""
        return all(gaps.satisfied for gaps in self.gaps.values())

    @property
    def warnings(self) -> tuple[str, ...]:
        """Every reason a gap scenario missed its requested design, deduplicated.

        Empty exactly when :attr:`satisfied` is true, so a report never says the
        design was not achieved without saying why.
        """
        seen: list[str] = []
        for gaps in self.gaps.values():
            for reason in gaps.shortfalls:
                if reason not in seen:
                    seen.append(reason)
        return tuple(seen)

    # -- tables --------------------------------------------------------------

    def metrics_frame(self) -> pd.DataFrame:
        """Return every metric as one tidy frame: target x gap class x subset."""
        combined: pd.DataFrame = pd.concat(
            [result.metrics_frame() for result in self.results.values()], ignore_index=True
        )
        return combined

    def predictions(self) -> pd.DataFrame:
        """Return every withheld observation and its prediction, stacked by target."""
        stacked: pd.DataFrame = pd.concat(
            [result.predictions.assign(target=target) for target, result in self.results.items()]
        )
        return stacked

    # -- provenance ----------------------------------------------------------

    def manifest(self) -> dict[str, Any]:
        """Return the full run manifest (method_spec.md section 7)."""
        return run_manifest(
            self.config,
            targets=list(self.targets),
            time_axis=self.time_axis.to_dict(),
            gaps={
                "shared": self.shared_gaps,
                "by_target": {target: gaps.to_dict() for target, gaps in self.gaps.items()},
            },
            features={target: result.features.to_dict() for target, result in self.results.items()},
            training={target: result.training.to_dict() for target, result in self.results.items()},
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation of the whole run."""
        return {
            "manifest": self.manifest(),
            "satisfied": self.satisfied,
            "warnings": list(self.warnings),
            "results": {target: result.to_dict() for target, result in self.results.items()},
            "energy_balance": (
                None if self.energy_balance is None else self.energy_balance.to_dict()
            ),
        }


# ---------------------------------------------------------------------------
# The orchestration
# ---------------------------------------------------------------------------


def validate_rfr(
    data: pd.DataFrame,
    *,
    targets: str | Sequence[str],
    mode: Mode | str | None = None,
    config: RFRConfig | None = None,
    column_map: ColumnMap | Mapping[str, str] | None = None,
    qc_columns: Mapping[str, str | None] | None = None,
    scenario: str | GapScenarioConfig | None = None,
    gaps: ArtificialGaps | None = None,
    hemisphere: str | None = None,
    latitude: float | None = None,
    frequency: str | None = None,
    site_id: str | None = None,
    random_state: int = 42,
    use_receptive_limiter: bool | None = None,
    timestamp: str | None = None,
    on_duplicates: DuplicatePolicy | str = DuplicatePolicy.ERROR,
    energy_balance_targets: tuple[str, str] | None = None,
) -> ValidationReport:
    """Run the paper's artificial-gap experiment and return everything it produced.

    >>> report = validate_rfr(                                   # doctest: +SKIP
    ...     df,
    ...     targets=["NEE", "H", "LE"],
    ...     mode="RFR10",
    ...     scenario="zhu2022",
    ...     latitude=51.5,
    ...     column_map=ColumnMap.fluxnet2015("RFR10"),
    ...     qc_columns={"NEE": "NEE_VUT_REF_QC", "H": "H_F_MDS_QC", "LE": "LE_F_MDS_QC"},
    ... )
    >>> report["NEE"].metrics.nighttime.r2                       # doctest: +SKIP

    ``targets`` names the flux columns to validate. Given several, they are
    scored on **identical** gap locations by default, as the paper's joint
    NEE/H/LE validation was (method_spec.md 4.4); a row must be genuinely
    measured for every target to be eligible for withholding, so the shared test
    set is the same set of half-hours for each.

    ``qc_columns`` maps a target to its quality flag. Without one, every finite
    value counts as a measurement - acceptable for cleaned data, wrong for a raw
    FLUXNET file, where it would hide already-filled values inside artificial
    gaps and then score predictions against them.

    Give either ``config`` or the individual settings (``mode``, ``hemisphere``
    or ``latitude``, ``frequency``, ``site_id``, ``random_state``,
    ``use_receptive_limiter``); mixing the two is rejected rather than silently
    resolved. ``scenario`` names an entry in :data:`SCENARIOS` or supplies a
    :class:`~rfrgapfill.config.GapScenarioConfig` directly. ``gaps`` reuses an
    already-generated scenario, which is how a paired RFR/ORF comparison puts
    both models on exactly the same gaps - see :func:`compare_receptive_limiter`.

    Raises :class:`ValidationError` for a run that cannot proceed. A run that
    completes but could not place the requested gap design is *not* an error: it
    returns with :attr:`ValidationReport.satisfied` false and the reasons in
    :attr:`ValidationReport.warnings` (ambiguity A7).
    """
    requested = _requested_targets(targets)
    settings = _resolve_config(
        config,
        mode=mode,
        column_map=column_map,
        scenario=scenario,
        hemisphere=hemisphere,
        latitude=latitude,
        frequency=frequency,
        site_id=site_id,
        random_state=random_state,
        use_receptive_limiter=use_receptive_limiter,
    )
    heat = _resolve_heat_targets(energy_balance_targets, requested, settings.validation)
    mapping = settings.require_column_map(column_map, include_ebr=heat is not None)

    frame, axis = prepare_time_index(
        data,
        timestamp=timestamp if timestamp is not None else mapping.timestamp,
        frequency=settings.time_step,
        on_duplicates=on_duplicates,
    )
    missing_targets = [name for name in requested if name not in frame.columns]
    if missing_targets:
        raise ValidationError(
            f"the data does not carry target column(s): {', '.join(missing_targets)}"
        )

    flags = dict(qc_columns or {})
    observed = {
        target: observed_mask(
            frame,
            target,
            qc_column=flags.get(target),
            observed_qc_values=settings.observed_qc_values,
        )
        for target in requested
    }

    scenarios, shared = _scenarios(
        gaps, axis=axis, observed=observed, config=settings, targets=requested
    )
    shortwave = pd.to_numeric(frame[mapping.column(SHORTWAVE)], errors="coerce")

    results: dict[str, TargetValidation] = {}
    for target in requested:
        results[target] = _validate_target(
            frame,
            target=target,
            config=settings,
            column_map=mapping,
            axis=axis,
            observed=observed[target],
            gaps=scenarios[target],
            shortwave=shortwave,
        )

    return ValidationReport(
        results=MappingProxyType(results),
        gaps=MappingProxyType(scenarios),
        energy_balance=_energy_balance(frame, results, column_map=mapping, heat=heat),
        config=settings,
        time_axis=axis,
        shared_gaps=shared,
    )


def _validate_target(
    frame: pd.DataFrame,
    *,
    target: str,
    config: RFRConfig,
    column_map: ColumnMap,
    axis: TimeAxis,
    observed: pd.Series,
    gaps: ArtificialGaps,
    shortwave: pd.Series,
) -> TargetValidation:
    """Withhold, refit, predict and score one target. The core of the experiment."""
    truth = pd.to_numeric(frame[target], errors="coerce")
    withheld = observed & gaps.mask
    available = observed & ~gaps.mask

    if not withheld.any():
        raise ValidationError(
            f"the artificial gaps withhold no observed value of {target!r}; there is nothing "
            "to score. Check the QC column and the observation mask."
        )
    if not available.any():
        raise ValidationError(
            f"the artificial gaps withhold every observed value of {target!r}, leaving no "
            "training data. Lower missing_fraction."
        )

    # The hidden truth leaves the frame entirely before features are built. The
    # `available` mask already excludes it from the daily statistics; removing the
    # values as well means no feature added later can reintroduce the leak.
    masked = frame.copy()
    masked.loc[gaps.mask, target] = np.nan

    features = build_feature_matrix(
        masked,
        target=target,
        config=config,
        column_map=column_map,
        available=available,
        origin=axis.start,
    )

    training_rows = available & features.complete
    model = RFRModel(config).fit(
        cast("pd.DataFrame", features.frame.loc[training_rows]), truth.loc[training_rows]
    )
    predicted = model.predict(cast("pd.DataFrame", features.frame.loc[withheld]))

    predictions = pd.DataFrame(
        {
            "measured": truth.loc[withheld],
            "predicted": predicted,
            "shortwave": shortwave.loc[withheld],
            "gap_id": gaps.gap_id.loc[withheld],
            "gap_class": gaps.gap_class.loc[withheld],
        }
    )

    return TargetValidation(
        target=target,
        metrics=metrics_from_config(
            predictions["measured"],
            predictions["predicted"],
            predictions["shortwave"],
            config=config.validation,
        ),
        by_gap_class=_metrics_by_gap_class(predictions, config=config.validation, gaps=gaps),
        predictions=predictions,
        gaps=gaps,
        training=model.report,
        features=features,
        model=model,
        n_withheld=int(withheld.sum()),
        n_predicted=int(predicted.notna().sum()),
    )


def _metrics_by_gap_class(
    predictions: pd.DataFrame,
    *,
    config: ValidationConfig,
    gaps: ArtificialGaps,
) -> Mapping[GapClass, GapClassMetrics]:
    """Return per-class metrics and per-gap biases (method_spec.md 6.3)."""
    if not config.report_by_gap_class:
        return MappingProxyType({})

    results: dict[GapClass, GapClassMetrics] = {}
    for gap_class in gaps.config.active_classes:
        rows = predictions["gap_class"] == gap_class.value
        if not rows.any():
            continue
        selected = predictions.loc[rows]
        biases = [
            bias(group["measured"], group["predicted"])
            for _, group in selected.groupby("gap_id", sort=True)
        ]
        results[gap_class] = GapClassMetrics(
            gap_class=gap_class,
            metrics=metrics_from_config(
                selected["measured"],
                selected["predicted"],
                selected["shortwave"],
                config=config,
            ),
            n_gaps=int(selected["gap_id"].nunique()),
            bias_by_gap=tuple(biases) if config.bias_iqr_by_gap_class else (),
        )
    return MappingProxyType(results)


def _energy_balance(
    frame: pd.DataFrame,
    results: Mapping[str, TargetValidation],
    *,
    column_map: ColumnMap,
    heat: tuple[str, str] | None,
) -> EnergyBalance | None:
    """Return measured and filled EBR over the shared artificial gaps (6.4).

    Computed only over rows where **both** heat fluxes were withheld and both
    were predicted, so the numerator's two terms always cover the same
    half-hours as each other and as the available-energy denominator.
    """
    if heat is None:
        return None
    sensible, latent = heat
    h = results[sensible].predictions
    le = results[latent].predictions

    rows = h.index.intersection(le.index)
    if len(rows) == 0:
        return None
    h = h.loc[rows]
    le = le.loc[rows]
    usable = (
        h["measured"].notna()
        & h["predicted"].notna()
        & le["measured"].notna()
        & le["predicted"].notna()
    )
    rows = rows[usable.to_numpy()]
    if len(rows) == 0:
        return None

    available_energy = frame.loc[rows]
    net_radiation = pd.to_numeric(
        available_energy[column_map.column(NET_RADIATION)], errors="coerce"
    )
    soil_heat = pd.to_numeric(available_energy[column_map.column(SOIL_HEAT_FLUX)], errors="coerce")

    return EnergyBalance(
        measured=energy_balance_ratio(
            h.loc[rows, "measured"], le.loc[rows, "measured"], net_radiation, soil_heat
        ),
        filled=energy_balance_ratio(
            h.loc[rows, "predicted"], le.loc[rows, "predicted"], net_radiation, soil_heat
        ),
        n_rows=len(rows),
        sensible_heat=sensible,
        latent_heat=latent,
    )


# ---------------------------------------------------------------------------
# The ORF benchmark (Supplementary Figure S1)
# ---------------------------------------------------------------------------


def compare_receptive_limiter(
    data: pd.DataFrame,
    *,
    targets: str | Sequence[str],
    **kwargs: Any,
) -> tuple[ValidationReport, ValidationReport, pd.DataFrame]:
    """Validate RFR against ORF on identical artificial gaps.

    Returns ``(rfr, orf, comparison)``. The two runs share the gap scenario, the
    seed, the driver set, the estimator family and the hyperparameter grid, and
    differ only in whether the receptive-limiter features are built - which is
    exactly the contrast Supplementary Figure S1 draws.

    ``comparison`` is the paired metrics table with an ``orf`` and an ``rfr``
    column per metric and their difference. Read it as a record, not a test: the
    supplement does not claim RFR improves every metric at every site, and
    neither does this function.
    """
    kwargs.pop("use_receptive_limiter", None)
    kwargs.pop("gaps", None)
    with_limiter, without_limiter = _paired_settings(kwargs)

    rfr = validate_rfr(data, targets=targets, **with_limiter)
    # The ORF run is handed the RFR run's own scenario rather than redrawing one,
    # so the paired metrics differ by the receptive limiter and nothing else.
    shared = next(iter(rfr.gaps.values())) if rfr.shared_gaps else None
    orf = validate_rfr(data, targets=targets, gaps=shared, **without_limiter)

    keys = ["target", "gap_class", "subset"]
    merged = rfr.metrics_frame().merge(orf.metrics_frame(), on=keys, suffixes=("_rfr", "_orf"))
    for metric in ("r2", "slope", "rmse", "bias"):
        merged[f"{metric}_difference"] = merged[f"{metric}_rfr"] - merged[f"{metric}_orf"]
    return rfr, orf, merged


def _paired_settings(kwargs: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return the RFR and ORF argument sets, differing only in the limiter switch.

    A prepared ``config=`` and the loose ``use_receptive_limiter=`` keyword are
    mutually exclusive in :func:`validate_rfr`, so the switch is flipped wherever
    the caller put the configuration rather than added alongside it.
    """
    base = kwargs.pop("config", None)
    if base is None:
        return (
            {**kwargs, "use_receptive_limiter": True},
            {**kwargs, "use_receptive_limiter": False},
        )
    return (
        {
            **kwargs,
            "config": base.replace(features=base.features.replace(use_receptive_limiter=True)),
        },
        {
            **kwargs,
            "config": base.replace(features=base.features.replace(use_receptive_limiter=False)),
        },
    )


# ---------------------------------------------------------------------------
# Resolution helpers
# ---------------------------------------------------------------------------


def _requested_targets(targets: str | Sequence[str]) -> tuple[str, ...]:
    """Return ``targets`` as a deduplicated tuple in request order."""
    names = [targets] if isinstance(targets, str) else list(targets)
    if not names:
        raise ValidationError("name at least one target to validate")
    ordered: list[str] = []
    for name in names:
        if not isinstance(name, str) or not name.strip():
            raise ValidationError(f"target names must be non-empty strings, got {name!r}")
        if name not in ordered:
            ordered.append(name)
    return tuple(ordered)


def _resolve_config(
    config: RFRConfig | None,
    *,
    mode: Mode | str | None,
    column_map: ColumnMap | Mapping[str, str] | None,
    scenario: str | GapScenarioConfig | None,
    hemisphere: str | None,
    latitude: float | None,
    frequency: str | None,
    site_id: str | None,
    random_state: int,
    use_receptive_limiter: bool | None,
) -> RFRConfig:
    """Return the run configuration, from ``config`` or from the loose settings."""
    supplied = {
        "mode": mode,
        "hemisphere": hemisphere,
        "latitude": latitude,
        "frequency": frequency,
        "site_id": site_id,
        "use_receptive_limiter": use_receptive_limiter,
    }
    given = sorted(name for name, value in supplied.items() if value is not None)

    if config is not None:
        if given:
            raise ConfigError(
                f"pass either config= or the individual setting(s) {', '.join(given)}, not "
                "both: two sources for the same field would leave the run manifest ambiguous"
            )
        settings = config
    else:
        if mode is None:
            raise ConfigError(
                "validate_rfr needs a driver set: pass mode='RFR3' or mode='RFR10', or a "
                "prepared config=RFRConfig(...). There is no default driver set."
            )
        features = FeatureConfig()
        if use_receptive_limiter is not None:
            features = features.replace(use_receptive_limiter=use_receptive_limiter)
        settings = RFRConfig(
            mode=mode,
            frequency=frequency,
            hemisphere=hemisphere,
            latitude=latitude,
            site_id=site_id,
            random_state=random_state,
            features=features,
            column_map=column_map,
        )

    if scenario is not None:
        settings = settings.replace(
            validation=settings.validation.replace(gaps=_scenario_config(scenario))
        )
    return settings


def _scenario_config(scenario: str | GapScenarioConfig) -> GapScenarioConfig:
    """Return the named or supplied artificial-gap scenario."""
    if isinstance(scenario, GapScenarioConfig):
        return scenario
    if isinstance(scenario, str):
        try:
            return SCENARIOS[scenario]
        except KeyError:
            raise ValidationError(
                f"unknown scenario {scenario!r}; registered scenarios are: "
                f"{', '.join(sorted(SCENARIOS))}. Pass a GapScenarioConfig for anything else."
            ) from None
    raise ValidationError(
        f"scenario must be a name or a GapScenarioConfig, got {type(scenario).__name__}"
    )


def _scenarios(
    gaps: ArtificialGaps | None,
    *,
    axis: TimeAxis,
    observed: Mapping[str, pd.Series],
    config: RFRConfig,
    targets: Sequence[str],
) -> tuple[dict[str, ArtificialGaps], bool]:
    """Return the artificial gaps each target is scored on, and whether they are shared.

    Shared by default, as the paper's joint validation was: one scenario drawn
    from the rows where **every** target is genuinely measured, so the same
    half-hours are withheld from each and the three results are comparable
    (method_spec.md 4.4).
    """
    if gaps is not None:
        return {target: gaps for target in targets}, True

    generator = GapScenarioGenerator(config.validation.gaps, random_state=config.random_state)
    if config.validation.gaps.shared_gaps_across_targets or len(targets) == 1:
        joint = observed[targets[0]].copy()
        for target in targets[1:]:
            joint &= observed[target]
        if not joint.any():
            raise ValidationError(
                "no timestamp carries a genuine measurement of every target, so no shared "
                f"gap scenario exists for {', '.join(targets)}. Validate the targets "
                "separately, or set shared_gaps_across_targets=False."
            )
        return {target: generator.generate(axis, joint) for target in targets}, True

    # Distinct scenarios: each target gets its own seed so the runs are not
    # accidentally identical where the observation masks happen to agree.
    scenarios = {}
    for offset, target in enumerate(targets):
        per_target = GapScenarioGenerator(
            config.validation.gaps, random_state=config.random_state + offset
        )
        scenarios[target] = per_target.generate(axis, observed[target])
    return scenarios, False


def _resolve_heat_targets(
    requested: tuple[str, str] | None,
    targets: Sequence[str],
    config: ValidationConfig,
) -> tuple[str, str] | None:
    """Return the (H, LE) target pair the EBR is computed for, or ``None``.

    Defaults to ``("H", "LE")`` when both were validated. An explicit pair names
    the columns at a site that calls them something else; naming a target that
    was not validated is an error rather than a silently skipped metric.
    """
    if not config.compute_energy_balance_ratio:
        return None
    if requested is None:
        pair = _DEFAULT_HEAT_TARGETS
        return pair if all(name in targets for name in pair) else None
    if len(requested) != 2:
        raise ValidationError(
            f"energy_balance_targets must name the sensible and latent heat targets, "
            f"got {requested!r}"
        )
    absent = [name for name in requested if name not in targets]
    if absent:
        raise ValidationError(
            f"energy_balance_targets names target(s) this run did not validate: {', '.join(absent)}"
        )
    return requested[0], requested[1]
