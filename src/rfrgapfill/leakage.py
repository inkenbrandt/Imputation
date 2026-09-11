"""Leakage-safe feature construction for artificial-gap validation.

The daily target statistics of the receptive limiter (``docs/method_spec.md``
section 3.4) are derived *from the target*, which makes them the one place where
the truth an artificial gap hides could be used to build its own predictors. This
module is the workflow that prevents it (method_spec.md section 3.5, ambiguity
A6), in the order the specification requires:

1. build the holdout mask;
2. mask the held-out target values;
3. compute target-derived daily statistics from the remaining visible
   observations only;
4. build the prediction features;
5. keep the untouched truth separately, for scoring alone.

:func:`build_validation_features` performs all five and returns a
:class:`ValidationFeatureSet`, which carries the feature matrix, the masks that
produced it, and the untouched truth as separate attributes - so "the values the
model may see" and "the values it is scored against" are different objects rather
than the same column at different points in time.

Two independent protections are applied, and either alone is sufficient:

* the held-out target values are removed from the frame the features are built
  from (:func:`hide_target`);
* the transformer is told which rows are visible through ``available_mask``, so
  the statistics exclude the held-out rows whatever the frame contains.

:func:`detect_target_leakage` verifies both by rebuilding the features with the
hidden truth replaced by absurd values and reporting any feature column that
moved; :func:`require_no_target_leakage` raises :class:`LeakageError` when one
does. That probe is the acceptance test of Step 6 and is available to callers as
a runtime check on their own configuration.

This module deliberately knows nothing about how the holdout was chosen. The
artificial-gap generator produces masks; operational filling of real gaps needs
none of this, because a genuinely missing value is already invisible.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from rfrgapfill.config import DEFAULT_OBSERVED_QC_VALUES, FeatureMode, RFRConfig
from rfrgapfill.features import (
    FeatureError,
    _as_boolean_mask,  # module-private helper, shared inside the package only
    build_feature_matrix,
    daily_statistic_names,
    describe_features,
    feature_names,
)
from rfrgapfill.legacy import LegacyFluxlibWarning, legacy_statistic_names
from rfrgapfill.schema import ColumnMap, ConfigError
from rfrgapfill.time import as_datetime_index, interval_mask

__all__ = [
    "LeakageError",
    "ValidationFeatureSet",
    "available_target_mask",
    "build_validation_features",
    "detect_target_leakage",
    "hide_target",
    "holdout_mask_from_intervals",
    "observed_target_mask",
    "require_no_target_leakage",
]

#: Replacement magnitude of :func:`detect_target_leakage`. Large enough that any
#: statistic touched by a hidden value moves far outside floating-point noise.
_PROBE_MAGNITUDE: float = 1e9


class LeakageError(RuntimeError):
    """Raised when hidden validation truth reached a feature used to predict it.

    A violated invariant rather than bad input: in ``paper_safe`` mode this must
    be unreachable, so it reports a defect in the feature path (or in a caller
    that bypassed :func:`build_validation_features`), not a user mistake.
    """


# ---------------------------------------------------------------------------
# Masks
# ---------------------------------------------------------------------------


def _mask_series(mask: object, index: pd.Index, *, field_name: str) -> pd.Series:
    """Return ``mask`` as a boolean Series aligned to ``index`` by label."""
    values = _as_boolean_mask(mask, index, field_name=field_name)
    aligned: pd.Series = pd.Series(values, index=index, name=field_name)
    return aligned


def observed_target_mask(
    data: pd.DataFrame,
    target: str,
    *,
    qc_column: str | None = None,
    config: RFRConfig | None = None,
    observed_qc_values: Sequence[int] | None = None,
) -> pd.Series:
    """Return the rows carrying a genuine, quality-controlled target measurement.

    A row is observed when its target value is present **and**, if a QC column is
    given, its flag is one of ``observed_qc_values`` (FLUXNET uses 0 for measured;
    anything else was already gap-filled before ingestion, method_spec.md section
    7). A missing or unrecognised flag counts as *not* observed, which is the
    conservative direction: a value of unknown provenance is not trained on and
    not scored against.

    Without a QC column every present value is treated as observed, and the
    result says so only as far as the data does - the distinction the paper draws
    between measured and pre-filled fluxes cannot be recovered from the values
    alone.
    """
    if config is not None and observed_qc_values is not None:
        raise ConfigError(
            "pass either config= or explicit observed_qc_values, not both: two sources "
            "of the same setting cannot be reconciled in the run manifest"
        )
    if observed_qc_values is None:
        observed_qc_values = (
            DEFAULT_OBSERVED_QC_VALUES if config is None else config.observed_qc_values
        )

    index = as_datetime_index(data.index, field_name="the data index")
    if target not in data.columns:
        raise FeatureError(f"target column {target!r} is not in the data")

    present: pd.Series = pd.Series(
        np.asarray(data[target].notna().to_numpy(), dtype=bool), index=index, name="observed"
    )
    if qc_column is None:
        return present
    if qc_column not in data.columns:
        raise FeatureError(f"QC column {qc_column!r} is not in the data")
    accepted = np.asarray(data[qc_column].isin(list(observed_qc_values)).to_numpy(), dtype=bool)
    observed: pd.Series = present & pd.Series(accepted, index=index)
    observed.name = "observed"
    return observed


def holdout_mask_from_intervals(
    values: object,
    intervals: Iterable[tuple[Any, Any]],
) -> pd.Series:
    """Return the mask of timestamps falling inside any ``(start, end)`` interval.

    Intervals are half-open ``[start, end)``, the convention of
    :func:`rfrgapfill.time.interval_mask`, so consecutive intervals of the same
    duration tile the axis without sharing a row. Overlapping intervals are
    simply unioned here; whether the generator is allowed to produce them is its
    own configuration.

    A convenience for turning an artificial-gap manifest - or a hand-written
    interval in a test - into the mask the rest of this module consumes.
    """
    index = as_datetime_index(values, field_name="timestamps")
    mask = pd.Series(False, index=index, name="holdout")
    for start, end in intervals:
        mask |= interval_mask(index, start, end=end)
    return mask


def available_target_mask(
    data: pd.DataFrame,
    target: str,
    *,
    holdout: object,
    qc_column: str | None = None,
    observed: object | None = None,
    config: RFRConfig | None = None,
    observed_qc_values: Sequence[int] | None = None,
) -> pd.Series:
    """Return the rows whose target value the model is allowed to see.

    ``available = observed and not held out``: a genuine quality-controlled
    measurement that the artificial-gap scenario has not withheld. This is the
    mask that must reach :func:`~rfrgapfill.features.daily_flux_statistics`;
    everything else in this module exists to make sure it does.

    Pass ``observed`` to reuse a mask already computed by
    :func:`observed_target_mask`, or ``qc_column`` to have one derived here.
    """
    index = as_datetime_index(data.index, field_name="the data index")
    if observed is None:
        visible = observed_target_mask(
            data,
            target,
            qc_column=qc_column,
            config=config,
            observed_qc_values=observed_qc_values,
        )
    else:
        if qc_column is not None:
            raise ConfigError(
                "pass either observed= or qc_column=, not both: an explicit observed "
                "mask already answers the question the QC column would be read for"
            )
        visible = _mask_series(observed, index, field_name="observed")
    withheld = _mask_series(holdout, index, field_name="holdout")
    available: pd.Series = visible & ~withheld
    available.name = "available"
    return available


def hide_target(data: pd.DataFrame, target: str, *, holdout: object) -> pd.DataFrame:
    """Return a copy of ``data`` with the held-out target values removed.

    Step 2 of the validation procedure. The returned frame is what features are
    built from, so held-out truth is not merely *unused* by the transformers - it
    is not in the frame they read. Only the target column is touched; drivers are
    left alone, because the paper's drivers arrive pre-filled and are available
    inside a gap.

    Defence in depth rather than the primary protection: the primary one is the
    ``available_mask`` threaded into the daily statistics, which holds even for a
    caller who builds features from the original frame.
    """
    index = as_datetime_index(data.index, field_name="the data index")
    if target not in data.columns:
        raise FeatureError(f"target column {target!r} is not in the data")
    withheld = _as_boolean_mask(holdout, index, field_name="holdout")
    hidden: pd.DataFrame = data.copy()
    hidden[target] = data[target].astype(float).where(~withheld)
    return hidden


# ---------------------------------------------------------------------------
# Validation feature set
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ValidationFeatureSet:
    """Features built with a holdout hidden, plus the truth kept back for scoring.

    The point of the type is that the truth and the features are separate
    attributes with different rules: :attr:`features` was built from
    :attr:`available_mask` alone, while :attr:`truth` is the untouched target and
    may be read **only** when scoring predictions. Nothing here re-derives
    features from the truth, and :meth:`training_target` never returns a held-out
    value.
    """

    #: Target flux these features were built for; the daily statistics are its own.
    target: str
    #: Feature matrix, in the deterministic order of :func:`feature_names`.
    features: pd.DataFrame
    #: The untouched target. Scoring only - never an input to a feature.
    truth: pd.Series
    #: Rows carrying a genuine quality-controlled measurement.
    observed_mask: pd.Series
    #: Rows withheld by the artificial-gap scenario.
    holdout_mask: pd.Series
    #: Rows the model may see: observed and not withheld.
    available_mask: pd.Series
    #: The configuration the features were built with.
    config: RFRConfig

    def __post_init__(self) -> None:
        index = self.features.index
        for name in ("truth", "observed_mask", "holdout_mask", "available_mask"):
            series = getattr(self, name)
            if not series.index.equals(index):
                raise FeatureError(f"{name} is not aligned with the feature index")
        overlap = int((self.available_mask & self.holdout_mask).sum())
        if overlap:
            raise LeakageError(
                f"{overlap} row(s) are both available to the model and withheld from it; "
                "the available mask must exclude every held-out row "
                "(docs/method_spec.md 3.5)"
            )

    # -- row selections ------------------------------------------------------

    @property
    def feature_names(self) -> tuple[str, ...]:
        """The feature column order, as recorded in the run manifest."""
        return tuple(self.features.columns)

    @property
    def training_mask(self) -> pd.Series:
        """Rows eligible for training: observed and not withheld.

        Row *eligibility* only. Dropping rows whose predictors are incomplete is
        the model layer's job, which reports what it excluded.
        """
        return self.available_mask

    @property
    def scoring_mask(self) -> pd.Series:
        """Rows that can be scored: withheld and carrying a real measurement.

        A held-out row whose target was missing anyway has no truth to compare a
        prediction against, so it is withheld from the metrics as well.
        """
        scoring: pd.Series = self.holdout_mask & self.observed_mask
        return scoring

    @property
    def complete_mask(self) -> pd.Series:
        """Rows whose features are all present, and which a model can therefore use.

        Worth reading before a validation run rather than after it. Under the
        default ``daily_statistic_strategy="missing"`` a day with no visible
        target observation has no daily statistics, so **every row of a gap
        longer than a day is incomplete** and would receive no prediction. That
        is the honest consequence of ambiguity A4 - the paper does not say what
        it did here - and the reaching strategies
        (``rolling_available``, ``neighbor_day_fallback``) are the configured way
        out. The count appears in :meth:`to_dict` so a run cannot quietly score
        far fewer rows than it withheld.
        """
        complete: pd.Series = self.features.notna().all(axis=1)
        complete.name = "complete"
        return complete

    def training_features(self) -> pd.DataFrame:
        """The feature rows the model may train on."""
        rows: pd.DataFrame = self.features.loc[self.training_mask.to_numpy()]
        return rows

    def training_target(self) -> pd.Series:
        """The target values the model may train on. Never includes held-out truth."""
        values: pd.Series = self.truth.loc[self.training_mask.to_numpy()]
        return values

    def holdout_features(self) -> pd.DataFrame:
        """The feature rows inside the artificial gaps, to predict."""
        rows: pd.DataFrame = self.features.loc[self.holdout_mask.to_numpy()]
        return rows

    def scoring_truth(self) -> pd.Series:
        """The measured values predictions are scored against."""
        values: pd.Series = self.truth.loc[self.scoring_mask.to_numpy()]
        return values

    # -- manifest ------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary for the run manifest."""
        rows = len(self.features)
        withheld = int(self.holdout_mask.sum())
        observed = int(self.observed_mask.sum())
        return {
            "target": self.target,
            "rows": rows,
            "observed_rows": observed,
            "holdout_rows": withheld,
            "training_rows": int(self.training_mask.sum()),
            "scoring_rows": int(self.scoring_mask.sum()),
            "training_rows_with_complete_features": int(
                (self.training_mask & self.complete_mask).sum()
            ),
            "holdout_rows_with_complete_features": int(
                (self.holdout_mask & self.complete_mask).sum()
            ),
            # Against observed rows, not against all rows: the paper's ~25% is a
            # share of the observations available for validation, not of the
            # calendar (method_spec.md 4.2).
            "withheld_fraction_of_observed": (withheld / observed if observed else None),
            "features": describe_features(self.config, target=self.target),
        }


# ---------------------------------------------------------------------------
# The validation workflow
# ---------------------------------------------------------------------------


def build_validation_features(
    data: pd.DataFrame,
    *,
    config: RFRConfig,
    target: str,
    holdout: object,
    qc_column: str | None = None,
    observed: object | None = None,
    column_map: ColumnMap | Mapping[str, str] | None = None,
    origin: pd.Timestamp | str | None = None,
    encode: bool = True,
    mask_target_values: bool = True,
) -> ValidationFeatureSet:
    """Build leakage-safe validation features for one target (method_spec.md 3.5).

    Performs the five specified steps in order: the holdout mask is taken as
    given (step 1, the artificial-gap generator's output), the held-out target
    values are removed from the working frame (step 2), the daily target
    statistics are computed from the visible observations only (step 3), the
    remaining features are assembled (step 4), and the untouched truth is kept
    beside them for scoring alone (step 5).

    Works unchanged for the ORF benchmark: with the receptive limiter off there
    are no target-derived features to protect, and the same masks and truth still
    define the same training rows and the same scored rows - which is what makes
    the paired comparison of Supplementary Figure S1 a comparison of the feature
    engineering and nothing else.

    With ``feature_mode="legacy_fluxlib"`` - and only then - steps 2 and 3 are
    skipped for the daily statistics, which are computed from every *observed*
    value, held-out ones included, as ``fluxlib`` computed them. Every such call
    emits a :class:`~rfrgapfill.legacy.LegacyFluxlibWarning`, and
    :func:`detect_target_leakage` reports the columns affected.

    :param holdout: boolean mask of the rows the scenario withholds. Build one
        from gap intervals with :func:`holdout_mask_from_intervals`.
    :param qc_column: QC/provenance column distinguishing measured from
        pre-filled target values. Without it, every present value counts as
        observed.
    :param observed: a mask from :func:`observed_target_mask`, when it has
        already been computed - for joint NEE/H/LE validation, for instance,
        where the holdout is shared but each target has its own QC.
    :param mask_target_values: whether the held-out values are also removed from
        the frame features are read from (the default). Setting it to ``False``
        leaves the ``available_mask`` as the only protection, which is how
        :func:`detect_target_leakage` checks that the mask alone is sufficient.
        There is no other reason to turn it off.
    """
    if not isinstance(config, RFRConfig):
        raise ConfigError(f"config must be an RFRConfig, got {type(config).__name__}")
    if not isinstance(data, pd.DataFrame):
        raise FeatureError(f"data must be a pandas DataFrame, got {type(data).__name__}")
    index = as_datetime_index(data.index, field_name="the data index")
    if target not in data.columns:
        raise FeatureError(f"target column {target!r} is not in the data")

    # Step 1: the holdout mask, before any target-derived feature exists.
    withheld = _mask_series(holdout, index, field_name="holdout")
    observed_mask = (
        observed_target_mask(data, target, qc_column=qc_column, config=config)
        if observed is None
        else _mask_series(observed, index, field_name="observed")
    )
    available = available_target_mask(
        data, target, holdout=withheld, observed=observed_mask, config=config
    )

    # Step 5, taken first so that nothing downstream can modify it: the truth is
    # copied out of the frame before the frame is altered.
    truth = pd.Series(data[target].astype(float).to_numpy(dtype=float), index=index, name=target)

    legacy = (
        config.features.mode is FeatureMode.LEGACY_FLUXLIB and config.features.use_receptive_limiter
    )
    if legacy:
        # The historical derivation, reproduced on purpose and nowhere else: fluxlib
        # computed the daily statistics from every observed value before the
        # holdout was applied, so steps 2 and 3 are skipped for them. The training
        # rows and the scored rows are still the available and withheld ones.
        warnings.warn(
            LegacyFluxlibWarning(
                "feature_mode='legacy_fluxlib' reproduces fluxlib's daily statistics, "
                "which are computed before the artificial gaps are hidden: held-out "
                f"{target} values reach the features used to predict them, so every "
                "score from this run is optimistic. Use it to measure that effect, "
                "never as a reproduction of the method (docs/fluxlib_audit.md)."
            ),
            stacklevel=2,
        )
        working = data
        statistics_mask = observed_mask
    else:
        # Step 2.
        working = hide_target(data, target, holdout=withheld) if mask_target_values else data
        statistics_mask = available

    # Steps 3 and 4.
    features = build_feature_matrix(
        working,
        config=config,
        target=target if config.features.use_receptive_limiter else None,
        column_map=column_map,
        available_mask=statistics_mask,
        origin=index.min() if origin is None else origin,
        encode=encode,
    )

    return ValidationFeatureSet(
        target=target,
        features=features,
        truth=truth,
        observed_mask=observed_mask,
        holdout_mask=withheld,
        available_mask=available,
        config=config,
    )


# ---------------------------------------------------------------------------
# The leakage probe
# ---------------------------------------------------------------------------


def _corrupt_hidden_truth(
    data: pd.DataFrame,
    target: str,
    *,
    holdout: np.ndarray,
    magnitude: float,
) -> pd.DataFrame:
    """Return ``data`` with every held-out target value replaced by an absurd one.

    The replacement varies from row to row rather than being a constant offset:
    a uniform shift would leave a shift-invariant statistic such as the standard
    deviation unchanged, and a leak that only moves the quartiles is still a leak
    worth failing on, but one that moved nothing at all would be missed.
    """
    corrupted: pd.DataFrame = data.copy()
    values = data[target].astype(float).to_numpy(dtype=float).copy()
    positions = np.flatnonzero(holdout)
    values[positions] = magnitude * (1.0 + np.arange(positions.size, dtype=float))
    corrupted[target] = values
    return corrupted


def detect_target_leakage(
    data: pd.DataFrame,
    *,
    config: RFRConfig,
    target: str,
    holdout: object,
    magnitude: float = _PROBE_MAGNITUDE,
    **kwargs: Any,
) -> tuple[str, ...]:
    """Return the feature columns that hidden truth can influence. Empty is correct.

    The probe of Step 6 and of acceptance tests 11-14: build the features, replace
    the held-out target values with absurd ones, rebuild, and report every column
    whose values moved anywhere in the matrix. Comparing the whole matrix rather
    than only the held-out rows is deliberate - a hidden value that shifted the
    statistics of a *neighbouring* day would corrupt the training data instead,
    which is no better.

    The rebuild is run twice: once through the ordinary workflow, and once with
    the frame-level masking of :func:`hide_target` disabled, so that the
    ``available_mask`` is left as the only protection and is shown to be
    sufficient on its own.

    ``kwargs`` are passed to :func:`build_validation_features` (``qc_column``,
    ``observed``, ``column_map``, ``origin``, ``encode``), so the probe checks the
    configuration the caller actually uses.
    """
    index = as_datetime_index(data.index, field_name="the data index")
    withheld = _as_boolean_mask(holdout, index, field_name="holdout")
    corrupted = _corrupt_hidden_truth(data, target, holdout=withheld, magnitude=magnitude)

    moved: list[str] = []
    for mask_target_values in (True, False):
        baseline = build_validation_features(
            data,
            config=config,
            target=target,
            holdout=withheld,
            mask_target_values=mask_target_values,
            **kwargs,
        ).features
        probed = build_validation_features(
            corrupted,
            config=config,
            target=target,
            holdout=withheld,
            mask_target_values=mask_target_values,
            **kwargs,
        ).features
        moved.extend(
            column
            for column in baseline.columns
            # `.equals` rather than `==`: NaN in the same place must compare
            # equal, since a statistic that is missing in both runs did not move.
            if column not in moved and not baseline[column].equals(probed[column])
        )
    return tuple(moved)


def require_no_target_leakage(
    data: pd.DataFrame,
    *,
    config: RFRConfig,
    target: str,
    holdout: object,
    magnitude: float = _PROBE_MAGNITUDE,
    **kwargs: Any,
) -> None:
    """Raise :class:`LeakageError` unless hidden truth reaches no feature at all.

    A runtime check of the invariant in method_spec.md section 3.5, usable on a
    real configuration before a validation run rather than only in the package's
    own tests. In ``paper_safe`` mode it must never fire; if it does, the feature
    path is defective and the validation result would be optimistic.
    """
    leaking = detect_target_leakage(
        data, config=config, target=target, holdout=holdout, magnitude=magnitude, **kwargs
    )
    if leaking and config.features.mode is FeatureMode.LEGACY_FLUXLIB:
        historical = legacy_statistic_names(target)
        raise LeakageError(
            f"held-out target values changed {len(leaking)} feature column(s): "
            f"{', '.join(leaking)}. That is what feature_mode='legacy_fluxlib' "
            "reproduces: fluxlib computed its daily statistics "
            f"({', '.join(name for name in leaking if name in historical)}) before the "
            "artificial gaps were hidden, so this check cannot pass in that mode "
            "(docs/fluxlib_audit.md). Use feature_mode='paper_safe' for a leakage-free "
            "validation."
        )
    if leaking:
        expected = daily_statistic_names(target)
        known = tuple(name for name in leaking if name in expected)
        order = feature_names(config, target=target)
        raise LeakageError(
            f"held-out target values changed {len(leaking)} feature column(s): "
            f"{', '.join(sorted(leaking, key=order.index))}. In feature_mode="
            f"{config.features.mode.value!r} no feature may depend on hidden truth "
            f"(docs/method_spec.md 3.5)."
            + (
                f" The affected columns are the target-derived daily statistics "
                f"({', '.join(known)}), so the available mask did not reach "
                f"daily_flux_statistics."
                if known
                else ""
            )
        )
