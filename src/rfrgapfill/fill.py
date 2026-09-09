"""Operational gap filling of real missing data (method_spec.md section 7).

The layer a practitioner actually calls. :class:`RFRGapFiller` assembles the
pieces the steps below it built - the validated configuration, the time axis, the
receptive-limiter features and :class:`~rfrgapfill.model.RFRModel` - into the two
operations an eddy-covariance workflow needs::

    filler = RFRGapFiller(config).fit(df, target="LE", qc_column="LE_QC")
    result = filler.fill(df)

and enforces the rules that make the output trustworthy:

* **training uses eligible observed rows only** - a genuine, quality-controlled
  measurement, never a value that arrived already gap-filled;
* **the caller's frame is never touched** - every stage works on a copy, and the
  original target column is carried through as ``<target>_original``;
* **only gaps are predicted** - a row that already has a value keeps it;
* **a row without complete predictors is not filled** - it is left missing and
  counted, or the fill raises, but nothing is imputed (section 7's "either fail
  clearly or leave predictions missing");
* **every value says where it came from** - the six provenance columns of
  :func:`fill_column_names`.

Leakage has nothing to prevent here and :mod:`rfrgapfill.leakage` is deliberately
not the entry point: a genuinely missing target value is already invisible to the
daily statistics, because there is no value to see. The one distinction that does
matter operationally is *observed* versus *pre-filled* - a FLUXNET target column
is often complete because it was gap-filled before ingestion - so the QC column
decides both what the model trains on and what the daily statistics are computed
from, exactly as in validation.

Two consequences are worth knowing before the first call, and both are reported
rather than hidden:

* Under the documented default ``daily_statistic_strategy="missing"`` (ambiguity
  A4) a gap covering a whole calendar day leaves every row of that day without
  daily statistics, so **nothing in it can be filled**. Filling a multi-day gap
  means choosing a reaching strategy deliberately.
* ``time_distance_hours`` is measured from the first timestamp of the series the
  model was *fitted* on, and :meth:`RFRGapFiller.fill` reuses that origin, so the
  feature means the same thing at fill time as it did at fit time.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Final

import numpy as np
import pandas as pd

from rfrgapfill.config import RFRConfig
from rfrgapfill.features import (
    DAILY_STATISTIC_SUFFIXES,
    build_feature_matrix,
    describe_features,
)
from rfrgapfill.leakage import observed_target_mask
from rfrgapfill.model import IncompletePolicy, NotFittedError, RFRModel
from rfrgapfill.schema import ColumnMap, ConfigError
from rfrgapfill.time import DuplicatePolicy, TimeAxis, prepare_time_index

__all__ = [
    "FILL_COLUMN_SUFFIXES",
    "FillError",
    "FillMethod",
    "FillReport",
    "FillResult",
    "RFRGapFiller",
    "fill_column_names",
    "method_label",
]

#: Suffixes of the provenance columns a fill emits, in the order of
#: ``docs/method_spec.md`` section 7. Appended to the target's own name, so
#: filling ``LE`` produces ``LE_original``, ``LE_filled`` and so on.
FILL_COLUMN_SUFFIXES: Final[tuple[str, ...]] = (
    "_original",
    "_filled",
    "_is_observed",
    "_is_filled",
    "_fill_method",
    "_model_version",
)


class FillError(RuntimeError):
    """Raised when a fill cannot be performed or described as asked.

    The fill layer's own contract: a target column that is not in the frame, a QC
    column the fit used and the fill frame lacks, provenance columns that would be
    overwritten. Configuration problems remain
    :class:`~rfrgapfill.schema.ConfigError` and an unfitted filler still raises
    :class:`~rfrgapfill.model.NotFittedError`, so the three stay distinguishable.
    """


class FillMethod(str, Enum):
    """Provenance of a value in ``<target>_filled``.

    The column answers one question - where did this number come from - so its
    values partition the series exactly. Model-produced values carry the arm's own
    label instead (:func:`method_label`: ``"RFR3"``, ``"RFR10"``, ``"ORF3"``,
    ``"ORF10"``), because "a Random Forest predicted it" is not specific enough to
    reproduce.
    """

    #: A genuine, quality-controlled measurement, carried through untouched.
    OBSERVED = "observed"
    #: A value present in the input that the QC flag says was already gap-filled
    #: before ingestion. Never overwritten unless ``refill_pre_filled=True``.
    PRE_FILLED = "pre_filled"
    #: No value: the target was missing and the row lacked a complete predictor, so
    #: nothing was invented for it (method_spec.md section 7).
    UNFILLED = "unfilled_incomplete_features"


def method_label(config: RFRConfig) -> str:
    """Return the short name of the arm ``config`` describes.

    ``"RFR3"``/``"RFR10"`` with the receptive limiter on, ``"ORF3"``/``"ORF10"``
    with it off (method_spec.md 3.6), so a filled column records which of the two
    produced it rather than only that a forest did.
    """
    if not isinstance(config, RFRConfig):
        raise ConfigError(f"config must be an RFRConfig, got {type(config).__name__}")
    return f"{'ORF' if config.is_orf else 'RFR'}{len(config.drivers)}"


def fill_column_names(target: str) -> tuple[str, ...]:
    """Return the six provenance column names a fill of ``target`` writes.

    In the order of :data:`FILL_COLUMN_SUFFIXES`, which is the order
    ``docs/method_spec.md`` section 7 lists them in.
    """
    if not isinstance(target, str) or not target.strip():
        raise FillError(f"target must be a non-empty string, got {target!r}")
    return tuple(f"{target}{suffix}" for suffix in FILL_COLUMN_SUFFIXES)


# ---------------------------------------------------------------------------
# Row accounting
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FillReport:
    """What happened to the rows offered to :meth:`RFRGapFiller.fill`.

    :attr:`observed_rows`, :attr:`pre_filled_rows` and :attr:`missing_rows`
    partition :attr:`rows`: every row either carries a quality-controlled
    measurement, carries a value of another provenance, or carries nothing.
    :attr:`filled_rows` and :attr:`unfilled_rows` partition
    :attr:`candidate_rows`, the rows this fill actually offered to the model.

    The report is the answer to "why is my gap still empty": under ambiguity A4's
    default a whole-day gap has no daily statistics, so its rows are candidates
    the model cannot reach. :attr:`worst_feature` names the column responsible and
    :meth:`summary` spells the consequence out.
    """

    #: The target flux that was filled.
    target: str
    #: The arm that produced the filled values (:func:`method_label`).
    method: str
    #: Identity of the fitted model that produced them.
    model_version: str
    #: Rows in the frame that was filled.
    rows: int
    #: Rows carrying a genuine, quality-controlled measurement.
    observed_rows: int
    #: Rows carrying a value the QC flag marks as already gap-filled.
    pre_filled_rows: int
    #: Rows carrying no target value at all.
    missing_rows: int
    #: Rows offered to the model: the gaps, plus the pre-filled rows when
    #: ``refill_pre_filled=True``.
    candidate_rows: int
    #: Candidate rows that received a prediction.
    filled_rows: int
    #: Candidate rows left alone because a predictor was missing.
    unfilled_rows: int
    #: Per feature, how many candidate rows lacked it.
    missing_by_feature: Mapping[str, int]
    #: Rows earlier than the first timestamp the model was fitted on, whose
    #: ``time_distance_hours`` is therefore negative and outside training range.
    rows_before_fit_origin: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "missing_by_feature", MappingProxyType(dict(self.missing_by_feature))
        )
        states = self.observed_rows + self.pre_filled_rows + self.missing_rows
        if states != self.rows:
            raise FillError(
                f"row states do not partition the frame: {states} classified against "
                f"{self.rows} row(s)"
            )
        if self.filled_rows + self.unfilled_rows != self.candidate_rows:
            raise FillError(
                f"{self.filled_rows} filled + {self.unfilled_rows} unfilled row(s) do not "
                f"account for {self.candidate_rows} candidate(s)"
            )

    @property
    def filled_fraction(self) -> float | None:
        """Share of candidate rows that were filled, or ``None`` when there were none."""
        return self.filled_rows / self.candidate_rows if self.candidate_rows else None

    @property
    def worst_feature(self) -> str | None:
        """The feature that blocked the most candidate rows, or ``None`` if none did."""
        if not self.missing_by_feature:
            return None
        name, count = max(self.missing_by_feature.items(), key=lambda item: (item[1], item[0]))
        return name if count else None

    def summary(self) -> str:
        """Return a one-line account of the fill, with the A4 hint when it applies."""
        if not self.candidate_rows:
            return (
                f"{self.target}: nothing to fill - {self.observed_rows} observed and "
                f"{self.pre_filled_rows} pre-filled row(s), no gaps."
            )
        text = (
            f"{self.target}: filled {self.filled_rows} of {self.candidate_rows} candidate "
            f"row(s) with {self.method}"
        )
        return text + _feature_hint(self)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary for the run manifest."""
        return {
            "target": self.target,
            "method": self.method,
            "model_version": self.model_version,
            "rows": self.rows,
            "observed_rows": self.observed_rows,
            "pre_filled_rows": self.pre_filled_rows,
            "missing_rows": self.missing_rows,
            "candidate_rows": self.candidate_rows,
            "filled_rows": self.filled_rows,
            "unfilled_rows": self.unfilled_rows,
            "filled_fraction": self.filled_fraction,
            "missing_by_feature": dict(self.missing_by_feature),
            "worst_feature": self.worst_feature,
            "rows_before_fit_origin": self.rows_before_fit_origin,
        }


def _feature_hint(report: FillReport) -> str:
    """Return a sentence naming the feature that blocked the most candidate rows."""
    worst = report.worst_feature
    if worst is None:
        return "."
    count = report.missing_by_feature[worst]
    hint = f"; the feature missing from most unfilled rows is {worst!r} ({count})"
    if worst.endswith(DAILY_STATISTIC_SUFFIXES):
        hint += (
            ". Under daily_statistic_strategy='missing' a day with no visible target "
            "observation has no daily statistics at all, so a gap covering whole days "
            "cannot be filled; choose a reaching strategy deliberately "
            "(docs/method_spec.md 3.4, A4)"
        )
    return hint + "."


# ---------------------------------------------------------------------------
# The result
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class FillResult:
    """A filled frame together with the accounting and configuration behind it.

    :attr:`frame` is a **new** frame - the caller's input is never modified - on
    the validated time axis, carrying the input's own columns plus the six
    provenance columns of :func:`fill_column_names`.

    Equality is identity-based (``eq=False``): the dataclass holds a DataFrame,
    whose ``==`` is element-wise.
    """

    #: The input columns on a validated time axis, plus the provenance columns.
    frame: pd.DataFrame
    #: Row accounting for the fill.
    report: FillReport
    #: The configuration the fill was performed under.
    config: RFRConfig
    #: The time axis of :attr:`frame`, recording the cadence the fill ran at.
    time_axis: TimeAxis

    @property
    def target(self) -> str:
        """The target flux that was filled."""
        return self.report.target

    @property
    def columns(self) -> tuple[str, ...]:
        """The six provenance column names this result added."""
        return fill_column_names(self.target)

    @property
    def original(self) -> pd.Series:
        """The target exactly as it arrived, before anything was filled."""
        return self._column("_original")

    @property
    def filled(self) -> pd.Series:
        """The best available series: observed where observed, predicted in the gaps.

        Missing wherever a gap could not be filled - :attr:`report` says why.
        """
        return self._column("_filled")

    @property
    def is_observed(self) -> pd.Series:
        """Rows carrying a genuine, quality-controlled measurement."""
        return self._column("_is_observed")

    @property
    def is_filled(self) -> pd.Series:
        """Rows whose value in :attr:`filled` was produced by the model."""
        return self._column("_is_filled")

    @property
    def fill_method(self) -> pd.Series:
        """Where each value in :attr:`filled` came from (:class:`FillMethod`)."""
        return self._column("_fill_method")

    def _column(self, suffix: str) -> pd.Series:
        series: pd.Series = self.frame[f"{self.target}{suffix}"]
        return series

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable description of the fill for the run manifest."""
        return {
            "fill": self.report.to_dict(),
            "columns": list(self.columns),
            "time_axis": self.time_axis.to_dict(),
            "config": self.config.to_dict(),
        }


# ---------------------------------------------------------------------------
# The filler
# ---------------------------------------------------------------------------


class RFRGapFiller:
    """Fit one target's Random Forest and fill that target's real gaps.

    One instance is one site and one target, as the paper requires
    (method_spec.md section 5). :meth:`fit` records everything the fill has to
    reproduce - the target, the column mapping, the QC column and the time origin
    of ``time_distance_hours`` - so :meth:`fill` cannot silently build features on
    different conventions from the ones the model learned.

    :param config: the validated run configuration. Supplies the mode and driver
        set, the seed, the feature conventions, the QC rule and the grid.
    """

    def __init__(self, config: RFRConfig) -> None:
        if not isinstance(config, RFRConfig):
            raise ConfigError(f"config must be an RFRConfig, got {type(config).__name__}")
        self._config = config
        self._target: str | None = None
        self._columns: ColumnMap | None = None
        self._qc_column: str | None = None
        self._timestamp: str | None = None
        self._origin: pd.Timestamp | None = None
        self._fit_axis: TimeAxis | None = None
        self._model: RFRModel | None = None

    # -- identity ------------------------------------------------------------

    def __repr__(self) -> str:
        state = "fitted" if self.is_fitted else "unfitted"
        return (
            f"{type(self).__name__}(method={method_label(self._config)}, "
            f"target={self._target!r}, {state})"
        )

    @property
    def config(self) -> RFRConfig:
        """The configuration this filler was built from."""
        return self._config

    @property
    def is_fitted(self) -> bool:
        """Whether :meth:`fit` has completed."""
        return self._model is not None

    @property
    def target(self) -> str:
        """The target flux this filler was fitted for."""
        self._require_fitted("read the target of")
        assert self._target is not None
        return self._target

    @property
    def model(self) -> RFRModel:
        """The fitted :class:`~rfrgapfill.model.RFRModel`."""
        self._require_fitted("read the model of")
        assert self._model is not None
        return self._model

    @property
    def method(self) -> str:
        """This filler's arm: ``"RFR3"``, ``"RFR10"``, ``"ORF3"`` or ``"ORF10"``."""
        return method_label(self._config)

    @property
    def origin(self) -> pd.Timestamp:
        """The timestamp ``time_distance_hours`` is measured from, fixed at :meth:`fit`."""
        self._require_fitted("read the time origin of")
        assert self._origin is not None
        return self._origin

    @property
    def model_version(self) -> str:
        """Identity stamped into ``<target>_model_version`` for every filled value.

        Package version, arm, target and the moment of the fit - enough to tell two
        fills of the same column apart and to find the run that produced one.
        """
        from rfrgapfill import __version__

        return f"rfr-gapfill/{__version__}/{self.method}/{self.target}@{self.model.fitted_at}"

    # -- fitting -------------------------------------------------------------

    def fit(
        self,
        data: pd.DataFrame,
        *,
        target: str,
        column_map: ColumnMap | Mapping[str, str] | None = None,
        qc_column: str | None = None,
        timestamp: str | None = None,
        on_duplicates: DuplicatePolicy | str = DuplicatePolicy.ERROR,
        min_training_rows: int | None = None,
    ) -> RFRGapFiller:
        """Fit the model on ``data``'s eligible observed rows, and return ``self``.

        Eligible means *observed*: the target value is present and finite and, when
        ``qc_column`` is given, its flag is one of ``config.observed_qc_values``. A
        value that arrived already gap-filled is neither trained on nor used to
        compute the daily target statistics, which is the same rule the validation
        workflow applies.

        ``data`` is not modified; it is copied onto a validated time axis first
        (:func:`~rfrgapfill.time.prepare_time_index`), so unsorted input is sorted
        and duplicate timestamps are resolved by ``on_duplicates`` rather than
        silently accepted.

        :param target: the target flux column in ``data``.
        :param column_map: mapping from canonical driver names to ``data``'s own
            columns. Falls back to ``config.column_map``; every driver the mode
            needs must be mapped.
        :param qc_column: the QC/provenance flag distinguishing measured from
            pre-filled target values. Strongly recommended: without it every
            present value counts as a measurement, and the paper's distinction
            cannot be recovered from the values alone.
        :param timestamp: the timestamp column, when ``data`` is not already
            indexed by one. Falls back to ``column_map.timestamp``.
        :param min_training_rows: an explicit floor on usable rows, passed to
            :meth:`~rfrgapfill.model.RFRModel.fit`.
        :raises rfrgapfill.model.InsufficientTrainingDataError: when too few
            eligible rows remain to fit or to cross-validate; its ``report`` says
            where they went.
        """
        if not isinstance(target, str) or not target.strip():
            raise FillError(f"target must be a non-empty string, got {target!r}")
        columns = self._config.require_column_map(column_map)
        frame, axis = self._prepare(
            data,
            columns=columns,
            timestamp=timestamp,
            on_duplicates=on_duplicates,
            context="fit",
        )
        self._require_columns(frame, target=target, qc_column=qc_column, context="fit")

        origin = frame.index.min()
        observed = self._observed(frame, target=target, qc_column=qc_column)
        features = self._features(
            frame, target=target, columns=columns, observed=observed, origin=origin
        )

        eligible = observed.to_numpy()
        model = RFRModel(self._config, target=target).fit(
            features.loc[eligible],
            frame[target].loc[eligible],
            min_training_rows=min_training_rows,
        )

        self._target = target
        self._columns = columns
        self._qc_column = qc_column
        self._timestamp = timestamp if timestamp is not None else columns.timestamp
        self._origin = origin
        self._fit_axis = axis
        self._model = model
        return self

    # -- filling -------------------------------------------------------------

    def fill(
        self,
        data: pd.DataFrame,
        *,
        on_incomplete: IncompletePolicy | str = IncompletePolicy.MISSING,
        refill_pre_filled: bool = False,
        timestamp: str | None = None,
        on_duplicates: DuplicatePolicy | str = DuplicatePolicy.ERROR,
    ) -> FillResult:
        """Predict ``data``'s target gaps and return a :class:`FillResult`.

        ``data`` is never modified. The result's frame is a copy on a validated
        time axis carrying the input's columns plus the six provenance columns: the
        original target, the filled series, the observed and filled masks, the
        per-row method and the model version.

        Observed values are copied through untouched, and so are values the QC flag
        marks as pre-filled unless ``refill_pre_filled=True``. Only rows with no
        value at all are predicted, and only those among them whose predictors are
        all present; the rest stay missing and are counted in
        :attr:`FillResult.report`.

        Features are built from this frame's *own* observed values but with the
        **fit's** time origin, so ``time_distance_hours`` keeps the meaning the
        model learned.

        :param on_incomplete: ``"missing"`` (the default) leaves a row without
            complete predictors unfilled; ``"raise"`` fails instead. Both are
            allowed by method_spec.md section 7; neither invents a driver.
        :param refill_pre_filled: also predict rows whose target value the QC flag
            marks as gap-filled before ingestion, replacing it. Off by default -
            nothing the caller supplied is overwritten unless asked for - but the
            way to run RFR over a FLUXNET target column that is already complete.
        """
        self._require_fitted("fill with")
        assert self._target is not None and self._columns is not None
        target, columns = self._target, self._columns
        policy = IncompletePolicy.coerce(on_incomplete)
        if refill_pre_filled and self._qc_column is None:
            raise ConfigError(
                "refill_pre_filled=True needs a QC column: without one every present value "
                "counts as a measurement, so there is nothing marked pre-filled to replace. "
                "Pass qc_column= to fit()."
            )

        frame, axis = self._prepare(
            data,
            columns=columns,
            timestamp=timestamp if timestamp is not None else self._timestamp,
            on_duplicates=on_duplicates,
            context="fill",
        )
        self._require_columns(frame, target=target, qc_column=self._qc_column, context="fill")
        provenance = fill_column_names(target)
        clashes = [name for name in provenance if name in frame.columns]
        if clashes:
            raise FillError(
                f"the frame already carries provenance column(s) {', '.join(clashes)}, which "
                "a fill would overwrite. Fill the original data rather than a previous "
                "result, or drop those columns first."
            )

        original: pd.Series = frame[target].astype(float)
        values = original.to_numpy(dtype=float)
        present = np.isfinite(values)
        observed = self._observed(frame, target=target, qc_column=self._qc_column)
        is_observed = observed.to_numpy()
        pre_filled = present & ~is_observed
        missing = ~present
        candidates = (missing | pre_filled) if refill_pre_filled else missing

        features = self._features(
            frame, target=target, columns=columns, observed=observed, origin=self.origin
        )
        candidate_features = features.loc[candidates]
        predictions = self.model.predict(candidate_features, on_incomplete=policy)

        filled = values.copy()
        is_filled = np.zeros(len(frame), dtype=bool)
        if len(predictions):
            predicted = predictions.to_numpy(dtype=float)
            usable = np.isfinite(predicted)
            positions = np.flatnonzero(candidates)[usable]
            filled[positions] = predicted[usable]
            is_filled[positions] = True

        method = np.where(
            is_filled,
            self.method,
            np.where(
                is_observed,
                FillMethod.OBSERVED.value,
                np.where(present, FillMethod.PRE_FILLED.value, FillMethod.UNFILLED.value),
            ),
        )
        version = self.model_version
        report = FillReport(
            target=target,
            method=self.method,
            model_version=version,
            rows=len(frame),
            observed_rows=int(is_observed.sum()),
            pre_filled_rows=int(pre_filled.sum()),
            missing_rows=int(missing.sum()),
            candidate_rows=int(candidates.sum()),
            filled_rows=int(is_filled.sum()),
            unfilled_rows=int(candidates.sum()) - int(is_filled.sum()),
            missing_by_feature=_missing_by_feature(candidate_features),
            rows_before_fit_origin=int((frame.index < self.origin).sum()),
        )
        if report.candidate_rows and not report.filled_rows:
            warnings.warn(f"no gap could be filled - {report.summary()}", UserWarning, stacklevel=2)

        # `frame` came out of prepare_time_index, which copies, so writing to it
        # cannot reach the caller's DataFrame.
        frame[provenance[0]] = values
        frame[provenance[1]] = filled
        frame[provenance[2]] = is_observed
        frame[provenance[3]] = is_filled
        frame[provenance[4]] = method
        frame[provenance[5]] = np.where(is_filled, version, None)
        return FillResult(frame=frame, report=report, config=self._config, time_axis=axis)

    # -- manifest ------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable description of the fitted filler.

        Everything method_spec.md section 7 asks a run to record about this stage:
        the configuration and column mapping, the QC rule, the feature conventions,
        the time axis, and the model's own grid, best parameters, seed and row
        accounting.
        """
        self._require_fitted("describe")
        assert self._columns is not None and self._fit_axis is not None
        return {
            "target": self.target,
            "method": self.method,
            "model_version": self.model_version,
            "qc_column": self._qc_column,
            "observed_qc_values": list(self._config.observed_qc_values),
            "timestamp_column": self._timestamp,
            "time_origin": self.origin.isoformat(),
            "column_map": self._columns.to_dict(),
            "time_axis": self._fit_axis.to_dict(),
            "features": describe_features(self._config, target=self.target),
            "model": self.model.to_dict(),
            "config": self._config.to_dict(),
        }

    # -- internals -----------------------------------------------------------

    def _require_fitted(self, action: str) -> None:
        """Raise :class:`~rfrgapfill.model.NotFittedError` unless :meth:`fit` has run."""
        if not self.is_fitted:
            raise NotFittedError(
                f"cannot {action} an unfitted filler; call fit(data, target=...) first"
            )

    def _prepare(
        self,
        data: pd.DataFrame,
        *,
        columns: ColumnMap,
        timestamp: str | None,
        on_duplicates: DuplicatePolicy | str,
        context: str,
    ) -> tuple[pd.DataFrame, TimeAxis]:
        """Return ``data`` copied onto a validated time axis, with that axis."""
        if not isinstance(data, pd.DataFrame):
            raise FillError(f"{context} needs a pandas DataFrame, got {type(data).__name__}")
        return prepare_time_index(
            data,
            timestamp=timestamp if timestamp is not None else columns.timestamp,
            frequency=self._config.time_step,
            on_duplicates=on_duplicates,
        )

    @staticmethod
    def _require_columns(
        frame: pd.DataFrame, *, target: str, qc_column: str | None, context: str
    ) -> None:
        """Raise :class:`FillError` unless the target and QC columns are present."""
        if target not in frame.columns:
            raise FillError(f"{context} needs target column {target!r}, which is not in the data")
        if qc_column is not None and qc_column not in frame.columns:
            raise FillError(
                f"{context} needs QC column {qc_column!r}, which is not in the data. The fit "
                "used it to tell measured values from pre-filled ones, and a fill that "
                "cannot make the same distinction would not be comparable."
            )

    def _observed(self, frame: pd.DataFrame, *, target: str, qc_column: str | None) -> pd.Series:
        """Return the rows carrying a finite, quality-controlled measurement.

        :func:`~rfrgapfill.leakage.observed_target_mask` plus a finiteness test, so
        that "observed" and "has a usable value" cannot disagree about an infinity.
        """
        observed = observed_target_mask(frame, target, qc_column=qc_column, config=self._config)
        finite = np.isfinite(frame[target].to_numpy(dtype=float))
        usable: pd.Series = observed & pd.Series(finite, index=observed.index)
        usable.name = "observed"
        return usable

    def _features(
        self,
        frame: pd.DataFrame,
        *,
        target: str,
        columns: ColumnMap,
        observed: pd.Series,
        origin: pd.Timestamp,
    ) -> pd.DataFrame:
        """Build the feature matrix, with the daily statistics restricted to observations."""
        return build_feature_matrix(
            frame,
            config=self._config,
            target=target if self._config.features.use_receptive_limiter else None,
            column_map=columns,
            available_mask=observed,
            origin=origin,
        )


def _missing_by_feature(features: pd.DataFrame) -> dict[str, int]:
    """Return, per feature, how many rows of ``features`` lack a finite value."""
    if features.empty:
        return {str(name): 0 for name in features.columns}
    finite = np.isfinite(features.to_numpy(dtype=float))
    return {
        str(name): int((~finite[:, position]).sum())
        for position, name in enumerate(features.columns)
    }
