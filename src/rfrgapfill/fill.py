"""The operational fit/fill API.

:class:`RFRGapFiller` is the interface for filling *real* missing data
(``docs/method_spec.md`` section 7, context section 10). It fits one model for
one target at one site on the genuinely observed rows, predicts the rows where
the target is missing, and returns a frame that keeps the two apart.

The rules it enforces are all conservative in the same direction:

* observed target values are never overwritten - the original column comes back
  untouched and the filled series is a new column;
* only rows whose target is actually missing receive a prediction;
* a row lacking a required driver stays missing rather than being filled from an
  imputed input, and the count is reported;
* the input frame is never mutated.

Artificial-gap validation is deliberately *not* done here: it needs the gap mask
built before the target-derived features, which is a different order of
operations. :meth:`RFRGapFiller.validate` hands that to
:func:`rfrgapfill.validation.validate_rfr`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from rfrgapfill.config import ColumnMap, RFRConfig
from rfrgapfill.features import FeatureMatrix, build_feature_matrix
from rfrgapfill.model import ModelError, RFRModel, TrainingReport, model_version
from rfrgapfill.provenance import (
    FILL_METHOD_RFR,
    fill_column_names,
    observed_mask,
    run_manifest,
)
from rfrgapfill.time import DuplicatePolicy, TimeAxis, prepare_time_index

if TYPE_CHECKING:  # pragma: no cover - import cycle broken for type checking only
    from rfrgapfill.validation import ValidationReport

__all__ = ["FillResult", "RFRGapFiller"]


@dataclass(frozen=True)
class FillResult:
    """The outcome of filling one target: the frame, and what was actually filled."""

    #: The input columns plus this target's provenance columns.
    frame: pd.DataFrame
    #: The target that was filled.
    target: str
    #: Rows carrying a genuine, quality-controlled measurement.
    n_observed: int
    #: Rows where the target was missing.
    n_missing: int
    #: Missing rows that received a prediction.
    n_filled: int
    #: Missing rows left unfilled because a required predictor was missing.
    n_unfilled: int
    #: Version string written into ``<target>_model_version``.
    model_version: str

    @property
    def fill_rate(self) -> float:
        """Share of the missing rows that could be filled, in [0, 1]."""
        if self.n_missing == 0:
            return 1.0
        return self.n_filled / self.n_missing

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary for the run manifest."""
        return {
            "target": self.target,
            "n_observed": self.n_observed,
            "n_missing": self.n_missing,
            "n_filled": self.n_filled,
            "n_unfilled": self.n_unfilled,
            "fill_rate": self.fill_rate,
            "model_version": self.model_version,
        }


class RFRGapFiller:
    """Fit a Random Forest on one site's observed flux and fill its real gaps.

    One instance is one site and one target (method_spec.md section 1); construct
    another for another target, and loop for another site. The configuration
    carries everything scientific - mode, hemisphere, seed, grid, receptive
    limiter - so the two calls below take only the data and the column names.

    >>> filler = RFRGapFiller(config)                        # doctest: +SKIP
    >>> filler.fit(df, target="LE", qc_col="LE_QC")          # doctest: +SKIP
    >>> result = filler.fill()                               # doctest: +SKIP
    """

    def __init__(self, config: RFRConfig) -> None:
        if not isinstance(config, RFRConfig):
            raise ModelError(f"config must be an RFRConfig, got {type(config).__name__}")
        self.config = config
        self._model: RFRModel | None = None
        self._target: str | None = None
        self._qc_col: str | None = None
        self._column_map: ColumnMap | None = None
        self._frame: pd.DataFrame | None = None
        self._axis: TimeAxis | None = None
        self._features: FeatureMatrix | None = None

    # -- state ---------------------------------------------------------------

    @property
    def is_fitted(self) -> bool:
        """Whether :meth:`fit` has completed."""
        return self._model is not None

    @property
    def model(self) -> RFRModel:
        """The fitted :class:`~rfrgapfill.model.RFRModel`."""
        if self._model is None:
            raise ModelError("this filler is not fitted yet; call fit() first")
        return self._model

    @property
    def target(self) -> str:
        """The target column this filler was fitted for."""
        if self._target is None:
            raise ModelError("this filler is not fitted yet; call fit() first")
        return self._target

    @property
    def training(self) -> TrainingReport:
        """The training report of the fitted model."""
        return self.model.report

    @property
    def time_axis(self) -> TimeAxis:
        """The validated time axis of the frame the filler was fitted on."""
        if self._axis is None:
            raise ModelError("this filler is not fitted yet; call fit() first")
        return self._axis

    # -- fitting -------------------------------------------------------------

    def fit(
        self,
        data: pd.DataFrame,
        *,
        target: str,
        column_map: ColumnMap | Mapping[str, str] | None = None,
        qc_col: str | None = None,
        timestamp: str | None = None,
        on_duplicates: DuplicatePolicy | str = DuplicatePolicy.ERROR,
    ) -> RFRGapFiller:
        """Fit on the genuinely observed rows of ``target``. Returns ``self``.

        ``qc_col`` names the quality flag distinguishing measured values from ones
        that arrived already gap-filled; without it every finite value is treated
        as observed. ``timestamp`` names the timestamp column when the frame has
        no :class:`~pandas.DatetimeIndex`.

        For operational filling the daily target statistics may see every observed
        value, because nothing is being held out. Validation is the case where
        that would be leakage, and it goes through
        :func:`rfrgapfill.validation.validate_rfr` instead.
        """
        mapping = self.config.require_column_map(column_map)
        frame, axis = prepare_time_index(
            data,
            timestamp=timestamp if timestamp is not None else mapping.timestamp,
            frequency=self.config.time_step,
            on_duplicates=on_duplicates,
        )
        observed = observed_mask(
            frame,
            target,
            qc_column=qc_col,
            observed_qc_values=self.config.observed_qc_values,
        )
        features = build_feature_matrix(
            frame,
            target=target,
            config=self.config,
            column_map=mapping,
            available=observed,
            origin=axis.start,
        )
        values = pd.to_numeric(frame[target], errors="coerce")

        model = RFRModel(self.config)
        model.fit(features.frame.loc[observed], values.loc[observed])

        self._model = model
        self._target = target
        self._qc_col = qc_col
        self._column_map = mapping
        self._frame = frame
        self._axis = axis
        self._features = features
        return self

    # -- filling -------------------------------------------------------------

    def fill(
        self,
        data: pd.DataFrame | None = None,
        *,
        include_uncertainty: bool = False,
    ) -> FillResult:
        """Fill the target's missing rows, preserving every observed value.

        Called without arguments it fills the frame the model was fitted on, which
        is the usual case: a site's own gaps. Passing another frame fills that one
        instead, with features rebuilt against it.

        The returned frame carries the input columns unchanged plus
        ``<target>_original``, ``<target>_filled``, ``<target>_is_observed``,
        ``<target>_is_filled``, ``<target>_fill_method`` and
        ``<target>_model_version``. ``<target>_filled`` is the series to use: the
        observed values where they exist, predictions where they did not, and
        still missing where a predictor was unavailable.

        ``include_uncertainty=True`` adds ``<target>_prediction_std``, the spread
        across the ensemble's trees. It is an enhancement over the published
        method and describes the forest, not the flux measurement.
        """
        model = self.model
        target = self.target
        frame, features = self._materialise(data)

        values = pd.to_numeric(frame[target], errors="coerce")
        observed = observed_mask(
            frame,
            target,
            qc_column=self._qc_col,
            observed_qc_values=self.config.observed_qc_values,
        )
        # Only rows without a usable measurement are candidates: an observed value
        # is never replaced, and a QC-rejected one is not treated as a gap to fill
        # unless it is also missing.
        missing = values.isna()
        predictions = model.predict(features.frame)
        fillable = missing & predictions.notna()

        filled = values.copy()
        filled[fillable] = predictions[fillable]

        version = model_version()
        names = fill_column_names(target)
        output = frame.copy()
        output[names[0]] = values
        output[names[1]] = filled
        output[names[2]] = observed
        output[names[3]] = fillable
        output[names[4]] = pd.Series(
            np.where(fillable.to_numpy(), FILL_METHOD_RFR, None), index=frame.index, dtype=object
        )
        output[names[5]] = pd.Series(
            np.where(fillable.to_numpy(), version, None), index=frame.index, dtype=object
        )
        if include_uncertainty:
            spread = model.predict_std(features.frame)
            output[f"{target}_prediction_std"] = spread.where(fillable)

        return FillResult(
            frame=output,
            target=target,
            n_observed=int(observed.sum()),
            n_missing=int(missing.sum()),
            n_filled=int(fillable.sum()),
            n_unfilled=int((missing & ~fillable).sum()),
            model_version=version,
        )

    def _materialise(self, data: pd.DataFrame | None) -> tuple[pd.DataFrame, FeatureMatrix]:
        """Return the frame to fill and its feature matrix, rebuilding if needed."""
        if data is None:
            if self._frame is None or self._features is None:  # pragma: no cover - guarded above
                raise ModelError("this filler is not fitted yet; call fit() first")
            return self._frame, self._features

        assert self._column_map is not None
        frame, _ = prepare_time_index(
            data,
            timestamp=self._column_map.timestamp,
            frequency=self.config.time_step,
        )
        observed = observed_mask(
            frame,
            self.target,
            qc_column=self._qc_col,
            observed_qc_values=self.config.observed_qc_values,
        )
        features = build_feature_matrix(
            frame,
            target=self.target,
            config=self.config,
            column_map=self._column_map,
            available=observed,
            origin=self.time_axis.start,
        )
        return frame, features

    # -- reporting -----------------------------------------------------------

    def manifest(self) -> dict[str, Any]:
        """Return the run manifest for this fit (method_spec.md section 7)."""
        assert self._features is not None
        return run_manifest(
            self.config,
            targets=[self.target],
            qc_columns={self.target: self._qc_col},
            time_axis=self.time_axis.to_dict(),
            features={self.target: self._features.to_dict()},
            training={self.target: self.training.to_dict()},
        )

    # -- validation ----------------------------------------------------------

    def validate(
        self,
        data: pd.DataFrame,
        *,
        targets: str | Sequence[str] | None = None,
        **kwargs: Any,
    ) -> ValidationReport:
        """Run the artificial-gap validation of method_spec.md section 4 on ``data``.

        A convenience wrapper over :func:`rfrgapfill.validation.validate_rfr` using
        this filler's configuration. It does not use the fitted model: validation
        must fit on the rows left after the gaps are cut, so it trains its own.
        """
        from rfrgapfill.validation import validate_rfr

        chosen = self._target if targets is None else targets
        if chosen is None:
            raise ModelError("pass targets=... or fit the filler first")
        kwargs.setdefault("column_map", self._column_map)
        if self._qc_col is not None and self._target is not None:
            kwargs.setdefault("qc_columns", {self._target: self._qc_col})
        return validate_rfr(data, targets=chosen, config=self.config, **kwargs)
