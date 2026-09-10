"""Random Forest fitting, tuning and persistence.

The estimator layer of ``docs/method_spec.md`` section 5: a
:class:`~sklearn.ensemble.RandomForestRegressor` tuned with
:class:`~sklearn.model_selection.GridSearchCV`, seeded, serialisable, and
deliberately ignorant of where its rows came from. :class:`RFRModel` knows
nothing about artificial gaps, targets or the receptive limiter - it takes a
design matrix and a target vector - which is what lets the same class serve
operational filling and paper validation without a mode flag.

Two things it does insist on.

* **Rows with missing predictors are dropped, and counted.** The paper used
  pre-filled meteorological drivers; this package never fabricates one
  (method_spec.md section 7). Fitting reports how many rows went and why, and
  prediction returns missing for a row it cannot support rather than a number
  built on an imputed input.
* **The grid is configuration, not a constant.** The article states that
  ``GridSearchCV`` was used but never enumerates the grid, so
  :data:`~rfrgapfill.config.DEFAULT_HYPERPARAMETER_GRID` is this package's
  documented default and is recorded as such in every manifest (ambiguity A1).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, cast

import numpy as np
import pandas as pd

from rfrgapfill.config import CVStrategy, RFRConfig

__all__ = [
    "ModelError",
    "RFRModel",
    "TrainingReport",
    "model_version",
]


class ModelError(ValueError):
    """Raised when a model cannot be fitted, used or restored.

    Reports insufficient or inconsistent *data* - too few training rows for the
    requested folds, a design matrix whose columns do not match the fitted
    model's - rather than an invalid configuration, which is
    :class:`~rfrgapfill.schema.ConfigError`.
    """


#: Attribute name under which a saved bundle records its layout version.
_BUNDLE_FORMAT: Final = 1


def model_version() -> str:
    """Return the version string recorded in fill provenance and run manifests.

    Names this package and scikit-learn together, because a prediction is
    reproducible only against both: the same seed and the same grid can still
    give different trees across scikit-learn releases.
    """
    from sklearn import __version__ as sklearn_version

    from rfrgapfill import __version__ as package_version

    return f"rfr-gapfill {package_version} / scikit-learn {sklearn_version}"


# ---------------------------------------------------------------------------
# Training report
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrainingReport:
    """What one fit did: how many rows it used, what it dropped, what it chose.

    Carried on the fitted model and copied into the run manifest. The dropped
    counts are the ones to read first when a fit behaves oddly: a model trained
    on a tenth of the rows offered is usually a column map pointing at a driver
    the site does not record.
    """

    #: Rows offered to :meth:`RFRModel.fit`.
    n_rows: int
    #: Rows dropped because the target was missing.
    n_dropped_missing_target: int
    #: Rows dropped because at least one predictor was missing.
    n_dropped_missing_features: int
    #: Rows actually used to fit.
    n_trained: int
    #: Feature columns, in matrix order.
    feature_names: tuple[str, ...]
    #: Best hyperparameters found by the grid search.
    best_params: Mapping[str, Any]
    #: Mean cross-validated R2 of the best candidate, on the training portion only.
    best_score: float
    #: The grid that was searched. Our documented default unless overridden (A1).
    hyperparameter_grid: Mapping[str, tuple[Any, ...]]
    #: Cross-validation strategy and fold count used inside the search (A5).
    cv_strategy: CVStrategy
    cv_folds: int
    #: Seed every stochastic component was given.
    random_state: int

    @property
    def n_dropped(self) -> int:
        """Rows dropped for any reason."""
        return self.n_dropped_missing_target + self.n_dropped_missing_features

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary for the run manifest."""
        return {
            "n_rows": self.n_rows,
            "n_dropped_missing_target": self.n_dropped_missing_target,
            "n_dropped_missing_features": self.n_dropped_missing_features,
            "n_trained": self.n_trained,
            "feature_names": list(self.feature_names),
            "best_params": dict(self.best_params),
            "best_score": self.best_score,
            "hyperparameter_grid": {
                key: list(values) for key, values in self.hyperparameter_grid.items()
            },
            "cv_strategy": self.cv_strategy.value,
            "cv_folds": self.cv_folds,
            "random_state": self.random_state,
        }


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class RFRModel:
    """A seeded, grid-searched Random Forest over a fixed feature matrix.

    One instance is one model for one target at one site (method_spec.md section
    1). Fit it with a design matrix from
    :func:`rfrgapfill.features.build_feature_matrix` and the target values on the
    same index; the feature names are learnt from the frame's columns and are
    enforced on every later call, so a matrix built from a different mode or a
    different receptive-limiter setting fails loudly instead of silently scoring
    the wrong columns.
    """

    def __init__(self, config: RFRConfig) -> None:
        if not isinstance(config, RFRConfig):
            raise ModelError(f"config must be an RFRConfig, got {type(config).__name__}")
        self.config = config
        self._estimator: Any | None = None
        self._report: TrainingReport | None = None

    # -- state ---------------------------------------------------------------

    @property
    def is_fitted(self) -> bool:
        """Whether :meth:`fit` has completed."""
        return self._estimator is not None

    @property
    def report(self) -> TrainingReport:
        """The :class:`TrainingReport` from the last fit."""
        return self._require_fitted()[1]

    def get_feature_names(self) -> tuple[str, ...]:
        """Return the feature columns this model expects, in order."""
        return self.report.feature_names

    def get_best_params(self) -> dict[str, Any]:
        """Return the hyperparameters the grid search selected."""
        return dict(self.report.best_params)

    def feature_importances(self) -> pd.Series:
        """Return impurity-based feature importances, indexed by feature name.

        A diagnostic, not a causal statement: importances are biased towards
        high-cardinality features, and the daily target statistics will usually
        dominate simply because they carry the target's own scale.
        """
        estimator, report = self._require_fitted()
        importances: pd.Series = pd.Series(
            np.asarray(estimator.feature_importances_, dtype=float),
            index=pd.Index(report.feature_names, name="feature"),
            name="importance",
        )
        return importances

    # -- fitting -------------------------------------------------------------

    def fit(self, X: pd.DataFrame, y: object) -> RFRModel:
        """Fit the forest on the rows of ``X`` that carry a target and every predictor.

        Returns ``self``. Rows missing the target or any predictor are dropped and
        counted in :attr:`report`; nothing is imputed. Raises :class:`ModelError`
        when too few rows survive to run the configured folds, because a grid
        search on three rows reports a number that means nothing.
        """
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import GridSearchCV

        frame, target = _aligned(X, y)
        names = tuple(str(column) for column in frame.columns)

        has_target = target.notna().to_numpy()
        complete = frame.notna().all(axis=1).to_numpy()
        usable = has_target & complete

        n_rows = len(frame)
        n_missing_target = int(np.sum(~has_target))
        n_missing_features = int(np.sum(has_target & ~complete))
        n_trained = int(np.sum(usable))

        minimum = max(self.config.cv_folds, 2)
        if n_trained < minimum:
            raise ModelError(
                f"only {n_trained} of {n_rows} row(s) can be trained on "
                f"({n_missing_target} missing the target, {n_missing_features} missing at "
                f"least one predictor), but cv_folds={self.config.cv_folds} needs at least "
                f"{minimum}. Check the column map, or lower cv_folds."
            )

        estimator = RandomForestRegressor(
            random_state=self.config.random_state,
            n_jobs=self.config.n_jobs,
        )
        search = GridSearchCV(
            estimator=estimator,
            param_grid={
                key: list(values) for key, values in self.config.hyperparameter_grid.items()
            },
            cv=self._cross_validator(n_trained),
            scoring="r2",
            n_jobs=self.config.n_jobs,
            refit=True,
        )
        search.fit(frame.to_numpy()[usable], target.to_numpy()[usable])

        self._estimator = search.best_estimator_
        self._report = TrainingReport(
            n_rows=n_rows,
            n_dropped_missing_target=n_missing_target,
            n_dropped_missing_features=n_missing_features,
            n_trained=n_trained,
            feature_names=names,
            best_params=dict(search.best_params_),
            best_score=float(search.best_score_),
            # A plain dict, not the configuration's read-only mapping: the report
            # travels inside saved model bundles, and mappingproxy cannot pickle.
            hyperparameter_grid={
                key: tuple(values) for key, values in self.config.hyperparameter_grid.items()
            },
            cv_strategy=self.config.cv,
            cv_folds=self.config.cv_folds,
            random_state=self.config.random_state,
        )
        return self

    def _cross_validator(self, n_samples: int) -> Any:
        """Return the fold generator for the configured strategy (A5)."""
        from sklearn.model_selection import KFold, TimeSeriesSplit

        folds = min(self.config.cv_folds, n_samples)
        if self.config.cv is CVStrategy.TIME_SERIES_SPLIT:
            return TimeSeriesSplit(n_splits=max(folds, 2))
        if self.config.cv_shuffle:
            return KFold(n_splits=folds, shuffle=True, random_state=self.config.random_state)
        return KFold(n_splits=folds, shuffle=False)

    # -- prediction ----------------------------------------------------------

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Return predictions for ``X``, missing on rows lacking a predictor.

        The result is aligned to ``X``'s index and is NaN wherever a feature is
        missing, so a caller can place predictions back into a series without
        first working out which rows were supportable (method_spec.md section 7).
        """
        estimator, report = self._require_fitted()
        frame = self._ordered(X, report.feature_names)
        complete = frame.notna().all(axis=1).to_numpy()

        values = np.full(len(frame), np.nan)
        if complete.any():
            values[complete] = estimator.predict(frame.to_numpy()[complete])
        predictions: pd.Series = pd.Series(values, index=frame.index, name="prediction")
        return predictions

    def predict_std(self, X: pd.DataFrame) -> pd.Series:
        """Return the standard deviation of the per-tree predictions for ``X``.

        An optional uncertainty diagnostic (context section 10) and an enhancement
        over the published method: it describes the ensemble's internal spread,
        not the flux measurement's uncertainty, and it is not the paper's bias
        confidence interval. Missing on the same rows as :meth:`predict`.
        """
        estimator, report = self._require_fitted()
        frame = self._ordered(X, report.feature_names)
        complete = frame.notna().all(axis=1).to_numpy()

        values = np.full(len(frame), np.nan)
        if complete.any():
            rows = frame.to_numpy()[complete]
            per_tree = np.stack([tree.predict(rows) for tree in estimator.estimators_])
            values[complete] = per_tree.std(axis=0, ddof=0)
        spread: pd.Series = pd.Series(values, index=frame.index, name="prediction_std")
        return spread

    def _ordered(self, X: pd.DataFrame, names: tuple[str, ...]) -> pd.DataFrame:
        """Return ``X``'s columns in the fitted order, raising on a mismatch."""
        if not isinstance(X, pd.DataFrame):
            raise ModelError(f"X must be a DataFrame of named features, got {type(X).__name__}")
        missing = [name for name in names if name not in X.columns]
        if missing:
            raise ModelError(
                f"the design matrix is missing feature(s) this model was fitted on: "
                f"{', '.join(missing)}. It was fitted on: {', '.join(names)}"
            )
        return cast("pd.DataFrame", X.loc[:, list(names)])

    def _require_fitted(self) -> tuple[Any, TrainingReport]:
        """Return the estimator and report, or raise if the model is not fitted."""
        if self._estimator is None or self._report is None:
            raise ModelError("this model is not fitted yet; call fit() first")
        return self._estimator, self._report

    # -- persistence ---------------------------------------------------------

    def save(self, path: str | Path) -> Path:
        """Write the fitted model, its configuration and its report to ``path``.

        Uses :mod:`joblib`. The bundle carries the configuration and the version
        string as well as the estimator, so a restored model can be traced back to
        the run that produced it rather than being an anonymous forest.
        """
        import joblib

        estimator, report = self._require_fitted()
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "format": _BUNDLE_FORMAT,
                "model_version": model_version(),
                "config": self.config,
                "estimator": estimator,
                "report": report,
            },
            destination,
        )
        return destination

    @classmethod
    def load(cls, path: str | Path) -> RFRModel:
        """Restore a model written by :meth:`save`.

        The restored model reproduces the saved one's predictions exactly: the
        fitted trees are restored, not refitted.
        """
        import joblib

        bundle = joblib.load(Path(path))
        if not isinstance(bundle, dict) or bundle.get("format") != _BUNDLE_FORMAT:
            raise ModelError(
                f"{path} is not an rfr-gapfill model bundle, or was written by an "
                "incompatible version"
            )
        model = cls(bundle["config"])
        model._estimator = bundle["estimator"]
        model._report = bundle["report"]
        return model


def _aligned(X: pd.DataFrame, y: object) -> tuple[pd.DataFrame, pd.Series]:
    """Return ``X`` and ``y`` as a frame and a float Series on a shared index."""
    if not isinstance(X, pd.DataFrame):
        raise ModelError(f"X must be a DataFrame of named features, got {type(X).__name__}")
    if X.shape[1] == 0:
        raise ModelError("X has no feature columns")

    if isinstance(y, pd.Series):
        if not y.index.equals(X.index):
            raise ModelError(
                "X and y must share an index; reindex the target onto the feature matrix "
                "rather than relying on row order"
            )
        target = y
    else:
        array = np.asarray(y)
        if array.ndim != 1 or array.size != len(X):
            raise ModelError(f"y must be 1-D of length {len(X)}, got shape {array.shape}")
        target = pd.Series(array, index=X.index)
    return X.astype(float), target.astype(float)
