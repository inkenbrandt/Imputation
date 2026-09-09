"""Random Forest fitting, hyperparameter search and persistence.

The low-level model layer of ``docs/method_spec.md`` section 5. It wraps
``sklearn.ensemble.RandomForestRegressor`` in a deterministic ``GridSearchCV``
whose grid comes from configuration, and it knows nothing about artificial gaps,
holdout masks or filling: it is handed a feature matrix and a target vector and
returns predictions. That separation is the point - the same class fits the
paper's validation arm, the ORF benchmark arm and an operational fill, and the
workflow above it decides which rows those are.

Three behaviours here are documented choices rather than paper statements:

* **The grid is ours** (ambiguity A1). The article states that ``GridSearchCV``
  was used and does not enumerate the grid, so
  :data:`~rfrgapfill.config.DEFAULT_HYPERPARAMETER_GRID` is a package default and
  is reported as one. An archived ``fluxlib`` grid may be added later as a named
  preset; nothing here may be labelled paper exact.
* **Rows with a non-finite predictor are dropped at fit and left unpredicted at
  predict.** Recent scikit-learn forests accept ``NaN`` natively, which would
  quietly substitute an undocumented imputation rule for the specification's
  "either fail clearly or leave predictions missing for rows lacking predictors"
  (method_spec.md section 7). :class:`FitReport` records how many rows went and
  which feature was responsible.
* **``n_jobs`` is given to the forest, not to the search.** One level of
  parallelism, so a configured ``n_jobs=-1`` cannot multiply into
  ``folds x candidates x trees`` oversubscribed workers, and prediction benefits
  from it too.

scikit-learn and joblib are imported inside the functions that need them, so that
``import rfrgapfill`` stays cheap - the same rule the rest of the package follows.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final

import numpy as np
import pandas as pd

from rfrgapfill.config import DEFAULT_HYPERPARAMETER_GRID, CVStrategy, RFRConfig
from rfrgapfill.features import DAILY_STATISTIC_SUFFIXES, feature_names
from rfrgapfill.schema import ConfigError, FrozenRecord, coerce_enum

if TYPE_CHECKING:  # pragma: no cover - typing only; sklearn is imported lazily
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import GridSearchCV

__all__ = [
    "SAVE_FORMAT_VERSION",
    "FitReport",
    "IncompletePolicy",
    "InsufficientTrainingDataError",
    "ModelError",
    "NotFittedError",
    "RFRModel",
]

#: Version of the ``joblib`` payload written by :meth:`RFRModel.save`. Bumped
#: whenever the stored layout changes; :meth:`RFRModel.load` refuses anything else
#: rather than reconstructing a model from fields it cannot interpret.
SAVE_FORMAT_VERSION: Final[int] = 1

#: Marker identifying a payload as this package's, so loading an arbitrary pickle
#: fails with a clear message instead of an obscure attribute error.
_PAYLOAD_KIND: Final[str] = "rfrgapfill.model.RFRModel"

#: Scoring used by the grid search: scikit-learn's default for a regressor, which
#: is ``RandomForestRegressor.score`` - the coefficient of determination.
_SCORING: Final[str] = "r2"


class ModelError(RuntimeError):
    """Raised when a model cannot be fitted, used or restored as asked.

    Distinct from :class:`~rfrgapfill.schema.ConfigError` (an invalid
    configuration) and :class:`~rfrgapfill.features.FeatureError` (input data a
    transformer cannot use): a :class:`ModelError` reports the model layer's own
    contract - an unfitted model, a feature matrix that does not match the one the
    model was fitted on, an unreadable model file.
    """


class NotFittedError(ModelError):
    """Raised when an operation needs a fitted model and the model is not fitted."""


class InsufficientTrainingDataError(ModelError):
    """Raised when too few usable rows remain to fit or to cross-validate.

    Carries the :class:`FitReport` that explains where the rows went, so a caller
    can tell "the target was never measured here" from "the daily statistics are
    missing because the gap covers whole days" (ambiguity A4).
    """

    def __init__(self, message: str, *, report: FitReport) -> None:
        super().__init__(message)
        #: Row accounting for the attempted fit.
        self.report = report


class IncompletePolicy(str, Enum):
    """What :meth:`RFRModel.predict` does with a row missing a predictor.

    The specification allows either behaviour (method_spec.md section 7): leave
    the prediction missing, or fail clearly. Never silently invent the absent
    driver.
    """

    #: Return ``NaN`` for the row. The default; the fill layer marks it unfilled.
    MISSING = "missing"
    #: Raise :class:`ModelError`, naming the rows and the features responsible.
    RAISE = "raise"

    @classmethod
    def coerce(cls, value: object) -> IncompletePolicy:
        """Return ``value`` as an :class:`IncompletePolicy`, accepting ``"nan"``."""
        aliases = {"nan": cls.MISSING, "skip": cls.MISSING, "error": cls.RAISE}
        return coerce_enum(cls, value, field_name="on_incomplete", aliases=aliases)


# ---------------------------------------------------------------------------
# Row accounting
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FitReport(FrozenRecord):
    """What happened to the rows offered to :meth:`RFRModel.fit`.

    The specification requires explicit reporting of rows excluded for missing
    predictors (method_spec.md section 5), so this is part of the fitted model and
    of the run manifest rather than a log line.

    :attr:`dropped_missing_target` and :attr:`dropped_missing_features` are
    exclusive categories - a row with neither a target nor complete features counts
    only against the target - so they and :attr:`fitted_rows` sum to :attr:`rows`.
    :attr:`missing_by_feature` is a per-column diagnostic over the rows that had a
    usable target, and its counts overlap where one row is missing several
    features.
    """

    #: Rows offered to ``fit``.
    rows: int
    #: Rows actually used to fit the forest.
    fitted_rows: int
    #: Rows dropped because the target was missing or not finite.
    dropped_missing_target: int
    #: Rows with a usable target dropped because a predictor was missing.
    dropped_missing_features: int
    #: Per feature, how many rows with a usable target lacked it.
    missing_by_feature: Mapping[str, int]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "missing_by_feature", MappingProxyType(dict(self.missing_by_feature))
        )

    @property
    def dropped_rows(self) -> int:
        """Rows excluded for any reason."""
        return self.rows - self.fitted_rows

    @property
    def usable_fraction(self) -> float | None:
        """Share of offered rows that were fitted, or ``None`` when none were offered."""
        return self.fitted_rows / self.rows if self.rows else None

    @property
    def worst_feature(self) -> str | None:
        """The feature responsible for the most exclusions, or ``None`` when none was."""
        if not self.missing_by_feature:
            return None
        name, count = max(self.missing_by_feature.items(), key=lambda item: (item[1], item[0]))
        return name if count else None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary for the run manifest."""
        return {
            "rows": self.rows,
            "fitted_rows": self.fitted_rows,
            "dropped_rows": self.dropped_rows,
            "dropped_missing_target": self.dropped_missing_target,
            "dropped_missing_features": self.dropped_missing_features,
            "usable_fraction": self.usable_fraction,
            "missing_by_feature": dict(self.missing_by_feature),
        }


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------


class RFRModel:
    """A Random Forest fitted through a configured ``GridSearchCV``.

    One instance is one target's model, as the paper requires (method_spec.md
    section 5). It holds the configuration it was built from, the feature order it
    was fitted on, the tuned estimator, and the row accounting of the fit.

    The expected feature order is taken from
    :func:`~rfrgapfill.features.feature_names` whenever it can be derived - which
    needs the target, because the daily statistics are target-specific - so a
    matrix built for a different target, or with the receptive limiter in the other
    state, is rejected at ``fit`` rather than quietly fitted on the wrong columns.

    :param config: the run configuration. Supplies the seed, ``n_jobs``, the
        hyperparameter grid and the CV policy, and travels with the saved model.
    :param target: the target flux this model predicts. Optional but strongly
        recommended: it names the daily-statistic features and the prediction
        series, and is recorded in the manifest.
    :param feature_names: the expected feature order, for when it cannot be derived
        from ``config`` and ``target`` (the receptive limiter is on and no target
        was given) or when a caller assembles a matrix itself. Checked against the
        derived order when both are available.
    """

    def __init__(
        self,
        config: RFRConfig,
        *,
        target: str | None = None,
        feature_names: Sequence[str] | None = None,
    ) -> None:
        if not isinstance(config, RFRConfig):
            raise ConfigError(f"config must be an RFRConfig, got {type(config).__name__}")
        if target is not None and (not isinstance(target, str) or not target.strip()):
            raise ConfigError(f"target must be a non-empty string or None, got {target!r}")

        self._config = config
        self._target = target
        self._expected_names = _resolve_expected_names(
            config, target=target, declared=feature_names
        )

        self._estimator: RandomForestRegressor | None = None
        self._search: GridSearchCV | None = None
        self._fitted_names: tuple[str, ...] | None = None
        self._report: FitReport | None = None
        self._fitted_at: str | None = None
        self._versions: Mapping[str, str] = MappingProxyType({})

    # -- identity ------------------------------------------------------------

    def __repr__(self) -> str:
        state = "fitted" if self.is_fitted else "unfitted"
        return (
            f"{type(self).__name__}(mode={self._config.rfr_mode.value}, "
            f"target={self._target!r}, {state})"
        )

    @property
    def config(self) -> RFRConfig:
        """The configuration this model was built from."""
        return self._config

    @property
    def target(self) -> str | None:
        """The target flux this model predicts, when it was named."""
        return self._target

    @property
    def is_fitted(self) -> bool:
        """Whether :meth:`fit` has completed."""
        return self._estimator is not None

    @property
    def fit_report(self) -> FitReport:
        """Row accounting for the fit (method_spec.md section 5)."""
        if self._report is None:
            raise NotFittedError("this model is not fitted yet, so it has no fit report")
        return self._report

    @property
    def estimator(self) -> RandomForestRegressor:
        """The tuned ``RandomForestRegressor``, refitted on the whole training set."""
        if self._estimator is None:
            raise NotFittedError("this model is not fitted yet; call fit(X, y) first")
        return self._estimator

    @property
    def best_score(self) -> float:
        """Mean cross-validated score of the winning candidate (r2)."""
        self._require_fitted("read the best score of")
        assert self._search is not None
        return float(self._search.best_score_)

    def get_feature_names(self) -> tuple[str, ...]:
        """Return the feature order this model was fitted on, or expects to be.

        Available before fitting whenever it could be derived from the
        configuration, so a caller can build a matrix in the right order rather
        than discover that order from a failure.
        """
        if self._fitted_names is not None:
            return self._fitted_names
        if self._expected_names is not None:
            return self._expected_names
        raise NotFittedError(
            "the feature order is not known yet: this model is unfitted and its feature "
            "names could not be derived from the configuration. Pass target= (the daily "
            "statistics are target-specific) or feature_names= to the constructor."
        )

    def get_best_params(self) -> dict[str, Any]:
        """Return the hyperparameters ``GridSearchCV`` selected.

        Only the searched parameters. The seed and ``n_jobs`` come from the
        configuration and are never searched, so :meth:`to_dict` reports those.
        """
        self._require_fitted("read the best parameters of")
        assert self._search is not None
        return dict(self._search.best_params_)

    def feature_importances(self) -> pd.Series:
        """Return the forest's impurity-based feature importances, in feature order.

        A diagnostic, not a scientific claim: impurity importances are biased
        towards high-cardinality features, and the paper makes no use of them.
        """
        importances: pd.Series = pd.Series(
            np.asarray(self.estimator.feature_importances_, dtype=float),
            index=list(self.get_feature_names()),
            name="importance",
        )
        return importances

    # -- fitting -------------------------------------------------------------

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray | Sequence[float],
        *,
        min_training_rows: int | None = None,
    ) -> RFRModel:
        """Fit the forest on the usable rows of ``X`` and ``y``, and return ``self``.

        Rows whose target or any predictor is missing or non-finite are excluded
        and reported in :attr:`fit_report`; nothing is imputed here. Which rows are
        *eligible* - observed, and not withheld by an artificial gap - was decided
        before this call, by the workflow that built ``X``.

        :param X: feature matrix in the order of :meth:`get_feature_names`,
            normally from :func:`~rfrgapfill.features.build_feature_matrix`.
        :param y: the target values, aligned with ``X`` by index when both carry
            one and by position otherwise.
        :param min_training_rows: an explicit floor on usable rows, for callers
            that will not accept a model fitted on very few. The
            cross-validation's own requirement always applies on top of it.
        """
        frame = self._as_feature_frame(X, context="fit")
        names = tuple(str(column) for column in frame.columns)
        self._check_names(names, expected=self._expected_names, context="fit")
        values = frame.to_numpy(dtype=float)
        target_values = _as_target_array(y, index=frame.index, rows=len(frame))

        finite_features = np.isfinite(values)
        complete_rows = finite_features.all(axis=1)
        target_ok = np.isfinite(target_values)
        usable = complete_rows & target_ok

        report = FitReport(
            rows=len(frame),
            fitted_rows=int(usable.sum()),
            dropped_missing_target=int((~target_ok).sum()),
            dropped_missing_features=int((target_ok & ~complete_rows).sum()),
            missing_by_feature={
                name: int((~finite_features[target_ok, position]).sum())
                for position, name in enumerate(names)
            },
        )
        self._require_enough_rows(report, min_training_rows=min_training_rows)

        search = self._build_search()
        search.fit(values[usable], target_values[usable])

        self._search = search
        self._estimator = search.best_estimator_
        self._fitted_names = names
        self._report = report
        self._fitted_at = datetime.now(tz=timezone.utc).isoformat(timespec="seconds")
        self._versions = MappingProxyType(_environment_versions())
        return self

    def predict(
        self,
        X: pd.DataFrame | np.ndarray,
        *,
        on_incomplete: IncompletePolicy | str = IncompletePolicy.MISSING,
    ) -> pd.Series:
        """Predict for every row of ``X``, in the order it was given.

        Returns a float series on ``X``'s own index, so predictions can be joined
        straight back onto the source frame. Rows missing a predictor are ``NaN``
        by default and never receive an invented value; pass
        ``on_incomplete="raise"`` to fail instead (method_spec.md section 7).
        """
        self._require_fitted("predict with")
        policy = IncompletePolicy.coerce(on_incomplete)
        frame = self._as_feature_frame(X, context="predict")
        names = tuple(str(column) for column in frame.columns)
        self._check_names(names, expected=self.get_feature_names(), context="predict")

        values = frame.to_numpy(dtype=float)
        complete = np.isfinite(values).all(axis=1)
        incomplete_count = int((~complete).sum())
        if incomplete_count and policy is IncompletePolicy.RAISE:
            culprits = [
                name
                for position, name in enumerate(names)
                if not np.isfinite(values[~complete, position]).all()
            ]
            raise ModelError(
                f"{incomplete_count} of {len(frame)} row(s) are missing a predictor and "
                f"cannot be predicted: {', '.join(culprits)}. Pass "
                'on_incomplete="missing" to leave those predictions missing instead '
                "(docs/method_spec.md section 7)."
            )

        predictions = np.full(len(frame), np.nan, dtype=float)
        if complete.any():
            predictions[complete] = np.asarray(
                self.estimator.predict(values[complete]), dtype=float
            )
        result: pd.Series = pd.Series(predictions, index=frame.index, name=self.prediction_name)
        return result

    @property
    def prediction_name(self) -> str:
        """Name given to the series :meth:`predict` returns."""
        return "prediction" if self._target is None else f"{self._target}_predicted"

    # -- persistence ---------------------------------------------------------

    def save(self, path: str | Path) -> Path:
        """Write the fitted model, its configuration and its provenance to ``path``.

        Uses ``joblib``, as the specification requires. The payload records the
        package, Python, scikit-learn, numpy and pandas versions it was written
        under, so a model restored in a different environment can be recognised as
        such.
        """
        import joblib

        self._require_fitted("save")
        destination = Path(path)
        if str(destination.parent) not in {"", "."}:
            destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "kind": _PAYLOAD_KIND,
            "format_version": SAVE_FORMAT_VERSION,
            "config": self._config,
            "target": self._target,
            "feature_names": list(self.get_feature_names()),
            "search": self._search,
            "fit_report": self._report,
            "fitted_at": self._fitted_at,
            "versions": dict(self._versions),
        }
        joblib.dump(payload, destination)
        return destination

    @classmethod
    def load(cls, path: str | Path) -> RFRModel:
        """Restore a model written by :meth:`save`.

        The configuration is revalidated as it is unpickled, so a file carrying
        settings the constructor would reject fails here rather than predicting
        under them. A scikit-learn version different from the one the model was
        fitted under is reported as a warning, not silently accepted: a pickled
        estimator is not guaranteed to survive a version change.
        """
        import joblib

        source = Path(path)
        payload = joblib.load(source)
        if not isinstance(payload, Mapping) or payload.get("kind") != _PAYLOAD_KIND:
            raise ModelError(f"{source} is not an rfrgapfill model file")
        stored_version = payload.get("format_version")
        if stored_version != SAVE_FORMAT_VERSION:
            raise ModelError(
                f"{source} was written in model format {stored_version!r}, and this version "
                f"of rfr-gapfill reads format {SAVE_FORMAT_VERSION}. Refit the model from "
                "its configuration rather than reinterpreting the stored fields."
            )

        model = cls(
            payload["config"],
            target=payload["target"],
            feature_names=payload["feature_names"],
        )
        search = payload["search"]
        model._search = search
        model._estimator = search.best_estimator_
        model._fitted_names = tuple(payload["feature_names"])
        model._report = payload["fit_report"]
        model._fitted_at = payload["fitted_at"]
        model._versions = MappingProxyType(dict(payload["versions"]))
        _warn_on_version_drift(source, stored=model._versions)
        return model

    # -- manifest ------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable description for the run manifest.

        Records the grid that was searched, the parameters it chose, the CV policy,
        the seed and the row accounting - everything section 7 of the specification
        requires of the model stage - and states explicitly that the grid is a
        package default rather than a paper value (A1).
        """
        config = self._config
        grid = {key: list(values) for key, values in config.hyperparameter_grid.items()}
        described: dict[str, Any] = {
            "estimator": "sklearn.ensemble.RandomForestRegressor",
            "target": self._target,
            "mode": config.rfr_mode.value,
            "feature_names": list(self._fitted_names or self._expected_names or ()),
            "hyperparameter_grid": grid,
            "hyperparameter_grid_is_package_default": (
                grid == {key: list(values) for key, values in DEFAULT_HYPERPARAMETER_GRID.items()}
            ),
            # A1: the article does not enumerate its grid, so neither the grid we
            # ship nor one a user supplies may be called the paper's search.
            "hyperparameter_grid_is_paper_exact": False,
            "cv_strategy": config.cv.value,
            "cv_folds": config.cv_folds,
            "cv_shuffle": config.cv_shuffle,
            "scoring": _SCORING,
            "random_state": config.random_state,
            "n_jobs": config.n_jobs,
            "is_fitted": self.is_fitted,
        }
        if self.is_fitted:
            described.update(
                {
                    "best_params": self.get_best_params(),
                    "best_score": self.best_score,
                    "estimator_params": _serialisable_params(self.estimator),
                    "fit": self.fit_report.to_dict(),
                    "fitted_at": self._fitted_at,
                    "versions": dict(self._versions),
                }
            )
        return described

    # -- internals -----------------------------------------------------------

    def _require_fitted(self, action: str) -> None:
        """Raise :class:`NotFittedError` unless the model has been fitted."""
        if not self.is_fitted:
            raise NotFittedError(f"cannot {action} an unfitted model; call fit(X, y) first")

    def _as_feature_frame(self, X: object, *, context: str) -> pd.DataFrame:
        """Return ``X`` as a float feature frame, naming an array's columns if needed."""
        if isinstance(X, pd.DataFrame):
            frame = X
        else:
            array = np.asarray(X, dtype=float)
            if array.ndim != 2:
                raise ModelError(
                    f"{context} needs a 2-dimensional feature matrix, got an array with "
                    f"{array.ndim} dimension(s)"
                )
            try:
                names = self.get_feature_names()
            except NotFittedError as error:
                raise ModelError(
                    f"{context} was given a plain array and this model does not know its "
                    "feature names yet; pass a DataFrame, or give target=/feature_names= "
                    "to the constructor"
                ) from error
            if array.shape[1] != len(names):
                raise ModelError(
                    f"{context} was given {array.shape[1]} column(s) but this model uses "
                    f"{len(names)}: {', '.join(names)}"
                )
            frame = pd.DataFrame(array, columns=list(names))

        if frame.columns.has_duplicates:
            duplicated = sorted({str(name) for name in frame.columns[frame.columns.duplicated()]})
            raise ModelError(
                f"{context} was given duplicate feature column(s): {', '.join(duplicated)}"
            )
        try:
            numeric: pd.DataFrame = frame.astype(float)
        except (TypeError, ValueError) as error:
            raise ModelError(
                f"{context} needs numeric features, and a column could not be read as float. "
                "The two categorical features are ordinal float codes when the matrix comes "
                "from build_feature_matrix(encode=True)."
            ) from error
        return numeric

    def _check_names(
        self,
        names: tuple[str, ...],
        *,
        expected: tuple[str, ...] | None,
        context: str,
    ) -> None:
        """Raise unless ``names`` is exactly ``expected``, in the same order."""
        if expected is None or names == expected:
            return
        if set(names) == set(expected):
            raise ModelError(
                f"{context} was given the right features in the wrong order: {list(names)} "
                f"instead of {list(expected)}. The order is part of the model - build the "
                "matrix with build_feature_matrix(), or reorder the columns to match "
                "feature_names()."
            )
        missing = [name for name in expected if name not in names]
        unexpected = [name for name in names if name not in expected]
        detail = []
        if missing:
            detail.append(f"missing {', '.join(missing)}")
        if unexpected:
            detail.append(f"unexpected {', '.join(unexpected)}")
        raise ModelError(
            f"{context} was given features that do not match this model: {'; '.join(detail)}. "
            f"Expected {list(expected)}."
        )

    def _minimum_rows(self) -> int:
        """Return the fewest rows the configured cross-validation can split."""
        folds = self._config.cv_folds
        # TimeSeriesSplit keeps a training block ahead of the first test fold, so it
        # needs one more sample than it has splits; KFold needs one per fold.
        return folds + 1 if self._config.cv is CVStrategy.TIME_SERIES_SPLIT else folds

    def _require_enough_rows(self, report: FitReport, *, min_training_rows: int | None) -> None:
        """Raise :class:`InsufficientTrainingDataError` unless the fit can proceed."""
        if min_training_rows is not None and (
            isinstance(min_training_rows, bool) or not isinstance(min_training_rows, int)
        ):
            raise ConfigError(
                f"min_training_rows must be an integer or None, got {min_training_rows!r}"
            )

        if report.fitted_rows == 0:
            raise InsufficientTrainingDataError(
                "no usable training rows: "
                f"{report.rows} row(s) offered, {report.dropped_missing_target} without a "
                f"target value and {report.dropped_missing_features} missing a predictor"
                + _feature_hint(report),
                report=report,
            )

        required = self._minimum_rows()
        if report.fitted_rows < required:
            raise InsufficientTrainingDataError(
                f"{report.fitted_rows} usable training row(s) cannot be split into "
                f"{self._config.cv_folds} {self._config.cv.value} fold(s), which needs at "
                f"least {required}. Supply more observations or lower cv_folds"
                + _feature_hint(report),
                report=report,
            )

        if min_training_rows is not None and report.fitted_rows < min_training_rows:
            raise InsufficientTrainingDataError(
                f"{report.fitted_rows} usable training row(s) is below the requested minimum "
                f"of {min_training_rows} ({report.rows} offered)" + _feature_hint(report),
                report=report,
            )

    def _build_estimator(self) -> RandomForestRegressor:
        """Return an unfitted forest carrying the configured seed and ``n_jobs``."""
        from sklearn.ensemble import RandomForestRegressor

        estimator: RandomForestRegressor = RandomForestRegressor(
            random_state=self._config.random_state,
            n_jobs=self._config.n_jobs,
        )
        return estimator

    def _build_cv(self) -> Any:
        """Return the configured cross-validation splitter (ambiguity A5)."""
        from sklearn.model_selection import KFold, TimeSeriesSplit

        config = self._config
        if config.cv is CVStrategy.TIME_SERIES_SPLIT:
            # A labelled enhancement, not the paper default: blocked folds never
            # train on data later than the block they score.
            return TimeSeriesSplit(n_splits=config.cv_folds)
        if config.cv_shuffle:
            return KFold(n_splits=config.cv_folds, shuffle=True, random_state=config.random_state)
        # KFold rejects random_state unless it shuffles, and it would mean nothing:
        # unshuffled folds are already deterministic.
        return KFold(n_splits=config.cv_folds, shuffle=False)

    def _build_search(self) -> GridSearchCV:
        """Return the configured ``GridSearchCV`` over the configured grid (A1)."""
        from sklearn.model_selection import GridSearchCV

        search: GridSearchCV = GridSearchCV(
            estimator=self._build_estimator(),
            param_grid={
                key: list(values) for key, values in self._config.hyperparameter_grid.items()
            },
            cv=self._build_cv(),
            scoring=None,  # the regressor's own score: r2
            refit=True,
            # n_jobs lives on the forest; see the module docstring.
            n_jobs=None,
            error_score="raise",
        )
        return search


# ---------------------------------------------------------------------------
# Module helpers
# ---------------------------------------------------------------------------


def _resolve_expected_names(
    config: RFRConfig,
    *,
    target: str | None,
    declared: Sequence[str] | None,
) -> tuple[str, ...] | None:
    """Return the feature order this model should see, or ``None`` when unknowable.

    Derivable whenever the daily statistics can be named - that is, with a target,
    or with the receptive limiter switched off for the ORF benchmark. A declared
    order is checked against the derived one so the two cannot drift apart.
    """
    derived: tuple[str, ...] | None = None
    if target is not None or not config.features.use_receptive_limiter:
        derived = feature_names(config, target=target)

    if declared is None:
        return derived

    if isinstance(declared, str) or not isinstance(declared, Sequence):
        raise ConfigError(f"feature_names must be a sequence of column names, got {declared!r}")
    names = tuple(str(name) for name in declared)
    if not names:
        raise ConfigError("feature_names must not be empty")
    if len(set(names)) != len(names):
        raise ConfigError(f"feature_names contains duplicates: {list(names)}")
    if derived is not None and names != derived:
        raise ConfigError(
            f"feature_names={list(names)} does not match the order this configuration "
            f"produces, {list(derived)}. Drop the argument to take the derived order, or "
            "fix the configuration it is meant to describe."
        )
    return names


def _as_target_array(y: object, *, index: pd.Index, rows: int) -> np.ndarray:
    """Return ``y`` as a float array, aligned to ``index`` by label when it has one."""
    if isinstance(y, pd.DataFrame):
        raise ModelError(f"y must be one-dimensional, got a DataFrame with {y.shape[1]} column(s)")
    if isinstance(y, pd.Series):
        if not y.index.equals(index):
            if not index.isin(y.index).all():
                raise ModelError(
                    "y is not aligned with X: its index does not cover every feature row"
                )
            y = y.reindex(index)
        values = np.asarray(y.to_numpy(), dtype=float)
    else:
        try:
            values = np.asarray(y, dtype=float)
        except (TypeError, ValueError) as error:
            raise ModelError(f"y must be numeric, got {type(y).__name__}") from error
    if values.ndim != 1:
        raise ModelError(f"y must be one-dimensional, got {values.ndim} dimension(s)")
    if len(values) != rows:
        raise ModelError(f"y has {len(values)} value(s) for {rows} feature row(s)")
    return values


def _feature_hint(report: FitReport) -> str:
    """Return a sentence pointing at the feature that cost the most rows, if any."""
    worst = report.worst_feature
    if worst is None:
        return "."
    count = report.missing_by_feature[worst]
    hint = f". The feature missing from most rows is {worst!r} ({count})"
    if worst.endswith(DAILY_STATISTIC_SUFFIXES):
        hint += (
            "; under daily_statistic_strategy='missing' a day with no visible target "
            "observation has no daily statistics at all, so a gap covering whole days "
            "leaves every row of those days incomplete (docs/method_spec.md 3.4, A4)"
        )
    return hint + "."


def _serialisable_params(estimator: RandomForestRegressor) -> dict[str, Any]:
    """Return the estimator's parameters, with non-JSON values rendered as text."""
    params: dict[str, Any] = {}
    for key, value in sorted(estimator.get_params().items()):
        params[key] = (
            value if isinstance(value, (bool, int, float, str, type(None))) else str(value)
        )
    return params


def _environment_versions() -> dict[str, str]:
    """Return the versions a reproduction of this fit would have to match."""
    import platform

    import sklearn

    from rfrgapfill import __version__

    return {
        "rfr_gapfill": __version__,
        "python": platform.python_version(),
        "scikit_learn": sklearn.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }


def _warn_on_version_drift(source: Path, *, stored: Mapping[str, str]) -> None:
    """Warn when the environment differs from the one the model was fitted under."""
    import warnings

    current = _environment_versions()
    drifted = {
        package: (stored[package], current[package])
        for package in ("rfr_gapfill", "scikit_learn")
        if package in stored and stored[package] != current[package]
    }
    if drifted:
        detail = ", ".join(
            f"{package} {was} -> {now}" for package, (was, now) in sorted(drifted.items())
        )
        warnings.warn(
            f"{source} was written under a different environment ({detail}); a pickled "
            "estimator is not guaranteed to behave identically across versions. Refit if the "
            "predictions matter.",
            UserWarning,
            stacklevel=3,
        )
