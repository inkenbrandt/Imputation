"""Random Forest wrapper tests. Covers acceptance tests 26-27.

The estimator layer is checked on its own, without gaps or features, because
that separation is the point of the module: the same class serves operational
filling and paper validation, so it must be correct without knowing which.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rfrgapfill.config import ColumnMap, CVStrategy, RFRConfig
from rfrgapfill.model import ModelError, RFRModel, model_version

#: A grid small enough to fit repeatedly in a unit test.
FAST_GRID = {"n_estimators": (20,), "min_samples_leaf": (1, 4)}


def config(**changes: object) -> RFRConfig:
    settings = {
        "mode": "RFR3",
        "frequency": "30min",
        "hemisphere": "north",
        "column_map": ColumnMap.fluxnet2015("RFR3"),
        "hyperparameter_grid": FAST_GRID,
        "cv_folds": 3,
    }
    settings.update(changes)
    return RFRConfig(**settings)  # type: ignore[arg-type]


def nonlinear(n: int = 400, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    """A learnable nonlinear surface: y = sin(a) * b^2 + noise."""
    rng = np.random.default_rng(seed)
    index = pd.date_range("2020-01-01", periods=n, freq="30min", name="timestamp")
    a = rng.uniform(0, np.pi, n)
    b = rng.uniform(-2, 2, n)
    frame = pd.DataFrame({"a": a, "b": b, "noise": rng.normal(0, 1, n)}, index=index)
    target = pd.Series(np.sin(a) * b**2 + rng.normal(0, 0.1, n), index=index, name="y")
    return frame, target


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------


def test_a_fitted_model_learns_a_nonlinear_surface() -> None:
    frame, target = nonlinear()
    model = RFRModel(config()).fit(frame, target)

    predictions = model.predict(frame)
    assert np.corrcoef(predictions, target)[0, 1] > 0.95


def test_fitting_reports_the_rows_it_dropped_and_why() -> None:
    frame, target = nonlinear(n=200)
    target.iloc[:10] = np.nan
    frame.iloc[20:35, 0] = np.nan

    report = RFRModel(config()).fit(frame, target).report
    assert report.n_rows == 200
    assert report.n_dropped_missing_target == 10
    assert report.n_dropped_missing_features == 15
    assert report.n_trained == 175
    assert report.n_dropped == 25


def test_too_few_usable_rows_names_the_reason_rather_than_failing_obscurely() -> None:
    frame, target = nonlinear(n=10)
    target.iloc[2:] = np.nan
    with pytest.raises(ModelError, match="cv_folds"):
        RFRModel(config()).fit(frame, target)


def test_the_grid_search_reports_the_parameters_it_chose() -> None:
    frame, target = nonlinear()
    model = RFRModel(config()).fit(frame, target)

    best = model.get_best_params()
    assert set(best) == set(FAST_GRID)
    assert best["min_samples_leaf"] in FAST_GRID["min_samples_leaf"]
    assert model.report.hyperparameter_grid == config().hyperparameter_grid


def test_feature_names_are_learnt_from_the_matrix_and_kept_in_order() -> None:
    frame, target = nonlinear()
    model = RFRModel(config()).fit(frame, target)
    assert model.get_feature_names() == ("a", "b", "noise")


def test_time_aware_folds_are_available_as_a_labelled_enhancement() -> None:
    frame, target = nonlinear()
    settings = config(cv_strategy=CVStrategy.TIME_SERIES_SPLIT)
    model = RFRModel(settings).fit(frame, target)

    assert model.report.cv_strategy is CVStrategy.TIME_SERIES_SPLIT
    assert not settings.is_paper_faithful


# ---------------------------------------------------------------------------
# Determinism (acceptance test 26)
# ---------------------------------------------------------------------------


def test_the_same_data_config_and_seed_give_identical_predictions() -> None:
    frame, target = nonlinear()
    first = RFRModel(config()).fit(frame, target).predict(frame)
    second = RFRModel(config()).fit(frame, target).predict(frame)

    pd.testing.assert_series_equal(first, second)


def test_a_different_seed_gives_a_different_forest() -> None:
    frame, target = nonlinear()
    first = RFRModel(config(random_state=1)).fit(frame, target).predict(frame)
    second = RFRModel(config(random_state=2)).fit(frame, target).predict(frame)

    assert not np.allclose(first.to_numpy(), second.to_numpy())


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------


def test_a_row_missing_a_predictor_gets_no_prediction_rather_than_an_imputed_one() -> None:
    frame, target = nonlinear()
    model = RFRModel(config()).fit(frame, target)

    incomplete = frame.copy()
    incomplete.iloc[5, 1] = np.nan
    predictions = model.predict(incomplete)

    assert np.isnan(predictions.iloc[5])
    assert predictions.drop(predictions.index[5]).notna().all()


def test_predicting_with_the_wrong_columns_fails_by_name() -> None:
    frame, target = nonlinear()
    model = RFRModel(config()).fit(frame, target)

    with pytest.raises(ModelError, match="missing feature"):
        model.predict(frame.drop(columns=["b"]))


def test_columns_in_a_different_order_are_reordered_not_misread() -> None:
    frame, target = nonlinear()
    model = RFRModel(config()).fit(frame, target)

    shuffled = frame.loc[:, ["noise", "b", "a"]]
    pd.testing.assert_series_equal(model.predict(shuffled), model.predict(frame))


def test_per_tree_spread_is_available_as_an_uncertainty_diagnostic() -> None:
    frame, target = nonlinear()
    model = RFRModel(config()).fit(frame, target)

    spread = model.predict_std(frame)
    assert (spread.dropna() >= 0).all()
    assert len(spread) == len(frame)


def test_an_unfitted_model_says_so() -> None:
    model = RFRModel(config())
    assert not model.is_fitted
    with pytest.raises(ModelError, match="not fitted"):
        model.predict(pd.DataFrame({"a": [1.0]}))


# ---------------------------------------------------------------------------
# Persistence (acceptance test 27)
# ---------------------------------------------------------------------------


def test_a_saved_and_reloaded_model_reproduces_its_predictions(tmp_path: Path) -> None:
    frame, target = nonlinear()
    model = RFRModel(config()).fit(frame, target)
    expected = model.predict(frame)

    model.save(tmp_path / "model.joblib")
    restored = RFRModel.load(tmp_path / "model.joblib")

    pd.testing.assert_series_equal(restored.predict(frame), expected)
    assert restored.get_best_params() == model.get_best_params()
    assert restored.get_feature_names() == model.get_feature_names()


def test_a_restored_model_carries_the_configuration_that_produced_it(tmp_path: Path) -> None:
    frame, target = nonlinear()
    model = RFRModel(config(site_id="GB-Ham")).fit(frame, target)
    model.save(tmp_path / "model.joblib")

    restored = RFRModel.load(tmp_path / "model.joblib")
    assert restored.config.site_id == "GB-Ham"
    assert restored.config.rfr_mode.value == "RFR3"


def test_loading_something_that_is_not_a_model_bundle_is_rejected(tmp_path: Path) -> None:
    import joblib

    joblib.dump({"not": "a model"}, tmp_path / "junk.joblib")
    with pytest.raises(ModelError, match="not an rfr-gapfill model bundle"):
        RFRModel.load(tmp_path / "junk.joblib")


def test_the_version_string_names_both_packages_a_prediction_depends_on() -> None:
    version = model_version()
    assert "rfr-gapfill" in version
    assert "scikit-learn" in version
