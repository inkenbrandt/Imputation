"""Model-layer tests. Covers Step 7 and acceptance tests 22-23, 26-27.

The four Step 7 requirements are the four sections below: a fit on synthetic
nonlinear data, identical results from the same data, configuration and seed,
predictions that survive a save/load round trip, and a useful error when there is
not enough usable training data.

The forests here are deliberately tiny (10-20 trees, 3 folds) so the suite stays
fast: nothing in this module is testing predictive quality beyond "the fit learned
the signal". See ``docs/method_spec.md`` section 5 and ambiguities A1 and A5.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rfrgapfill.config import CVStrategy, FeatureConfig, RFRConfig
from rfrgapfill.features import build_feature_matrix, feature_names
from rfrgapfill.model import (
    SAVE_FORMAT_VERSION,
    FitReport,
    IncompletePolicy,
    InsufficientTrainingDataError,
    ModelError,
    NotFittedError,
    RFRModel,
)
from rfrgapfill.schema import ColumnMap, ConfigError

HALF_HOURLY = "30min"

COLUMN_MAP = ColumnMap({"shortwave": "SW", "vpd": "VPD", "air_temperature": "TA"})

#: Small enough to fit in milliseconds, wide enough that GridSearchCV has a choice.
TEST_GRID = {"n_estimators": (10, 20)}


def site_frame(days: int = 30, *, seed: int = 0) -> pd.DataFrame:
    """A half-hourly site frame whose target is a nonlinear function of its drivers.

    ``LE`` is a product of radiation and temperature plus a saturating VPD term,
    so a Random Forest can learn it and a mean predictor cannot.
    """
    index = pd.date_range("2020-06-01", periods=days * 48, freq=HALF_HOURLY)
    hour = index.hour + index.minute / 60.0
    shortwave = np.clip(600.0 * np.sin(np.pi * (hour - 6.0) / 12.0), 0.0, None)
    air_temperature = 12.0 + 8.0 * np.sin(2.0 * np.pi * index.dayofyear / 365.0) + 0.4 * hour
    vpd = np.clip(0.02 * shortwave + 0.3 * air_temperature, 0.0, None)
    noise = np.random.default_rng(seed).normal(0.0, 2.0, len(index))
    latent_heat = 0.03 * shortwave * air_temperature + 20.0 * np.sqrt(vpd) + noise
    return pd.DataFrame(
        {
            "SW": shortwave,
            "VPD": vpd,
            "TA": air_temperature,
            "LE": latent_heat,
            "LE_QC": 0.0,
        },
        index=index,
    )


def rfr_config(**changes: object) -> RFRConfig:
    """An RFR3 configuration with a small grid and few folds."""
    settings: dict[str, object] = {
        "mode": "RFR3",
        "frequency": HALF_HOURLY,
        "hemisphere": "north",
        "random_state": 42,
        "hyperparameter_grid": TEST_GRID,
        "cv_folds": 3,
        "column_map": COLUMN_MAP,
    }
    settings.update(changes)
    return RFRConfig(**settings)  # type: ignore[arg-type]


def r2(measured: np.ndarray, predicted: np.ndarray) -> float:
    """Coefficient of determination, for asserting that a fit learned something."""
    residual = float(np.sum((measured - predicted) ** 2))
    total = float(np.sum((measured - measured.mean()) ** 2))
    return 1.0 - residual / total


@pytest.fixture(scope="module")
def data() -> pd.DataFrame:
    return site_frame()


@pytest.fixture(scope="module")
def config() -> RFRConfig:
    return rfr_config()


@pytest.fixture(scope="module")
def features(data: pd.DataFrame, config: RFRConfig) -> pd.DataFrame:
    return build_feature_matrix(data, config=config, target="LE")


@pytest.fixture(scope="module")
def fitted(data: pd.DataFrame, config: RFRConfig, features: pd.DataFrame) -> RFRModel:
    """One fitted model shared by the read-only tests."""
    return RFRModel(config, target="LE").fit(features, data["LE"])


# ---------------------------------------------------------------------------
# Fitting synthetic nonlinear data
# ---------------------------------------------------------------------------


def test_fit_learns_a_nonlinear_signal(data: pd.DataFrame, features: pd.DataFrame) -> None:
    """Train on the first three weeks, score the rest: skill must be real, not memorised."""
    split = 20 * 48
    model = RFRModel(rfr_config(), target="LE").fit(features.iloc[:split], data["LE"].iloc[:split])

    predicted = model.predict(features.iloc[split:]).to_numpy()
    measured = data["LE"].iloc[split:].to_numpy()

    assert not np.isnan(predicted).any()
    assert r2(measured, predicted) > 0.8


def test_fit_returns_self_and_reports_fitted_state(
    data: pd.DataFrame, features: pd.DataFrame
) -> None:
    model = RFRModel(rfr_config(), target="LE")
    assert model.is_fitted is False
    assert "unfitted" in repr(model)

    assert model.fit(features, data["LE"]) is model
    assert model.is_fitted is True
    assert "fitted" in repr(model)


def test_best_params_come_from_the_configured_grid(fitted: RFRModel) -> None:
    best = fitted.get_best_params()

    assert set(best) == {"n_estimators"}
    assert best["n_estimators"] in TEST_GRID["n_estimators"]
    # The seed and n_jobs are configuration, never search dimensions (A1 forbids
    # inventing a grid; the package's own settings are not part of one).
    assert "random_state" not in best
    assert fitted.estimator.get_params()["n_estimators"] == best["n_estimators"]
    assert isinstance(fitted.best_score, float)


def test_the_configured_seed_and_n_jobs_reach_the_forest(fitted: RFRModel) -> None:
    assert fitted.estimator.get_params()["random_state"] == 42

    estimator = RFRModel(rfr_config(n_jobs=-1), target="LE")._build_estimator()
    assert estimator.get_params()["n_jobs"] == -1


def test_feature_order_is_the_configured_order(config: RFRConfig, fitted: RFRModel) -> None:
    """Acceptance tests 22-23: the driver set is exactly the mode's, in spec order."""
    assert fitted.get_feature_names() == feature_names(config, target="LE")
    assert fitted.get_feature_names()[:3] == ("shortwave", "vpd", "air_temperature")

    rfr10 = rfr_config(mode="RFR10", column_map=ColumnMap.fluxnet2015("RFR10"))
    assert len(RFRModel(rfr10, target="LE").get_feature_names()) == 10 + 7


def test_the_orf_benchmark_model_fits_the_drivers_alone(data: pd.DataFrame) -> None:
    """The ORF arm needs no target to know its features: it has none derived from one."""
    orf = rfr_config().as_orf()
    matrix = build_feature_matrix(data, config=orf)

    model = RFRModel(orf).fit(matrix, data["LE"])

    assert model.get_feature_names() == ("shortwave", "vpd", "air_temperature")
    assert model.predict(matrix).notna().all()


def test_feature_names_are_known_before_fitting_when_they_can_be_derived(
    config: RFRConfig,
) -> None:
    assert RFRModel(config, target="LE").get_feature_names() == feature_names(config, target="LE")


def test_feature_names_are_unknown_before_fitting_without_a_target(config: RFRConfig) -> None:
    """The daily statistics are target-specific, so an unnamed target cannot name them."""
    model = RFRModel(config)

    with pytest.raises(NotFittedError, match="target-specific"):
        model.get_feature_names()


def test_declared_feature_names_must_match_the_configured_order(config: RFRConfig) -> None:
    with pytest.raises(ConfigError, match="does not match the order"):
        RFRModel(config, target="LE", feature_names=["shortwave", "vpd"])


def test_a_declared_order_is_accepted_when_it_cannot_be_derived(
    data: pd.DataFrame, config: RFRConfig, features: pd.DataFrame
) -> None:
    names = list(feature_names(config, target="LE"))
    model = RFRModel(config, feature_names=names)

    assert model.get_feature_names() == tuple(names)
    assert model.fit(features, data["LE"]).predict(features).notna().all()


# ---------------------------------------------------------------------------
# Row accounting (method_spec.md section 5)
# ---------------------------------------------------------------------------


def test_fit_report_accounts_for_every_offered_row(
    data: pd.DataFrame, features: pd.DataFrame
) -> None:
    matrix = features.copy()
    matrix.iloc[:5, 0] = np.nan  # five rows lose a driver
    target = data["LE"].copy()
    target.iloc[100:110] = np.nan  # ten rows lose the target
    target.iloc[3] = np.nan  # one row loses both: counted against the target only

    model = RFRModel(rfr_config(), target="LE").fit(matrix, target)
    report = model.fit_report

    assert report.rows == len(matrix)
    assert report.dropped_missing_target == 11
    assert report.dropped_missing_features == 4
    assert report.fitted_rows == report.rows - 15
    assert report.dropped_rows == 15
    assert report.usable_fraction == pytest.approx(report.fitted_rows / report.rows)
    # The per-feature diagnostic counts rows with a usable target only.
    assert report.missing_by_feature["shortwave"] == 4
    assert report.worst_feature == "shortwave"
    assert json.dumps(report.to_dict())


def test_a_non_finite_predictor_is_excluded_like_a_missing_one(
    data: pd.DataFrame, features: pd.DataFrame
) -> None:
    """Recent forests accept NaN natively; the specification does not, so we drop."""
    matrix = features.copy()
    matrix.iloc[:7, 1] = np.inf

    report = RFRModel(rfr_config(), target="LE").fit(matrix, data["LE"]).fit_report

    assert report.dropped_missing_features == 7


def test_fit_report_survives_a_report_with_nothing_missing(fitted: RFRModel) -> None:
    report = fitted.fit_report

    assert report.dropped_rows == 0
    assert report.worst_feature is None
    assert isinstance(report, FitReport)


# ---------------------------------------------------------------------------
# Determinism (acceptance test 26)
# ---------------------------------------------------------------------------


def test_the_same_data_configuration_and_seed_give_identical_predictions(
    data: pd.DataFrame, features: pd.DataFrame
) -> None:
    first = RFRModel(rfr_config(), target="LE").fit(features, data["LE"])
    second = RFRModel(rfr_config(), target="LE").fit(features, data["LE"])

    np.testing.assert_array_equal(
        first.predict(features).to_numpy(), second.predict(features).to_numpy()
    )
    assert first.get_best_params() == second.get_best_params()
    assert first.best_score == second.best_score


def test_a_different_seed_gives_a_different_forest(
    data: pd.DataFrame, features: pd.DataFrame
) -> None:
    first = RFRModel(rfr_config(random_state=1), target="LE").fit(features, data["LE"])
    second = RFRModel(rfr_config(random_state=2), target="LE").fit(features, data["LE"])

    assert not np.array_equal(
        first.predict(features).to_numpy(), second.predict(features).to_numpy()
    )


def test_the_cross_validation_splitter_follows_the_configuration() -> None:
    """Ambiguity A5: conventional folds by default, blocked folds as an enhancement."""
    from sklearn.model_selection import KFold, TimeSeriesSplit

    default = RFRModel(rfr_config(), target="LE")._build_cv()
    assert isinstance(default, KFold)
    assert default.shuffle is False
    assert default.get_n_splits() == 3

    shuffled = RFRModel(rfr_config(cv_shuffle=True), target="LE")._build_cv()
    assert isinstance(shuffled, KFold)
    assert shuffled.random_state == 42

    blocked = RFRModel(
        rfr_config(cv_strategy=CVStrategy.TIME_SERIES_SPLIT), target="LE"
    )._build_cv()
    assert isinstance(blocked, TimeSeriesSplit)


# ---------------------------------------------------------------------------
# Predicting
# ---------------------------------------------------------------------------


def test_predict_returns_a_series_on_the_given_index(
    fitted: RFRModel, features: pd.DataFrame
) -> None:
    predictions = fitted.predict(features)

    assert isinstance(predictions, pd.Series)
    assert predictions.index.equals(features.index)
    assert predictions.name == "LE_predicted"
    assert predictions.dtype == float


def test_rows_missing_a_predictor_are_left_missing_by_default(
    fitted: RFRModel, features: pd.DataFrame
) -> None:
    matrix = features.copy()
    matrix.iloc[10:13, 2] = np.nan

    predictions = fitted.predict(matrix)

    assert predictions.iloc[10:13].isna().all()
    assert predictions.drop(predictions.index[10:13]).notna().all()


def test_rows_missing_a_predictor_can_be_made_an_error(
    fitted: RFRModel, features: pd.DataFrame
) -> None:
    matrix = features.copy()
    matrix.iloc[0, 2] = np.nan

    with pytest.raises(ModelError, match="air_temperature"):
        fitted.predict(matrix, on_incomplete="raise")

    assert fitted.predict(matrix, on_incomplete=IncompletePolicy.MISSING).isna().sum() == 1


def test_an_unknown_incomplete_policy_is_rejected(fitted: RFRModel, features: pd.DataFrame) -> None:
    with pytest.raises(ConfigError, match="on_incomplete"):
        fitted.predict(features, on_incomplete="impute")


def test_predicting_before_fitting_says_so(config: RFRConfig, features: pd.DataFrame) -> None:
    with pytest.raises(NotFittedError, match="fit"):
        RFRModel(config, target="LE").predict(features)


def test_features_in_the_wrong_order_are_rejected(fitted: RFRModel, features: pd.DataFrame) -> None:
    reordered = features[list(reversed(features.columns))]

    with pytest.raises(ModelError, match="wrong order"):
        fitted.predict(reordered)


def test_features_that_do_not_match_the_model_are_named(
    fitted: RFRModel, features: pd.DataFrame
) -> None:
    renamed = features.rename(columns={"season": "quarter"})

    with pytest.raises(ModelError, match="missing season; unexpected quarter"):
        fitted.predict(renamed)


def test_a_plain_array_is_accepted_in_the_models_own_feature_order(
    fitted: RFRModel, features: pd.DataFrame
) -> None:
    from_frame = fitted.predict(features).to_numpy()
    from_array = fitted.predict(features.to_numpy()).to_numpy()

    np.testing.assert_array_equal(from_frame, from_array)


def test_an_array_of_the_wrong_width_is_rejected(fitted: RFRModel, features: pd.DataFrame) -> None:
    with pytest.raises(ModelError, match="column"):
        fitted.predict(features.to_numpy()[:, :4])


def test_feature_importances_are_reported_in_feature_order(fitted: RFRModel) -> None:
    importances = fitted.feature_importances()

    assert list(importances.index) == list(fitted.get_feature_names())
    assert importances.sum() == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Target alignment
# ---------------------------------------------------------------------------


def test_a_target_series_is_aligned_by_label_not_position(
    data: pd.DataFrame, features: pd.DataFrame
) -> None:
    shuffled = data["LE"].sample(frac=1.0, random_state=0)

    ordered = RFRModel(rfr_config(), target="LE").fit(features, data["LE"])
    scrambled = RFRModel(rfr_config(), target="LE").fit(features, shuffled)

    np.testing.assert_array_equal(
        ordered.predict(features).to_numpy(), scrambled.predict(features).to_numpy()
    )


def test_a_target_that_does_not_cover_every_feature_row_is_rejected(
    data: pd.DataFrame, features: pd.DataFrame
) -> None:
    with pytest.raises(ModelError, match="not aligned"):
        RFRModel(rfr_config(), target="LE").fit(features, data["LE"].iloc[:-5])


def test_a_target_of_the_wrong_length_is_rejected(features: pd.DataFrame) -> None:
    with pytest.raises(ModelError, match="feature row"):
        RFRModel(rfr_config(), target="LE").fit(features, np.zeros(len(features) - 1))


def test_a_two_dimensional_target_is_rejected(data: pd.DataFrame, features: pd.DataFrame) -> None:
    with pytest.raises(ModelError, match="one-dimensional"):
        RFRModel(rfr_config(), target="LE").fit(features, data[["LE"]])


# ---------------------------------------------------------------------------
# Insufficient training data
# ---------------------------------------------------------------------------


def test_no_usable_rows_reports_where_they_went(features: pd.DataFrame) -> None:
    with pytest.raises(InsufficientTrainingDataError) as raised:
        RFRModel(rfr_config(), target="LE").fit(features, np.full(len(features), np.nan))

    assert "no usable training rows" in str(raised.value)
    assert raised.value.report.dropped_missing_target == len(features)
    assert raised.value.report.fitted_rows == 0


def test_missing_daily_statistics_point_at_ambiguity_a4(
    data: pd.DataFrame, features: pd.DataFrame
) -> None:
    """The A4 consequence is the most likely cause of an empty long-gap fit."""
    matrix = features.copy()
    matrix["LE_daily_std"] = np.nan

    with pytest.raises(InsufficientTrainingDataError, match="A4"):
        RFRModel(rfr_config(), target="LE").fit(matrix, data["LE"])


def test_too_few_rows_for_the_configured_folds_says_which_knob_to_turn(
    data: pd.DataFrame, features: pd.DataFrame
) -> None:
    with pytest.raises(InsufficientTrainingDataError, match="cv_folds"):
        RFRModel(rfr_config(), target="LE").fit(features.iloc[:2], data["LE"].iloc[:2])


def test_blocked_folds_need_one_more_row_than_they_have_splits(
    data: pd.DataFrame, features: pd.DataFrame
) -> None:
    config = rfr_config(cv_strategy="time_series_split", cv_folds=3)

    with warnings.catch_warnings():
        # Four rows is the floor, not a sensible fit: one-sample test folds leave
        # scikit-learn's r2 undefined, which it warns about and we do not hide.
        warnings.simplefilter("ignore")
        RFRModel(config, target="LE").fit(features.iloc[:4], data["LE"].iloc[:4])
    with pytest.raises(InsufficientTrainingDataError, match="at least 4"):
        RFRModel(config, target="LE").fit(features.iloc[:3], data["LE"].iloc[:3])


def test_an_explicit_row_floor_is_enforced_on_top_of_the_folds(
    data: pd.DataFrame, features: pd.DataFrame
) -> None:
    model = RFRModel(rfr_config(), target="LE")

    with pytest.raises(InsufficientTrainingDataError, match="below the requested minimum"):
        model.fit(features.iloc[:10], data["LE"].iloc[:10], min_training_rows=100)

    assert model.fit(features, data["LE"], min_training_rows=100).is_fitted


# ---------------------------------------------------------------------------
# Persistence (acceptance test 27)
# ---------------------------------------------------------------------------


def test_a_reloaded_model_reproduces_its_predictions(
    fitted: RFRModel, features: pd.DataFrame, tmp_path: Path
) -> None:
    path = fitted.save(tmp_path / "models" / "LE.joblib")
    assert path.exists()

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # the same environment must not warn
        restored = RFRModel.load(path)

    np.testing.assert_array_equal(
        fitted.predict(features).to_numpy(), restored.predict(features).to_numpy()
    )
    assert restored.get_feature_names() == fitted.get_feature_names()
    assert restored.get_best_params() == fitted.get_best_params()
    assert restored.target == fitted.target
    assert restored.config == fitted.config
    assert restored.fit_report.to_dict() == fitted.fit_report.to_dict()


def test_a_reloaded_configuration_is_revalidated_not_trusted(
    fitted: RFRModel, tmp_path: Path
) -> None:
    """A frozen configuration round-trips through pickle and keeps its invariants."""
    import pickle

    restored = pickle.loads(pickle.dumps(fitted.config))

    assert restored == fitted.config
    assert restored.hyperparameter_grid == fitted.config.hyperparameter_grid
    with pytest.raises(ConfigError):
        object.__new__(RFRConfig).__setstate__({**fitted.config.__getstate__(), "cv_folds": 1})


def test_saving_an_unfitted_model_is_refused(config: RFRConfig, tmp_path: Path) -> None:
    with pytest.raises(NotFittedError, match="save"):
        RFRModel(config, target="LE").save(tmp_path / "unfitted.joblib")


def test_loading_something_that_is_not_a_model_file_says_so(tmp_path: Path) -> None:
    import joblib

    path = tmp_path / "not-a-model.joblib"
    joblib.dump({"kind": "something else"}, path)

    with pytest.raises(ModelError, match="not an rfrgapfill model file"):
        RFRModel.load(path)


def test_loading_an_unknown_format_version_is_refused(fitted: RFRModel, tmp_path: Path) -> None:
    import joblib

    path = fitted.save(tmp_path / "LE.joblib")
    payload = joblib.load(path)
    payload["format_version"] = SAVE_FORMAT_VERSION + 1
    joblib.dump(payload, path)

    with pytest.raises(ModelError, match="model format"):
        RFRModel.load(path)


def test_a_model_fitted_elsewhere_warns_about_its_environment(
    fitted: RFRModel, tmp_path: Path
) -> None:
    import joblib

    path = fitted.save(tmp_path / "LE.joblib")
    payload = joblib.load(path)
    payload["versions"]["scikit_learn"] = "0.0.0-not-a-real-version"
    joblib.dump(payload, path)

    with pytest.warns(UserWarning, match="different environment"):
        RFRModel.load(path)


# ---------------------------------------------------------------------------
# Run manifest (method_spec.md section 7)
# ---------------------------------------------------------------------------


def test_the_manifest_records_the_search_and_never_claims_the_paper_grid(
    fitted: RFRModel,
) -> None:
    manifest = fitted.to_dict()

    assert manifest["estimator"] == "sklearn.ensemble.RandomForestRegressor"
    assert manifest["target"] == "LE"
    assert manifest["mode"] == "RFR3"
    assert manifest["feature_names"] == list(fitted.get_feature_names())
    assert manifest["hyperparameter_grid"] == {"n_estimators": [10, 20]}
    # A1: the article never enumerates its grid, so nothing may claim to be it.
    assert manifest["hyperparameter_grid_is_paper_exact"] is False
    assert manifest["hyperparameter_grid_is_package_default"] is False
    assert manifest["cv_strategy"] == "kfold"
    assert manifest["cv_folds"] == 3
    assert manifest["random_state"] == 42
    assert manifest["best_params"] == fitted.get_best_params()
    assert manifest["fit"] == fitted.fit_report.to_dict()
    assert json.dumps(manifest)


def test_the_manifest_marks_the_shipped_grid_as_the_package_default(
    config: RFRConfig,
) -> None:
    default_grid = RFRConfig(mode="RFR3", hemisphere="north", column_map=COLUMN_MAP)

    assert RFRModel(default_grid, target="LE").to_dict()["hyperparameter_grid_is_package_default"]
    assert not RFRModel(config, target="LE").to_dict()["hyperparameter_grid_is_package_default"]


def test_an_unfitted_manifest_reports_the_plan_without_results(config: RFRConfig) -> None:
    manifest = RFRModel(config, target="LE").to_dict()

    assert manifest["is_fitted"] is False
    assert "best_params" not in manifest
    assert manifest["feature_names"] == list(feature_names(config, target="LE"))


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_the_model_requires_a_real_configuration() -> None:
    with pytest.raises(ConfigError, match="RFRConfig"):
        RFRModel({"mode": "RFR3"})  # type: ignore[arg-type]


def test_the_target_must_be_a_non_empty_name(config: RFRConfig) -> None:
    with pytest.raises(ConfigError, match="target"):
        RFRModel(config, target="  ")


def test_duplicate_feature_columns_are_rejected(data: pd.DataFrame, features: pd.DataFrame) -> None:
    doubled = pd.concat([features, features[["season"]]], axis=1)

    with pytest.raises(ModelError, match="duplicate"):
        RFRModel(rfr_config(), target="LE").fit(doubled, data["LE"])


def test_a_receptive_limiter_model_and_its_orf_counterpart_share_everything_else(
    data: pd.DataFrame,
) -> None:
    """Acceptance tests 7-8 at the model layer: same forest, same rows, fewer features."""
    rfr = rfr_config(features=FeatureConfig())
    orf = rfr.as_orf()

    rfr_model = RFRModel(rfr, target="LE").fit(
        build_feature_matrix(data, config=rfr, target="LE"), data["LE"]
    )
    orf_model = RFRModel(orf, target="LE").fit(build_feature_matrix(data, config=orf), data["LE"])

    assert type(rfr_model.estimator) is type(orf_model.estimator)
    assert rfr_model.fit_report.fitted_rows == orf_model.fit_report.fitted_rows
    assert set(orf_model.get_feature_names()) < set(rfr_model.get_feature_names())
    assert (
        rfr_model.estimator.get_params()["random_state"]
        == (orf_model.estimator.get_params()["random_state"])
    )
