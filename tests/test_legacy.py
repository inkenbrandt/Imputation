"""Historical ``fluxlib`` compatibility (Step 20).

Covers ``feature_mode="legacy_fluxlib"`` and the ``fluxlib`` hyperparameter
presets. The reference for the daily statistics is :func:`fluxlib_set_stats`, a
transcription of ``GFiller.set_stats`` from ``fluxlib`` 0.0.23 (commit
``da53256``), changed only where pandas has since renamed the ``"15T"`` alias and
taking the flux as one column name rather than a one-element list. Everything
else is hand-calculated. See ``docs/fluxlib_audit.md``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rfrgapfill.config import (
    DEFAULT_HYPERPARAMETER_GRID,
    HYPERPARAMETER_PRESETS,
    LEGACY_FLUXLIB_FIXED_HYPERPARAMETERS,
    LEGACY_FLUXLIB_HYPERPARAMETER_GRID,
    FeatureConfig,
    HyperparameterPreset,
    RFRConfig,
)
from rfrgapfill.features import build_feature_matrix, feature_names, season_tag
from rfrgapfill.fill import method_label
from rfrgapfill.leakage import (
    LeakageError,
    build_validation_features,
    detect_target_leakage,
    holdout_mask_from_intervals,
    require_no_target_leakage,
)
from rfrgapfill.legacy import (
    LEGACY_CALENDAR_FEATURES,
    LegacyFluxlibWarning,
    legacy_calendar,
    legacy_daily_statistics,
    legacy_feature_names,
    legacy_radiation_rank,
    legacy_season_code,
    legacy_statistic_names,
)
from rfrgapfill.model import ModelError, RFRModel
from rfrgapfill.provenance import ambiguity_choices
from rfrgapfill.schema import ColumnMap, ConfigError
from rfrgapfill.synthetic import synthetic_site
from rfrgapfill.validation import validate_rfr

HALF_HOURLY = "30min"
COLUMN_MAP = ColumnMap({"shortwave": "SW", "vpd": "VPD", "air_temperature": "TA"})
DRIVERS = ("shortwave", "vpd", "air_temperature")
GAP = ("2020-06-06", "2020-06-13")

#: Every build below warns by design; tests that assert the warning say so.
quiet = pytest.mark.filterwarnings("ignore::rfrgapfill.legacy.LegacyFluxlibWarning")


def site_frame(days: int = 20, *, start: str = "2020-06-01") -> pd.DataFrame:
    """A half-hourly site frame with a diurnal driver and a day-varying target."""
    index = pd.date_range(start, periods=days * 48, freq=HALF_HOURLY)
    hour = index.hour + index.minute / 60.0
    daylight = np.clip(600.0 * np.sin(np.pi * (hour - 6.0) / 12.0), 0.0, None)
    return pd.DataFrame(
        {
            "SW": daylight,
            "VPD": 5.0 + 0.1 * hour,
            "TA": 12.0 + 0.5 * hour,
            "LE": 10.0 * index.dayofyear + hour,
            "LE_QC": 0.0,
        },
        index=index,
    )


def config(**changes: object) -> RFRConfig:
    """An RFR3 configuration in ``legacy_fluxlib`` mode."""
    settings: dict[str, object] = {
        "mode": "RFR3",
        "hemisphere": "north",
        "column_map": COLUMN_MAP,
        "features": FeatureConfig(feature_mode="legacy_fluxlib"),
    }
    settings.update(changes)
    return RFRConfig(**settings)  # type: ignore[arg-type]


def fluxlib_set_stats(df: pd.DataFrame, flux: str, scale: str = "15min") -> pd.DataFrame:
    """``GFiller.set_stats`` of fluxlib 0.0.23, transcribed."""
    flux_orig = df[flux].copy()
    df[flux] = df[flux].interpolate()
    flux_max = df[flux].resample("D").max()
    df["flux_max"] = flux_max.resample(scale).bfill()
    flux_min = df[flux].resample("D").min()
    df["flux_min"] = flux_min.resample(scale).bfill()
    flux_mean = df[flux].resample("D").mean()
    df["flux_mean"] = flux_mean.resample(scale).bfill()
    flux_std = df[flux].resample("D").std()
    df["flux_std"] = flux_std.resample(scale).bfill()
    flux_p25 = df[flux].resample("D").quantile(0.25)
    df["flux_p25"] = flux_p25.resample(scale).bfill()
    flux_p50 = df[flux].resample("D").quantile(0.50)
    df["flux_p50"] = flux_p50.resample(scale).bfill()
    flux_p75 = df[flux].resample("D").quantile(0.75)
    df["flux_p75"] = flux_p75.resample(scale).bfill()
    df = df.interpolate()
    df[flux] = flux_orig
    return df


# ---------------------------------------------------------------------------
# Daily statistics
# ---------------------------------------------------------------------------


def test_daily_statistics_match_a_transcription_of_fluxlib() -> None:
    index = pd.date_range("2020-03-01", periods=10 * 48, freq=HALF_HOURLY)
    values = np.random.default_rng(3).normal(5.0, 2.0, index.size)
    values[:7] = np.nan  # leading gap: stays missing in both
    values[96:144] = np.nan  # a whole calendar day
    values[300:311] = np.nan  # a short real gap
    target = pd.Series(values, index=index, name="LE")

    ours = legacy_daily_statistics(target)
    theirs = fluxlib_set_stats(pd.DataFrame({"LE": values}, index=index), "LE")

    reference = [
        "flux_max",
        "flux_min",
        "flux_mean",
        "flux_std",
        "flux_p25",
        "flux_p50",
        "flux_p75",
    ]
    assert list(ours.columns) == list(legacy_statistic_names("LE"))
    np.testing.assert_allclose(
        ours.to_numpy(), theirs[reference].to_numpy(), rtol=1e-12, equal_nan=True
    )


def test_a_row_at_midnight_keeps_its_day_and_every_other_row_takes_the_next() -> None:
    index = pd.date_range("2020-01-01", periods=3 * 48, freq=HALF_HOURLY)
    target = pd.Series(np.repeat([1.0, 2.0, 3.0], 48), index=index, name="NEE")
    daily_max = legacy_daily_statistics(target)["NEE_legacy_max"]

    assert daily_max["2020-01-01 00:00"] == 1.0
    assert daily_max["2020-01-01 00:30"] == 2.0  # the next day's value
    assert daily_max["2020-01-02 00:00"] == 2.0
    assert daily_max["2020-01-02 23:30"] == 3.0
    # After the last midnight the closing interpolation repeats the last day.
    assert daily_max["2020-01-03 00:30"] == daily_max["2020-01-03 23:30"] == 3.0


def test_a_day_inside_a_gap_gets_statistics_from_the_interpolated_target() -> None:
    index = pd.date_range("2020-01-01", periods=3 * 48, freq=HALF_HOURLY)
    values = np.repeat([1.0, np.nan, 3.0], 48)
    target = pd.Series(values, index=index, name="NEE")
    daily_mean = legacy_daily_statistics(target)["NEE_legacy_mean"]
    # Day two is a straight line from 1 to 3 over 49 steps; its mean is 2, and it
    # reaches the rows of day one through the look-ahead.
    assert daily_mean["2020-01-01 12:00"] == pytest.approx(2.0)
    assert daily_mean.notna().all()


def test_the_mask_decides_what_the_statistics_read() -> None:
    index = pd.date_range("2020-01-01", periods=2 * 48, freq=HALF_HOURLY)
    target = pd.Series(np.r_[np.full(48, 1.0), np.full(48, 5.0)], index=index, name="LE")
    hidden = pd.Series(index.day == 2, index=index)
    masked = legacy_daily_statistics(target, available_mask=~hidden)
    # Day two is hidden and has no later value, so interpolation repeats day one.
    assert masked["LE_legacy_max"].iloc[-1] == 1.0


# ---------------------------------------------------------------------------
# Tags
# ---------------------------------------------------------------------------


def test_radiation_rank_puts_exact_thresholds_and_missing_values_in_class_zero() -> None:
    ranks = legacy_radiation_rank([5.0, 10.0, 50.0, 100.0, 150.0, np.nan])
    assert ranks.tolist() == [1.0, 0.0, 2.0, 0.0, 3.0, 0.0]


@pytest.mark.parametrize("hemisphere", ["north", "south"])
def test_season_code_is_the_season_tag_ordinal_plus_one(hemisphere: str) -> None:
    index = pd.date_range("2021-01-15", periods=12, freq="MS") + pd.Timedelta(days=14)
    legacy = legacy_season_code(index, hemisphere=hemisphere)
    ours = season_tag(index, hemisphere=hemisphere).cat.codes.to_numpy()
    np.testing.assert_array_equal(legacy.to_numpy(), ours + 1)


def test_season_code_follows_the_fluxlib_formulas() -> None:
    months = pd.DatetimeIndex(["2021-01-10", "2021-04-10", "2021-07-10", "2021-10-10"])
    assert legacy_season_code(months, hemisphere="north").tolist() == [1.0, 2.0, 3.0, 4.0]
    assert legacy_season_code(months, hemisphere="south").tolist() == [3.0, 4.0, 1.0, 2.0]


def test_calendar_features_are_day_of_year_and_year() -> None:
    calendar = legacy_calendar(pd.DatetimeIndex(["2020-12-31 23:30"]))
    assert calendar.iloc[0].tolist() == [366.0, 2020.0]


# ---------------------------------------------------------------------------
# Feature names and the matrix
# ---------------------------------------------------------------------------


def test_feature_order_is_fluxlibs_after_the_drivers() -> None:
    names = feature_names(config(), target="LE")
    assert names == DRIVERS + legacy_statistic_names("LE") + LEGACY_CALENDAR_FEATURES
    assert legacy_feature_names(None) == LEGACY_CALENDAR_FEATURES


def test_legacy_and_paper_safe_features_share_only_the_drivers() -> None:
    safe = RFRConfig(mode="RFR3", hemisphere="north", column_map=COLUMN_MAP)
    shared = set(feature_names(config(), target="LE")) & set(feature_names(safe, target="LE"))
    assert shared == set(DRIVERS)


def test_the_matrix_carries_the_legacy_columns_and_no_elapsed_hours() -> None:
    data = site_frame(4)
    matrix = build_feature_matrix(data, config=config(), target="LE")
    assert tuple(matrix.columns) == feature_names(config(), target="LE")
    assert "time_distance_hours" not in matrix.columns
    assert matrix["legacy_season"].eq(3.0).all()  # June, northern summer


def test_a_paper_safe_model_rejects_a_legacy_matrix() -> None:
    data = site_frame(4)
    matrix = build_feature_matrix(data, config=config(), target="LE")
    safe = RFRConfig(
        mode="RFR3",
        hemisphere="north",
        column_map=COLUMN_MAP,
        hyperparameter_grid={"n_estimators": (5,)},
        cv_folds=2,
    )
    with pytest.raises(ModelError):
        RFRModel(safe, target="LE").fit(matrix, data["LE"])


def test_orf_has_no_legacy_features_and_keeps_its_label() -> None:
    orf = config().as_orf()
    assert feature_names(orf) == DRIVERS
    assert method_label(orf) == "ORF3"
    assert method_label(config()) == "RFR3-legacy"
    assert method_label(config(mode="RFR10", column_map=None)) == "RFR10-legacy"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def test_legacy_mode_is_accepted_and_never_paper_faithful() -> None:
    legacy = config()
    assert legacy.features.mode.value == "legacy_fluxlib"
    assert legacy.is_paper_faithful is False


@pytest.mark.parametrize(
    "setting",
    [
        {"radiation_thresholds": (5.0, 100.0)},
        {"boundary_convention": "medium_exclusive"},
        {"min_daily_observations": 3},
        {"daily_statistic_strategy": "within_day_available"},
        {"daily_statistic_strategy": "rolling_available"},
        {"daily_std_ddof": 0},
    ],
)
def test_settings_fluxlib_replaces_cannot_be_changed(setting: dict[str, object]) -> None:
    with pytest.raises(ConfigError, match="would have no effect"):
        FeatureConfig(feature_mode="legacy_fluxlib", **setting)  # type: ignore[arg-type]


def test_the_manifest_reports_replaced_settings_as_not_applicable() -> None:
    described = FeatureConfig(feature_mode="legacy_fluxlib").to_dict()
    assert described["boundary_convention"] is None
    assert described["daily_statistic_strategy"] is None
    assert described["min_daily_observations"] is None
    assert FeatureConfig().to_dict()["boundary_convention"] == "medium_inclusive"


def test_ambiguity_choices_record_fluxlibs_rules_and_the_leak() -> None:
    choices = ambiguity_choices(config())["choices"]
    assert choices["A6"]["settings"]["is_leakage_safe"] is False
    assert "fluxlib_audit" in choices["A6"]["settings"]["evidence"]
    assert "set_rg_tag" in choices["A2"]["settings"]["rule"]
    assert "set_stats" in choices["A4"]["settings"]["rule"]

    safe = ambiguity_choices(RFRConfig(mode="RFR3", hemisphere="north"))["choices"]
    assert safe["A6"]["settings"]["is_leakage_safe"] is True
    assert safe["A2"]["settings"]["boundary_convention"] == "medium_inclusive"


# ---------------------------------------------------------------------------
# Leakage: reproduced on purpose, and reported
# ---------------------------------------------------------------------------


def test_building_legacy_validation_features_warns() -> None:
    data = site_frame()
    with pytest.warns(LegacyFluxlibWarning, match="held-out LE values"):
        build_validation_features(
            data,
            config=config(),
            target="LE",
            holdout=holdout_mask_from_intervals(data.index, [GAP]),
            qc_column="LE_QC",
        )


@quiet
def test_the_probe_finds_exactly_the_legacy_daily_statistics() -> None:
    data = site_frame()
    holdout = holdout_mask_from_intervals(data.index, [GAP])
    leaking = detect_target_leakage(data, config=config(), target="LE", holdout=holdout)
    assert set(leaking) == set(legacy_statistic_names("LE"))

    safe = RFRConfig(mode="RFR3", hemisphere="north", column_map=COLUMN_MAP)
    assert detect_target_leakage(data, config=safe, target="LE", holdout=holdout) == ()


@quiet
def test_requiring_no_leakage_explains_the_legacy_mode() -> None:
    data = site_frame()
    holdout = holdout_mask_from_intervals(data.index, [GAP])
    with pytest.raises(LeakageError, match="legacy_fluxlib"):
        require_no_target_leakage(data, config=config(), target="LE", holdout=holdout)


@quiet
def test_every_withheld_row_has_complete_legacy_features() -> None:
    # The practical face of the leak: under paper_safe's default a 7-day gap has
    # no daily statistics at all, while fluxlib's are complete inside it.
    data = site_frame()
    holdout = holdout_mask_from_intervals(data.index, [GAP])
    legacy = build_validation_features(data, config=config(), target="LE", holdout=holdout)
    safe = build_validation_features(
        data,
        config=RFRConfig(mode="RFR3", hemisphere="north", column_map=COLUMN_MAP),
        target="LE",
        holdout=holdout,
    )
    summary = legacy.to_dict()
    assert summary["holdout_rows_with_complete_features"] == summary["holdout_rows"]
    assert safe.to_dict()["holdout_rows_with_complete_features"] == 0
    assert summary["features"]["held_out_truth_in_features"] is True


# ---------------------------------------------------------------------------
# Hyperparameter presets
# ---------------------------------------------------------------------------


def test_the_default_grid_is_the_package_default_preset() -> None:
    default = RFRConfig(mode="RFR3", hemisphere="north")
    assert default.preset is HyperparameterPreset.PACKAGE_DEFAULT
    assert dict(default.grid) == dict(DEFAULT_HYPERPARAMETER_GRID)
    assert default.to_dict()["hyperparameter_preset"] == "package_default"


def test_the_fluxlib_preset_is_the_auto_optimize_grid_verbatim() -> None:
    legacy = RFRConfig(mode="RFR3", hemisphere="north", hyperparameter_preset="legacy_fluxlib")
    assert dict(legacy.grid) == dict(LEGACY_FLUXLIB_HYPERPARAMETER_GRID)
    assert dict(legacy.grid) == {
        "bootstrap": (True,),
        "max_depth": (80, 90, 100, 110),
        "max_features": (2, 3),
        "min_samples_leaf": (3, 4, 5),
        "min_samples_split": (8, 10, 12),
        "n_estimators": (100, 200, 300, 1000),
    }


def test_the_fixed_preset_is_one_point() -> None:
    fixed = RFRConfig(mode="RFR3", hemisphere="north", hyperparameter_preset="legacy_fluxlib_fixed")
    assert dict(fixed.grid) == dict(LEGACY_FLUXLIB_FIXED_HYPERPARAMETERS)
    assert all(len(values) == 1 for values in fixed.grid.values())
    assert set(HYPERPARAMETER_PRESETS) == set(HyperparameterPreset)


def test_a_grid_equal_to_a_preset_is_recognised_and_a_custom_grid_has_none() -> None:
    recognised = RFRConfig(
        mode="RFR3", hemisphere="north", hyperparameter_grid=LEGACY_FLUXLIB_HYPERPARAMETER_GRID
    )
    assert recognised.preset is HyperparameterPreset.LEGACY_FLUXLIB
    custom = RFRConfig(mode="RFR3", hemisphere="north", hyperparameter_grid={"n_estimators": (7,)})
    assert custom.preset is None
    assert custom.to_dict()["hyperparameter_preset"] is None


def test_a_preset_and_a_disagreeing_grid_are_rejected() -> None:
    with pytest.raises(ConfigError, match="does not match"):
        RFRConfig(
            mode="RFR3",
            hemisphere="north",
            hyperparameter_preset="legacy_fluxlib",
            hyperparameter_grid={"n_estimators": (7,)},
        )


def test_switching_preset_on_an_existing_configuration() -> None:
    default = RFRConfig(mode="RFR3", hemisphere="north")
    switched = default.with_hyperparameter_preset("legacy_fluxlib_fixed")
    assert switched.preset is HyperparameterPreset.LEGACY_FLUXLIB_FIXED
    # An ordinary replace keeps the resolved pair consistent.
    assert switched.replace(random_state=7).preset is HyperparameterPreset.LEGACY_FLUXLIB_FIXED


def test_the_preset_reaches_the_model_manifest_and_the_ambiguity_block() -> None:
    legacy = RFRConfig(mode="RFR3", hemisphere="north", hyperparameter_preset="legacy_fluxlib")
    described = RFRModel(legacy, target="LE").to_dict()
    assert described["hyperparameter_preset"] == "legacy_fluxlib"
    assert described["hyperparameter_grid_is_package_default"] is False
    choices = ambiguity_choices(legacy)["choices"]
    assert choices["A1"]["settings"]["hyperparameter_preset"] == "legacy_fluxlib"


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------


def test_a_legacy_validation_runs_end_to_end_and_is_labelled_as_such() -> None:
    site = synthetic_site()
    legacy = site.config(
        "RFR3",
        hyperparameter_grid={"n_estimators": (15,)},
        cv_folds=3,
        features=FeatureConfig(feature_mode="legacy_fluxlib"),
    )
    with pytest.warns(LegacyFluxlibWarning):
        report = validate_rfr(
            site.frame,
            config=legacy,
            targets="NEE",
            qc_columns=site.qc_column("NEE"),
            gaps=site.known_gaps(),
        )
    table = report.to_frame()
    assert set(table["method"]) == {"RFR3-legacy"}
    # Complete legacy features inside every gap class, even the 30-day one.
    assert table.loc[table["gap_class"] == "very_long", "n"].max() > 0

    manifest = report.manifest("NEE").to_dict()
    assert manifest["run"]["feature_mode"] == "legacy_fluxlib"
    assert manifest["run"]["is_paper_faithful"] is False
