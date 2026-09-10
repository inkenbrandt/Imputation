"""Metric tests against hand-calculated fixtures. Covers acceptance tests 28-33.

Every core metric is checked against a fixture whose expected value is worked
out by hand in the test itself, so a refactor that changes a definition fails
here rather than quietly shifting a published number. The day/night tests pin
the paper's 20 W m-2 split, including the boundary value, which the prose leaves
to the ``>`` in ``day = shortwave > 20.0``.

See ``docs/method_spec.md`` for the contract and
``docs/supplement_benchmarks.md`` for benchmark provenance.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from rfrgapfill.config import MetricSubset, ValidationConfig
from rfrgapfill.metrics import (
    DEFAULT_SUBSETS,
    CoreMetrics,
    MetricError,
    SubsetMetrics,
    bias,
    core_metrics,
    daytime_mask,
    energy_balance_ratio,
    metrics_by_subset,
    metrics_from_config,
    nighttime_mask,
    r2,
    regression_slope,
    rmse,
    subset_mask,
)

#: A pair whose every metric is hand-calculable. Measured mean is 3.0, so
#: SS_tot = 4 + 1 + 0 + 1 + 4 = 10; residuals are +1, -1, +1, -1, +1, so
#: SS_res = 5, R2 = 0.5, RMSE = 1.0 and bias = (16 - 15) / 5 = 0.2.
MEASURED = [1.0, 2.0, 3.0, 4.0, 5.0]
FILLED = [2.0, 1.0, 4.0, 3.0, 6.0]


def half_hourly(periods: int) -> pd.DatetimeIndex:
    return pd.date_range("2020-06-01", periods=periods, freq="30min")


# ---------------------------------------------------------------------------
# Core metrics (acceptance tests 28-31)
# ---------------------------------------------------------------------------


def test_r2_matches_hand_calculated_fixture() -> None:
    assert r2(MEASURED, FILLED) == pytest.approx(0.5)


def test_r2_is_one_for_a_perfect_prediction_and_penalises_bias() -> None:
    assert r2(MEASURED, MEASURED) == pytest.approx(1.0)
    shifted = [value + 1.0 for value in MEASURED]
    # A constant offset correlates perfectly but is not explained variance:
    # SS_res = 5, SS_tot = 10, so R2 = 0.5 rather than the squared correlation 1.
    assert r2(MEASURED, shifted) == pytest.approx(0.5)


def test_regression_slope_uses_measured_on_x_and_filled_on_y() -> None:
    # cov(x, y) = (2 + 1 + 0 + 1 + 4 + ...) worked out below:
    # centred x = [-2, -1, 0, 1, 2]; filled mean is 3.2, centred y =
    # [-1.2, -2.2, 0.8, -0.2, 2.8]; dot = 2.4 + 2.2 + 0 - 0.2 + 5.6 = 10.0,
    # and var(x) = 10, so the slope is 1.0.
    assert regression_slope(MEASURED, FILLED) == pytest.approx(1.0)


def test_regression_slope_is_orientation_sensitive() -> None:
    # Filled values with half the amplitude of the measurements: measured on x
    # gives 0.5. Swapping the arguments gives 2.0, which is why the orientation
    # is fixed by the signature rather than left to the caller.
    damped = [3.0 + (value - 3.0) * 0.5 for value in MEASURED]
    assert regression_slope(MEASURED, damped) == pytest.approx(0.5)
    assert regression_slope(damped, MEASURED) == pytest.approx(2.0)


def test_rmse_matches_hand_calculated_fixture() -> None:
    assert rmse(MEASURED, FILLED) == pytest.approx(1.0)
    assert rmse(MEASURED, MEASURED) == pytest.approx(0.0)


def test_bias_matches_the_paper_definition() -> None:
    expected = (sum(FILLED) - sum(MEASURED)) / len(MEASURED)
    assert bias(MEASURED, FILLED) == pytest.approx(expected)
    assert bias(MEASURED, FILLED) == pytest.approx(0.2)


def test_bias_is_signed_towards_over_prediction() -> None:
    assert bias(MEASURED, [value + 2.0 for value in MEASURED]) == pytest.approx(2.0)
    assert bias(MEASURED, [value - 2.0 for value in MEASURED]) == pytest.approx(-2.0)


# ---------------------------------------------------------------------------
# Degenerate inputs
# ---------------------------------------------------------------------------


def test_non_finite_pairs_are_dropped_rather_than_scored_as_zero_error() -> None:
    measured = [1.0, 2.0, np.nan, 4.0, 5.0]
    filled = [2.0, 1.0, 4.0, np.nan, 6.0]
    # Only rows 0, 1 and 4 can be scored: measured [1, 2, 5], filled [2, 1, 6].
    assert core_metrics(measured, filled).n == 3
    assert bias(measured, filled) == pytest.approx((9.0 - 8.0) / 3.0)


def test_empty_input_scores_as_n_zero_with_missing_metrics() -> None:
    metrics = core_metrics([], [])
    assert metrics.n == 0
    assert metrics.is_empty
    for value in (metrics.r2, metrics.slope, metrics.rmse, metrics.bias):
        assert math.isnan(value)


def test_constant_measurements_leave_r2_and_slope_undefined() -> None:
    metrics = core_metrics([2.0, 2.0, 2.0], [1.0, 2.0, 3.0])
    assert math.isnan(metrics.r2)
    assert math.isnan(metrics.slope)
    # RMSE and bias remain perfectly well defined.
    assert metrics.rmse == pytest.approx(math.sqrt(2.0 / 3.0))
    assert metrics.bias == pytest.approx(0.0)


def test_single_pair_leaves_regression_quantities_undefined() -> None:
    metrics = core_metrics([3.0], [4.0])
    assert metrics.n == 1
    assert math.isnan(metrics.r2)
    assert math.isnan(metrics.slope)
    assert metrics.rmse == pytest.approx(1.0)


def test_mismatched_lengths_and_indexes_raise() -> None:
    with pytest.raises(MetricError, match="has 4 value"):
        core_metrics(MEASURED, FILLED[:4])
    left = pd.Series(MEASURED, index=half_hourly(5))
    right = pd.Series(FILLED, index=half_hourly(5) + pd.Timedelta("1D"))
    with pytest.raises(MetricError, match="does not share an index"):
        core_metrics(left, right)


def test_non_numeric_input_raises_a_metric_error() -> None:
    with pytest.raises(MetricError, match="not numeric"):
        core_metrics(["a", "b"], [1.0, 2.0])


def test_two_dimensional_input_is_rejected() -> None:
    with pytest.raises(MetricError, match="one-dimensional"):
        core_metrics(np.zeros((2, 2)), np.zeros((2, 2)))
    with pytest.raises(MetricError, match="got a DataFrame"):
        core_metrics(pd.DataFrame({"LE": MEASURED}), FILLED)


# ---------------------------------------------------------------------------
# Day/night masks (acceptance test 32)
# ---------------------------------------------------------------------------

#: Values straddling the paper's 20 W m-2 threshold, including the boundary.
SHORTWAVE = [0.0, 19.999, 20.0, 20.001, 500.0, np.nan]


def test_daytime_is_strictly_above_the_threshold_and_night_takes_the_boundary() -> None:
    day = daytime_mask(SHORTWAVE).to_numpy()
    night = nighttime_mask(SHORTWAVE).to_numpy()
    assert list(day) == [False, False, False, True, True, False]
    assert list(night) == [True, True, True, False, False, False]


def test_missing_radiation_is_neither_day_nor_night() -> None:
    day = daytime_mask([np.nan]).to_numpy()
    night = nighttime_mask([np.nan]).to_numpy()
    assert not day[0]
    assert not night[0]


def test_day_and_night_partition_every_observed_row() -> None:
    observed = [value for value in SHORTWAVE if not math.isnan(value)]
    day = daytime_mask(observed).to_numpy()
    night = nighttime_mask(observed).to_numpy()
    assert list(day ^ night) == [True] * len(observed)


def test_the_threshold_is_configurable() -> None:
    # At a 50 W m-2 threshold the 20.001 value becomes nighttime.
    day = daytime_mask(SHORTWAVE, threshold=50.0).to_numpy()
    assert list(day) == [False, False, False, False, True, False]
    assert daytime_mask([0.5], threshold=0.0).to_numpy()[0]


def test_masks_preserve_a_timestamp_index() -> None:
    index = half_hourly(len(SHORTWAVE))
    mask = daytime_mask(pd.Series(SHORTWAVE, index=index))
    assert mask.index.equals(index)
    assert mask.dtype == bool


def test_subset_mask_selects_every_row_for_the_all_subset() -> None:
    selected = subset_mask(SHORTWAVE, MetricSubset.ALL).to_numpy()
    # Including the row whose radiation is missing: "all" does not need a class.
    assert list(selected) == [True] * len(SHORTWAVE)


def test_subset_mask_accepts_day_and_night_spellings() -> None:
    assert list(subset_mask(SHORTWAVE, "day").to_numpy()) == list(daytime_mask(SHORTWAVE))
    assert list(subset_mask(SHORTWAVE, "night").to_numpy()) == list(nighttime_mask(SHORTWAVE))


def test_a_non_finite_threshold_is_rejected() -> None:
    with pytest.raises(MetricError, match="finite"):
        daytime_mask(SHORTWAVE, threshold=float("nan"))
    with pytest.raises(MetricError, match="must be a number"):
        daytime_mask(SHORTWAVE, threshold="20")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# All/day/night reporting (method_spec.md 6.2)
# ---------------------------------------------------------------------------


def diel_fixture() -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return measured, filled and radiation series straddling the threshold.

    Three daytime rows, three nighttime rows (one of them exactly on the 20
    W m-2 boundary) and one row whose radiation is missing.
    """
    index = half_hourly(7)
    shortwave = pd.Series([0.0, 5.0, 20.0, 20.001, 300.0, 600.0, np.nan], index=index)
    measured = pd.Series([0.5, 1.0, 1.5, 10.0, 20.0, 30.0, 4.0], index=index)
    filled = pd.Series([1.5, 0.0, 2.5, 12.0, 18.0, 33.0, 5.0], index=index)
    return measured, filled, shortwave


def test_reporting_returns_all_day_and_night_by_default() -> None:
    measured, filled, shortwave = diel_fixture()
    report = metrics_by_subset(measured, filled, shortwave)
    assert tuple(report.metrics) == DEFAULT_SUBSETS
    assert report.all.n == 7
    assert report.daytime.n == 3
    assert report.nighttime.n == 3
    assert report.n_missing_shortwave == 1
    assert report.daytime_threshold == 20.0


def test_scored_counts_account_for_every_row() -> None:
    measured, filled, shortwave = diel_fixture()
    report = metrics_by_subset(measured, filled, shortwave)
    assert report.all.n == report.daytime.n + report.nighttime.n + report.n_missing_shortwave


def test_subset_metrics_are_computed_over_their_own_rows_only() -> None:
    measured, filled, shortwave = diel_fixture()
    report = metrics_by_subset(measured, filled, shortwave)
    night_rows = nighttime_mask(shortwave).to_numpy()
    assert report.nighttime == core_metrics(measured[night_rows], filled[night_rows])
    # Night bias here is (1.5 + 0.0 + 2.5) - (0.5 + 1.0 + 1.5) = 1.0 over 3 rows.
    assert report.nighttime.bias == pytest.approx(1.0 / 3.0)


def test_day_and_night_metrics_differ_from_the_aggregate() -> None:
    measured, filled, shortwave = diel_fixture()
    report = metrics_by_subset(measured, filled, shortwave)
    # The point of the split: strong daytime skill hides weak nighttime skill.
    assert report.daytime.r2 > report.nighttime.r2
    assert report.all.r2 != pytest.approx(report.nighttime.r2)


def test_a_boundary_value_is_scored_as_night() -> None:
    index = half_hourly(4)
    shortwave = pd.Series([20.0, 20.0, 20.001, 20.001], index=index)
    measured = pd.Series([1.0, 2.0, 3.0, 4.0], index=index)
    filled = pd.Series([1.0, 2.0, 3.0, 4.0], index=index)
    report = metrics_by_subset(measured, filled, shortwave)
    assert report.nighttime.n == 2
    assert report.daytime.n == 2


def test_an_empty_subset_is_reported_rather_than_dropped() -> None:
    index = half_hourly(3)
    shortwave = pd.Series([0.0, 1.0, 2.0], index=index)
    report = metrics_by_subset(
        pd.Series([1.0, 2.0, 3.0], index=index),
        pd.Series([1.5, 2.5, 3.5], index=index),
        shortwave,
    )
    assert MetricSubset.DAYTIME in report
    assert report.daytime.n == 0
    assert math.isnan(report.daytime.rmse)
    assert report.nighttime.n == 3


def test_the_reported_threshold_is_configurable_through_the_call() -> None:
    measured, filled, shortwave = diel_fixture()
    report = metrics_by_subset(measured, filled, shortwave, threshold=100.0)
    assert report.daytime_threshold == 100.0
    assert report.daytime.n == 2
    assert report.nighttime.n == 4


def test_the_reported_threshold_and_subsets_come_from_the_validation_config() -> None:
    measured, filled, shortwave = diel_fixture()
    config = ValidationConfig(daytime_threshold=100.0, subsets=["all", "night"])
    report = metrics_from_config(measured, filled, shortwave, config=config)
    assert report.daytime_threshold == config.daytime_threshold
    assert tuple(report.metrics) == (MetricSubset.ALL, MetricSubset.NIGHTTIME)
    assert report.nighttime.n == 4


def test_requesting_a_radiation_subset_without_radiation_is_an_error() -> None:
    measured, filled, _ = diel_fixture()
    with pytest.raises(MetricError, match="shortwave radiation is required"):
        metrics_by_subset(measured, filled)


def test_the_all_subset_alone_needs_no_radiation() -> None:
    measured, filled, _ = diel_fixture()
    report = metrics_by_subset(measured, filled, subsets=["all"])
    assert report.all.n == 7
    assert MetricSubset.DAYTIME not in report


def test_membership_is_false_for_an_unknown_subset_name() -> None:
    measured, filled, shortwave = diel_fixture()
    report = metrics_by_subset(measured, filled, shortwave)
    assert "day" in report
    assert "dusk" not in report


def test_an_unreported_subset_raises_on_access() -> None:
    measured, filled, _ = diel_fixture()
    report = metrics_by_subset(measured, filled, subsets=["all"])
    with pytest.raises(KeyError, match="daytime was not reported"):
        report.daytime


def test_subsets_must_be_a_non_empty_sequence_of_known_names() -> None:
    measured, filled, shortwave = diel_fixture()
    with pytest.raises(MetricError, match="at least one"):
        metrics_by_subset(measured, filled, shortwave, subsets=[])
    with pytest.raises(ValueError, match="not a recognised MetricSubset"):
        metrics_by_subset(measured, filled, shortwave, subsets=["dusk"])


def test_reporting_is_serialisable_and_tabular() -> None:
    measured, filled, shortwave = diel_fixture()
    report = metrics_by_subset(measured, filled, shortwave)
    payload = report.to_dict()
    assert payload["daytime_threshold"] == 20.0
    assert payload["n_missing_shortwave"] == 1
    assert set(payload["subsets"]) == {"all", "daytime", "nighttime"}
    assert set(payload["subsets"]["nighttime"]) == {"n", "r2", "slope", "rmse", "bias"}
    frame = report.to_frame()
    assert list(frame.index) == ["all", "daytime", "nighttime"]
    assert frame.loc["daytime", "n"] == 3


def test_reported_results_are_immutable() -> None:
    measured, filled, shortwave = diel_fixture()
    report = metrics_by_subset(measured, filled, shortwave)
    with pytest.raises(TypeError):
        report.metrics[MetricSubset.ALL] = CoreMetrics(0, 0.0, 0.0, 0.0, 0.0)  # type: ignore[index]
    with pytest.raises(AttributeError):
        report.daytime_threshold = 5.0  # type: ignore[misc]


def test_reporting_accepts_plain_sequences_as_well_as_series() -> None:
    measured, filled, shortwave = diel_fixture()
    from_series = metrics_by_subset(measured, filled, shortwave)
    from_lists = metrics_by_subset(measured.tolist(), filled.tolist(), shortwave.tolist())
    assert isinstance(from_lists, SubsetMetrics)
    assert from_lists.to_dict() == from_series.to_dict()


# ---------------------------------------------------------------------------
# Energy-balance ratio (acceptance test 33)
# ---------------------------------------------------------------------------


def test_ebr_matches_a_hand_calculated_fixture() -> None:
    sensible = [50.0, 60.0]
    latent = [100.0, 90.0]
    net_radiation = [200.0, 190.0]
    soil_heat = [10.0, 10.0]
    # (50 + 100 + 60 + 90) / ((200 - 10) + (190 - 10)) = 300 / 370
    expected = 300.0 / 370.0
    assert energy_balance_ratio(sensible, latent, net_radiation, soil_heat) == pytest.approx(
        expected
    )


def test_ebr_ignores_rows_where_any_component_is_missing() -> None:
    ratio = energy_balance_ratio([50.0, np.nan], [100.0, 90.0], [200.0, 190.0], [10.0, 10.0])
    assert ratio == pytest.approx(150.0 / 190.0)


def test_ebr_is_undefined_for_a_zero_or_absent_available_energy() -> None:
    assert math.isnan(energy_balance_ratio([50.0], [100.0], [10.0], [10.0]))
    assert math.isnan(energy_balance_ratio([np.nan], [np.nan], [np.nan], [np.nan]))
    assert math.isnan(energy_balance_ratio([], [], [], []))


def test_ebr_difference_between_measured_and_filled_is_reportable() -> None:
    net_radiation = [200.0, 190.0]
    soil_heat = [10.0, 10.0]
    measured = energy_balance_ratio([50.0, 60.0], [100.0, 90.0], net_radiation, soil_heat)
    filled = energy_balance_ratio([55.0, 60.0], [100.0, 90.0], net_radiation, soil_heat)
    assert filled - measured == pytest.approx(5.0 / 370.0)
