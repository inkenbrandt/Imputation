"""Metric tests against hand-calculated fixtures. Covers Step 11 and acceptance
tests 28-31 and 33.

Every expected number in this module is derivable with a pencil. The core
fixture is deliberately tiny and its arithmetic is written out in
:data:`HAND` below, so a failure here says the formula changed, not that a
reference implementation drifted. Acceptance test 32 (the day/night split at the
configured threshold) belongs to the validation layer, which does the subsetting.

See ``docs/method_spec.md`` section 6 for the contract and ambiguity A12 for the
two readings of ``R2``.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from rfrgapfill.config import R2Definition, ValidationConfig
from rfrgapfill.metrics import (
    CoreMetrics,
    EnergyBalanceComparison,
    MetricError,
    bias,
    compare_energy_balance,
    core_metrics,
    energy_balance_ratio,
    r2,
    regression_slope,
    rmse,
)

# ---------------------------------------------------------------------------
# The hand-calculated fixture
# ---------------------------------------------------------------------------

#: measured = 1, 2, 3, 4; predicted = 2, 2, 4, 4.
#:
#: mean(measured) = 2.5, mean(predicted) = 3.0
#: SS_tot  = 1.5^2 + 0.5^2 + 0.5^2 + 1.5^2               = 5.0
#: SS_res  = 1^2 + 0^2 + 1^2 + 0^2                       = 2.0
#: residual R2      = 1 - 2/5                            = 0.6
#: cov     = (-1.5)(-1) + (-0.5)(-1) + (0.5)(1) + (1.5)(1) = 4.0
#: var(pred, sum)   = 1 + 1 + 1 + 1                      = 4.0
#: slope   = cov / SS_tot = 4/5                          = 0.8
#: r^2     = cov^2 / (SS_tot * SS_pred) = 16 / 20        = 0.8
#: RMSE    = sqrt(2/4)                                   = sqrt(0.5)
#: bias    = (12 - 10) / 4                               = 0.5
MEASURED = [1.0, 2.0, 3.0, 4.0]
PREDICTED = [2.0, 2.0, 4.0, 4.0]

HAND = {
    "r2_residual": 0.6,
    "r2_squared_correlation": 0.8,
    "slope": 0.8,
    "rmse": math.sqrt(0.5),
    "bias": 0.5,
}


@pytest.fixture
def index() -> pd.DatetimeIndex:
    """A half-hourly index the length of the fixture."""
    return pd.date_range("2020-06-01", periods=len(MEASURED), freq="30min")


@pytest.fixture
def measured(index: pd.DatetimeIndex) -> pd.Series:
    return pd.Series(MEASURED, index=index, name="LE")


@pytest.fixture
def predicted(index: pd.DatetimeIndex) -> pd.Series:
    return pd.Series(PREDICTED, index=index, name="LE_filled")


# ---------------------------------------------------------------------------
# Acceptance tests 28-31: each metric against the hand calculation
# ---------------------------------------------------------------------------


def test_r2_matches_the_hand_calculation(measured: pd.Series, predicted: pd.Series) -> None:
    assert r2(measured, predicted) == pytest.approx(HAND["r2_residual"])


def test_r2_defaults_to_the_residual_definition(measured: pd.Series, predicted: pd.Series) -> None:
    assert r2(measured, predicted) == r2(measured, predicted, definition="residual")


def test_squared_correlation_is_the_other_reading_of_r2(
    measured: pd.Series, predicted: pd.Series
) -> None:
    """A12: the two definitions differ, and this fixture separates them."""
    residual = r2(measured, predicted, definition=R2Definition.RESIDUAL)
    correlation = r2(measured, predicted, definition=R2Definition.SQUARED_CORRELATION)
    assert residual == pytest.approx(HAND["r2_residual"])
    assert correlation == pytest.approx(HAND["r2_squared_correlation"])
    assert residual != correlation


def test_regression_slope_matches_the_hand_calculation(
    measured: pd.Series, predicted: pd.Series
) -> None:
    assert regression_slope(measured, predicted) == pytest.approx(HAND["slope"])


def test_rmse_matches_the_hand_calculation(measured: pd.Series, predicted: pd.Series) -> None:
    assert rmse(measured, predicted) == pytest.approx(HAND["rmse"])


def test_bias_matches_the_published_formula(measured: pd.Series, predicted: pd.Series) -> None:
    """Acceptance test 31: bias is ``(sum(pred) - sum(obs)) / n`` exactly."""
    expected = (sum(PREDICTED) - sum(MEASURED)) / len(MEASURED)
    assert bias(measured, predicted) == pytest.approx(expected)
    assert bias(measured, predicted) == pytest.approx(HAND["bias"])


# ---------------------------------------------------------------------------
# Definitions and orientation
# ---------------------------------------------------------------------------


def test_residual_r2_goes_negative_for_a_prediction_worse_than_the_mean() -> None:
    """SS_tot = 5; SS_res = 11.5^2 + 10.5^2 + 9.5^2 + 8.5^2 = 405; R2 = 1 - 81 = -80."""
    assert r2(MEASURED, [12.5, 12.5, 12.5, 12.5]) == pytest.approx(-80.0)


def test_squared_correlation_cannot_see_a_systematic_offset(
    measured: pd.Series, predicted: pd.Series
) -> None:
    """The reason ``residual`` is the default (A12)."""
    offset = predicted + 100.0
    assert r2(measured, offset, definition="squared_correlation") == pytest.approx(
        r2(measured, predicted, definition="squared_correlation")
    )
    assert r2(measured, offset) < r2(measured, predicted)


def test_a_perfect_prediction_scores_one_on_both_definitions(measured: pd.Series) -> None:
    assert r2(measured, measured) == pytest.approx(1.0)
    assert r2(measured, measured, definition="squared_correlation") == pytest.approx(1.0)
    assert regression_slope(measured, measured) == pytest.approx(1.0)
    assert rmse(measured, measured) == pytest.approx(0.0)
    assert bias(measured, measured) == pytest.approx(0.0)


def test_the_slope_regresses_predicted_on_measured_not_the_reverse(
    measured: pd.Series, predicted: pd.Series
) -> None:
    """Orientation is fixed: x = measured, y = predicted (method_spec.md 6.1).

    The reverse fit gives cov / SS_pred = 4/4 = 1.0, so a swapped orientation
    would be visible rather than harmless.
    """
    assert regression_slope(measured, predicted) == pytest.approx(0.8)
    assert regression_slope(predicted, measured) == pytest.approx(1.0)


def test_the_slope_can_be_forced_through_the_origin(
    measured: pd.Series, predicted: pd.Series
) -> None:
    """sum(xy) / sum(x^2) = (2 + 4 + 12 + 16) / (1 + 4 + 9 + 16) = 34/30."""
    assert regression_slope(measured, predicted, fit_intercept=False) == pytest.approx(34 / 30)


def test_an_unknown_r2_definition_is_rejected(measured: pd.Series, predicted: pd.Series) -> None:
    with pytest.raises(ValueError, match="r2_definition"):
        r2(measured, predicted, definition="pearson")


def test_bias_cancels_where_rmse_does_not() -> None:
    """Equal and opposite errors: bias 0, RMSE 1."""
    assert bias([1.0, 2.0], [2.0, 1.0]) == pytest.approx(0.0)
    assert rmse([1.0, 2.0], [2.0, 1.0]) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Empty and undefined subsets
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("metric", [r2, regression_slope, rmse, bias])
def test_an_empty_subset_is_undefined_rather_than_an_error(metric: object) -> None:
    """Step 11 requires empty subsets to be handled safely."""
    assert metric([], []) is None  # type: ignore[operator]


@pytest.mark.parametrize("metric", [r2, regression_slope, rmse, bias])
def test_a_subset_of_only_incomplete_pairs_is_undefined(metric: object) -> None:
    assert metric([np.nan, np.nan], [1.0, 2.0]) is None  # type: ignore[operator]


def test_constant_measurements_leave_r2_and_slope_undefined() -> None:
    """No variance to explain and no slope to fit; RMSE and bias still exist."""
    constant = [3.0, 3.0, 3.0]
    assert r2(constant, [1.0, 2.0, 3.0]) is None
    assert r2(constant, [1.0, 2.0, 3.0], definition="squared_correlation") is None
    assert regression_slope(constant, [1.0, 2.0, 3.0]) is None
    assert rmse(constant, [3.0, 3.0, 4.0]) == pytest.approx(math.sqrt(1 / 3))
    assert bias(constant, [3.0, 3.0, 4.0]) == pytest.approx(1 / 3)


def test_a_single_pair_leaves_r2_and_slope_undefined() -> None:
    assert r2([2.0], [3.0]) is None
    assert regression_slope([2.0], [3.0]) is None
    assert rmse([2.0], [3.0]) == pytest.approx(1.0)
    assert bias([2.0], [3.0]) == pytest.approx(1.0)


def test_a_constant_prediction_leaves_squared_correlation_undefined() -> None:
    assert r2(MEASURED, [2.0, 2.0, 2.0, 2.0], definition="squared_correlation") is None
    assert r2(MEASURED, [2.0, 2.0, 2.0, 2.0]) is not None


# ---------------------------------------------------------------------------
# Pairing rules
# ---------------------------------------------------------------------------


def test_incomplete_pairs_are_dropped_from_both_sides() -> None:
    """An unfilled row leaves the score entirely, denominator included."""
    measured = [*MEASURED, 5.0]
    predicted = [*PREDICTED, np.nan]
    # The surviving pairs are exactly the hand-calculated fixture.
    assert bias(measured, predicted) == pytest.approx(HAND["bias"])
    assert rmse(measured, predicted) == pytest.approx(HAND["rmse"])
    assert r2(measured, predicted) == pytest.approx(HAND["r2_residual"])
    assert regression_slope(measured, predicted) == pytest.approx(HAND["slope"])


def test_a_missing_measurement_drops_its_pair_too() -> None:
    measured = [1.0, 2.0, np.nan, 3.0, 4.0]
    predicted = [2.0, 2.0, 99.0, 4.0, 4.0]  # survivors are the fixture again
    assert bias(measured, predicted) == pytest.approx(HAND["bias"])


def test_infinities_count_as_incomplete() -> None:
    assert bias([*MEASURED, 5.0], [*PREDICTED, np.inf]) == pytest.approx(HAND["bias"])


def test_series_indexed_differently_are_refused_rather_than_aligned(
    measured: pd.Series, predicted: pd.Series
) -> None:
    """Silent realignment would compare a prediction against the wrong half hour."""
    shifted = predicted.copy()
    shifted.index = shifted.index + pd.Timedelta("1h")
    with pytest.raises(MetricError, match="indexed differently"):
        bias(measured, shifted)


def test_series_of_different_lengths_are_refused(measured: pd.Series) -> None:
    with pytest.raises(MetricError, match="same length"):
        bias(measured, [1.0, 2.0])


def test_non_numeric_values_are_refused(measured: pd.Series) -> None:
    with pytest.raises(MetricError, match="numeric"):
        bias(measured, ["a", "b", "c", "d"])


def test_a_frame_is_refused(measured: pd.Series) -> None:
    with pytest.raises(MetricError, match="DataFrame"):
        bias(measured, pd.DataFrame({"a": MEASURED}))


def test_plain_sequences_and_arrays_score_the_same_as_series(
    measured: pd.Series, predicted: pd.Series
) -> None:
    from_series = bias(measured, predicted)
    from_lists = bias(MEASURED, PREDICTED)
    from_arrays = bias(np.array(MEASURED), np.array(PREDICTED))
    assert from_series == from_lists == from_arrays


def test_an_unindexed_input_pairs_positionally_with_a_series(
    measured: pd.Series,
) -> None:
    """A bare array carries no index to check, so only the length is required."""
    assert bias(measured, PREDICTED) == pytest.approx(HAND["bias"])


# ---------------------------------------------------------------------------
# CoreMetrics
# ---------------------------------------------------------------------------


def test_core_metrics_agrees_with_the_individual_functions(
    measured: pd.Series, predicted: pd.Series
) -> None:
    """The exit criterion's "independent" functions must not disagree with the bundle."""
    scored = core_metrics(measured, predicted)
    assert scored.r2 == r2(measured, predicted)
    assert scored.slope == regression_slope(measured, predicted)
    assert scored.rmse == rmse(measured, predicted)
    assert scored.bias == bias(measured, predicted)


def test_core_metrics_carries_the_r2_definition_it_used(
    measured: pd.Series, predicted: pd.Series
) -> None:
    scored = core_metrics(measured, predicted, definition="squared_correlation")
    assert scored.r2_kind is R2Definition.SQUARED_CORRELATION
    assert scored.r2 == pytest.approx(HAND["r2_squared_correlation"])
    assert scored.to_dict()["r2_definition"] == "squared_correlation"


def test_core_metrics_reports_what_the_pairing_cost() -> None:
    scored = core_metrics([1.0, 2.0, 3.0, 4.0, 5.0], [2.0, 2.0, np.nan, 4.0, 4.0])
    assert scored.n == 4
    assert scored.n_offered == 5
    assert scored.dropped_incomplete == 1
    assert not scored.is_empty


def test_core_metrics_on_an_empty_subset_is_empty_not_zero() -> None:
    scored = core_metrics([], [])
    assert scored.is_empty
    assert scored.n == 0 and scored.n_offered == 0
    assert scored.r2 is None
    assert scored.slope is None
    assert scored.rmse is None
    assert scored.bias is None


def test_core_metrics_to_dict_is_json_serialisable(
    measured: pd.Series, predicted: pd.Series
) -> None:
    import json

    payload = core_metrics(measured, predicted).to_dict()
    assert json.loads(json.dumps(payload)) == payload
    assert payload["bias"] == pytest.approx(HAND["bias"])


def test_an_undefined_metric_is_null_in_the_manifest_not_nan() -> None:
    import json

    payload = core_metrics([], []).to_dict()
    assert json.dumps(payload).count("null") == 4
    assert "NaN" not in json.dumps(payload)


def test_core_metrics_rejects_impossible_row_accounting() -> None:
    with pytest.raises(MetricError, match="cannot increase"):
        CoreMetrics(n=5, n_offered=4, r2=None, slope=None, rmse=None, bias=None)


# ---------------------------------------------------------------------------
# Acceptance test 33: energy-balance ratio
# ---------------------------------------------------------------------------

#: H = 30, 50; LE = 20, 30; NETRAD = 100, 150; G = 10, 20.
#: sum(H + LE)      = 50 + 80    = 130
#: sum(NETRAD - G)  = 90 + 130   = 220
#: EBR              = 130 / 220  = 0.590909...
EBR_H = [30.0, 50.0]
EBR_LE = [20.0, 30.0]
EBR_NETRAD = [100.0, 150.0]
EBR_G = [10.0, 20.0]
EBR_EXPECTED = 130.0 / 220.0


def test_energy_balance_ratio_matches_the_hand_calculation() -> None:
    assert energy_balance_ratio(
        sensible_heat=EBR_H,
        latent_heat=EBR_LE,
        net_radiation=EBR_NETRAD,
        soil_heat_flux=EBR_G,
    ) == pytest.approx(EBR_EXPECTED)


def test_energy_balance_ratio_is_a_ratio_of_sums_not_a_mean_of_ratios() -> None:
    """Per-row ratios are 50/90 and 80/130; their mean is not the EBR."""
    per_row_mean = ((50 / 90) + (80 / 130)) / 2
    assert per_row_mean != pytest.approx(EBR_EXPECTED)


def test_a_zero_denominator_is_undefined_rather_than_a_division_error() -> None:
    """Available energy cancelling to zero is the case Step 11 calls out."""
    assert (
        energy_balance_ratio(
            sensible_heat=[10.0, 10.0],
            latent_heat=[5.0, 5.0],
            net_radiation=[100.0, 20.0],
            soil_heat_flux=[20.0, 100.0],
        )
        is None
    )


def test_an_empty_energy_balance_interval_is_undefined() -> None:
    assert (
        energy_balance_ratio(sensible_heat=[], latent_heat=[], net_radiation=[], soil_heat_flux=[])
        is None
    )


def test_a_row_missing_any_component_leaves_the_energy_balance_entirely() -> None:
    """Otherwise the numerator and denominator would cover different half hours."""
    ratio = energy_balance_ratio(
        sensible_heat=[*EBR_H, 999.0],
        latent_heat=[*EBR_LE, 999.0],
        net_radiation=[*EBR_NETRAD, np.nan],
        soil_heat_flux=[*EBR_G, 5.0],
    )
    assert ratio == pytest.approx(EBR_EXPECTED)


def test_energy_balance_ratio_is_keyword_only() -> None:
    with pytest.raises(TypeError):
        energy_balance_ratio(EBR_H, EBR_LE, EBR_NETRAD, EBR_G)  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Measured against filled (method_spec.md 6.4)
# ---------------------------------------------------------------------------


def test_compare_energy_balance_returns_both_values_and_their_difference() -> None:
    """Filled H is doubled: sum(H + LE) = 80 + 130 = 210, so EBR = 210/220."""
    comparison = compare_energy_balance(
        measured_sensible_heat=EBR_H,
        measured_latent_heat=EBR_LE,
        filled_sensible_heat=[60.0, 100.0],
        filled_latent_heat=EBR_LE,
        net_radiation=EBR_NETRAD,
        soil_heat_flux=EBR_G,
    )
    assert comparison.measured == pytest.approx(EBR_EXPECTED)
    assert comparison.filled == pytest.approx(210.0 / 220.0)
    assert comparison.difference == pytest.approx(210.0 / 220.0 - EBR_EXPECTED)
    assert comparison.n == 2 and comparison.n_offered == 2


def test_an_unchanged_fill_leaves_the_ratio_and_the_difference_alone() -> None:
    comparison = compare_energy_balance(
        measured_sensible_heat=EBR_H,
        measured_latent_heat=EBR_LE,
        filled_sensible_heat=EBR_H,
        filled_latent_heat=EBR_LE,
        net_radiation=EBR_NETRAD,
        soil_heat_flux=EBR_G,
    )
    assert comparison.measured == comparison.filled
    assert comparison.difference == pytest.approx(0.0)


def test_both_ratios_are_computed_over_one_shared_row_set() -> None:
    """A row the fill did not reach is excluded from the measured ratio as well.

    Without the shared row set the measured ratio would keep row three and the
    filled ratio would not, and the difference would mix the change in flux with
    the change in interval.
    """
    comparison = compare_energy_balance(
        measured_sensible_heat=[*EBR_H, 70.0],
        measured_latent_heat=[*EBR_LE, 40.0],
        filled_sensible_heat=[*EBR_H, np.nan],
        filled_latent_heat=[*EBR_LE, 40.0],
        net_radiation=[*EBR_NETRAD, 200.0],
        soil_heat_flux=[*EBR_G, 30.0],
    )
    assert comparison.n == 2
    assert comparison.n_offered == 3
    assert comparison.measured == pytest.approx(EBR_EXPECTED)
    assert comparison.difference == pytest.approx(0.0)


def test_an_empty_comparison_reports_no_difference() -> None:
    comparison = compare_energy_balance(
        measured_sensible_heat=[],
        measured_latent_heat=[],
        filled_sensible_heat=[],
        filled_latent_heat=[],
        net_radiation=[],
        soil_heat_flux=[],
    )
    assert comparison.is_empty
    assert comparison.measured is None
    assert comparison.filled is None
    assert comparison.difference is None


def test_energy_balance_comparison_to_dict_names_both_ratios() -> None:
    import json

    comparison = compare_energy_balance(
        measured_sensible_heat=EBR_H,
        measured_latent_heat=EBR_LE,
        filled_sensible_heat=EBR_H,
        filled_latent_heat=EBR_LE,
        net_radiation=EBR_NETRAD,
        soil_heat_flux=EBR_G,
    )
    payload = comparison.to_dict()
    assert set(payload) == {
        "n",
        "n_offered",
        "n_missing_measured",
        "n_missing_filled",
        "n_missing_available_energy",
        "available_energy",
        "measured_ebr",
        "filled_ebr",
        "difference",
    }
    assert json.loads(json.dumps(payload)) == payload


# ---------------------------------------------------------------------------
# Step 19: the denominator, and what missing data costs
# ---------------------------------------------------------------------------


def test_the_comparison_reports_its_shared_denominator() -> None:
    comparison = compare_energy_balance(
        measured_sensible_heat=EBR_H,
        measured_latent_heat=EBR_LE,
        filled_sensible_heat=EBR_H,
        filled_latent_heat=EBR_LE,
        net_radiation=EBR_NETRAD,
        soil_heat_flux=EBR_G,
    )
    assert comparison.available_energy == pytest.approx(220.0)


def test_a_negative_denominator_is_undefined_rather_than_a_backwards_closure() -> None:
    """A night-only interval: -60 / -40 = 1.5 would read as over-closure.

    Worse, raising the turbulent flux there would *lower* the ratio, so the
    filled-minus-measured difference would report a fill's effect with its sign
    flipped. The ratio is undefined; the denominator is still reported.
    """
    night = {
        "net_radiation": [-50.0, -40.0],
        "soil_heat_flux": [-30.0, -20.0],
    }
    assert (
        energy_balance_ratio(sensible_heat=[-20.0, -20.0], latent_heat=[-10.0, -10.0], **night)
        is None
    )
    comparison = compare_energy_balance(
        measured_sensible_heat=[-20.0, -20.0],
        measured_latent_heat=[-10.0, -10.0],
        filled_sensible_heat=[-10.0, -10.0],
        filled_latent_heat=[-10.0, -10.0],
        **night,
    )
    assert comparison.n == 2
    assert comparison.available_energy == pytest.approx(-40.0)
    assert comparison.measured is None
    assert comparison.filled is None
    assert comparison.difference is None


def test_a_night_row_inside_a_positive_interval_still_counts() -> None:
    """The rule is on the interval's sum, not on each row's sign."""
    ratio = energy_balance_ratio(
        sensible_heat=[*EBR_H, -5.0],
        latent_heat=[*EBR_LE, 0.0],
        net_radiation=[*EBR_NETRAD, -40.0],
        soil_heat_flux=[*EBR_G, -10.0],
    )
    assert ratio == pytest.approx(125.0 / 190.0)


def test_every_dropped_row_is_attributed_to_what_it_was_missing() -> None:
    """Rows 3-6 are each missing something different; row 6 is missing two things."""
    comparison = compare_energy_balance(
        measured_sensible_heat=[*EBR_H, np.nan, 1.0, 1.0, np.nan],
        measured_latent_heat=[*EBR_LE, 1.0, 1.0, 1.0, 1.0],
        filled_sensible_heat=[*EBR_H, 1.0, 1.0, 1.0, 1.0],
        filled_latent_heat=[*EBR_LE, 1.0, np.nan, 1.0, 1.0],
        net_radiation=[*EBR_NETRAD, 50.0, 50.0, np.nan, 50.0],
        soil_heat_flux=[*EBR_G, 5.0, 5.0, 5.0, np.nan],
    )
    assert comparison.n == 2
    assert comparison.n_offered == 6
    assert comparison.dropped_incomplete == 4
    assert comparison.n_missing_measured == 2
    assert comparison.n_missing_filled == 1
    assert comparison.n_missing_available_energy == 2
    # The two complete rows are the hand fixture, untouched by the four others.
    assert comparison.measured == pytest.approx(EBR_EXPECTED)
    assert comparison.available_energy == pytest.approx(220.0)


def test_an_empty_comparison_has_no_denominator() -> None:
    comparison = compare_energy_balance(
        measured_sensible_heat=[np.nan],
        measured_latent_heat=[1.0],
        filled_sensible_heat=[1.0],
        filled_latent_heat=[1.0],
        net_radiation=[1.0],
        soil_heat_flux=[0.0],
    )
    assert comparison.is_empty
    assert comparison.available_energy is None
    assert comparison.n_missing_measured == 1


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"n": 3, "n_offered": 2}, "cannot increase"),
        ({"n": -1, "n_offered": 2}, "negative"),
        ({"n": 1, "n_offered": 3, "n_missing_filled": 1}, "unexplained"),
        ({"n": 1, "n_offered": 2, "n_missing_measured": 2}, "more rows"),
    ],
)
def test_the_comparison_rejects_impossible_row_accounting(fields, message) -> None:
    values = {"measured": None, "filled": None, "difference": None, **fields}
    with pytest.raises(MetricError, match=message):
        EnergyBalanceComparison(**values)


def test_compare_energy_balance_accepts_series_on_one_index() -> None:
    index = pd.date_range("2020-06-01", periods=2, freq="30min")
    comparison = compare_energy_balance(
        measured_sensible_heat=pd.Series(EBR_H, index=index),
        measured_latent_heat=pd.Series(EBR_LE, index=index),
        filled_sensible_heat=pd.Series(EBR_H, index=index),
        filled_latent_heat=pd.Series(EBR_LE, index=index),
        net_radiation=pd.Series(EBR_NETRAD, index=index),
        soil_heat_flux=pd.Series(EBR_G, index=index),
    )
    assert comparison.measured == pytest.approx(EBR_EXPECTED)


def test_the_energy_balance_comparison_checks_alignment_across_all_six_series() -> None:
    index = pd.date_range("2020-06-01", periods=2, freq="30min")
    with pytest.raises(MetricError, match="indexed differently"):
        compare_energy_balance(
            measured_sensible_heat=pd.Series(EBR_H, index=index),
            measured_latent_heat=pd.Series(EBR_LE, index=index),
            filled_sensible_heat=pd.Series(EBR_H, index=index),
            filled_latent_heat=pd.Series(EBR_LE, index=index),
            net_radiation=pd.Series(EBR_NETRAD, index=index + pd.Timedelta("1h")),
            soil_heat_flux=pd.Series(EBR_G, index=index),
        )


# ---------------------------------------------------------------------------
# The configuration carries the A12 choice into the run manifest
# ---------------------------------------------------------------------------


def test_the_validation_config_defaults_to_the_bias_sensitive_r2() -> None:
    config = ValidationConfig()
    assert config.r2 is R2Definition.RESIDUAL
    assert config.r2.is_bias_sensitive
    assert config.to_dict()["r2_definition"] == "residual"


def test_the_r2_definition_is_configurable_and_recorded() -> None:
    config = ValidationConfig(r2_definition="squared_correlation")
    assert config.r2 is R2Definition.SQUARED_CORRELATION
    assert not config.r2.is_bias_sensitive
    assert config.to_dict()["r2_definition"] == "squared_correlation"


def test_an_unknown_configured_r2_definition_is_rejected_at_construction() -> None:
    with pytest.raises(ValueError, match="r2_definition"):
        ValidationConfig(r2_definition="adjusted")


def test_a_scored_subset_round_trips_through_the_configured_definition(
    measured: pd.Series, predicted: pd.Series
) -> None:
    """How the validation layer will pass the choice through (Step 13)."""
    config = ValidationConfig(r2_definition="squared_correlation")
    scored = core_metrics(measured, predicted, definition=config.r2)
    assert scored.r2 == pytest.approx(HAND["r2_squared_correlation"])


def test_the_records_survive_a_pickle_round_trip(measured: pd.Series, predicted: pd.Series) -> None:
    """FrozenRecord revalidates on unpickle, so a corrupted payload fails loudly."""
    import pickle

    scored = core_metrics(measured, predicted)
    assert pickle.loads(pickle.dumps(scored)) == scored
    comparison = EnergyBalanceComparison(n=2, n_offered=2, measured=0.5, filled=0.6, difference=0.1)
    assert pickle.loads(pickle.dumps(comparison)) == comparison
