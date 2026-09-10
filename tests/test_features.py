"""Receptive-limiter feature tests. Covers acceptance tests 1-6 and 11-14.

Every expected value here is worked out by hand in the test, and the radiation
and season tests walk the boundaries the paper's prose leaves implicit: 9.9, 10,
50, 100 and 100.1 W m-2, and all twelve months in both hemispheres.

The leakage tests at the bottom are the ones that matter most. They alter hidden
truth by an absurd amount and assert that nothing a model would see moves, which
is the property ``feature_mode="paper_safe"`` exists to guarantee
(``docs/method_spec.md`` 3.5, ambiguity A6).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from rfrgapfill.config import (
    BoundaryConvention,
    ColumnMap,
    DailyStatisticStrategy,
    FeatureConfig,
    RFRConfig,
)
from rfrgapfill.features import (
    RADIATION_CATEGORIES,
    SEASONS,
    TIME_DISTANCE_HOURS,
    FeatureError,
    build_feature_matrix,
    daily_flux_statistics,
    daily_statistic_names,
    feature_names,
    radiation_tag,
    season_names,
    season_tag,
    time_distance_hours,
)
from rfrgapfill.schema import RFR3_DRIVERS, RFR10_DRIVERS


def half_hourly(periods: int, start: str = "2020-06-01") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=periods, freq="30min", name="timestamp")


# ---------------------------------------------------------------------------
# Radiation category (acceptance test 1)
# ---------------------------------------------------------------------------

#: The values the implementation plan names, spanning both thresholds.
BOUNDARY_VALUES = [9.9, 10.0, 50.0, 100.0, 100.1]


def test_radiation_categories_at_and_around_the_thresholds() -> None:
    tag = radiation_tag(BOUNDARY_VALUES)
    assert list(tag) == ["weak", "medium", "medium", "medium", "strong"]


def test_medium_exclusive_moves_both_boundary_values_outwards() -> None:
    tag = radiation_tag(BOUNDARY_VALUES, convention=BoundaryConvention.MEDIUM_EXCLUSIVE)
    assert list(tag) == ["weak", "weak", "medium", "strong", "strong"]


def test_radiation_category_is_missing_rather_than_defaulted() -> None:
    tag = radiation_tag([np.nan, 50.0])
    assert tag.isna().tolist() == [True, False]


def test_radiation_categories_are_ordered_weak_to_strong() -> None:
    tag = radiation_tag([5.0, 50.0, 500.0])
    assert list(tag.cat.categories) == list(RADIATION_CATEGORIES)
    assert tag.cat.ordered
    assert tag.cat.codes.tolist() == [0, 1, 2]


def test_radiation_thresholds_are_configurable() -> None:
    tag = radiation_tag([4.0, 6.0, 250.0], thresholds=(5.0, 200.0))
    assert list(tag) == ["weak", "medium", "strong"]


def test_increasing_radiation_thresholds_are_required() -> None:
    with pytest.raises(FeatureError, match="increasing"):
        radiation_tag([1.0], thresholds=(100.0, 10.0))


# ---------------------------------------------------------------------------
# Seasons (acceptance tests 2-3)
# ---------------------------------------------------------------------------

NORTHERN_SEASONS = {
    1: "winter",
    2: "winter",
    3: "spring",
    4: "spring",
    5: "spring",
    6: "summer",
    7: "summer",
    8: "summer",
    9: "autumn",
    10: "autumn",
    11: "autumn",
    12: "winter",
}
SOUTHERN_SEASONS = {
    1: "summer",
    2: "summer",
    3: "autumn",
    4: "autumn",
    5: "autumn",
    6: "winter",
    7: "winter",
    8: "winter",
    9: "spring",
    10: "spring",
    11: "spring",
    12: "summer",
}


@pytest.mark.parametrize("month", range(1, 13))
def test_northern_hemisphere_season_of_every_month(month: int) -> None:
    stamp = pd.DatetimeIndex([f"2021-{month:02d}-15"])
    assert season_tag(stamp, "north").iloc[0] == NORTHERN_SEASONS[month]


@pytest.mark.parametrize("month", range(1, 13))
def test_southern_hemisphere_season_of_every_month(month: int) -> None:
    stamp = pd.DatetimeIndex([f"2021-{month:02d}-15"])
    assert season_tag(stamp, "south").iloc[0] == SOUTHERN_SEASONS[month]


def test_the_hemispheres_are_exact_opposites_all_year() -> None:
    months = pd.DatetimeIndex([f"2021-{month:02d}-15" for month in range(1, 13)])
    north = season_tag(months, "north")
    south = season_tag(months, "south")
    assert not (north.to_numpy() == south.to_numpy()).any()


def test_season_categories_are_the_documented_four_in_a_fixed_order() -> None:
    tag = season_tag(pd.DatetimeIndex(["2021-01-01"]), "north")
    assert list(tag.cat.categories) == list(SEASONS)


# ---------------------------------------------------------------------------
# Elapsed hours (acceptance test 4)
# ---------------------------------------------------------------------------


def test_time_distance_is_elapsed_hours_at_half_hourly_cadence() -> None:
    hours = time_distance_hours(half_hourly(5))
    assert hours.tolist() == [0.0, 0.5, 1.0, 1.5, 2.0]
    assert hours.name == TIME_DISTANCE_HOURS


def test_time_distance_follows_the_clock_across_a_missing_stretch() -> None:
    index = pd.DatetimeIndex(["2020-01-01 00:00", "2020-01-01 00:30", "2020-01-08 00:00"])
    # The third row is a week in, not the third row's worth of half-hours.
    assert time_distance_hours(index).tolist() == [0.0, 0.5, 168.0]


def test_an_explicit_origin_keeps_a_subset_on_the_full_series_clock() -> None:
    index = half_hourly(10)
    subset = index[6:]
    assert time_distance_hours(subset, origin=index[0]).iloc[0] == 3.0
    assert time_distance_hours(subset).iloc[0] == 0.0


# ---------------------------------------------------------------------------
# Daily target statistics (acceptance test 5)
# ---------------------------------------------------------------------------


def test_daily_statistics_match_hand_calculated_values() -> None:
    # Four values on one day: 1, 2, 3, 4. Q1 = 1.75, Q2 = 2.5, Q3 = 3.25 under
    # pandas' linear interpolation; the sample standard deviation is
    # sqrt(5/3) = 1.29099...
    index = pd.date_range("2020-03-01", periods=4, freq="6h")
    target = pd.Series([1.0, 2.0, 3.0, 4.0], index=index, name="NEE")

    frame = daily_flux_statistics(target, name="NEE").frame
    assert frame.columns.tolist() == list(daily_statistic_names("NEE"))
    assert frame["NEE_daily_q1"].tolist() == [1.75] * 4
    assert frame["NEE_daily_q2"].tolist() == [2.5] * 4
    assert frame["NEE_daily_q3"].tolist() == [3.25] * 4
    assert frame["NEE_daily_std"].iloc[0] == pytest.approx(math.sqrt(5 / 3))


def test_each_day_gets_its_own_statistics_joined_to_all_of_its_rows() -> None:
    index = pd.DatetimeIndex(
        ["2020-03-01 00:00", "2020-03-01 12:00", "2020-03-02 00:00", "2020-03-02 12:00"]
    )
    target = pd.Series([0.0, 10.0, 100.0, 110.0], index=index, name="LE")

    frame = daily_flux_statistics(target, name="LE").frame
    assert frame["LE_daily_q2"].tolist() == [5.0, 5.0, 105.0, 105.0]


def test_a_day_below_the_minimum_has_no_statistics_of_its_own() -> None:
    index = pd.DatetimeIndex(["2020-03-01 00:00", "2020-03-02 00:00", "2020-03-02 12:00"])
    target = pd.Series([7.0, 1.0, 3.0], index=index, name="H")

    result = daily_flux_statistics(target, name="H", strategy=DailyStatisticStrategy.WITHIN_DAY)
    assert result.n_days_observed == 1
    assert result.n_days_missing == 1
    assert result.frame["H_daily_q2"].tolist()[1:] == [2.0, 2.0]
    assert math.isnan(result.frame["H_daily_q2"].iloc[0])


def test_the_nearest_visible_day_fills_a_day_that_has_nothing() -> None:
    # 1 March has one value (too few for a standard deviation); 2 March has two.
    index = pd.DatetimeIndex(["2020-03-01 00:00", "2020-03-02 00:00", "2020-03-02 12:00"])
    target = pd.Series([7.0, 1.0, 3.0], index=index, name="H")

    result = daily_flux_statistics(target, name="H")
    assert result.n_days_from_neighbour == 1
    assert result.n_days_missing == 0
    # 1 March borrows 2 March's median of 2.0 rather than reporting its own 7.0.
    assert result.frame["H_daily_q2"].tolist() == [2.0, 2.0, 2.0]


def test_the_nearest_visible_day_breaks_a_tie_towards_the_earlier_day() -> None:
    index = pd.DatetimeIndex(
        [
            "2020-03-01 00:00",
            "2020-03-01 12:00",
            "2020-03-02 00:00",
            "2020-03-03 00:00",
            "2020-03-03 12:00",
        ]
    )
    target = pd.Series([0.0, 2.0, np.nan, 100.0, 102.0], index=index, name="H")

    result = daily_flux_statistics(target, name="H")
    # 2 March is one day from both neighbours; the earlier one wins.
    assert result.frame["H_daily_q2"].iloc[2] == 1.0


def test_statistics_are_all_missing_when_nothing_at_all_is_visible() -> None:
    index = pd.date_range("2020-03-01", periods=4, freq="6h")
    target = pd.Series([1.0, 2.0, 3.0, 4.0], index=index, name="NEE")

    result = daily_flux_statistics(target, name="NEE", available=[False] * 4)
    assert result.n_days_missing == 1
    assert result.frame.isna().all().all()


# ---------------------------------------------------------------------------
# Feature matrix order (acceptance test 6)
# ---------------------------------------------------------------------------


def rfr_config(mode: str = "RFR10", **changes: object) -> RFRConfig:
    return RFRConfig(
        mode=mode,
        frequency="30min",
        hemisphere="north",
        column_map=ColumnMap.fluxnet2015(mode),
        **changes,  # type: ignore[arg-type]
    )


def small_frame(periods: int = 96) -> pd.DataFrame:
    index = half_hourly(periods, start="2020-06-01")
    rng = np.random.default_rng(0)
    columns = {
        "SW_IN_F": np.tile(np.concatenate([np.zeros(24), np.linspace(0, 600, 24)]), periods // 48),
        "VPD_F_MDS": rng.uniform(1, 20, periods),
        "TA_F_MDS": rng.uniform(5, 25, periods),
        "NETRAD": rng.uniform(-50, 500, periods),
        "WS": rng.uniform(0.5, 5, periods),
        "WD": rng.uniform(0, 360, periods),
        "G_F_MDS": rng.uniform(-20, 40, periods),
        "TS_F_MDS": rng.uniform(5, 20, periods),
        "RH": rng.uniform(30, 95, periods),
        "SWC_F_MDS": rng.uniform(10, 40, periods),
        "NEE": rng.normal(0, 3, periods),
    }
    return pd.DataFrame(columns, index=index)


def test_feature_names_are_deterministic_and_documented_in_order() -> None:
    names = feature_names("NEE", mode="RFR10")
    assert names[:10] == RFR10_DRIVERS
    assert names[10:12] == ("radiation_class", TIME_DISTANCE_HOURS)
    assert names[12:16] == season_names()
    assert names[16:] == daily_statistic_names("NEE")


def test_rfr3_requires_exactly_the_three_canonical_drivers() -> None:
    matrix = build_feature_matrix(small_frame(), target="NEE", config=rfr_config("RFR3"))
    assert matrix.names[:3] == RFR3_DRIVERS
    assert len(matrix.names) == 3 + 2 + 4 + 4


def test_rfr10_requires_the_ten_canonical_drivers() -> None:
    matrix = build_feature_matrix(small_frame(), target="NEE", config=rfr_config("RFR10"))
    assert matrix.names[:10] == RFR10_DRIVERS


def test_the_matrix_column_order_matches_feature_names() -> None:
    config = rfr_config("RFR10")
    matrix = build_feature_matrix(small_frame(), target="NEE", config=config)
    assert matrix.names == feature_names("NEE", mode="RFR10", features=config.features)


def test_drivers_are_renamed_to_canonical_names_not_station_names() -> None:
    matrix = build_feature_matrix(small_frame(), target="NEE", config=rfr_config("RFR3"))
    assert "SW_IN_F" not in matrix.names
    assert "shortwave" in matrix.names


def test_a_mistyped_driver_column_fails_before_fitting() -> None:
    frame = small_frame().drop(columns=["VPD_F_MDS"])
    with pytest.raises(FeatureError, match="VPD_F_MDS"):
        build_feature_matrix(frame, target="NEE", config=rfr_config("RFR3"))


def test_a_missing_target_column_is_reported_by_name() -> None:
    with pytest.raises(FeatureError, match="'LE'"):
        build_feature_matrix(small_frame(), target="LE", config=rfr_config("RFR3"))


# ---------------------------------------------------------------------------
# The ORF benchmark (acceptance tests 7-9)
# ---------------------------------------------------------------------------


def test_orf_omits_the_receptive_limiter_but_keeps_the_drivers() -> None:
    orf = rfr_config("RFR10", features=FeatureConfig(use_receptive_limiter=False))
    matrix = build_feature_matrix(small_frame(), target="NEE", config=orf)

    assert matrix.names == RFR10_DRIVERS
    assert matrix.daily_statistics is None
    assert not matrix.use_receptive_limiter


def test_rfr_and_orf_share_a_driver_set_and_differ_only_by_engineered_features() -> None:
    rfr = build_feature_matrix(small_frame(), target="NEE", config=rfr_config("RFR3"))
    orf = build_feature_matrix(
        small_frame(),
        target="NEE",
        config=rfr_config("RFR3", features=FeatureConfig(use_receptive_limiter=False)),
    )
    assert set(orf.names) < set(rfr.names)
    assert orf.names == rfr.names[: len(orf.names)]
    pd.testing.assert_frame_equal(orf.frame, rfr.frame.loc[:, list(orf.names)])


# ---------------------------------------------------------------------------
# Leakage (acceptance tests 11-14)
# ---------------------------------------------------------------------------


def hidden_interval(index: pd.DatetimeIndex) -> pd.Series:
    """Mark one contiguous day as held out."""
    mask = (index >= "2020-06-02 00:00") & (index < "2020-06-03 00:00")
    return pd.Series(mask, index=index)


def test_hidden_truth_cannot_reach_the_features_built_to_predict_it() -> None:
    frame = small_frame(periods=96 * 3)
    hidden = hidden_interval(frame.index)
    available = frame["NEE"].notna() & ~hidden
    config = rfr_config("RFR10")

    before = build_feature_matrix(frame, target="NEE", config=config, available=available).frame

    # Alter the hidden truth by an absurd amount and rebuild from scratch.
    altered = frame.copy()
    altered.loc[hidden, "NEE"] = 1e6

    after = build_feature_matrix(altered, target="NEE", config=config, available=available).frame

    pd.testing.assert_frame_equal(before, after)


def test_the_leakage_guarantee_covers_the_held_out_rows_specifically() -> None:
    frame = small_frame(periods=96 * 3)
    hidden = hidden_interval(frame.index)
    available = frame["NEE"].notna() & ~hidden
    config = rfr_config("RFR10")

    before = build_feature_matrix(frame, target="NEE", config=config, available=available)
    altered = frame.copy()
    altered.loc[hidden, "NEE"] *= -1000.0
    after = build_feature_matrix(altered, target="NEE", config=config, available=available)

    pd.testing.assert_frame_equal(before.frame.loc[hidden], after.frame.loc[hidden])


def test_without_an_available_mask_the_hidden_values_do_move_the_features() -> None:
    # The complement of the guarantee: omitting the mask is exactly the leak the
    # validation workflow must not have, and this proves the mask is what stops it.
    frame = small_frame(periods=96 * 3)
    hidden = hidden_interval(frame.index)
    config = rfr_config("RFR10")

    before = build_feature_matrix(frame, target="NEE", config=config).frame
    altered = frame.copy()
    altered.loc[hidden, "NEE"] = 1e6
    after = build_feature_matrix(altered, target="NEE", config=config).frame

    assert not before.loc[hidden, "NEE_daily_q2"].equals(after.loc[hidden, "NEE_daily_q2"])


def test_daily_statistics_read_only_the_available_observations() -> None:
    index = pd.date_range("2020-03-01", periods=4, freq="6h")
    target = pd.Series([1.0, 2.0, 3.0, 400.0], index=index, name="NEE")

    visible = daily_flux_statistics(target, name="NEE", available=[True, True, True, False]).frame
    assert visible["NEE_daily_q2"].iloc[0] == 2.0  # median of 1, 2, 3 - not of 400
