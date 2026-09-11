"""Receptive-limiter feature tests. Covers acceptance tests 1-6 (radiation
categories, both hemispheres' seasons, elapsed hours, daily statistics,
deterministic feature order).

Expected values are hand-calculated on small fixtures rather than compared
against the implementation's own output, so a change in binning, quantile
convention or column order fails here instead of being ratified.

The full leakage tests 11-14 live in ``tests/test_leakage.py``, which drives the
Step 6 validation workflow; what is pinned here is the transformer-level contract
they rest on: masked-out target values cannot influence
:func:`daily_flux_statistics`, whichever thinly-observed-day strategy is in use.

See ``docs/method_spec.md`` for the contract and
``docs/supplement_benchmarks.md`` for benchmark provenance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rfrgapfill.config import FeatureConfig, RFRConfig
from rfrgapfill.features import (
    DAILY_STATISTIC_SUFFIXES,
    RADIATION_CATEGORY,
    SEASON,
    TIME_DISTANCE_HOURS,
    FeatureError,
    RadiationClass,
    Season,
    build_feature_matrix,
    daily_flux_statistics,
    daily_statistic_names,
    describe_features,
    feature_names,
    radiation_tag,
    receptive_limiter_features,
    season_tag,
    time_distance_hours,
)
from rfrgapfill.schema import ColumnMap, ConfigError, Hemisphere
from rfrgapfill.time import TimestampError

HALF_HOURLY = "30min"


def half_hourly(periods: int = 48, start: str = "2020-06-01") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=periods, freq=HALF_HOURLY)


def rfr3_config(**changes: object) -> RFRConfig:
    """An RFR3 configuration wired to short, obviously non-FLUXNET column names."""
    base = RFRConfig(
        mode="RFR3",
        hemisphere="north",
        column_map=ColumnMap({"shortwave": "SW", "vpd": "VPD", "air_temperature": "TA"}),
    )
    return base.replace(**changes) if changes else base


def driver_frame(index: pd.DatetimeIndex, **columns: object) -> pd.DataFrame:
    """A frame carrying the RFR3 drivers under the names of :func:`rfr3_config`."""
    data: dict[str, object] = {
        "SW": np.linspace(0.0, 400.0, len(index)),
        "VPD": 5.0,
        "TA": 12.0,
        "LE": np.arange(len(index), dtype=float),
    }
    data.update(columns)
    return pd.DataFrame(data, index=index)


# ---------------------------------------------------------------------------
# 1. Radiation category (acceptance test 1)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (9.9, "weak"),  # below the lower threshold
        (10.0, "medium"),  # exactly on it: A2 puts the boundary in medium
        (50.0, "medium"),  # between
        (100.0, "medium"),  # exactly on the upper threshold
        (100.1, "strong"),  # above
    ],
)
def test_radiation_boundaries_follow_the_documented_convention(value: float, expected: str) -> None:
    assert radiation_tag([value]).iloc[0] == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (9.9, "weak"),
        (10.0, "weak"),
        (50.0, "medium"),
        (100.0, "strong"),
        (100.1, "strong"),
    ],
)
def test_medium_exclusive_convention_moves_both_boundaries_outward(
    value: float, expected: str
) -> None:
    # The alternative convention exists so legacy behaviour can be reproduced;
    # it must differ from the default at exactly 10 and 100, and nowhere else.
    tagged = radiation_tag([value], convention="medium_exclusive")
    assert tagged.iloc[0] == expected


def test_radiation_bins_are_exhaustive_over_the_reals() -> None:
    values = [-1000.0, -1e-9, 0.0, 9.999999, 10.000001, 99.9, 1e6, np.inf, -np.inf]
    tagged = radiation_tag(values)
    assert tagged.notna().all(), "every finite or infinite value must land in a bin"


def test_missing_radiation_yields_a_missing_category_not_a_default_class() -> None:
    tagged = radiation_tag([np.nan, 5.0, np.nan])
    assert pd.isna(tagged.iloc[0]) and pd.isna(tagged.iloc[2])
    assert tagged.iloc[1] == RadiationClass.WEAK.value


def test_radiation_categories_are_ordered_weak_medium_strong() -> None:
    tagged = radiation_tag([5.0, 50.0, 500.0])
    assert tagged.cat.ordered
    assert list(tagged.cat.categories) == ["weak", "medium", "strong"]
    assert tagged.cat.codes.tolist() == [0, 1, 2]
    assert tagged.sort_values().tolist() == ["weak", "medium", "strong"]


def test_radiation_thresholds_are_configurable() -> None:
    config = FeatureConfig(radiation_thresholds=(20.0, 200.0))
    tagged = radiation_tag([15.0, 150.0, 250.0], config=config)
    assert tagged.tolist() == ["weak", "medium", "strong"]


def test_radiation_preserves_its_input_index_and_name() -> None:
    index = half_hourly(4)
    tagged = radiation_tag(pd.Series([0.0, 50.0, 500.0, np.nan], index=index))
    assert tagged.index.equals(index)
    assert tagged.name == RADIATION_CATEGORY


def test_radiation_rejects_two_sources_of_the_same_setting() -> None:
    with pytest.raises(ConfigError, match="not both"):
        radiation_tag([1.0], thresholds=(1.0, 2.0), config=FeatureConfig())


def test_radiation_rejects_non_numeric_input() -> None:
    with pytest.raises(FeatureError, match="numeric"):
        radiation_tag(["bright", "dark"])


# ---------------------------------------------------------------------------
# 2-3. Seasons in both hemispheres (acceptance tests 2 and 3)
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
def test_northern_seasons_for_every_month(month: int) -> None:
    stamp = pd.DatetimeIndex([f"2021-{month:02d}-15 12:00"])
    assert season_tag(stamp, hemisphere="north").iloc[0] == NORTHERN_SEASONS[month]


@pytest.mark.parametrize("month", range(1, 13))
def test_southern_seasons_for_every_month(month: int) -> None:
    stamp = pd.DatetimeIndex([f"2021-{month:02d}-15 12:00"])
    assert season_tag(stamp, hemisphere="south").iloc[0] == SOUTHERN_SEASONS[month]


@pytest.mark.parametrize("month", range(1, 13))
def test_hemispheres_are_exact_six_month_mirrors(month: int) -> None:
    # Not redundant with the two tables above: it pins the relationship between
    # them, so a typo that happened to be copied into both tables still fails.
    opposite = {"winter": "summer", "summer": "winter", "spring": "autumn", "autumn": "spring"}
    assert SOUTHERN_SEASONS[month] == opposite[NORTHERN_SEASONS[month]]


def test_season_covers_month_boundaries_not_just_mid_month() -> None:
    edges = pd.DatetimeIndex(
        ["2021-02-28 23:30", "2021-03-01 00:00", "2021-11-30 23:30", "2021-12-01 00:00"]
    )
    assert season_tag(edges, hemisphere="north").tolist() == [
        "winter",
        "spring",
        "autumn",
        "winter",
    ]


@pytest.mark.parametrize(
    ("latitude", "expected"),
    [(51.5, Hemisphere.NORTH), (0.0, Hemisphere.NORTH), (-0.1, Hemisphere.SOUTH)],
)
def test_latitude_infers_the_hemisphere_by_the_documented_rule(
    latitude: float, expected: Hemisphere
) -> None:
    # A9: latitude >= 0 -> north is a declared tie-break, so 0.0 must be north.
    january = pd.DatetimeIndex(["2021-01-15"])
    reference = season_tag(january, hemisphere=expected).iloc[0]
    assert season_tag(january, latitude=latitude).iloc[0] == reference


def test_season_requires_a_hemisphere() -> None:
    with pytest.raises(ConfigError, match="hemisphere"):
        season_tag(half_hourly(2))


def test_season_rejects_hemisphere_and_latitude_together() -> None:
    with pytest.raises(ConfigError, match="not both"):
        season_tag(half_hourly(2), hemisphere="north", latitude=-20.0)


def test_season_categories_are_stable_regardless_of_which_months_appear() -> None:
    # The encoding must not depend on the data: a summer-only frame still has to
    # carry all four categories, or its codes would disagree with a fitted model.
    summer_only = pd.DatetimeIndex(["2021-07-01", "2021-07-02"])
    tagged = season_tag(summer_only, hemisphere="north")
    assert list(tagged.cat.categories) == [member.value for member in Season]
    assert tagged.name == SEASON


# ---------------------------------------------------------------------------
# 4. Elapsed hours (acceptance test 4)
# ---------------------------------------------------------------------------


def test_elapsed_hours_at_half_hourly_cadence() -> None:
    hours = time_distance_hours(half_hourly(5))
    assert hours.tolist() == [0.0, 0.5, 1.0, 1.5, 2.0]
    assert hours.name == TIME_DISTANCE_HOURS


def test_elapsed_hours_measure_time_not_row_position() -> None:
    # Two rows either side of a 7-day gap: row distance 1, elapsed distance 168 h.
    index = pd.DatetimeIndex(["2020-01-01 00:00", "2020-01-08 00:00"])
    assert time_distance_hours(index).tolist() == [0.0, 168.0]


def test_elapsed_hours_can_measure_from_an_explicit_series_origin() -> None:
    # A subset must carry the same values it has in the whole series.
    full = half_hourly(48)
    subset = full[10:20]
    from_full = time_distance_hours(full).loc[subset]
    from_subset = time_distance_hours(subset, origin=full.min())
    pd.testing.assert_series_equal(from_full, from_subset)
    assert from_subset.iloc[0] == 5.0


def test_elapsed_hours_accepts_a_frame_and_uses_its_index() -> None:
    frame = driver_frame(half_hourly(4))
    assert time_distance_hours(frame).tolist() == [0.0, 0.5, 1.0, 1.5]


def test_elapsed_hours_rejects_numeric_pseudo_timestamps() -> None:
    with pytest.raises(TimestampError):
        time_distance_hours([0, 1, 2, 3])


# ---------------------------------------------------------------------------
# 5. Daily target statistics (acceptance test 5)
# ---------------------------------------------------------------------------


def one_day(values: list[float], *, start: str = "2020-03-01") -> pd.Series:
    """A single calendar day of ``values``, evenly spaced, named ``LE``."""
    index = pd.date_range(start, periods=len(values), freq=f"{24 * 60 // len(values)}min")
    return pd.Series(values, index=index, name="LE")


def test_daily_statistics_match_hand_calculated_values() -> None:
    # 1..8: q1 = 2.75, median = 4.5, q3 = 6.25 by linear interpolation;
    # sample variance = 42/7 = 6, so std = sqrt(6) = 2.449489742783178.
    statistics = daily_flux_statistics(one_day([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]))
    first = statistics.iloc[0]
    assert first["LE_daily_q1"] == pytest.approx(2.75)
    assert first["LE_daily_q2"] == pytest.approx(4.5)
    assert first["LE_daily_q3"] == pytest.approx(6.25)
    assert first["LE_daily_std"] == pytest.approx(np.sqrt(6.0))


def test_daily_statistics_are_joined_back_to_every_row_of_their_day() -> None:
    statistics = daily_flux_statistics(one_day([1.0, 2.0, 3.0, 4.0]))
    assert len(statistics) == 4
    assert statistics.nunique().eq(1).all(), "all rows of a day share its statistics"


def test_daily_statistics_are_computed_per_calendar_day() -> None:
    index = pd.DatetimeIndex(
        ["2020-03-01 00:00", "2020-03-01 23:30", "2020-03-02 00:00", "2020-03-02 12:00"]
    )
    series = pd.Series([0.0, 10.0, 100.0, 300.0], index=index, name="LE")
    medians = daily_flux_statistics(series)["LE_daily_q2"]
    assert medians.tolist() == [5.0, 5.0, 200.0, 200.0]


def test_daily_std_uses_the_configured_degrees_of_freedom() -> None:
    # A11: the paper names the statistic, not the convention. Population std of
    # 1..8 is sqrt(42/8) = sqrt(5.25); the sample std is sqrt(6).
    values = one_day([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    sample = daily_flux_statistics(values)["LE_daily_std"].iloc[0]
    population = daily_flux_statistics(values, ddof=0)["LE_daily_std"].iloc[0]
    assert sample == pytest.approx(np.sqrt(6.0))
    assert population == pytest.approx(np.sqrt(5.25))


def test_missing_target_values_are_excluded_rather_than_treated_as_zero() -> None:
    with_gaps = one_day([2.0, np.nan, 4.0, np.nan])
    statistics = daily_flux_statistics(with_gaps)
    assert statistics["LE_daily_q2"].iloc[0] == pytest.approx(3.0)


def test_a_day_with_no_visible_observations_yields_missing_statistics() -> None:
    # A4: never imputed, never borrowed from a neighbouring day.
    empty = one_day([np.nan, np.nan, np.nan, np.nan])
    statistics = daily_flux_statistics(empty)
    assert statistics.isna().all().all()


def test_min_observations_suppresses_a_thinly_observed_day() -> None:
    sparse = one_day([5.0, np.nan, np.nan, np.nan])
    permissive = daily_flux_statistics(sparse, min_observations=1)
    strict = daily_flux_statistics(sparse, min_observations=2)
    assert permissive["LE_daily_q2"].iloc[0] == pytest.approx(5.0)
    assert strict.isna().all().all()


def test_a_single_observation_defines_the_quartiles_but_not_the_sample_std() -> None:
    single = one_day([5.0, np.nan, np.nan, np.nan])
    statistics = daily_flux_statistics(single).iloc[0]
    assert statistics["LE_daily_q1"] == statistics["LE_daily_q3"] == pytest.approx(5.0)
    assert pd.isna(statistics["LE_daily_std"]), "sample std of one value is undefined"


def test_daily_statistic_columns_are_named_after_the_target() -> None:
    assert daily_statistic_names("NEE") == (
        "NEE_daily_q1",
        "NEE_daily_q2",
        "NEE_daily_q3",
        "NEE_daily_std",
    )
    statistics = daily_flux_statistics(one_day([1.0, 2.0]), target_name="H")
    assert list(statistics.columns) == list(daily_statistic_names("H"))
    assert [name.removeprefix("H") for name in statistics.columns] == list(DAILY_STATISTIC_SUFFIXES)


def test_daily_statistics_need_a_target_name() -> None:
    unnamed = pd.Series([1.0, 2.0], index=half_hourly(2))
    with pytest.raises(ConfigError, match="target_name"):
        daily_flux_statistics(unnamed)


# -- strategies for a thinly observed day (A4) -----------------------------


def three_days(start: str = "2020-03-01") -> pd.Series:
    """Three six-hourly days whose middle day holds a single observation.

    Day 1 is 1, 2, 3, 4; day 2 is 10 and three missing values; day 3 is 5, 6, 7, 8.
    With ``min_observations=2`` the middle day is below the minimum, which is what
    the four strategies disagree about.
    """
    index = pd.date_range(start, periods=12, freq="6h")
    values = [1.0, 2.0, 3.0, 4.0, 10.0, np.nan, np.nan, np.nan, 5.0, 6.0, 7.0, 8.0]
    return pd.Series(values, index=index, name="LE")


def middle_day(statistics: pd.DataFrame) -> pd.Series:
    row: pd.Series = statistics.loc["2020-03-02"].iloc[0]
    return row


def test_missing_is_the_default_strategy() -> None:
    assert FeatureConfig().statistic_strategy.value == "missing"
    assert FeatureConfig().fallback_window is None


def test_missing_leaves_a_thinly_observed_day_without_statistics() -> None:
    statistics = daily_flux_statistics(three_days(), min_observations=2, strategy="missing")
    assert middle_day(statistics).isna().all()
    # The neighbouring days are unaffected: only the deficient day is suppressed.
    assert statistics.loc["2020-03-01", "LE_daily_q2"].iloc[0] == pytest.approx(2.5)


def test_within_day_available_uses_what_the_day_has_and_ignores_the_minimum() -> None:
    statistics = daily_flux_statistics(
        three_days(), min_observations=2, strategy="within_day_available"
    )
    row = middle_day(statistics)
    assert row["LE_daily_q1"] == row["LE_daily_q2"] == row["LE_daily_q3"] == pytest.approx(10.0)
    assert pd.isna(row["LE_daily_std"]), "the sample std of one value is still undefined"


def test_neighbor_day_fallback_borrows_the_nearest_qualifying_day() -> None:
    statistics = daily_flux_statistics(
        three_days(),
        min_observations=2,
        strategy="neighbor_day_fallback",
        fallback_window_days=1,
    )
    row = middle_day(statistics)
    # Both neighbours are one day away, so the tie resolves to the earlier one:
    # day 1 is 1, 2, 3, 4 with q1 1.75, median 2.5, q3 3.25 and sample std
    # sqrt(5/3) = 1.29099...
    assert row["LE_daily_q1"] == pytest.approx(1.75)
    assert row["LE_daily_q2"] == pytest.approx(2.5)
    assert row["LE_daily_q3"] == pytest.approx(3.25)
    assert row["LE_daily_std"] == pytest.approx(np.sqrt(5.0 / 3.0))


def test_neighbor_day_fallback_measures_distance_in_calendar_days() -> None:
    # The qualifying days sit a fortnight away in time but are adjacent rows in
    # the frame; a window of one day must not reach them.
    index = pd.DatetimeIndex(
        ["2020-03-01 00:00", "2020-03-01 06:00", "2020-03-15 00:00", "2020-03-29 00:00"]
    )
    series = pd.Series([1.0, 3.0, 50.0, 90.0], index=index, name="LE")
    statistics = daily_flux_statistics(
        series, min_observations=2, strategy="neighbor_day_fallback", fallback_window_days=1
    )
    assert statistics.loc["2020-03-15"].isna().all().all()
    assert statistics.loc["2020-03-01", "LE_daily_q2"].iloc[0] == pytest.approx(2.0)


def test_neighbor_day_fallback_gives_up_beyond_its_window() -> None:
    days = three_days()
    isolated = pd.Series([np.nan], index=pd.DatetimeIndex(["2020-03-20"]), name="LE")
    statistics = daily_flux_statistics(
        pd.concat([days, isolated]),
        min_observations=2,
        strategy="neighbor_day_fallback",
        fallback_window_days=1,
    )
    assert middle_day(statistics).notna().all(), "one day away is within reach"
    assert statistics.loc["2020-03-20"].isna().all().all(), "eighteen days away is not"


def test_rolling_available_pools_the_surrounding_visible_observations() -> None:
    statistics = daily_flux_statistics(
        three_days(),
        min_observations=2,
        strategy="rolling_available",
        fallback_window_days=1,
    )
    row = middle_day(statistics)
    # Pool = 1, 2, 3, 4, 10, 5, 6, 7, 8 -> sorted 1..8 plus 10, so with linear
    # interpolation over nine values q1 = 3, median = 5 and q3 = 7.
    assert row["LE_daily_q1"] == pytest.approx(3.0)
    assert row["LE_daily_q2"] == pytest.approx(5.0)
    assert row["LE_daily_q3"] == pytest.approx(7.0)
    assert row["LE_daily_std"] == pytest.approx(np.std([1.0, 2, 3, 4, 10, 5, 6, 7, 8], ddof=1))


def test_rolling_available_still_respects_the_minimum() -> None:
    index = pd.DatetimeIndex(["2020-03-01 00:00", "2020-03-10 00:00"])
    series = pd.Series([1.0, 2.0], index=index, name="LE")
    statistics = daily_flux_statistics(
        series, min_observations=2, strategy="rolling_available", fallback_window_days=2
    )
    assert statistics.isna().all().all(), "a pool of one is not a statistic"


def test_a_fallback_window_is_rejected_where_it_would_do_nothing() -> None:
    with pytest.raises(ConfigError, match="meaningful only"):
        daily_flux_statistics(three_days(), strategy="missing", fallback_window_days=3)


def test_a_configured_strategy_and_an_explicit_one_are_not_both_accepted() -> None:
    with pytest.raises(ConfigError, match="not both"):
        daily_flux_statistics(three_days(), config=FeatureConfig(), strategy="within_day_available")


def test_an_unknown_strategy_is_rejected_by_name() -> None:
    with pytest.raises(ConfigError, match="daily_statistic_strategy"):
        daily_flux_statistics(three_days(), strategy="borrow_from_next_year")


# -- the leakage contract the Step 6 validation rests on --------------------


def test_masked_target_values_do_not_reach_the_daily_statistics() -> None:
    values = one_day([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    visible = pd.Series(values.index < values.index[4], index=values.index)

    masked = daily_flux_statistics(values, available_mask=visible)
    # Only 1..4 are visible: q1 = 1.75, median = 2.5, q3 = 3.25.
    assert masked["LE_daily_q2"].iloc[0] == pytest.approx(2.5)

    # Corrupting the hidden half by an enormous amount must change nothing.
    corrupted = values.copy()
    corrupted.iloc[4:] = 1e9
    recomputed = daily_flux_statistics(corrupted, available_mask=visible)
    pd.testing.assert_frame_equal(masked, recomputed)


def test_masking_every_observation_of_a_day_leaves_it_missing() -> None:
    values = one_day([1.0, 2.0, 3.0, 4.0])
    hidden = pd.Series(False, index=values.index)
    assert daily_flux_statistics(values, available_mask=hidden).isna().all().all()


def test_an_available_mask_aligns_by_timestamp_not_by_position() -> None:
    values = one_day([1.0, 2.0, 3.0, 4.0])
    shuffled = pd.Series([False, True, True, False], index=values.index)[::-1]
    aligned = daily_flux_statistics(values, available_mask=shuffled)
    assert aligned["LE_daily_q2"].iloc[0] == pytest.approx(2.5)


def test_a_mask_that_does_not_cover_a_row_treats_it_as_unavailable() -> None:
    # Conservative direction for a leakage guard: unknown means not visible.
    values = one_day([1.0, 2.0, 3.0, 4.0])
    partial = pd.Series(True, index=values.index[:2])
    covered = daily_flux_statistics(values, available_mask=partial)
    assert covered["LE_daily_q2"].iloc[0] == pytest.approx(1.5)


def test_a_positional_mask_of_the_wrong_length_is_rejected() -> None:
    values = one_day([1.0, 2.0, 3.0, 4.0])
    with pytest.raises(FeatureError, match="shape"):
        daily_flux_statistics(values, available_mask=np.array([True, False]))


# ---------------------------------------------------------------------------
# 6. Feature matrix and deterministic order (acceptance test 6)
# ---------------------------------------------------------------------------

RFR3_FEATURES = (
    "shortwave",
    "vpd",
    "air_temperature",
    RADIATION_CATEGORY,
    TIME_DISTANCE_HOURS,
    SEASON,
    "LE_daily_q1",
    "LE_daily_q2",
    "LE_daily_q3",
    "LE_daily_std",
)


def test_feature_order_is_deterministic_and_predicted_without_data() -> None:
    config = rfr3_config()
    matrix = build_feature_matrix(driver_frame(half_hourly()), config=config, target="LE")
    assert tuple(matrix.columns) == RFR3_FEATURES
    assert feature_names(config, target="LE") == RFR3_FEATURES


def test_rfr10_adds_its_seven_drivers_ahead_of_the_limiter_features() -> None:
    config = RFRConfig(
        mode="RFR10",
        hemisphere="north",
        column_map=ColumnMap.fluxnet2015("RFR10"),
    )
    names = feature_names(config, target="NEE")
    assert names[:10] == config.drivers
    assert names[10:] == (
        RADIATION_CATEGORY,
        TIME_DISTANCE_HOURS,
        SEASON,
        "NEE_daily_q1",
        "NEE_daily_q2",
        "NEE_daily_q3",
        "NEE_daily_std",
    )


def test_orf_keeps_the_drivers_and_omits_every_limiter_feature() -> None:
    # method_spec.md 3.6: ORF is the same driver set, minus sections 3.1-3.4.
    orf = rfr3_config(features=FeatureConfig(use_receptive_limiter=False))
    matrix = build_feature_matrix(driver_frame(half_hourly()), config=orf)
    assert tuple(matrix.columns) == orf.drivers
    assert feature_names(orf) == orf.drivers


def test_drivers_are_renamed_to_canonical_names() -> None:
    # The matrix must not depend on the station's column names, only on the map.
    index = half_hourly(8)
    frame = driver_frame(index)
    renamed = frame.rename(columns={"SW": "Rg", "VPD": "vpd_hPa", "TA": "Tair"})
    config = rfr3_config(
        column_map=ColumnMap({"shortwave": "Rg", "vpd": "vpd_hPa", "air_temperature": "Tair"})
    )
    pd.testing.assert_frame_equal(
        build_feature_matrix(frame, config=rfr3_config(), target="LE"),
        build_feature_matrix(renamed, config=config, target="LE"),
    )


def test_categorical_features_are_encoded_as_documented_ordinal_codes() -> None:
    index = pd.DatetimeIndex(["2020-01-15 12:00", "2020-04-15 12:00", "2020-07-15 12:00"])
    frame = driver_frame(index, SW=[5.0, 50.0, 500.0])
    matrix = build_feature_matrix(frame, config=rfr3_config(), target="LE")
    assert matrix[RADIATION_CATEGORY].tolist() == [0.0, 1.0, 2.0]
    assert matrix[SEASON].tolist() == [0.0, 1.0, 2.0]  # winter, spring, summer


def test_unencoded_matrix_keeps_the_categories_readable() -> None:
    index = pd.DatetimeIndex(["2020-01-15 12:00", "2020-07-15 12:00"])
    frame = driver_frame(index, SW=[5.0, 500.0])
    matrix = build_feature_matrix(frame, config=rfr3_config(), target="LE", encode=False)
    assert matrix[RADIATION_CATEGORY].tolist() == ["weak", "strong"]
    assert matrix[SEASON].tolist() == ["winter", "summer"]


def test_missing_categories_encode_as_nan_not_as_a_class() -> None:
    index = half_hourly(2)
    frame = driver_frame(index, SW=[np.nan, 50.0])
    matrix = build_feature_matrix(frame, config=rfr3_config(), target="LE")
    assert pd.isna(matrix[RADIATION_CATEGORY].iloc[0])
    assert matrix[RADIATION_CATEGORY].iloc[1] == 1.0


def test_the_matrix_keeps_every_row_and_imputes_nothing() -> None:
    # Deciding which rows are usable is the model layer's job, and it reports it.
    index = half_hourly(48)
    frame = driver_frame(index)
    frame.loc[frame.index[3], "TA"] = np.nan
    matrix = build_feature_matrix(frame, config=rfr3_config(), target="LE")
    assert matrix.index.equals(index)
    assert pd.isna(matrix["air_temperature"].iloc[3])


def test_the_matrix_measures_elapsed_hours_from_the_requested_origin() -> None:
    full = half_hourly(48)
    frame = driver_frame(full)
    subset = frame.iloc[10:20]
    from_subset = build_feature_matrix(subset, config=rfr3_config(), target="LE", origin=full.min())
    assert from_subset[TIME_DISTANCE_HOURS].iloc[0] == 5.0
    standalone = build_feature_matrix(subset, config=rfr3_config(), target="LE")
    assert standalone[TIME_DISTANCE_HOURS].iloc[0] == 0.0


def test_the_matrix_passes_the_available_mask_through_to_the_daily_statistics() -> None:
    index = pd.date_range("2020-06-01", periods=8, freq="3h")
    frame = driver_frame(index, LE=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    visible = pd.Series(index < index[4], index=index)

    masked = build_feature_matrix(frame, config=rfr3_config(), target="LE", available_mask=visible)
    corrupted = frame.copy()
    corrupted.iloc[4:, corrupted.columns.get_loc("LE")] = 1e9
    recomputed = build_feature_matrix(
        corrupted, config=rfr3_config(), target="LE", available_mask=visible
    )
    pd.testing.assert_frame_equal(masked, recomputed)
    assert masked["LE_daily_q2"].iloc[0] == pytest.approx(2.5)


def test_the_matrix_is_reproducible() -> None:
    frame = driver_frame(half_hourly())
    pd.testing.assert_frame_equal(
        build_feature_matrix(frame, config=rfr3_config(), target="LE"),
        build_feature_matrix(frame, config=rfr3_config(), target="LE"),
    )


def test_a_target_is_required_while_the_receptive_limiter_is_on() -> None:
    with pytest.raises(ConfigError, match="target"):
        build_feature_matrix(driver_frame(half_hourly()), config=rfr3_config())


def test_an_incomplete_driver_map_is_rejected_before_any_data_is_touched() -> None:
    # RFRConfig validates its own map, so an RFR10 run cannot even be configured
    # with RFR3 columns; the matrix never sees a half-mapped configuration.
    with pytest.raises(ConfigError, match="net_radiation"):
        RFRConfig(mode="RFR10", hemisphere="north", column_map=ColumnMap.fluxnet2015("RFR3"))


def test_an_unmapped_driver_in_an_override_map_is_reported_by_name() -> None:
    partial = ColumnMap({"shortwave": "SW", "vpd": "VPD"})
    with pytest.raises(ConfigError, match="air_temperature"):
        build_feature_matrix(
            driver_frame(half_hourly()), config=rfr3_config(), target="LE", column_map=partial
        )


def test_a_mapped_column_missing_from_the_data_is_reported_by_name() -> None:
    frame = driver_frame(half_hourly()).drop(columns=["VPD"])
    with pytest.raises(FeatureError, match="VPD"):
        build_feature_matrix(frame, config=rfr3_config(), target="LE")


def test_a_missing_target_column_is_reported() -> None:
    frame = driver_frame(half_hourly()).drop(columns=["LE"])
    with pytest.raises(FeatureError, match="LE"):
        build_feature_matrix(frame, config=rfr3_config(), target="LE")


def test_orf_builds_without_a_hemisphere_or_a_target() -> None:
    # The season feature is what makes the hemisphere mandatory (enforced by
    # RFRConfig); ORF builds neither, so it must run without either input.
    orf = RFRConfig(
        mode="RFR3",
        column_map=ColumnMap({"shortwave": "SW", "vpd": "VPD", "air_temperature": "TA"}),
        features=FeatureConfig(use_receptive_limiter=False),
    )
    matrix = build_feature_matrix(driver_frame(half_hourly()), config=orf)
    assert tuple(matrix.columns) == orf.drivers


# ---------------------------------------------------------------------------
# Manifest description
# ---------------------------------------------------------------------------


def test_the_feature_description_records_every_convention_it_applied() -> None:
    described = describe_features(rfr3_config(), target="LE")
    assert described["feature_names"] == list(RFR3_FEATURES)
    assert described["radiation_thresholds"] == [10.0, 100.0]
    assert described["boundary_convention"] == "medium_inclusive"
    assert described["feature_mode"] == "paper_safe"
    assert described["hemisphere"] == "north"
    assert described["daily_std_ddof"] == 1
    assert described["radiation_category_codes"] == {"weak": 0, "medium": 1, "strong": 2}
    assert described["season_codes"] == {"winter": 0, "spring": 1, "summer": 2, "autumn": 3}


def test_the_orf_description_advertises_no_limiter_conventions() -> None:
    orf = rfr3_config(features=FeatureConfig(use_receptive_limiter=False))
    described = describe_features(orf)
    assert described["use_receptive_limiter"] is False
    assert described["feature_names"] == list(orf.drivers)
    assert "radiation_thresholds" not in described


# ---------------------------------------------------------------------------
# ORF benchmark: the receptive-limiter contribution (acceptance tests 7-9)
# ---------------------------------------------------------------------------


def test_the_limiter_contributes_exactly_the_features_orf_drops() -> None:
    # Acceptance tests 8 and 9 as one identity: the two arms' feature sets differ
    # by the limiter tuple and by nothing else.
    rfr = rfr3_config()
    orf = rfr.as_orf()

    assert receptive_limiter_features(rfr, target="LE") == (
        RADIATION_CATEGORY,
        TIME_DISTANCE_HOURS,
        SEASON,
        "LE_daily_q1",
        "LE_daily_q2",
        "LE_daily_q3",
        "LE_daily_std",
    )
    assert receptive_limiter_features(orf, target="LE") == ()
    assert feature_names(rfr, target="LE") == feature_names(orf) + receptive_limiter_features(
        rfr, target="LE"
    )


def test_both_arms_are_built_on_the_identical_driver_columns() -> None:
    # Acceptance test 7 at the feature layer: same drivers, same values, same
    # order; the ORF matrix is the RFR matrix with the limiter block removed.
    rfr = rfr3_config()
    orf = rfr.as_orf()
    frame = driver_frame(half_hourly())

    rfr_matrix = build_feature_matrix(frame, config=rfr, target="LE")
    orf_matrix = build_feature_matrix(frame, config=orf)

    assert tuple(orf_matrix.columns) == rfr.drivers
    pd.testing.assert_frame_equal(rfr_matrix[list(rfr.drivers)], orf_matrix)


def test_orf_needs_no_target_because_it_derives_nothing_from_one() -> None:
    orf = rfr3_config().as_orf()
    frame = driver_frame(half_hourly()).drop(columns=["LE"])
    matrix = build_feature_matrix(frame, config=orf)
    assert tuple(matrix.columns) == orf.drivers


def test_orf_features_cannot_depend_on_the_target_at_all() -> None:
    # The strongest statement of "omits the receptive-limiter-derived features":
    # corrupting the target leaves the ORF matrix bit-identical, so no ORF
    # feature can carry target information by any route.
    orf = rfr3_config().as_orf()
    frame = driver_frame(half_hourly())
    corrupted = frame.copy()
    corrupted["LE"] = 1e9

    pd.testing.assert_frame_equal(
        build_feature_matrix(frame, config=orf),
        build_feature_matrix(corrupted, config=orf),
    )
