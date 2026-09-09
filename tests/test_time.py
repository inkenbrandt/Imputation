"""Timestamp, cadence and elapsed-time tests. Covers acceptance test 4.

Exercises 30-minute and hourly cadences, series with missing rows, and series
with duplicate timestamps, and pins the rule the whole package depends on:
temporal quantities come from elapsed time, never from row position (one day is
not always 48 rows).

See ``docs/method_spec.md`` for the contract and
``docs/supplement_benchmarks.md`` for benchmark provenance.
"""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from rfrgapfill.schema import ConfigError
from rfrgapfill.time import (
    TIME_DISTANCE_HOURS,
    DuplicatePolicy,
    TimeAxis,
    TimestampError,
    as_datetime_index,
    duplicate_timestamps,
    duration_to_periods,
    elapsed_hours,
    infer_time_step,
    interval_bounds,
    interval_mask,
    iso_duration,
    periods_to_duration,
    prepare_time_index,
    resolve_duplicates,
    resolve_time_step,
    set_time_index,
    sort_by_time,
    to_timedelta,
)

HALF_HOURLY = "30min"
HOURLY = "60min"


def frame(index: pd.DatetimeIndex, **columns: object) -> pd.DataFrame:
    """Return a small frame carrying ``index`` and a recognisable value column."""
    data: dict[str, object] = {"LE": np.arange(len(index), dtype=float)}
    data.update(columns)
    return pd.DataFrame(data, index=index)


def half_hourly(periods: int = 48, start: str = "2020-01-01") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=periods, freq=HALF_HOURLY)


def hourly(periods: int = 24, start: str = "2020-01-01") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=periods, freq=HOURLY)


# ---------------------------------------------------------------------------
# Duration parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("30min", timedelta(minutes=30)),
        ("24h", timedelta(hours=24)),
        ("7d", timedelta(days=7)),
        ("30d", timedelta(days=30)),
        ("1D", timedelta(days=1)),
        (timedelta(hours=1), timedelta(hours=1)),
    ],
)
def test_paper_durations_parse_to_timedeltas(value: object, expected: timedelta) -> None:
    assert to_timedelta(value) == expected


def test_bare_numbers_are_rejected_as_durations() -> None:
    # A unitless 48 is exactly the row-count-as-duration confusion to prevent.
    with pytest.raises(ConfigError, match="not a bare number"):
        to_timedelta(48, field_name="duration")


@pytest.mark.parametrize("value", ["-1d", "0h", "sometimes", None])
def test_invalid_durations_are_rejected(value: object) -> None:
    with pytest.raises(ConfigError):
        to_timedelta(value)


def test_iso_duration_is_manifest_ready() -> None:
    assert iso_duration(timedelta(days=30)) == "P30DT0H0M0S"


# ---------------------------------------------------------------------------
# Timestamp validation
# ---------------------------------------------------------------------------


def test_datetime_index_passes_through_unchanged() -> None:
    index = half_hourly(4)
    assert as_datetime_index(index) is index


def test_date_strings_are_parsed() -> None:
    parsed = as_datetime_index(["2020-01-01 00:00", "2020-01-01 00:30"])
    assert isinstance(parsed, pd.DatetimeIndex)
    assert parsed[1] == pd.Timestamp("2020-01-01 00:30")


def test_numeric_values_are_refused_rather_than_read_as_epoch_nanoseconds() -> None:
    with pytest.raises(TimestampError, match="numeric dtype"):
        as_datetime_index(pd.Index([0, 1, 2]))


def test_missing_timestamps_are_rejected() -> None:
    with pytest.raises(TimestampError, match="missing timestamp"):
        as_datetime_index(pd.DatetimeIndex(["2020-01-01", None]))


def test_empty_input_is_rejected() -> None:
    with pytest.raises(TimestampError, match="empty"):
        as_datetime_index(pd.DatetimeIndex([]))


def test_timestamp_column_becomes_the_index_and_leaves_no_duplicate_column() -> None:
    index = half_hourly(6)
    data = pd.DataFrame({"TIMESTAMP": index.astype(str), "LE": np.arange(6.0)})
    prepared = set_time_index(data, timestamp="TIMESTAMP")
    assert isinstance(prepared.index, pd.DatetimeIndex)
    assert prepared.index.equals(index)
    assert list(prepared.columns) == ["LE"]
    assert list(data.columns) == ["TIMESTAMP", "LE"]  # input untouched


def test_a_mistyped_timestamp_column_names_the_columns_that_exist() -> None:
    data = frame(half_hourly(4))
    with pytest.raises(TimestampError, match="TIMESTAMP_END"):
        set_time_index(data, timestamp="TIMESTAMP_END")


def test_a_non_datetime_index_without_a_timestamp_column_is_rejected() -> None:
    data = pd.DataFrame({"LE": [1.0, 2.0, 3.0]})
    with pytest.raises(TimestampError, match="numeric dtype"):
        set_time_index(data)


def test_sorting_is_stable_and_leaves_sorted_frames_alone() -> None:
    index = half_hourly(4)
    shuffled = frame(index).iloc[[2, 0, 3, 1]]
    ordered = sort_by_time(shuffled)
    assert ordered.index.equals(index)
    assert list(ordered["LE"]) == [0.0, 1.0, 2.0, 3.0]

    already = frame(index)
    assert sort_by_time(already).equals(already)


# ---------------------------------------------------------------------------
# Duplicate timestamps
# ---------------------------------------------------------------------------


def duplicated_frame() -> pd.DataFrame:
    stamps = pd.DatetimeIndex(
        [
            "2020-01-01 00:00",
            "2020-01-01 00:30",
            "2020-01-01 00:30",
            "2020-01-01 01:00",
        ]
    )
    return frame(stamps)


def test_duplicate_timestamps_are_reported_once_each_in_time_order() -> None:
    repeated = duplicate_timestamps(duplicated_frame().index)
    assert list(repeated) == [pd.Timestamp("2020-01-01 00:30")]


def test_duplicates_are_an_error_by_default() -> None:
    with pytest.raises(TimestampError, match="occur more than once"):
        resolve_duplicates(duplicated_frame())


@pytest.mark.parametrize(
    ("policy", "kept"),
    [("keep_first", 1.0), ("keep_last", 2.0), (DuplicatePolicy.KEEP_FIRST, 1.0)],
)
def test_documented_duplicate_policies_keep_the_named_row(policy: object, kept: float) -> None:
    resolved, removed = resolve_duplicates(duplicated_frame(), policy=policy)
    assert len(resolved) == 3
    assert resolved.index.is_unique
    assert resolved.loc[pd.Timestamp("2020-01-01 00:30"), "LE"] == kept
    assert list(removed) == [pd.Timestamp("2020-01-01 00:30")]


def test_an_unknown_duplicate_policy_is_rejected() -> None:
    with pytest.raises(ConfigError, match="on_duplicates"):
        resolve_duplicates(duplicated_frame(), policy="average")


def test_prepare_records_which_duplicates_were_resolved_away() -> None:
    prepared, axis = prepare_time_index(duplicated_frame(), on_duplicates="keep_first")
    assert len(prepared) == 3
    assert list(axis.duplicates_removed) == [pd.Timestamp("2020-01-01 00:30")]
    assert axis.to_dict()["n_duplicates_removed"] == 1


# ---------------------------------------------------------------------------
# Cadence inference
# ---------------------------------------------------------------------------


def test_half_hourly_and_hourly_cadences_are_inferred() -> None:
    assert infer_time_step(half_hourly(48)) == timedelta(minutes=30)
    assert infer_time_step(hourly(24)) == timedelta(hours=1)


def test_cadence_is_inferred_through_missing_rows() -> None:
    index = half_hourly(48).delete([10, 11, 12, 30])
    assert infer_time_step(index) == timedelta(minutes=30)


def test_a_series_with_no_single_cadence_refuses_to_guess() -> None:
    index = pd.DatetimeIndex(
        ["2020-01-01 00:00", "2020-01-01 00:30", "2020-01-01 01:00", "2020-01-01 01:15"]
    )
    with pytest.raises(TimestampError, match="no single cadence"):
        infer_time_step(index)


def test_unsorted_timestamps_do_not_yield_a_cadence() -> None:
    index = half_hourly(6)[[0, 2, 1, 3, 4, 5]]
    with pytest.raises(TimestampError, match="sorted"):
        infer_time_step(index)


def test_a_single_timestamp_cannot_imply_a_cadence() -> None:
    with pytest.raises(TimestampError, match="cannot be inferred"):
        infer_time_step(half_hourly(1))


def test_a_declared_frequency_overrides_inference_and_is_recorded() -> None:
    index = pd.DatetimeIndex(
        ["2020-01-01 00:00", "2020-01-01 00:30", "2020-01-01 01:00", "2020-01-01 01:15"]
    )
    step, source = resolve_time_step(index, frequency="15min")
    assert step == timedelta(minutes=15)
    assert source == "declared"

    step, source = resolve_time_step(half_hourly(10))
    assert step == timedelta(minutes=30)
    assert source == "inferred"


# ---------------------------------------------------------------------------
# Elapsed hours (method_spec.md 3.2)
# ---------------------------------------------------------------------------


def test_elapsed_hours_at_half_hourly_cadence_counts_real_time() -> None:
    hours = elapsed_hours(half_hourly(5))
    assert hours.name == TIME_DISTANCE_HOURS
    assert list(hours) == [0.0, 0.5, 1.0, 1.5, 2.0]


def test_elapsed_hours_at_hourly_cadence_counts_real_time() -> None:
    assert list(elapsed_hours(hourly(4))) == [0.0, 1.0, 2.0, 3.0]


def test_elapsed_hours_skip_missing_rows_rather_than_counting_positions() -> None:
    # Rows 2 and 3 (01:00, 01:30) are absent; the row after the hole is still
    # 2 hours into the series, not 1 hour as a row count would say.
    index = half_hourly(6).delete([2, 3])
    hours = elapsed_hours(index)
    assert list(hours) == [0.0, 0.5, 2.0, 2.5]


def test_a_day_is_not_always_forty_eight_rows() -> None:
    index = half_hourly(48).delete([5, 6, 7])
    hours = elapsed_hours(index)
    assert len(index) == 45
    assert hours.iloc[-1] == 23.5  # last stamp of the day, whatever the row count


def test_an_explicit_origin_keeps_a_subset_on_the_full_series_axis() -> None:
    index = half_hourly(10)
    whole = elapsed_hours(index)
    subset = elapsed_hours(index[4:], origin=index[0])
    pd.testing.assert_series_equal(subset, whole.iloc[4:])
    # Without the origin the subset would restart at zero.
    assert elapsed_hours(index[4:]).iloc[0] == 0.0


def test_elapsed_hours_reject_an_origin_of_the_wrong_timezone_awareness() -> None:
    index = pd.date_range("2020-01-01", periods=3, freq=HALF_HOURLY, tz="UTC")
    with pytest.raises(TimestampError, match="timezone-aware"):
        elapsed_hours(index, origin=pd.Timestamp("2020-01-01"))


def test_elapsed_hours_are_real_elapsed_hours_across_a_daylight_saving_change() -> None:
    # 2020-03-29 01:00 UTC is the spring-forward instant in Europe/London: the
    # wall clock jumps from 01:00 to 02:00, elapsed time does not.
    index = pd.date_range("2020-03-29 00:00", periods=4, freq=HOURLY, tz="UTC").tz_convert(
        "Europe/London"
    )
    assert list(elapsed_hours(index)) == [0.0, 1.0, 2.0, 3.0]
    assert index[1].hour - index[0].hour == 2  # wall clock skipped an hour


# ---------------------------------------------------------------------------
# Duration <-> row conversion
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("duration", "step", "rows"),
    [
        ("24h", "30min", 48),
        ("24h", "60min", 24),
        ("7d", "30min", 336),
        ("30d", "30min", 1440),
        ("30d", "60min", 720),
    ],
)
def test_paper_gap_durations_convert_to_rows_at_the_given_cadence(
    duration: str, step: str, rows: int
) -> None:
    assert duration_to_periods(duration, step) == rows


def test_a_duration_that_does_not_fit_the_cadence_is_refused() -> None:
    with pytest.raises(TimestampError, match="whole number"):
        duration_to_periods("24h", "50min")


def test_rows_convert_back_to_elapsed_duration() -> None:
    assert periods_to_duration(48, "30min") == timedelta(hours=24)
    assert periods_to_duration(0, "30min") == timedelta(0)
    with pytest.raises(TimestampError, match="negative"):
        periods_to_duration(-1, "30min")


# ---------------------------------------------------------------------------
# Intervals
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("duration", "end"),
    [("24h", "2020-06-02 00:00"), ("7d", "2020-06-08 00:00"), ("30d", "2020-07-01 00:00")],
)
def test_gap_durations_translate_to_timestamp_intervals(duration: str, end: str) -> None:
    start, stop = interval_bounds("2020-06-01", duration)
    assert start == pd.Timestamp("2020-06-01")
    assert stop == pd.Timestamp(end)


def test_a_24_hour_interval_selects_24_elapsed_hours_at_either_cadence() -> None:
    for index, rows in ((half_hourly(96), 48), (hourly(48), 24)):
        mask = interval_mask(index, index[0], "24h")
        selected = index[mask.to_numpy()]
        assert len(selected) == rows
        assert selected[-1] - selected[0] < timedelta(hours=24)


def test_intervals_are_half_open_so_neighbours_do_not_share_a_row() -> None:
    index = half_hourly(96)
    first = interval_mask(index, index[0], "24h").to_numpy()
    second = interval_mask(index, pd.Timestamp("2020-01-02"), "24h").to_numpy()
    assert not (first & second).any()
    assert (first | second).all()


def test_interval_endpoints_follow_the_closed_argument() -> None:
    index = half_hourly(6)
    start, stop = index[0], index[2]
    counts = {
        closed: int(interval_mask(index, start, end=stop, closed=closed).sum())
        for closed in ("left", "right", "both", "neither")
    }
    assert counts == {"left": 2, "right": 2, "both": 3, "neither": 1}


def test_an_interval_over_missing_rows_selects_only_the_rows_that_exist() -> None:
    index = half_hourly(96).delete(range(10, 20))
    mask = interval_mask(index, index[0], "24h")
    assert int(mask.sum()) == 38  # 48 grid points, 10 of them absent


def test_interval_selection_needs_exactly_one_of_duration_or_end() -> None:
    index = half_hourly(4)
    with pytest.raises(TimestampError, match="exactly one"):
        interval_mask(index, index[0])
    with pytest.raises(TimestampError, match="exactly one"):
        interval_mask(index, index[0], "24h", end=index[-1])


# ---------------------------------------------------------------------------
# TimeAxis
# ---------------------------------------------------------------------------


def test_a_complete_series_is_regular_at_both_cadences() -> None:
    for index, step in ((half_hourly(48), timedelta(minutes=30)), (hourly(24), timedelta(hours=1))):
        axis = TimeAxis.from_index(index)
        assert axis.time_step == step
        assert axis.step_source == "inferred"
        assert axis.is_regular
        assert len(axis) == len(index)
        assert axis.n_expected == len(index)
        assert axis.coverage == 1.0
        assert len(axis.missing) == 0


def test_missing_rows_are_reported_without_making_the_axis_unusable() -> None:
    index = half_hourly(48).delete([10, 11, 12])
    axis = TimeAxis.from_index(index)
    assert not axis.is_regular
    assert len(axis.missing) == 3
    assert len(axis.off_grid) == 0
    assert axis.n_expected == 48
    assert axis.coverage == pytest.approx(45 / 48)
    assert list(axis.missing) == list(half_hourly(48)[[10, 11, 12]])
    # The axis still answers temporal questions; it is not a failure state.
    assert axis.periods("24h") == 48
    assert axis.elapsed_hours().iloc[-1] == 23.5


def test_off_grid_timestamps_are_reported_separately_from_missing_rows() -> None:
    index = pd.DatetimeIndex(
        ["2020-01-01 00:00", "2020-01-01 00:30", "2020-01-01 00:45", "2020-01-01 01:00"]
    )
    axis = TimeAxis.from_index(index, frequency=HALF_HOURLY)
    assert list(axis.off_grid) == [pd.Timestamp("2020-01-01 00:45")]
    assert len(axis.missing) == 0
    assert not axis.is_regular


def test_require_regular_names_both_kinds_of_defect() -> None:
    TimeAxis.from_index(half_hourly(10)).require_regular()  # does not raise

    axis = TimeAxis.from_index(half_hourly(10).delete([4]))
    with pytest.raises(TimestampError, match="missing grid point"):
        axis.require_regular()


def test_an_axis_refuses_unsorted_or_duplicated_timestamps() -> None:
    with pytest.raises(TimestampError, match="sorted"):
        TimeAxis.from_index(half_hourly(4)[[0, 2, 1, 3]])
    with pytest.raises(TimestampError, match="unique"):
        TimeAxis.from_index(duplicated_frame().index)


def test_a_wildly_wrong_declared_frequency_fails_before_building_a_huge_grid() -> None:
    index = pd.date_range("1990-01-01", periods=100, freq=HALF_HOURLY)
    with pytest.raises(TimestampError, match="almost certainly wrong"):
        TimeAxis.from_index(index, frequency="1ms")


def test_axis_reports_its_span_timezone_and_manifest_summary() -> None:
    index = pd.date_range("2020-01-01", periods=48, freq=HALF_HOURLY, tz="UTC")
    axis = TimeAxis.from_index(index)
    assert axis.start == index[0]
    assert axis.end == index[-1]
    assert axis.span == timedelta(hours=23, minutes=30)
    assert axis.timezone == "UTC"

    manifest = axis.to_dict()
    assert manifest["time_step"] == "P0DT0H30M0S"
    assert manifest["time_step_source"] == "inferred"
    assert manifest["timezone"] == "UTC"
    assert manifest["n_timestamps"] == 48
    assert manifest["is_regular"] is True
    assert manifest["n_missing"] == 0


def test_axis_duration_helpers_use_the_configured_cadence() -> None:
    axis = TimeAxis.from_index(hourly(48))
    assert axis.periods("24h") == 24
    assert axis.duration(24) == timedelta(hours=24)
    assert axis.bounds(axis.start, "7d")[1] == axis.start + timedelta(days=7)
    assert int(axis.mask(axis.start, "24h").sum()) == 24


# ---------------------------------------------------------------------------
# prepare_time_index
# ---------------------------------------------------------------------------


def test_prepare_sorts_indexes_and_describes_the_axis_in_one_call() -> None:
    index = half_hourly(48)
    data = pd.DataFrame(
        {"TIMESTAMP": index.astype(str), "LE": np.arange(48.0)},
    ).iloc[::-1]

    prepared, axis = prepare_time_index(data, timestamp="TIMESTAMP")

    assert prepared.index.equals(index)
    assert list(prepared["LE"]) == list(np.arange(48.0))
    assert axis.time_step == timedelta(minutes=30)
    assert axis.is_regular
    assert axis.index.equals(prepared.index)


def test_prepare_tolerates_missing_rows_by_default_and_can_demand_a_full_grid() -> None:
    data = frame(half_hourly(48).delete([20]))

    prepared, axis = prepare_time_index(data)
    assert len(prepared) == 47
    assert not axis.is_regular

    with pytest.raises(TimestampError, match="not a regular"):
        prepare_time_index(data, require_regular=True)


def test_prepare_accepts_a_declared_frequency_for_an_irregular_series() -> None:
    index = pd.DatetimeIndex(
        ["2020-01-01 00:00", "2020-01-01 00:30", "2020-01-01 00:45", "2020-01-01 01:00"]
    )
    _, axis = prepare_time_index(frame(index), frequency=HALF_HOURLY)
    assert axis.time_step == timedelta(minutes=30)
    assert axis.step_source == "declared"
    assert len(axis.off_grid) == 1
