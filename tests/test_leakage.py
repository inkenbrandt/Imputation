"""Leakage tests for artificial-gap validation. Covers acceptance tests 11-14.

The required test of Step 6 is the last group: hide an interval, alter the hidden
truth by an enormous amount, rebuild every feature used to predict that interval,
and require that nothing moved. It is run for each daily-statistic strategy, not
only the default, because the two strategies that reach into neighbouring days
are exactly the ones where a leak could hide.

Expected values are hand-calculated on small fixtures. See ``docs/method_spec.md``
section 3.5 and ambiguities A4 and A6.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rfrgapfill import leakage
from rfrgapfill.config import DailyStatisticStrategy, FeatureConfig, RFRConfig
from rfrgapfill.features import FeatureError, build_feature_matrix, daily_statistic_names
from rfrgapfill.leakage import (
    LeakageError,
    ValidationFeatureSet,
    available_target_mask,
    build_validation_features,
    detect_target_leakage,
    hide_target,
    holdout_mask_from_intervals,
    observed_target_mask,
    require_no_target_leakage,
)
from rfrgapfill.schema import ColumnMap, ConfigError

HALF_HOURLY = "30min"

#: A gap wide enough that whole calendar days vanish inside it, which is the case
#: the daily statistics are sensitive to. The series is long enough on both sides
#: for the reaching strategies to have somewhere to reach.
GAP = ("2020-06-06", "2020-06-13")

COLUMN_MAP = ColumnMap({"shortwave": "SW", "vpd": "VPD", "air_temperature": "TA"})


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
            # Distinct per day and per half hour, so a statistic that borrowed
            # from the wrong day is visibly wrong rather than coincidentally equal.
            "LE": 10.0 * index.dayofyear + hour,
            "LE_QC": 0.0,
        },
        index=index,
    )


def config(**feature_changes: object) -> RFRConfig:
    """An RFR3 configuration, optionally with a different daily-statistic policy."""
    return RFRConfig(
        mode="RFR3",
        hemisphere="north",
        column_map=COLUMN_MAP,
        features=FeatureConfig(**feature_changes),
    )


def gap_mask(data: pd.DataFrame, interval: tuple[str, str] = GAP) -> pd.Series:
    return holdout_mask_from_intervals(data.index, [interval])


def corrupt(data: pd.DataFrame, mask: pd.Series, target: str = "LE") -> pd.DataFrame:
    """Return ``data`` with the masked target values replaced by absurd ones."""
    corrupted = data.copy()
    corrupted.loc[mask.to_numpy(), target] = -1e9
    return corrupted


# ---------------------------------------------------------------------------
# Masks (step 1: build the holdout mask)
# ---------------------------------------------------------------------------


def test_intervals_are_half_open_so_adjacent_gaps_do_not_share_a_row() -> None:
    data = site_frame(4)
    first = holdout_mask_from_intervals(data.index, [("2020-06-02", "2020-06-03")])
    second = holdout_mask_from_intervals(data.index, [("2020-06-03", "2020-06-04")])
    assert first.sum() == second.sum() == 48
    assert not (first & second).any()
    assert data.index[first.to_numpy()][-1] == pd.Timestamp("2020-06-02 23:30")


def test_several_intervals_are_unioned() -> None:
    data = site_frame(6)
    mask = holdout_mask_from_intervals(
        data.index, [("2020-06-02", "2020-06-03"), ("2020-06-05", "2020-06-06")]
    )
    assert mask.sum() == 96


def test_a_qc_flag_marks_a_value_as_already_gap_filled() -> None:
    data = site_frame(2)
    data.loc[data.index[:10], "LE_QC"] = 1
    observed = observed_target_mask(data, "LE", qc_column="LE_QC")
    assert observed.sum() == len(data) - 10
    assert not observed.iloc[:10].any()


def test_a_missing_qc_flag_counts_as_not_observed() -> None:
    # Conservative direction: a value of unknown provenance is neither trained on
    # nor scored against.
    data = site_frame(2)
    data.loc[data.index[0], "LE_QC"] = np.nan
    assert not observed_target_mask(data, "LE", qc_column="LE_QC").iloc[0]


def test_without_a_qc_column_every_present_value_counts_as_observed() -> None:
    data = site_frame(2)
    data.loc[data.index[3], "LE"] = np.nan
    observed = observed_target_mask(data, "LE")
    assert observed.sum() == len(data) - 1
    assert not observed.iloc[3]


def test_qc_values_are_configurable() -> None:
    data = site_frame(2)
    data["LE_QC"] = 1
    assert not observed_target_mask(data, "LE", qc_column="LE_QC").any()
    permissive = observed_target_mask(data, "LE", qc_column="LE_QC", observed_qc_values=(0, 1))
    assert permissive.all()


def test_available_is_observed_minus_held_out() -> None:
    data = site_frame(6)
    holdout = gap_mask(data, ("2020-06-03", "2020-06-04"))
    data.loc[data.index[0], "LE_QC"] = 2
    available = available_target_mask(data, "LE", holdout=holdout, qc_column="LE_QC")
    assert not (available & holdout).any()
    assert not available.iloc[0], "a pre-filled value is not available either"
    assert int(available.sum()) == len(data) - 48 - 1


def test_a_mask_and_a_qc_column_are_not_both_accepted() -> None:
    data = site_frame(2)
    with pytest.raises(ConfigError, match="not both"):
        available_target_mask(
            data,
            "LE",
            holdout=gap_mask(data, ("2020-06-01", "2020-06-02")),
            qc_column="LE_QC",
            observed=pd.Series(True, index=data.index),
        )


# ---------------------------------------------------------------------------
# Hiding the truth (step 2)
# ---------------------------------------------------------------------------


def test_hiding_removes_the_held_out_target_and_nothing_else() -> None:
    data = site_frame(6)
    holdout = gap_mask(data, ("2020-06-03", "2020-06-04"))
    hidden = hide_target(data, "LE", holdout=holdout)

    assert hidden.loc[holdout.to_numpy(), "LE"].isna().all()
    assert hidden.loc[~holdout.to_numpy(), "LE"].equals(data.loc[~holdout.to_numpy(), "LE"])
    for driver in ("SW", "VPD", "TA"):
        # Drivers arrive pre-filled in the paper and stay available inside a gap.
        assert hidden[driver].equals(data[driver])


def test_hiding_does_not_touch_the_caller_s_frame() -> None:
    data = site_frame(4)
    before = data["LE"].copy()
    hide_target(data, "LE", holdout=gap_mask(data, ("2020-06-02", "2020-06-03")))
    assert data["LE"].equals(before)


# ---------------------------------------------------------------------------
# The feature set (steps 3-5)
# ---------------------------------------------------------------------------


def test_the_held_out_days_statistics_are_not_computed_from_hidden_truth() -> None:
    data = site_frame()
    holdout = gap_mask(data)
    features = build_validation_features(
        data, config=config(), target="LE", holdout=holdout, qc_column="LE_QC"
    ).features
    # Under the default strategy a fully hidden day has no visible observation
    # left, so its statistics are missing rather than computed from the truth.
    hidden_rows = features.loc[holdout.to_numpy(), list(daily_statistic_names("LE"))]
    assert hidden_rows.isna().all().all()


def test_visible_days_keep_their_own_statistics() -> None:
    data = site_frame()
    holdout = gap_mask(data)
    features = build_validation_features(
        data, config=config(), target="LE", holdout=holdout, qc_column="LE_QC"
    ).features
    # 2020-06-01 is untouched: LE = 10*doy + hour, so the visible day spans
    # 10*153 + 0.0 to 10*153 + 23.5 and its median is 1530 + 11.75.
    day = features.loc["2020-06-01"]
    assert day["LE_daily_q2"].iloc[0] == pytest.approx(1541.75)


def test_the_truth_is_kept_untouched_beside_the_features() -> None:
    data = site_frame()
    holdout = gap_mask(data)
    result = build_validation_features(
        data, config=config(), target="LE", holdout=holdout, qc_column="LE_QC"
    )
    pd.testing.assert_series_equal(result.truth, data["LE"].astype(float), check_names=False)
    assert result.truth.loc[holdout.to_numpy()].notna().all(), "truth survives for scoring"


def test_the_training_target_never_contains_a_held_out_value() -> None:
    data = site_frame()
    holdout = gap_mask(data)
    result = build_validation_features(
        data, config=config(), target="LE", holdout=holdout, qc_column="LE_QC"
    )
    trained_on = result.training_target().index
    assert not holdout.loc[trained_on].any()
    assert len(trained_on) + int(holdout.sum()) == len(data)


def test_scoring_uses_only_withheld_rows_that_were_really_measured() -> None:
    data = site_frame()
    holdout = gap_mask(data)
    # One value inside the gap was already gap-filled before ingestion: it is
    # withheld like the rest, but there is no measurement to score against.
    first_hidden = data.index[holdout.to_numpy()][0]
    data.loc[first_hidden, "LE_QC"] = 1
    result = build_validation_features(
        data, config=config(), target="LE", holdout=holdout, qc_column="LE_QC"
    )
    assert int(result.scoring_mask.sum()) == int(holdout.sum()) - 1
    assert first_hidden not in result.scoring_truth().index


def test_the_manifest_reports_what_was_withheld_and_what_is_predictable() -> None:
    data = site_frame()
    holdout = gap_mask(data)
    summary = build_validation_features(
        data, config=config(), target="LE", holdout=holdout, qc_column="LE_QC"
    ).to_dict()
    assert summary["holdout_rows"] == int(holdout.sum())
    assert summary["withheld_fraction_of_observed"] == pytest.approx(int(holdout.sum()) / len(data))
    # A4 made visible: under the default strategy a multi-day gap leaves every
    # held-out row without daily statistics, so none of it can be predicted.
    assert summary["holdout_rows_with_complete_features"] == 0
    assert summary["features"]["daily_statistic_strategy"] == "missing"


@pytest.mark.parametrize("strategy", ["rolling_available", "neighbor_day_fallback"])
def test_a_reaching_strategy_makes_a_multi_day_gap_predictable(strategy: str) -> None:
    data = site_frame()
    holdout = gap_mask(data)
    summary = build_validation_features(
        data,
        config=config(daily_statistic_strategy=strategy, fallback_window_days=7),
        target="LE",
        holdout=holdout,
        qc_column="LE_QC",
    ).to_dict()
    assert summary["holdout_rows_with_complete_features"] == int(holdout.sum())


def test_an_available_row_may_never_also_be_withheld() -> None:
    data = site_frame(2)
    index = data.index
    everything = pd.Series(True, index=index)
    with pytest.raises(LeakageError, match="both available"):
        ValidationFeatureSet(
            target="LE",
            features=pd.DataFrame({"shortwave": data["SW"]}, index=index),
            truth=data["LE"].astype(float),
            observed_mask=everything,
            holdout_mask=everything,
            available_mask=everything,
            config=config(),
        )


def test_a_misaligned_mask_is_rejected() -> None:
    data = site_frame(2)
    index = data.index
    with pytest.raises(FeatureError, match="aligned"):
        ValidationFeatureSet(
            target="LE",
            features=pd.DataFrame({"shortwave": data["SW"]}, index=index),
            truth=data["LE"].astype(float),
            observed_mask=pd.Series(True, index=index[:-1]),
            holdout_mask=pd.Series(False, index=index),
            available_mask=pd.Series(True, index=index),
            config=config(),
        )


def test_the_orf_arm_uses_the_same_masks_and_the_same_truth() -> None:
    data = site_frame()
    holdout = gap_mask(data)
    rfr = build_validation_features(
        data, config=config(), target="LE", holdout=holdout, qc_column="LE_QC"
    )
    orf = build_validation_features(
        data,
        config=config().as_orf(),
        target="LE",
        holdout=holdout,
        qc_column="LE_QC",
    )
    assert orf.feature_names == ("shortwave", "vpd", "air_temperature")
    pd.testing.assert_series_equal(rfr.truth, orf.truth)
    pd.testing.assert_series_equal(rfr.scoring_mask, orf.scoring_mask)
    pd.testing.assert_series_equal(rfr.training_mask, orf.training_mask)
    for driver in orf.feature_names:
        pd.testing.assert_series_equal(rfr.features[driver], orf.features[driver])


# ---------------------------------------------------------------------------
# The required leakage test (acceptance tests 11-14)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("strategy", list(DailyStatisticStrategy))
def test_altering_the_hidden_truth_changes_no_feature(
    strategy: DailyStatisticStrategy,
) -> None:
    # 11-14: hide an interval, alter the hidden truth dramatically, recompute
    # every feature used to predict that interval, and require that nothing
    # moved - including the features of the rows that are not held out, since a
    # hidden value that shifted a neighbouring day would corrupt the training
    # data instead.
    data = site_frame()
    holdout = gap_mask(data)
    window = 7 if strategy.uses_other_days else None
    settings = config(daily_statistic_strategy=strategy, fallback_window_days=window)

    before = build_validation_features(
        data, config=settings, target="LE", holdout=holdout, qc_column="LE_QC"
    ).features
    after = build_validation_features(
        corrupt(data, holdout),
        config=settings,
        target="LE",
        holdout=holdout,
        qc_column="LE_QC",
    ).features
    pd.testing.assert_frame_equal(before, after)


@pytest.mark.parametrize("strategy", list(DailyStatisticStrategy))
def test_the_available_mask_alone_is_enough_even_without_hiding_the_values(
    strategy: DailyStatisticStrategy,
) -> None:
    # The two protections are independent. With the frame-level masking off, the
    # available mask is the only thing standing between hidden truth and the
    # daily statistics, and it must still hold.
    data = site_frame()
    holdout = gap_mask(data)
    window = 7 if strategy.uses_other_days else None
    settings = config(daily_statistic_strategy=strategy, fallback_window_days=window)

    before = build_validation_features(
        data,
        config=settings,
        target="LE",
        holdout=holdout,
        qc_column="LE_QC",
        mask_target_values=False,
    ).features
    after = build_validation_features(
        corrupt(data, holdout),
        config=settings,
        target="LE",
        holdout=holdout,
        qc_column="LE_QC",
        mask_target_values=False,
    ).features
    pd.testing.assert_frame_equal(before, after)


@pytest.mark.parametrize("strategy", list(DailyStatisticStrategy))
def test_the_probe_reports_no_leaking_feature_in_paper_safe_mode(
    strategy: DailyStatisticStrategy,
) -> None:
    data = site_frame()
    window = 7 if strategy.uses_other_days else None
    settings = config(daily_statistic_strategy=strategy, fallback_window_days=window)
    assert (
        detect_target_leakage(
            data,
            config=settings,
            target="LE",
            holdout=gap_mask(data),
            qc_column="LE_QC",
        )
        == ()
    )
    require_no_target_leakage(
        data, config=settings, target="LE", holdout=gap_mask(data), qc_column="LE_QC"
    )


def test_the_corruption_is_detectable_when_nothing_protects_the_features() -> None:
    # Negative control. The tests above are only worth something if the altered
    # truth *would* have shown up: building the features straight from the frame,
    # with no mask and no hiding - what a leaking implementation does - moves
    # exactly the four target-derived columns and nothing else.
    data = site_frame()
    holdout = gap_mask(data)
    settings = config()
    before = build_feature_matrix(data, config=settings, target="LE")
    after = build_feature_matrix(corrupt(data, holdout), config=settings, target="LE")

    moved = {column for column in before.columns if not before[column].equals(after[column])}
    assert moved == set(daily_statistic_names("LE"))


def test_a_detected_leak_is_reported_with_the_offending_columns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # In paper_safe mode the probe cannot be made to fire through the public API,
    # which is the point of the mode; the reporting path is exercised directly so
    # that a future feature_mode cannot fail silently.
    monkeypatch.setattr(
        leakage, "detect_target_leakage", lambda *args, **kwargs: ("LE_daily_q1", "LE_daily_std")
    )
    data = site_frame(2)
    with pytest.raises(LeakageError) as failure:
        require_no_target_leakage(
            data,
            config=config(),
            target="LE",
            holdout=gap_mask(data, ("2020-06-01", "2020-06-02")),
        )
    message = str(failure.value)
    assert "LE_daily_q1, LE_daily_std" in message
    assert "paper_safe" in message
    assert "daily_flux_statistics" in message
