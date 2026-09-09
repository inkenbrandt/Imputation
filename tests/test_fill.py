"""Gap-filling and provenance tests. Covers Step 8 and acceptance tests 24-25.

The four Step 8 rules are the four sections below: observed values are never
changed, gaps are filled where the predictors exist, a row lacking a required
driver is left alone or raises according to configuration, and the caller's
DataFrame is never mutated. The rest of the module covers the provenance columns,
the observed-versus-pre-filled distinction, the fixed time origin, and the
ambiguity-A4 consequence that a whole-day gap cannot be filled under the
documented default.

The forests here are deliberately tiny (10 trees, 3 folds): nothing in this module
tests predictive quality. See ``docs/method_spec.md`` sections 3.4 and 7.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from rfrgapfill.config import FeatureConfig, RFRConfig
from rfrgapfill.fill import (
    FILL_COLUMN_SUFFIXES,
    FillError,
    FillMethod,
    FillReport,
    FillResult,
    RFRGapFiller,
    fill_column_names,
    method_label,
)
from rfrgapfill.model import ModelError, NotFittedError
from rfrgapfill.schema import ColumnMap, ColumnMapError, ConfigError
from rfrgapfill.time import TimestampError

HALF_HOURLY = "30min"

COLUMN_MAP = ColumnMap({"shortwave": "SW", "vpd": "VPD", "air_temperature": "TA"})

#: One candidate per grid point is enough; this module never scores a prediction.
TEST_GRID = {"n_estimators": (10,)}

#: A gap short enough to leave its calendar day plenty of visible observations, so
#: the daily statistics of method_spec.md 3.4 exist and the rows can be filled.
GAP_START = "2020-06-20 06:00"
GAP_END = "2020-06-20 12:00"


def site_frame(days: int = 60, *, seed: int = 0) -> pd.DataFrame:
    """A half-hourly site frame whose target is a nonlinear function of its drivers.

    The same construction as ``tests/test_model.py``: ``LE`` is a product of
    radiation and temperature plus a saturating VPD term, with a ``LE_QC`` column
    flagging every value as measured (FLUXNET's 0).
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


def with_gap(data: pd.DataFrame, start: str = GAP_START, end: str = GAP_END) -> pd.DataFrame:
    """Return a copy of ``data`` with the target missing over ``[start, end)``."""
    gapped = data.copy()
    inside = gap_mask(gapped, start, end)
    gapped.loc[inside, "LE"] = np.nan
    gapped.loc[inside, "LE_QC"] = np.nan
    return gapped


def gap_mask(data: pd.DataFrame, start: str = GAP_START, end: str = GAP_END) -> np.ndarray:
    """Return the boolean mask of ``data``'s rows inside the half-open interval."""
    index = pd.DatetimeIndex(data.index)
    mask: np.ndarray = (index >= pd.Timestamp(start)) & (index < pd.Timestamp(end))
    return mask


def rfr_config(**changes: object) -> RFRConfig:
    """An RFR3 configuration with a one-candidate grid and few folds."""
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


@pytest.fixture(scope="module")
def data() -> pd.DataFrame:
    """A site frame with one six-hour gap in the target."""
    return with_gap(site_frame())


@pytest.fixture(scope="module")
def filler(data: pd.DataFrame) -> RFRGapFiller:
    """One fitted filler shared by the read-only tests."""
    return RFRGapFiller(rfr_config()).fit(data, target="LE", qc_column="LE_QC")


@pytest.fixture(scope="module")
def result(filler: RFRGapFiller, data: pd.DataFrame) -> FillResult:
    """The fill of the frame the filler was fitted on."""
    return filler.fill(data)


# ---------------------------------------------------------------------------
# Step 8, rule 1: no observed value is changed (acceptance test 24)
# ---------------------------------------------------------------------------


def test_observed_values_are_never_changed(result: FillResult) -> None:
    observed = result.is_observed.to_numpy(dtype=bool)
    assert observed.any()
    np.testing.assert_array_equal(
        result.filled.to_numpy(dtype=float)[observed],
        result.original.to_numpy(dtype=float)[observed],
    )


def test_observed_rows_are_never_marked_filled(result: FillResult) -> None:
    both = result.is_observed.to_numpy(dtype=bool) & result.is_filled.to_numpy(dtype=bool)
    assert not both.any(), "a row cannot be both an observation and a model prediction"


def test_original_column_is_the_untouched_input(result: FillResult, data: pd.DataFrame) -> None:
    pd.testing.assert_series_equal(
        result.original.astype(float),
        data["LE"].astype(float),
        check_names=False,
    )


def test_the_target_column_itself_is_carried_through_unchanged(
    result: FillResult, data: pd.DataFrame
) -> None:
    """Filling adds columns; it never writes into the target column in place."""
    pd.testing.assert_series_equal(result.frame["LE"], data["LE"], check_names=False)


# ---------------------------------------------------------------------------
# Step 8, rule 2: gaps are filled where the predictors exist (acceptance test 25)
# ---------------------------------------------------------------------------


def test_the_gap_is_filled(result: FillResult, data: pd.DataFrame) -> None:
    inside = gap_mask(data)
    assert inside.sum() == 12
    assert result.is_filled.to_numpy(dtype=bool)[inside].all()
    assert np.isfinite(result.filled.to_numpy(dtype=float)[inside]).all()


def test_only_rows_without_a_value_are_filled(result: FillResult, data: pd.DataFrame) -> None:
    missing = data["LE"].isna().to_numpy()
    np.testing.assert_array_equal(result.is_filled.to_numpy(dtype=bool), missing)


def test_filled_values_are_plausible(result: FillResult, data: pd.DataFrame) -> None:
    """A sanity floor only: the predictions sit inside the observed range."""
    inside = gap_mask(data)
    predictions = result.filled.to_numpy(dtype=float)[inside]
    observed = data["LE"].dropna().to_numpy(dtype=float)
    assert predictions.min() >= observed.min()
    assert predictions.max() <= observed.max()


def test_fill_method_labels_every_row(result: FillResult) -> None:
    methods = set(result.fill_method.unique())
    assert methods == {FillMethod.OBSERVED.value, "RFR3"}


def test_model_version_is_stamped_on_filled_rows_only(
    result: FillResult, filler: RFRGapFiller
) -> None:
    version = result.frame["LE_model_version"]
    filled = result.is_filled.to_numpy(dtype=bool)
    assert set(version[filled].unique()) == {filler.model_version}
    assert version[~filled].isna().all()


# ---------------------------------------------------------------------------
# Step 8, rule 3: a row missing a required driver is left alone, or raises
# ---------------------------------------------------------------------------


def driverless_frame() -> pd.DataFrame:
    """A frame whose gap has no shortwave driver in its second half."""
    frame = with_gap(site_frame())
    inside = gap_mask(frame)
    positions = np.flatnonzero(inside)[6:]
    frame.iloc[positions, frame.columns.get_loc("SW")] = np.nan
    return frame


@pytest.fixture(scope="module")
def driverless() -> pd.DataFrame:
    return driverless_frame()


@pytest.fixture(scope="module")
def driverless_filler(driverless: pd.DataFrame) -> RFRGapFiller:
    return RFRGapFiller(rfr_config()).fit(driverless, target="LE", qc_column="LE_QC")


def test_rows_without_a_driver_are_left_unfilled(
    driverless_filler: RFRGapFiller, driverless: pd.DataFrame
) -> None:
    result = driverless_filler.fill(driverless)
    inside = gap_mask(driverless)
    positions = np.flatnonzero(inside)
    filled = result.is_filled.to_numpy(dtype=bool)

    assert filled[positions[:6]].all()
    assert not filled[positions[6:]].any()
    assert np.isnan(result.filled.to_numpy(dtype=float)[positions[6:]]).all()
    assert (result.fill_method.to_numpy()[positions[6:]] == FillMethod.UNFILLED.value).all()


def test_unfilled_rows_are_reported_with_the_feature_that_blocked_them(
    driverless_filler: RFRGapFiller, driverless: pd.DataFrame
) -> None:
    report = driverless_filler.fill(driverless).report
    assert report.candidate_rows == 12
    assert report.filled_rows == 6
    assert report.unfilled_rows == 6
    assert report.filled_fraction == pytest.approx(0.5)
    assert report.missing_by_feature["shortwave"] == 6
    assert report.worst_feature in {"shortwave", "radiation_category"}


def test_missing_drivers_raise_when_the_caller_asks(
    driverless_filler: RFRGapFiller, driverless: pd.DataFrame
) -> None:
    with pytest.raises(ModelError, match="missing a predictor"):
        driverless_filler.fill(driverless, on_incomplete="raise")


def test_a_driver_missing_outside_a_gap_does_not_block_the_fill(
    driverless_filler: RFRGapFiller, driverless: pd.DataFrame
) -> None:
    """``on_incomplete='raise'`` judges the rows being predicted, not the whole frame."""
    frame = driverless.copy()
    inside = gap_mask(frame)
    frame.loc[~inside & (frame.index.hour == 3), "SW"] = np.nan
    frame.loc[gap_mask(frame), "SW"] = driverless.loc[gap_mask(driverless), "SW"].to_numpy()
    frame.iloc[np.flatnonzero(inside)[6:], frame.columns.get_loc("SW")] = frame["SW"].iloc[0]
    assert driverless_filler.fill(frame, on_incomplete="raise").report.filled_rows == 12


# ---------------------------------------------------------------------------
# Step 8, rule 4: the caller's DataFrame is not mutated
# ---------------------------------------------------------------------------


def test_fit_does_not_mutate_the_input() -> None:
    frame = with_gap(site_frame())
    before = frame.copy()
    RFRGapFiller(rfr_config()).fit(frame, target="LE", qc_column="LE_QC")
    pd.testing.assert_frame_equal(frame, before)


def test_fill_does_not_mutate_the_input(filler: RFRGapFiller, data: pd.DataFrame) -> None:
    before = data.copy()
    filler.fill(data)
    pd.testing.assert_frame_equal(data, before)
    assert not any(name in data.columns for name in fill_column_names("LE"))


def test_writing_to_the_result_does_not_reach_the_input(
    filler: RFRGapFiller, data: pd.DataFrame
) -> None:
    result = filler.fill(data)
    result.frame.loc[result.frame.index[0], "LE"] = -999.0
    assert data["LE"].iloc[0] != -999.0


# ---------------------------------------------------------------------------
# Provenance columns (method_spec.md section 7)
# ---------------------------------------------------------------------------


def test_fill_column_names_are_the_six_the_specification_lists() -> None:
    assert fill_column_names("LE") == (
        "LE_original",
        "LE_filled",
        "LE_is_observed",
        "LE_is_filled",
        "LE_fill_method",
        "LE_model_version",
    )
    assert len(FILL_COLUMN_SUFFIXES) == 6


def test_the_result_carries_the_input_columns_and_the_provenance_columns(
    result: FillResult, data: pd.DataFrame
) -> None:
    assert list(result.frame.columns) == list(data.columns) + list(fill_column_names("LE"))
    assert result.columns == fill_column_names("LE")


def test_the_masks_are_boolean(result: FillResult) -> None:
    assert result.is_observed.dtype == bool
    assert result.is_filled.dtype == bool


@pytest.mark.parametrize(
    ("mode", "limiter", "expected"),
    [
        ("RFR3", True, "RFR3"),
        ("RFR10", True, "RFR10"),
        ("RFR3", False, "ORF3"),
        ("RFR10", False, "ORF10"),
    ],
)
def test_method_label_names_the_arm(mode: str, limiter: bool, expected: str) -> None:
    config = RFRConfig(
        mode=mode,
        hemisphere="north",
        column_map=ColumnMap.fluxnet2015(mode),
        features=FeatureConfig(use_receptive_limiter=limiter),
    )
    assert method_label(config) == expected


def test_the_orf_arm_labels_its_own_fills(data: pd.DataFrame) -> None:
    """The ORF benchmark fills through the same class and says so in the column."""
    orf = RFRGapFiller(rfr_config().as_orf()).fit(data, target="LE", qc_column="LE_QC")
    result = orf.fill(data)
    assert orf.method == "ORF3"
    assert set(result.fill_method.unique()) == {FillMethod.OBSERVED.value, "ORF3"}
    assert result.report.filled_rows == 12


# ---------------------------------------------------------------------------
# Observed versus pre-filled (method_spec.md section 7)
# ---------------------------------------------------------------------------


def pre_filled_frame() -> pd.DataFrame:
    """A frame whose target is complete but whose first morning is QC-flagged as filled."""
    frame = site_frame()
    flagged = gap_mask(frame, "2020-06-10 00:00", "2020-06-10 06:00")
    frame.loc[flagged, "LE_QC"] = 2.0
    return frame


def test_pre_filled_values_are_not_trained_on() -> None:
    frame = pre_filled_frame()
    filler = RFRGapFiller(rfr_config()).fit(frame, target="LE", qc_column="LE_QC")
    assert filler.model.fit_report.rows == len(frame) - 12


def test_pre_filled_values_are_kept_and_labelled() -> None:
    frame = pre_filled_frame()
    result = RFRGapFiller(rfr_config()).fit(frame, target="LE", qc_column="LE_QC").fill(frame)

    assert result.report.pre_filled_rows == 12
    assert result.report.candidate_rows == 0
    assert result.report.filled_rows == 0
    flagged = gap_mask(frame, "2020-06-10 00:00", "2020-06-10 06:00")
    assert (result.fill_method.to_numpy()[flagged] == FillMethod.PRE_FILLED.value).all()
    np.testing.assert_array_equal(
        result.filled.to_numpy(dtype=float)[flagged],
        frame["LE"].to_numpy(dtype=float)[flagged],
    )


def test_refill_pre_filled_replaces_them() -> None:
    frame = pre_filled_frame()
    filler = RFRGapFiller(rfr_config()).fit(frame, target="LE", qc_column="LE_QC")
    result = filler.fill(frame, refill_pre_filled=True)

    flagged = gap_mask(frame, "2020-06-10 00:00", "2020-06-10 06:00")
    assert result.report.filled_rows == 12
    assert (result.fill_method.to_numpy()[flagged] == "RFR3").all()
    assert not np.allclose(
        result.filled.to_numpy(dtype=float)[flagged],
        result.original.to_numpy(dtype=float)[flagged],
    )
    # The untouched original is still there to compare against.
    np.testing.assert_array_equal(
        result.original.to_numpy(dtype=float), frame["LE"].to_numpy(dtype=float)
    )


def test_refill_pre_filled_needs_a_qc_column(filler: RFRGapFiller, data: pd.DataFrame) -> None:
    unflagged = RFRGapFiller(rfr_config()).fit(data, target="LE")
    with pytest.raises(ConfigError, match="needs a QC column"):
        unflagged.fill(data, refill_pre_filled=True)


def test_without_a_qc_column_every_present_value_counts_as_observed(data: pd.DataFrame) -> None:
    result = RFRGapFiller(rfr_config()).fit(data, target="LE").fill(data)
    assert result.report.pre_filled_rows == 0
    assert result.report.observed_rows == int(data["LE"].notna().sum())


# ---------------------------------------------------------------------------
# The time origin of ``time_distance_hours``
# ---------------------------------------------------------------------------


def test_fill_reuses_the_fit_time_origin(filler: RFRGapFiller, data: pd.DataFrame) -> None:
    """A later slice must get the predictions it had in the whole series.

    ``time_distance_hours`` is elapsed hours since the start of the series the
    model was fitted on. Rebuilding it from the slice's own first timestamp would
    shift every value, so this is the test that the origin is carried.
    """
    whole = filler.fill(data)
    later = data.loc["2020-06-15":]
    slice_result = filler.fill(later)

    assert filler.origin == data.index.min()
    assert filler.origin != later.index.min(), "the slice must start after the fit origin"
    assert slice_result.report.filled_rows == 12, "the slice must contain the gap"
    np.testing.assert_array_equal(
        slice_result.filled.to_numpy(dtype=float),
        whole.filled.loc[later.index].to_numpy(dtype=float),
    )


def test_rows_before_the_fit_origin_are_reported(data: pd.DataFrame) -> None:
    """Filling earlier data than the fit saw means negative elapsed hours; say so."""
    later = data.loc["2020-06-15":]
    filler = RFRGapFiller(rfr_config()).fit(later, target="LE", qc_column="LE_QC")
    result = filler.fill(data)
    assert result.report.rows_before_fit_origin == int((data.index < later.index.min()).sum())
    assert result.report.rows_before_fit_origin > 0


# ---------------------------------------------------------------------------
# Ambiguity A4: a whole-day gap has no daily statistics
# ---------------------------------------------------------------------------


def whole_day_frame() -> pd.DataFrame:
    """A frame whose target is missing for one entire calendar day."""
    return with_gap(site_frame(), "2020-06-20", "2020-06-21")


def test_a_whole_day_gap_cannot_be_filled_under_the_default_strategy() -> None:
    """The honest consequence of A4, reported rather than hidden."""
    frame = whole_day_frame()
    filler = RFRGapFiller(rfr_config()).fit(frame, target="LE", qc_column="LE_QC")
    with pytest.warns(UserWarning, match="no gap could be filled"):
        result = filler.fill(frame)

    assert result.report.candidate_rows == 48
    assert result.report.filled_rows == 0
    assert result.report.worst_feature is not None
    assert result.report.worst_feature.startswith("LE_daily_")
    assert "A4" in result.report.summary()
    assert (
        result.fill_method.to_numpy()[gap_mask(frame, "2020-06-20", "2020-06-21")]
        == FillMethod.UNFILLED.value
    ).all()


def test_a_reaching_strategy_fills_a_whole_day_gap() -> None:
    """The configured way out of A4: statistics from neighbouring visible observations."""
    frame = whole_day_frame()
    config = rfr_config(
        features=FeatureConfig(daily_statistic_strategy="rolling_available", fallback_window_days=7)
    )
    result = RFRGapFiller(config).fit(frame, target="LE", qc_column="LE_QC").fill(frame)
    assert result.report.candidate_rows == 48
    assert result.report.filled_rows == 48


def test_a_short_gap_needs_no_reaching_strategy(result: FillResult) -> None:
    """The default is only a problem for gaps that cover whole days."""
    assert result.report.filled_fraction == 1.0


# ---------------------------------------------------------------------------
# Row accounting
# ---------------------------------------------------------------------------


def test_the_report_partitions_the_rows(result: FillResult, data: pd.DataFrame) -> None:
    report = result.report
    assert report.rows == len(data)
    assert report.observed_rows + report.pre_filled_rows + report.missing_rows == report.rows
    assert report.filled_rows + report.unfilled_rows == report.candidate_rows
    assert report.missing_rows == 12
    assert report.target == "LE"
    assert report.method == "RFR3"


def valid_report(**changes: object) -> FillReport:
    settings: dict[str, object] = {
        "target": "LE",
        "method": "RFR3",
        "model_version": "test",
        "rows": 10,
        "observed_rows": 7,
        "pre_filled_rows": 1,
        "missing_rows": 2,
        "candidate_rows": 2,
        "filled_rows": 1,
        "unfilled_rows": 1,
        "missing_by_feature": {"shortwave": 1},
        "rows_before_fit_origin": 0,
    }
    settings.update(changes)
    return FillReport(**settings)  # type: ignore[arg-type]


def test_the_report_rejects_row_states_that_do_not_partition_the_frame() -> None:
    with pytest.raises(FillError, match="do not partition"):
        valid_report(observed_rows=6)


def test_the_report_rejects_candidates_that_are_not_accounted_for() -> None:
    with pytest.raises(FillError, match="do not account for"):
        valid_report(filled_rows=2)


def test_the_report_has_no_worst_feature_when_nothing_was_missing() -> None:
    assert valid_report(missing_by_feature={"shortwave": 0}).worst_feature is None


def test_the_summary_says_so_when_there_is_nothing_to_fill() -> None:
    report = valid_report(
        missing_rows=0, observed_rows=9, candidate_rows=0, filled_rows=0, unfilled_rows=0
    )
    assert "nothing to fill" in report.summary()
    assert report.filled_fraction is None


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


def test_an_unfitted_filler_cannot_fill(data: pd.DataFrame) -> None:
    with pytest.raises(NotFittedError, match="call fit"):
        RFRGapFiller(rfr_config()).fill(data)


def test_an_unfitted_filler_has_no_target_or_manifest() -> None:
    unfitted = RFRGapFiller(rfr_config())
    assert not unfitted.is_fitted
    assert "unfitted" in repr(unfitted)
    with pytest.raises(NotFittedError):
        unfitted.to_dict()


def test_the_config_must_be_an_rfr_config() -> None:
    with pytest.raises(ConfigError, match="must be an RFRConfig"):
        RFRGapFiller({"mode": "RFR3"})  # type: ignore[arg-type]


def test_a_missing_target_column_is_named(data: pd.DataFrame) -> None:
    with pytest.raises(FillError, match="target column 'H'"):
        RFRGapFiller(rfr_config()).fit(data, target="H")


def test_an_empty_target_name_is_rejected(data: pd.DataFrame) -> None:
    with pytest.raises(FillError, match="non-empty string"):
        RFRGapFiller(rfr_config()).fit(data, target="  ")


def test_the_qc_column_must_still_be_there_at_fill(
    filler: RFRGapFiller, data: pd.DataFrame
) -> None:
    with pytest.raises(FillError, match="QC column 'LE_QC'"):
        filler.fill(data.drop(columns=["LE_QC"]))


def test_filling_a_previous_result_is_refused(filler: RFRGapFiller, data: pd.DataFrame) -> None:
    """The provenance columns would be overwritten; fill the original instead."""
    once = filler.fill(data)
    with pytest.raises(FillError, match="already carries provenance column"):
        filler.fill(once.frame)


def test_an_unmapped_driver_is_refused(data: pd.DataFrame) -> None:
    config = RFRConfig(mode="RFR3", frequency=HALF_HOURLY, hemisphere="north")
    with pytest.raises(ColumnMapError):
        RFRGapFiller(config).fit(data, target="LE")


def test_duplicate_timestamps_are_not_accepted_silently(data: pd.DataFrame) -> None:
    doubled = pd.concat([data, data.iloc[:2]])
    with pytest.raises(TimestampError, match=r"duplicate|repeat"):
        RFRGapFiller(rfr_config()).fit(doubled, target="LE", qc_column="LE_QC")


def test_unsorted_input_is_sorted_rather_than_rejected(data: pd.DataFrame) -> None:
    shuffled = data.iloc[::-1]
    result = RFRGapFiller(rfr_config()).fit(shuffled, target="LE", qc_column="LE_QC").fill(shuffled)
    assert result.frame.index.is_monotonic_increasing
    assert result.report.filled_rows == 12


def test_fill_needs_a_dataframe(filler: RFRGapFiller) -> None:
    with pytest.raises(FillError, match="pandas DataFrame"):
        filler.fill([1, 2, 3])  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def test_the_filler_manifest_is_json_serialisable(filler: RFRGapFiller) -> None:
    manifest = filler.to_dict()
    text = json.dumps(manifest)
    assert json.loads(text) == manifest

    assert manifest["target"] == "LE"
    assert manifest["method"] == "RFR3"
    assert manifest["qc_column"] == "LE_QC"
    assert manifest["observed_qc_values"] == [0]
    assert manifest["time_origin"].startswith("2020-06-01")
    assert manifest["column_map"]["variables"]["shortwave"] == "SW"
    assert manifest["features"]["feature_names"][0] == "shortwave"
    # A1 travels with every manifest this package writes.
    assert manifest["model"]["hyperparameter_grid_is_paper_exact"] is False
    assert manifest["config"]["random_state"] == 42


def test_the_result_manifest_is_json_serialisable(result: FillResult) -> None:
    manifest = result.to_dict()
    assert json.loads(json.dumps(manifest)) == manifest
    assert manifest["fill"]["filled_rows"] == 12
    assert manifest["columns"] == list(fill_column_names("LE"))
    assert manifest["time_axis"]["time_step"] == "P0DT0H30M0S"


def test_the_model_version_identifies_the_run(filler: RFRGapFiller) -> None:
    version = filler.model_version
    assert version.startswith("rfr-gapfill/")
    assert "/RFR3/LE@" in version
    assert filler.model.fitted_at is not None
    assert version.endswith(filler.model.fitted_at)


def test_the_result_knows_its_own_target_and_config(result: FillResult) -> None:
    assert result.target == "LE"
    assert result.config.rfr_mode.value == "RFR3"
    assert result.time_axis.time_step == pd.Timedelta(HALF_HOURLY).to_pytimedelta()
