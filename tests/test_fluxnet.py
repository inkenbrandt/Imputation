"""FLUXNET2015 adapter tests.

Two things are being protected here. First, the adapter must actually work on
the shape of a real FULLSET file: integer ``YYYYMMDDHHMM`` timestamps, the
``-9999`` sentinel, and soil columns that carry a depth index. Second, the
coupling must stay one-way - the adapter knows FLUXNET names, the core package
still only sees a :class:`~rfrgapfill.schema.ColumnMap`.

The QC semantics asserted here are the ones documented in
``docs/fluxnet_adapter.md``: flag 0 is a measurement, everything else arrived
already gap-filled, and a fractional flag means the file is an aggregated
product whose flags mean something else entirely.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from fixtures import synthetic_site
from rfrgapfill.config import RFRConfig
from rfrgapfill.fill import RFRGapFiller
from rfrgapfill.fluxnet import (
    FLUXNET2015_FLUX_QC,
    FLUXNET2015_FLUXES,
    FLUXNET2015_MISSING_VALUE,
    FLUXNET2015_QC_FLAGS,
    OBSERVED_QC_VALUES,
    FluxnetError,
    fluxnet_column_map,
    inspect_fluxnet,
    prepare_fluxnet_frame,
    qc_summary,
    read_fluxnet_csv,
)
from rfrgapfill.leakage import observed_target_mask
from rfrgapfill.schema import ColumnMapError, Mode

FAST_GRID = {"n_estimators": (25,), "min_samples_leaf": (2,)}

FLUXNET_RENAMES = {
    "NEE": "NEE_VUT_REF",
    "H": "H_F_MDS",
    "LE": "LE_F_MDS",
    "NEE_QC": "NEE_VUT_REF_QC",
    "H_QC": "H_F_MDS_QC",
    "LE_QC": "LE_F_MDS_QC",
}


def raw_fluxnet(days: int = 20, seed: int = 5) -> pd.DataFrame:
    """Return a synthetic site in the shape a FULLSET CSV arrives in.

    FLUXNET column names, integer ``YYYYMMDDHHMM`` timestamp columns instead of
    an index, and ``-9999`` where a value is missing.
    """
    frame = synthetic_site(days=days, seed=seed).rename(columns=FLUXNET_RENAMES)
    index = frame.index
    starts = [int(stamp.strftime("%Y%m%d%H%M")) for stamp in index]
    ends = [int((stamp + pd.Timedelta("30min")).strftime("%Y%m%d%H%M")) for stamp in index]
    frame = frame.reset_index(drop=True)
    frame.insert(0, "TIMESTAMP_START", starts)
    frame.insert(1, "TIMESTAMP_END", ends)
    return frame.fillna(FLUXNET2015_MISSING_VALUE)


@pytest.fixture(scope="module")
def raw() -> pd.DataFrame:
    return raw_fluxnet()


@pytest.fixture(scope="module")
def prepared(raw: pd.DataFrame) -> pd.DataFrame:
    return prepare_fluxnet_frame(raw)


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------


def test_prepare_indexes_by_the_interval_start(raw: pd.DataFrame, prepared: pd.DataFrame) -> None:
    expected = pd.to_datetime(raw["TIMESTAMP_START"].astype(str), format="%Y%m%d%H%M")
    assert isinstance(prepared.index, pd.DatetimeIndex)
    assert prepared.index.name == "timestamp"
    assert list(prepared.index) == list(expected)


def test_prepare_drops_both_raw_timestamp_columns(prepared: pd.DataFrame) -> None:
    assert "TIMESTAMP_START" not in prepared.columns
    assert "TIMESTAMP_END" not in prepared.columns


def test_prepare_can_index_by_the_interval_end(raw: pd.DataFrame) -> None:
    by_end = prepare_fluxnet_frame(raw, timestamp="TIMESTAMP_END")
    by_start = prepare_fluxnet_frame(raw)
    assert list(by_end.index) == [stamp + pd.Timedelta("30min") for stamp in by_start.index]


def test_prepare_turns_the_sentinel_into_nan(raw: pd.DataFrame, prepared: pd.DataFrame) -> None:
    sentinel_rows = raw["LE_F_MDS"] == FLUXNET2015_MISSING_VALUE
    assert sentinel_rows.any(), "fixture must contain sentinel values to be meaningful"
    assert prepared.loc[sentinel_rows.to_numpy(), "LE_F_MDS"].isna().all()
    numeric = prepared.select_dtypes(include="number")
    assert not (numeric == FLUXNET2015_MISSING_VALUE).to_numpy().any()


def test_prepare_leaves_measurements_untouched(raw: pd.DataFrame, prepared: pd.DataFrame) -> None:
    measured = (raw["LE_F_MDS"] != FLUXNET2015_MISSING_VALUE).to_numpy()
    assert np.allclose(
        prepared.loc[measured, "LE_F_MDS"].to_numpy(),
        raw.loc[measured, "LE_F_MDS"].to_numpy(),
    )


def test_prepare_does_not_mutate_the_caller_frame(raw: pd.DataFrame) -> None:
    before = raw.copy()
    prepare_fluxnet_frame(raw)
    pd.testing.assert_frame_equal(raw, before)


def test_prepare_can_leave_the_sentinel_alone(raw: pd.DataFrame) -> None:
    kept = prepare_fluxnet_frame(raw, sentinel=None)
    assert (kept["LE_F_MDS"] == FLUXNET2015_MISSING_VALUE).any()


def test_prepare_accepts_a_frame_that_is_already_indexed(prepared: pd.DataFrame) -> None:
    again = prepare_fluxnet_frame(prepared, timestamp=None)
    assert list(again.index) == list(prepared.index)


def test_prepare_rejects_a_missing_timestamp_column(prepared: pd.DataFrame) -> None:
    with pytest.raises(FluxnetError, match="TIMESTAMP_START"):
        prepare_fluxnet_frame(prepared.reset_index(drop=True))


def test_prepare_rejects_timestamps_in_another_form(raw: pd.DataFrame) -> None:
    broken = raw.copy()
    broken["TIMESTAMP_START"] = "2004-01-01 00:30"
    with pytest.raises(FluxnetError, match=r"%Y%m%d%H%M"):
        prepare_fluxnet_frame(broken)


def test_prepare_rejects_a_non_frame() -> None:
    with pytest.raises(FluxnetError, match="DataFrame"):
        prepare_fluxnet_frame({"TIMESTAMP_START": [200401010000]})  # type: ignore[arg-type]


def test_read_csv_round_trips_a_local_file(raw: pd.DataFrame, tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "FLX_XX-Site_FLUXNET2015_FULLSET_HH.csv"
    raw.to_csv(path, index=False)
    frame = read_fluxnet_csv(path)
    assert isinstance(frame.index, pd.DatetimeIndex)
    assert frame["LE_F_MDS"].isna().any()
    assert not (frame.select_dtypes(include="number") == FLUXNET2015_MISSING_VALUE).to_numpy().any()


# ---------------------------------------------------------------------------
# Column mapping
# ---------------------------------------------------------------------------


def test_reference_column_map_names_every_driver_and_the_timestamp() -> None:
    mapping = fluxnet_column_map()
    assert mapping.column("shortwave") == "SW_IN_F"
    assert mapping.column("soil_water_content") == "SWC_F_MDS"
    assert mapping.timestamp == "TIMESTAMP_START"
    assert len(mapping) == 10


def test_reference_column_map_can_be_restricted_to_a_mode() -> None:
    mapping = fluxnet_column_map("RFR3")
    assert len(mapping) == 3
    assert mapping.missing(Mode.RFR10.drivers)


def test_inspect_finds_every_driver_flux_and_flag(prepared: pd.DataFrame) -> None:
    info = inspect_fluxnet(prepared)
    assert info.rfr10_ready and info.rfr3_ready
    assert info.best_mode is Mode.RFR10
    assert not info.missing_drivers
    assert dict(info.fluxes) == dict(FLUXNET2015_FLUXES)
    assert dict(info.qc) == dict(FLUXNET2015_FLUX_QC)


def test_inspect_reads_a_bare_iterable_of_column_names() -> None:
    info = inspect_fluxnet(FLUXNET2015_FLUXES.values())
    assert not info.rfr3_ready
    assert info.best_mode is None
    assert set(info.fluxes) == set(FLUXNET2015_FLUXES)


def test_inspect_resolves_depth_indexed_soil_columns(prepared: pd.DataFrame) -> None:
    frame = prepared.rename(columns={"TS_F_MDS": "TS_F_MDS_2", "SWC_F_MDS": "SWC_F_MDS_1"})
    info = inspect_fluxnet(frame)
    assert info.rfr10_ready
    mapping = info.column_map("RFR10")
    assert mapping.column("soil_temperature") == "TS_F_MDS_2"
    assert mapping.column("soil_water_content") == "SWC_F_MDS_1"


def test_shallowest_depth_wins_when_several_are_present(prepared: pd.DataFrame) -> None:
    frame = prepared.rename(columns={"SWC_F_MDS": "SWC_F_MDS_3"})
    frame["SWC_F_MDS_1"] = frame["SWC_F_MDS_3"]
    assert inspect_fluxnet(frame).column_map().column("soil_water_content") == "SWC_F_MDS_1"


def test_a_site_without_the_extended_drivers_falls_back_to_rfr3(prepared: pd.DataFrame) -> None:
    frame = prepared.drop(columns=["NETRAD", "SWC_F_MDS"])
    info = inspect_fluxnet(frame)
    assert info.rfr3_ready and not info.rfr10_ready
    assert info.best_mode is Mode.RFR3
    assert set(info.missing_drivers) == {"net_radiation", "soil_water_content"}
    assert len(info.column_map()) == 3


def test_requesting_an_unsupported_mode_names_the_missing_columns(prepared: pd.DataFrame) -> None:
    info = inspect_fluxnet(prepared.drop(columns=["NETRAD"]))
    with pytest.raises(ColumnMapError, match=r"net_radiation.*NETRAD"):
        info.column_map("RFR10")


def test_a_site_without_shortwave_supports_no_mode(prepared: pd.DataFrame) -> None:
    info = inspect_fluxnet(prepared.drop(columns=["SW_IN_F"]))
    assert info.best_mode is None
    with pytest.raises(ColumnMapError, match="SW_IN_F"):
        info.column_map()


def test_overrides_pin_a_column_the_search_would_not_find(prepared: pd.DataFrame) -> None:
    frame = prepared.rename(columns={"NETRAD": "NETRAD_CORRECTED"})
    info = inspect_fluxnet(frame, overrides={"net_radiation": "NETRAD_CORRECTED"})
    assert info.column_map("RFR10").column("net_radiation") == "NETRAD_CORRECTED"


def test_a_different_nee_variant_can_be_named(prepared: pd.DataFrame) -> None:
    frame = prepared.rename(columns={"NEE_VUT_REF": "NEE_CUT_REF"})
    info = inspect_fluxnet(frame, fluxes={"NEE": "NEE_CUT_REF"})
    assert info.target_columns() == ("NEE_CUT_REF",)


def test_the_timestamp_column_travels_into_the_column_map(raw: pd.DataFrame) -> None:
    assert inspect_fluxnet(raw).column_map().timestamp == "TIMESTAMP_START"


def test_an_indexed_frame_needs_no_timestamp_column(prepared: pd.DataFrame) -> None:
    assert inspect_fluxnet(prepared).column_map().timestamp is None


# ---------------------------------------------------------------------------
# Missing-variable reporting
# ---------------------------------------------------------------------------


def test_target_and_qc_columns_are_shaped_for_validate_rfr(prepared: pd.DataFrame) -> None:
    info = inspect_fluxnet(prepared)
    assert set(info.target_columns()) == set(FLUXNET2015_FLUXES.values())
    assert info.qc_columns() == {
        "NEE_VUT_REF": "NEE_VUT_REF_QC",
        "H_F_MDS": "H_F_MDS_QC",
        "LE_F_MDS": "LE_F_MDS_QC",
    }


def test_heat_targets_name_the_ebr_pair_under_fluxnet_naming(prepared: pd.DataFrame) -> None:
    info = inspect_fluxnet(prepared)
    assert info.heat_targets() == ("H_F_MDS", "LE_F_MDS")
    assert inspect_fluxnet(prepared.drop(columns=["H_F_MDS"])).heat_targets() is None


def test_a_flux_without_its_flag_maps_to_none(prepared: pd.DataFrame) -> None:
    info = inspect_fluxnet(prepared.drop(columns=["LE_F_MDS_QC"]))
    assert info.qc_columns(["LE"]) == {"LE_F_MDS": None}
    assert info.missing_qc["LE"] == "LE_F_MDS_QC"


def test_asking_for_an_absent_flux_is_an_error(prepared: pd.DataFrame) -> None:
    info = inspect_fluxnet(prepared.drop(columns=["H_F_MDS"]))
    with pytest.raises(FluxnetError, match="H"):
        info.target_columns(["H"])


def test_require_reports_a_missing_driver_and_a_missing_flag(prepared: pd.DataFrame) -> None:
    info = inspect_fluxnet(prepared.drop(columns=["NETRAD", "LE_F_MDS_QC"]))
    info.require("RFR3", fluxes=["NEE"])
    with pytest.raises(FluxnetError, match="net_radiation"):
        info.require("RFR10", fluxes=["NEE"])
    with pytest.raises(FluxnetError, match="missing QC flag for LE"):
        info.require("RFR3")
    info.require("RFR3", require_qc=False)


def test_summary_and_dict_report_what_was_found(prepared: pd.DataFrame) -> None:
    info = inspect_fluxnet(prepared.drop(columns=["NETRAD"]))
    text = info.summary()
    assert "RFR3 supported" in text
    assert "NETRAD" in text
    payload = info.to_dict()
    assert payload["best_mode"] == "RFR3"
    assert payload["rfr10_ready"] is False
    assert json.loads(json.dumps(payload))["missing_drivers"] == {"net_radiation": "NETRAD"}


# ---------------------------------------------------------------------------
# QC flag semantics
# ---------------------------------------------------------------------------


def test_only_flag_zero_counts_as_a_measurement() -> None:
    assert OBSERVED_QC_VALUES == (0,)
    assert FLUXNET2015_QC_FLAGS[0] == "measured"
    assert all("gap fill" in FLUXNET2015_QC_FLAGS[flag] for flag in (1, 2, 3))
    assert tuple(RFRConfig(mode="RFR3", latitude=0.0).observed_qc_values) == OBSERVED_QC_VALUES


def test_qc_summary_counts_each_documented_flag() -> None:
    frame = pd.DataFrame(
        {
            "LE_F_MDS": [1.0, 2.0, 3.0, 4.0, np.nan],
            "LE_F_MDS_QC": [0, 0, 1, 3, 0],
        }
    )
    summary = qc_summary(frame, "LE_F_MDS_QC", target="LE_F_MDS")
    assert dict(summary.counts) == {0: 3, 1: 1, 3: 1}
    assert summary.n_measured == 2  # the third flag-0 row has no value
    assert summary.n_prefilled == 2
    assert summary.n_missing_value == 1
    assert summary.measured_fraction == pytest.approx(0.4)
    assert "good-quality gap fill" in summary.describe()
    assert json.loads(json.dumps(summary.to_dict()))["n_measured"] == 2


def test_qc_summary_agrees_with_the_mask_the_package_actually_uses(
    prepared: pd.DataFrame,
) -> None:
    summary = qc_summary(prepared, "NEE_VUT_REF_QC", target="NEE_VUT_REF")
    mask = observed_target_mask(prepared, "NEE_VUT_REF", qc_column="NEE_VUT_REF_QC")
    assert summary.n_measured == int(mask.sum())


def test_qc_summary_rejects_the_aggregated_products_fractional_flag() -> None:
    frame = pd.DataFrame({"LE_F_MDS_QC": [0.0, 0.75, 1.0]})
    with pytest.raises(FluxnetError, match="fraction"):
        qc_summary(frame, "LE_F_MDS_QC")


def test_qc_summary_counts_unflagged_rows_separately() -> None:
    frame = pd.DataFrame({"LE_F_MDS_QC": [0, 1, np.nan]})
    summary = qc_summary(frame, "LE_F_MDS_QC")
    assert summary.n_unflagged == 1
    assert summary.n_measured == 1


def test_qc_summary_requires_the_columns_it_describes(prepared: pd.DataFrame) -> None:
    with pytest.raises(FluxnetError, match="QC column"):
        qc_summary(prepared, "NOT_A_COLUMN")
    with pytest.raises(FluxnetError, match="flux column"):
        qc_summary(prepared, "LE_F_MDS_QC", target="NOT_A_COLUMN")


# ---------------------------------------------------------------------------
# The exit criterion: a FLUXNET frame reaches the generic API in a few lines
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_a_raw_fluxnet_frame_can_be_filled_in_a_few_lines(raw: pd.DataFrame) -> None:
    frame = prepare_fluxnet_frame(raw)
    info = inspect_fluxnet(frame)
    config = RFRConfig(
        mode=info.best_mode,
        frequency="30min",
        latitude=51.5,
        column_map=info.column_map(),
        hyperparameter_grid=FAST_GRID,
        cv_folds=3,
    )
    target = FLUXNET2015_FLUXES["LE"]
    filler = RFRGapFiller(config).fit(frame, target=target, qc_column=info.qc_columns()[target])
    result = filler.fill(frame)

    assert result.report.filled_rows > 0
    measured = observed_target_mask(frame, target, qc_column=FLUXNET2015_FLUX_QC["LE"])
    assert np.allclose(
        result.frame.loc[measured, f"{target}_filled"].to_numpy(),
        frame.loc[measured, target].to_numpy(),
    )
