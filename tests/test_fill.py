"""Operational fit/fill tests. Covers acceptance tests 22-25.

The invariants here are the conservative ones: an observed value is never
replaced, a row without predictors is never filled from an imputed input, and
the caller's frame is never mutated (``docs/method_spec.md`` section 7).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fixtures import synthetic_site
from rfrgapfill.config import ColumnMap, RFRConfig
from rfrgapfill.features import FeatureError
from rfrgapfill.fill import RFRGapFiller
from rfrgapfill.provenance import fill_column_names, observed_mask
from rfrgapfill.schema import ColumnMapError

FAST_GRID = {"n_estimators": (25,), "min_samples_leaf": (2,)}


def config(mode: str = "RFR3", **changes: object) -> RFRConfig:
    settings: dict[str, object] = {
        "mode": mode,
        "frequency": "30min",
        "latitude": 51.5,
        "column_map": ColumnMap.fluxnet2015(mode),
        "hyperparameter_grid": FAST_GRID,
        "cv_folds": 3,
    }
    settings.update(changes)
    return RFRConfig(**settings)  # type: ignore[arg-type]


@pytest.fixture(scope="module")
def site() -> pd.DataFrame:
    return synthetic_site(days=60, seed=3)


@pytest.fixture(scope="module")
def fitted(site: pd.DataFrame) -> RFRGapFiller:
    return RFRGapFiller(config()).fit(site, target="LE", qc_col="LE_QC")


# ---------------------------------------------------------------------------
# Driver requirements (acceptance tests 22-23)
# ---------------------------------------------------------------------------


def test_rfr3_needs_exactly_its_three_drivers_mapped(site: pd.DataFrame) -> None:
    partial = ColumnMap(variables={"shortwave": "SW_IN_F", "vpd": "VPD_F_MDS"})
    with pytest.raises(ColumnMapError, match="air_temperature"):
        RFRGapFiller(config(column_map=None)).fit(site, target="LE", column_map=partial)


def test_rfr10_needs_all_ten_drivers_mapped(site: pd.DataFrame) -> None:
    with pytest.raises(ColumnMapError, match="net_radiation"):
        RFRGapFiller(config("RFR10", column_map=None)).fit(
            site, target="LE", column_map=ColumnMap.fluxnet2015("RFR3")
        )


def test_a_mapped_column_absent_from_the_data_fails_before_fitting(
    site: pd.DataFrame,
) -> None:
    with pytest.raises(FeatureError, match="TA_F_MDS"):
        RFRGapFiller(config()).fit(site.drop(columns=["TA_F_MDS"]), target="LE")


# ---------------------------------------------------------------------------
# Preservation (acceptance tests 24-25)
# ---------------------------------------------------------------------------


def test_observed_values_are_never_replaced(site: pd.DataFrame, fitted: RFRGapFiller) -> None:
    result = fitted.fill()
    observed = site["LE"].notna()

    pd.testing.assert_series_equal(
        result.frame.loc[observed, "LE_filled"],
        site.loc[observed, "LE"],
        check_names=False,
    )


def test_only_missing_rows_receive_a_prediction(site: pd.DataFrame, fitted: RFRGapFiller) -> None:
    result = fitted.fill()
    observed = site["LE"].notna()

    assert not result.frame.loc[observed, "LE_is_filled"].any()
    assert result.n_filled > 0
    assert result.n_filled == int(result.frame["LE_is_filled"].sum())


def test_the_original_column_is_returned_untouched(
    site: pd.DataFrame, fitted: RFRGapFiller
) -> None:
    result = fitted.fill()
    pd.testing.assert_series_equal(result.frame["LE_original"], site["LE"], check_names=False)


def test_the_input_frame_is_not_mutated(site: pd.DataFrame) -> None:
    before = site.copy()
    filler = RFRGapFiller(config()).fit(site, target="LE", qc_col="LE_QC")
    filler.fill()
    pd.testing.assert_frame_equal(site, before)


def test_a_row_missing_a_driver_stays_unfilled(site: pd.DataFrame) -> None:
    frame = site.copy()
    gap = frame["LE"].isna().to_numpy().nonzero()[0][:20]
    frame.iloc[gap, frame.columns.get_loc("VPD_F_MDS")] = np.nan

    result = RFRGapFiller(config()).fit(frame, target="LE", qc_col="LE_QC").fill()

    assert not result.frame["LE_is_filled"].to_numpy()[gap].any()
    assert result.n_unfilled >= len(gap)
    assert (
        result.frame["LE_filled"].to_numpy()[gap[0]] != result.frame["LE_filled"].to_numpy()[gap[0]]
    )  # still NaN


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


def test_filling_returns_every_documented_provenance_column(fitted: RFRGapFiller) -> None:
    result = fitted.fill()
    for name in fill_column_names("LE"):
        assert name in result.frame.columns


def test_the_fill_method_and_model_version_are_written_only_where_filled(
    fitted: RFRGapFiller,
) -> None:
    result = fitted.fill()
    filled = result.frame["LE_is_filled"]

    assert (result.frame.loc[filled, "LE_fill_method"] == "RFR").all()
    assert result.frame.loc[~filled, "LE_fill_method"].isna().all()
    assert set(result.frame.loc[filled, "LE_model_version"]) == {result.model_version}


def test_the_qc_flag_decides_which_values_count_as_measured(site: pd.DataFrame) -> None:
    # Every LE value flagged 1 was already removed by the fixture, so the mask
    # with and without the flag agree here; a value present but flagged must not.
    frame = site.copy()
    frame.loc[frame.index[:5], "LE"] = 42.0
    frame.loc[frame.index[:5], "LE_QC"] = 1

    with_flag = observed_mask(frame, "LE", qc_column="LE_QC")
    without_flag = observed_mask(frame, "LE")

    assert not with_flag.iloc[:5].any()
    assert without_flag.iloc[:5].all()


def test_the_manifest_records_the_configuration_and_the_training_report(
    fitted: RFRGapFiller,
) -> None:
    import json

    manifest = fitted.manifest()
    assert json.loads(json.dumps(manifest)) == manifest
    assert manifest["config"]["mode"] == "RFR3"
    assert manifest["training"]["LE"]["n_trained"] > 0
    assert manifest["environment"]["scikit_learn"]
    assert manifest["qc_columns"] == {"LE": "LE_QC"}


def test_uncertainty_is_optional_and_reported_only_where_filled(
    fitted: RFRGapFiller,
) -> None:
    result = fitted.fill(include_uncertainty=True)
    spread = result.frame["LE_prediction_std"]

    assert spread.notna().sum() == result.n_filled
    assert (spread.dropna() >= 0).all()
