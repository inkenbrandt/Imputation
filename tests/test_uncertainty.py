"""Bias-spread uncertainty diagnostics. Covers Step 18A and acceptance test 40.

Three parts. The first re-reads ``docs/supplement_benchmarks.md`` and checks the
Table S8 ranges against it, exactly as ``test_benchmarks.py`` does for Table S3,
and checks that no published range can be taken for a reproduction target - the
exit criterion of Step 18A.

The second builds tidy tables by hand, because what :func:`bias_iqr` gets right
or wrong is grouping and quartiles, and five sites with biases chosen to land on
order statistics say that more plainly than a fitted model does.

The last runs the real thing, so the hand-built shape is proven to be the shape
a validation run has.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd
import pytest

import rfrgapfill
from rfrgapfill.benchmarks import PUBLISHED_BENCHMARKS
from rfrgapfill.config import FeatureConfig
from rfrgapfill.schema import GapClass
from rfrgapfill.sensitivity import GAP_CLASS_ORDER, SensitivityError, gap_length_table
from rfrgapfill.synthetic import synthetic_site
from rfrgapfill.uncertainty import (
    BIAS_IQR_TABLE_COLUMNS,
    EXPERIMENTAL_STATUS,
    POOLED_IGBP,
    TABLE_S8_RANGES,
    UNCERTAINTY_RANGE_TABLE_COLUMNS,
    PublishedUncertaintyRange,
    UncertaintyError,
    bias_iqr,
    published_uncertainty_table,
)
from rfrgapfill.validation import BIAS_SPREAD_TABLE_COLUMNS, validate_rfr

BENCHMARK_DOC = Path(__file__).resolve().parents[1] / "docs" / "supplement_benchmarks.md"

FLUXES = ("NEE", "H", "LE")
METHODS = ("MDS", "RFR3", "RFR10")

#: One grid point and three folds. Nothing here scores predictive quality.
TEST_GRID = {"n_estimators": (15,)}

#: The reaching strategy a long-gap run has to choose deliberately (A4).
REACHING = FeatureConfig(daily_statistic_strategy="rolling_available")


# ---------------------------------------------------------------------------
# Table S8 as documentation
# ---------------------------------------------------------------------------


def documented_s8() -> dict[tuple[str, str], tuple[float, float]]:
    """Return the Table S8 ranges as ``docs/supplement_benchmarks.md`` records them."""
    text = BENCHMARK_DOC.read_text(encoding="utf-8")
    blocks = [block for block in re.split(r"^## ", text, flags=re.MULTILINE)]
    matching = [block for block in blocks if block.startswith("7. Uncertainty diagnostics")]
    assert len(matching) == 1, "the benchmark document has no section 7"
    found: dict[tuple[str, str], tuple[float, float]] = {}
    for line in matching[0].splitlines():
        if not line.startswith("|"):
            continue
        row = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(row) != 4 or row[0] not in FLUXES:
            continue
        for method, interval in zip(METHODS, row[1:], strict=True):
            # The document writes each interval with an en dash.
            lower, upper = interval.split("\N{EN DASH}")
            found[row[0], method] = (float(lower), float(upper))
    assert len(found) == 9, f"expected 9 Table S8 ranges, read {len(found)}"
    return found


class TestTableS8IsTraceable:
    """The constants and the benchmark document say the same thing."""

    def test_every_range_matches_the_document(self):
        documented = documented_s8()
        carried = {(r.target, r.method): (r.lower, r.upper) for r in TABLE_S8_RANGES}
        assert carried == documented

    def test_every_range_is_the_very_long_class_of_table_s8(self):
        for record in TABLE_S8_RANGES:
            assert record.gap_class == GapClass.VERY_LONG.value
            assert "Table S8" in record.source


class TestTableS8CannotBeMistakenForAReproduction:
    """Step 18A's exit criterion, made checkable."""

    def test_every_published_range_is_experimental(self):
        assert {record.status for record in TABLE_S8_RANGES} == {EXPERIMENTAL_STATUS}

    def test_the_status_is_not_a_setting(self):
        # There is no field to set: a record cannot be built as "verified".
        with pytest.raises(TypeError):
            PublishedUncertaintyRange(  # type: ignore[call-arg]
                target="NEE",
                method="RFR3",
                gap_class="very_long",
                lower=1.0,
                upper=2.0,
                status="verified",
            )

    def test_every_row_of_the_table_carries_the_caveat(self):
        table = published_uncertainty_table()
        assert tuple(table.columns) == UNCERTAINTY_RANGE_TABLE_COLUMNS
        assert set(table["status"]) == {EXPERIMENTAL_STATUS}
        assert all("A8" in caveat for caveat in table["caveat"])

    def test_the_ranges_are_not_reproduction_benchmarks(self):
        # compare_to_benchmarks() reads PUBLISHED_BENCHMARKS only, so keeping
        # Table S8 out of it is what stops a run being differenced against it.
        assert not any("S8" in record.source for record in PUBLISHED_BENCHMARKS)

    def test_no_public_name_computes_a_normalized_ratio_unlabelled(self):
        # The tripwire for whoever implements the Table S8 ratio: it may only
        # arrive with "experimental" in its name until reproduced (A8).
        normalized = [name for name in rfrgapfill.__all__ if "normali" in name.lower()]
        assert all("experimental" in name.lower() for name in normalized), normalized

    def test_a_swapped_interval_is_refused(self):
        with pytest.raises(UncertaintyError, match="swapped"):
            PublishedUncertaintyRange(
                target="NEE", method="RFR3", gap_class="very_long", lower=2.97, upper=2.90
            )


# ---------------------------------------------------------------------------
# Hand-built results
# ---------------------------------------------------------------------------


def arm(
    *,
    bias: float | None,
    target: str = "NEE",
    method: str = "RFR3",
    gap_classes: tuple[str, ...] = ("all", "very_long"),
    subsets: tuple[str, ...] = ("all",),
) -> pd.DataFrame:
    """Return the tidy table one arm of one site's run would produce."""
    return pd.DataFrame(
        [
            {
                "target": target,
                "method": method,
                "mode": "RFR3" if method.endswith("3") else "RFR10",
                "gap_class": gap_class,
                "subset": subset,
                "n": 100,
                "n_offered": 105,
                "r2": 0.8,
                "slope": 0.9,
                "rmse": 2.5,
                "bias": bias,
            }
            for gap_class in gap_classes
            for subset in subsets
        ]
    )


#: Five sites whose biases land exactly on the quartile order statistics:
#: q1 = -1, median = 0, q3 = 1, IQR = 2.
FIVE = {"A": -2.0, "B": -1.0, "C": 0.0, "D": 1.0, "E": 4.0}


def five_sites(**overrides: float | None) -> dict[str, pd.DataFrame]:
    return {site: arm(bias=overrides.get(site, bias)) for site, bias in FIVE.items()}


def cell(table: pd.DataFrame, **where: object) -> pd.Series:
    """Return the single row of ``table`` matching ``where``."""
    selected = table
    for column, value in where.items():
        selected = selected.loc[selected[column].astype(str) == str(value)]
    assert len(selected) == 1, f"expected one row for {where}, found {len(selected)}"
    return selected.iloc[0]


# ---------------------------------------------------------------------------
# Bias IQR across sites
# ---------------------------------------------------------------------------


class TestBiasIQR:
    """What :func:`bias_iqr` guarantees."""

    def test_the_quartiles_are_those_of_the_per_site_biases(self):
        row = cell(bias_iqr(five_sites()), gap_class="very_long", igbp="all")
        assert row["bias_q1"] == pytest.approx(-1.0)
        assert row["bias_median"] == pytest.approx(0.0)
        assert row["bias_q3"] == pytest.approx(1.0)
        assert row["bias_iqr"] == pytest.approx(2.0)
        assert row["n_sites"] == 5

    def test_the_quartiles_use_linear_interpolation(self):
        # A11: the numpy/pandas default, the same convention as every quantile
        # the package reports.
        biases = [0.3, -1.7, 2.2, 0.9]
        table = bias_iqr({str(i): arm(bias=value) for i, value in enumerate(biases)})
        q1, q3 = np.percentile(biases, (25.0, 75.0))
        assert cell(table, gap_class="all", igbp="all")["bias_iqr"] == pytest.approx(q3 - q1)

    def test_columns_are_the_documented_order(self):
        assert tuple(bias_iqr(five_sites()).columns) == BIAS_IQR_TABLE_COLUMNS

    def test_gap_classes_are_kept_apart_and_in_duration_order(self):
        sites = {
            site: pd.concat(
                [
                    arm(bias=bias, gap_classes=("short",)),
                    arm(bias=10 * bias, gap_classes=("very_long",)),
                ],
                ignore_index=True,
            )
            for site, bias in FIVE.items()
        }
        table = bias_iqr(sites)
        assert list(table["gap_class"]) == ["short", "very_long"]
        assert cell(table, gap_class="short")["bias_iqr"] == pytest.approx(2.0)
        assert cell(table, gap_class="very_long")["bias_iqr"] == pytest.approx(20.0)

    def test_methods_and_targets_are_kept_apart(self):
        table = bias_iqr(
            {
                "A": [arm(bias=0.0), arm(bias=0.0, method="RFR10"), arm(bias=5.0, target="H")],
                "B": [arm(bias=2.0), arm(bias=1.0, method="RFR10"), arm(bias=9.0, target="H")],
            }
        )
        pooled = table.loc[table["gap_class"] == "all"]
        assert cell(pooled, target="NEE", method="RFR3")["bias_iqr"] == pytest.approx(1.0)
        assert cell(pooled, target="NEE", method="RFR10")["bias_iqr"] == pytest.approx(0.5)
        assert cell(pooled, target="H", method="RFR3")["bias_iqr"] == pytest.approx(2.0)

    def test_units_are_never_pooled_across(self):
        table = pd.concat(
            [
                gap_length_table(arm(bias=1.0), site="A"),
                gap_length_table(arm(bias=2.0), site="B", units={"NEE": "g C m-2 d-1"}),
            ],
            ignore_index=True,
        )
        pooled = bias_iqr(table).loc[lambda frame: frame["gap_class"] == "all"]
        assert set(pooled["units"]) == {"umol m-2 s-1", "g C m-2 d-1"}
        assert bool(pooled["bias_iqr"].isna().all())

    def test_a_site_without_a_bias_is_skipped_not_counted_as_zero(self):
        row = cell(
            bias_iqr({"A": arm(bias=None), "B": arm(bias=1.0), "C": arm(bias=3.0)}), gap_class="all"
        )
        assert row["n_sites"] == 2
        assert row["n_sites_offered"] == 3
        assert row["bias_iqr"] == pytest.approx(1.0)

    def test_one_site_has_quartiles_but_no_spread(self):
        row = cell(bias_iqr({"A": arm(bias=0.7)}), gap_class="all")
        assert row["bias_q1"] == row["bias_median"] == row["bias_q3"] == pytest.approx(0.7)
        assert pd.isna(row["bias_iqr"])

    def test_no_site_with_a_bias_leaves_everything_undefined(self):
        table = bias_iqr({"A": arm(bias=None), "B": arm(bias=None)})
        for column in ("bias_q1", "bias_median", "bias_q3", "bias_iqr"):
            assert table[column].dtype == "float64"
            assert bool(table[column].isna().all())

    def test_an_unlabelled_row_is_refused(self):
        # One unlabelled run is one site; its spread is across gaps, and the
        # message points there.
        with pytest.raises(UncertaintyError, match="bias_spread_frame"):
            bias_iqr(arm(bias=1.0))

    def test_one_site_contributing_a_cell_twice_is_refused(self):
        with pytest.raises(UncertaintyError, match="twice"):
            bias_iqr({"A": [arm(bias=1.0), arm(bias=2.0)], "B": arm(bias=0.0)})

    def test_its_errors_are_sensitivity_errors(self):
        assert issubclass(UncertaintyError, SensitivityError)

    def test_nothing_at_all_gives_an_empty_table_of_the_right_shape(self):
        table = bias_iqr([])
        assert table.empty
        assert tuple(table.columns) == BIAS_IQR_TABLE_COLUMNS
        assert table["bias_iqr"].dtype == "float64"


class TestBiasIQRByEcosystem:
    """IGBP stratification, when the metadata is available."""

    IGBP: ClassVar[dict[str, str]] = {"A": "ENF", "B": "ENF", "C": "ENF", "D": "GRA", "E": "GRA"}

    def test_without_metadata_every_row_pools_every_class(self):
        assert set(bias_iqr(five_sites())["igbp"]) == {POOLED_IGBP}

    def test_each_class_is_spread_separately(self):
        table = bias_iqr(five_sites(), igbp=self.IGBP)
        enf = cell(table, gap_class="all", igbp="ENF")
        gra = cell(table, gap_class="all", igbp="GRA")
        assert (enf["bias_q1"], enf["bias_q3"], enf["bias_iqr"]) == pytest.approx((-1.5, -0.5, 1.0))
        assert (gra["bias_q1"], gra["bias_q3"], gra["bias_iqr"]) == pytest.approx((1.75, 3.25, 1.5))
        assert enf["n_sites"] == 3

    def test_the_pooled_row_is_unchanged_by_stratifying(self):
        plain = cell(bias_iqr(five_sites()), gap_class="all", igbp="all")
        stratified = cell(bias_iqr(five_sites(), igbp=self.IGBP), gap_class="all", igbp="all")
        assert stratified["bias_iqr"] == pytest.approx(plain["bias_iqr"])

    def test_the_pooled_row_comes_first(self):
        table = bias_iqr(five_sites(), igbp=self.IGBP)
        pooled = table.loc[table["gap_class"] == "all", "igbp"]
        assert list(pooled) == ["all", "ENF", "GRA"]

    def test_a_site_without_a_class_is_pooled_only(self):
        table = bias_iqr(five_sites(), igbp={"A": "ENF", "B": "ENF", "C": None})
        assert set(table["igbp"]) == {"all", "ENF"}
        assert cell(table, gap_class="all", igbp="ENF")["n_sites_offered"] == 2
        assert cell(table, gap_class="all", igbp="all")["n_sites_offered"] == 5

    def test_site_metadata_can_be_the_supplement_table(self):
        metadata = pd.DataFrame(
            {"site_id": list(self.IGBP), "IGBP": list(self.IGBP.values()), "country": "x"}
        )
        from_frame = bias_iqr(five_sites(), igbp=metadata)
        from_mapping = bias_iqr(five_sites(), igbp=self.IGBP)
        pd.testing.assert_frame_equal(from_frame, from_mapping)

    def test_metadata_without_an_igbp_column_is_refused(self):
        with pytest.raises(UncertaintyError, match="IGBP column"):
            bias_iqr(five_sites(), igbp=pd.DataFrame({"site_id": ["A"], "biome": ["ENF"]}))

    def test_a_site_given_two_classes_is_refused(self):
        metadata = pd.DataFrame({"site": ["A", "A"], "igbp": ["ENF", "GRA"]})
        with pytest.raises(UncertaintyError, match="two IGBP classes"):
            bias_iqr(five_sites(), igbp=metadata)

    def test_the_pooled_label_is_not_a_class(self):
        with pytest.raises(UncertaintyError, match="pooling every class"):
            bias_iqr(five_sites(), igbp={"A": "all"})


# ---------------------------------------------------------------------------
# Real validation results
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def report():
    """One RFR3 arm over NEE on the reference synthetic site's known gaps."""
    site = synthetic_site()
    return validate_rfr(
        site.frame,
        config=site.config("RFR3", hyperparameter_grid=TEST_GRID, cv_folds=3, features=REACHING),
        targets=["NEE"],
        qc_columns={"NEE": site.qc_column("NEE")},
        gaps=site.known_gaps(),
    )


class TestFromValidationRuns:
    """Both populations, from a real run."""

    def test_the_within_site_frame_is_the_bias_spread_the_result_carries(self, report):
        result = report["NEE"]
        frame = result.bias_spread_frame()
        assert tuple(frame.columns) == BIAS_SPREAD_TABLE_COLUMNS
        for gap_class, spread in result.bias_spread.items():
            row = cell(frame, gap_class=gap_class.value)
            assert row["n_gaps"] == spread.n_gaps
            for column, value in (("bias_median", spread.median), ("bias_iqr", spread.iqr)):
                if value is None:
                    assert pd.isna(row[column])
                else:
                    assert row[column] == pytest.approx(value)

    def test_the_within_site_frame_lists_the_classes_in_duration_order(self, report):
        classes = list(report.bias_spread_frame()["gap_class"])
        assert classes == [name for name in GAP_CLASS_ORDER if name in classes]

    def test_the_run_is_one_site_of_an_across_site_table(self, report):
        # The same run relabelled as two sites: identical biases, so a spread of
        # exactly zero, with both sites counted.
        table = bias_iqr({"A": report, "B": report})
        row = cell(table, gap_class="all", subset="all")
        assert row["n_sites"] == 2
        assert row["bias_median"] == pytest.approx(report["NEE"].overall.bias)
        assert row["bias_iqr"] == pytest.approx(0.0)
