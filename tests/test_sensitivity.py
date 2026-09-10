"""Gap-length sensitivity tables. Covers Step 18's tidy-table requirement.

Two halves. Most of the module builds tidy frames by hand, because the questions
here are about *labelling, ordering and aggregating* - a median across sites is
wrong or right regardless of which forest produced the numbers, and a fixture
that fits on a screen says so more clearly than a fitted model does.

The last class runs the real thing: two arms of :func:`validate_rfr` over the
synthetic site, aggregated into one table, checked cell by cell against the
metrics the results themselves carry. That is what makes the hand-built fixtures
above legitimate - it proves the shape they imitate is the shape a real run has.
"""

from __future__ import annotations

import pandas as pd
import pytest

from rfrgapfill.config import FeatureConfig, MetricSubset
from rfrgapfill.schema import GapClass
from rfrgapfill.sensitivity import (
    DEFAULT_TARGET_UNITS,
    GAP_CLASS_ORDER,
    SENSITIVITY_TABLE_COLUMNS,
    SITE_MEDIAN_TABLE_COLUMNS,
    SUBSET_ORDER,
    SensitivityError,
    gap_length_pivot,
    gap_length_table,
    median_across_sites,
)
from rfrgapfill.synthetic import synthetic_site
from rfrgapfill.validation import validate_rfr

#: One grid point and three folds. Nothing here scores predictive quality.
TEST_GRID = {"n_estimators": (15,)}

#: The reaching strategy a long-gap run has to choose deliberately (A4).
REACHING = FeatureConfig(daily_statistic_strategy="rolling_available")


# ---------------------------------------------------------------------------
# Hand-built results
# ---------------------------------------------------------------------------


def tidy(
    *,
    target: str = "NEE",
    method: str = "RFR3",
    mode: str | None = None,
    r2: float | None = 0.80,
    slope: float | None = 0.90,
    rmse: float | None = 2.5,
    bias: float | None = -0.01,
    n: int = 100,
    gap_classes: tuple[str, ...] = GAP_CLASS_ORDER,
    subsets: tuple[str, ...] = SUBSET_ORDER,
) -> pd.DataFrame:
    """Return the tidy table one arm of one run would produce."""
    return pd.DataFrame(
        [
            {
                "target": target,
                "method": method,
                "mode": mode or ("RFR3" if method.endswith("3") else "RFR10"),
                "gap_class": gap_class,
                "subset": subset,
                "n": n,
                "n_offered": n + 5,
                "r2": r2,
                "slope": slope,
                "rmse": rmse,
                "bias": bias,
            }
            for gap_class in gap_classes
            for subset in subsets
        ]
    )


def cell(table: pd.DataFrame, **where: object) -> pd.Series:
    """Return the single row of ``table`` matching ``where``."""
    selected = table
    for column, value in where.items():
        selected = selected.loc[selected[column].astype(str) == str(value)]
    assert len(selected) == 1, f"expected one row for {where}, found {len(selected)}"
    return selected.iloc[0]


# ---------------------------------------------------------------------------
# The tidy table
# ---------------------------------------------------------------------------


class TestGapLengthTable:
    """What :func:`gap_length_table` guarantees about its output."""

    def test_columns_are_the_documented_order(self):
        table = gap_length_table(tidy(), site="US-Ha1")
        assert tuple(table.columns) == SENSITIVITY_TABLE_COLUMNS

    def test_gap_classes_come_back_in_duration_order(self):
        # The whole point of the table: alphabetical order would put `long`
        # before `short` and make every sensitivity plot read backwards.
        table = gap_length_table(tidy(subsets=("all",)), site="US-Ha1")
        assert list(table["gap_class"]) == list(GAP_CLASS_ORDER)

    def test_subsets_come_back_in_reporting_order(self):
        table = gap_length_table(tidy(gap_classes=("short",)), site="US-Ha1")
        assert list(table["subset"]) == list(SUBSET_ORDER)

    def test_methods_come_back_in_published_order(self):
        table = gap_length_table(
            [tidy(method="ORF3"), tidy(method="RFR10"), tidy(method="RFR3")],
            site="US-Ha1",
        )
        assert list(dict.fromkeys(table["method"])) == ["RFR3", "RFR10", "ORF3"]

    def test_an_all_undefined_metric_is_still_a_float_column(self):
        # A run that scored nothing anywhere gives a column of `None`, which
        # pandas would otherwise hand on as object dtype - and a later median
        # over object dtype raises instead of reporting "no value".
        table = gap_length_table(tidy(r2=None), site="US-Ha1")
        assert table["r2"].dtype == "float64"
        assert bool(table["r2"].isna().all())

    def test_undefined_stays_undefined_rather_than_zero(self):
        table = gap_length_table(tidy(rmse=None), site="US-Ha1")
        assert bool(table["rmse"].isna().all())
        assert not bool((table["rmse"] == 0.0).any())

    def test_units_default_to_the_model_units_of_each_flux(self):
        table = gap_length_table([tidy(target="NEE"), tidy(target="H")], site="US-Ha1")
        assert (
            cell(table, target="NEE", gap_class="all", subset="all")["units"]
            == (DEFAULT_TARGET_UNITS["NEE"])
        )
        assert cell(table, target="H", gap_class="all", subset="all")["units"] == "W m-2"

    def test_units_can_be_stated_by_the_caller(self):
        table = gap_length_table(tidy(target="NEE"), site="US-Ha1", units={"NEE": "g C m-2 d-1"})
        assert set(table["units"]) == {"g C m-2 d-1"}

    def test_a_target_with_no_documented_units_gets_none(self):
        table = gap_length_table(tidy(target="CH4"), site="US-Ha1")
        assert bool(table["units"].isna().all())

    def test_a_mapping_labels_every_result_with_its_site(self):
        table = gap_length_table({"US-Ha1": tidy(), "FI-Hyy": [tidy(method="RFR10")]})
        assert set(table["site"]) == {"US-Ha1", "FI-Hyy"}
        assert set(table.loc[table["site"] == "FI-Hyy", "method"]) == {"RFR10"}

    def test_an_unlabelled_run_has_no_site(self):
        table = gap_length_table(tidy())
        assert bool(table["site"].isna().all())

    def test_a_site_label_overrides_the_one_a_saved_table_carried(self):
        saved = gap_length_table(tidy(), site="old")
        relabelled = gap_length_table(saved, site="new")
        assert set(relabelled["site"]) == {"new"}

    def test_a_saved_tables_own_labels_survive(self):
        saved = gap_length_table(tidy(), site="US-Ha1")
        assert set(gap_length_table(saved)["site"]) == {"US-Ha1"}

    def test_site_and_a_mapping_together_are_refused(self):
        with pytest.raises(SensitivityError, match="not both"):
            gap_length_table({"US-Ha1": tidy()}, site="FI-Hyy")

    def test_nothing_at_all_gives_an_empty_table_of_the_right_shape(self):
        table = gap_length_table([])
        assert table.empty
        assert tuple(table.columns) == SENSITIVITY_TABLE_COLUMNS
        assert table["r2"].dtype == "float64"

    def test_a_frame_missing_a_required_column_names_it(self):
        incomplete = tidy().drop(columns=["slope"])
        with pytest.raises(SensitivityError, match="slope"):
            gap_length_table(incomplete)

    def test_something_that_is_not_a_result_is_refused(self):
        with pytest.raises(SensitivityError, match="ValidationReport"):
            gap_length_table(42)

    def test_a_nested_mapping_is_refused(self):
        with pytest.raises(SensitivityError, match="nested mappings"):
            gap_length_table({"US-Ha1": {"RFR3": tidy()}})


# ---------------------------------------------------------------------------
# Across sites
# ---------------------------------------------------------------------------


class TestMedianAcrossSites:
    """The aggregation Supplementary Table S3 reports."""

    @staticmethod
    def three_sites() -> pd.DataFrame:
        return gap_length_table(
            {
                "A": tidy(r2=0.60, n=10),
                "B": tidy(r2=0.90, n=20),
                "C": tidy(r2=0.70, n=30),
            }
        )

    def test_the_median_is_the_middle_site(self):
        medians = median_across_sites(self.three_sites())
        assert cell(medians, gap_class="all", subset="all")["r2"] == pytest.approx(0.70)

    def test_columns_are_the_documented_order(self):
        medians = median_across_sites(self.three_sites())
        assert tuple(medians.columns) == SITE_MEDIAN_TABLE_COLUMNS

    def test_site_and_row_counts_are_carried(self):
        row = cell(median_across_sites(self.three_sites()), gap_class="all", subset="all")
        assert row["n_sites"] == 3
        assert row["n"] == 60

    def test_a_site_without_a_value_is_skipped_not_counted_as_zero(self):
        table = gap_length_table({"A": tidy(r2=0.60), "B": tidy(r2=None), "C": tidy(r2=0.80)})
        row = cell(median_across_sites(table), gap_class="all", subset="all")
        # The median of the two sites that had a value, not of three with a zero.
        assert row["r2"] == pytest.approx(0.70)
        assert row["n_sites"] == 3

    def test_a_metric_no_site_defined_stays_undefined(self):
        table = gap_length_table({"A": tidy(r2=None), "B": tidy(r2=None)})
        assert bool(median_across_sites(table)["r2"].isna().all())

    def test_arms_are_kept_apart(self):
        table = gap_length_table(
            {
                "A": [tidy(method="RFR3", r2=0.60), tidy(method="RFR10", r2=0.80)],
                "B": [tidy(method="RFR3", r2=0.70), tidy(method="RFR10", r2=0.90)],
            }
        )
        medians = median_across_sites(table)
        assert cell(medians, method="RFR3", gap_class="all", subset="all")["r2"] == pytest.approx(
            0.65
        )
        assert cell(medians, method="RFR10", gap_class="all", subset="all")["r2"] == pytest.approx(
            0.85
        )

    def test_gap_classes_are_kept_apart(self):
        table = gap_length_table(
            {
                site: pd.concat(
                    [
                        tidy(gap_classes=("short",), r2=short),
                        tidy(gap_classes=("very_long",), r2=long_),
                    ],
                    ignore_index=True,
                )
                for site, short, long_ in (("A", 0.90, 0.50), ("B", 0.80, 0.40))
            }
        )
        medians = median_across_sites(table)
        assert cell(medians, gap_class="short", subset="all")["r2"] == pytest.approx(0.85)
        assert cell(medians, gap_class="very_long", subset="all")["r2"] == pytest.approx(0.45)

    def test_units_are_never_averaged_across(self):
        # Two sites reporting NEE in different units are two rows, not one
        # meaningless median between molar and carbon flux.
        table = pd.concat(
            [
                gap_length_table(tidy(rmse=2.5), site="A"),
                gap_length_table(tidy(rmse=2.6), site="B", units={"NEE": "g C m-2 d-1"}),
            ],
            ignore_index=True,
        )
        medians = median_across_sites(table)
        assert set(medians["units"]) == {"umol m-2 s-1", "g C m-2 d-1"}

    def test_an_unlabelled_row_is_refused(self):
        with pytest.raises(SensitivityError, match="no site label"):
            median_across_sites(gap_length_table(tidy()))

    def test_one_site_contributing_a_cell_twice_is_refused(self):
        # Two runs of one arm at one site would weight that site twice.
        doubled = gap_length_table({"A": [tidy(), tidy()]})
        with pytest.raises(SensitivityError, match="weight those sites twice"):
            median_across_sites(doubled)

    def test_an_empty_table_aggregates_to_an_empty_table(self):
        medians = median_across_sites(gap_length_table([]))
        assert medians.empty
        assert tuple(medians.columns) == SITE_MEDIAN_TABLE_COLUMNS


# ---------------------------------------------------------------------------
# The readable view
# ---------------------------------------------------------------------------


class TestGapLengthPivot:
    """The target-by-method view the paper's comparison is quoted in."""

    @staticmethod
    def two_arms() -> pd.DataFrame:
        return gap_length_table(
            [
                tidy(target="NEE", method="RFR3", r2=0.78),
                tidy(target="NEE", method="RFR10", r2=0.84),
                tidy(target="H", method="RFR3", r2=0.78),
                tidy(target="H", method="RFR10", r2=0.90),
            ],
            site="US-Ha1",
        )

    def test_one_class_gives_targets_down_and_methods_across(self):
        wide = gap_length_pivot(self.two_arms(), metric="r2", gap_class="all")
        assert list(wide.index) == ["H", "NEE"]
        assert list(wide.columns) == ["RFR3", "RFR10"]
        assert wide.loc["H", "RFR10"] == pytest.approx(0.90)

    def test_every_class_keeps_the_durations_in_order(self):
        wide = gap_length_pivot(self.two_arms(), metric="r2")
        classes = [str(gap_class) for _, gap_class in wide.index]
        assert classes[: len(GAP_CLASS_ORDER)] == list(GAP_CLASS_ORDER)

    def test_targets_appear_in_the_order_they_were_asked_for(self):
        wide = gap_length_pivot(self.two_arms(), metric="r2", gap_class="all", targets=["NEE", "H"])
        assert list(wide.index) == ["NEE", "H"]

    def test_an_unknown_metric_is_refused(self):
        with pytest.raises(SensitivityError, match="unknown metric"):
            gap_length_pivot(self.two_arms(), metric="mape")

    def test_an_empty_selection_says_what_was_asked_for(self):
        with pytest.raises(SensitivityError, match="nothing to show"):
            gap_length_pivot(self.two_arms(), metric="r2", gap_class="short", targets=["LE"])

    def test_a_multi_site_table_is_refused_rather_than_averaged(self):
        # Silently averaging two sites into one cell is the error this exists to
        # surface: the published values are medians, and a mean is not one.
        table = gap_length_table({"A": tidy(r2=0.60), "B": tidy(r2=0.90)})
        with pytest.raises(SensitivityError, match="median_across_sites"):
            gap_length_pivot(table, metric="r2", gap_class="all")


# ---------------------------------------------------------------------------
# Real validation results
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def site():
    """The reference synthetic site: one year of half-hourly data, seeded."""
    return synthetic_site()


@pytest.fixture(scope="module")
def arms(site) -> dict[str, object]:
    """RFR3 and RFR10 over NEE, on one shared set of artificial intervals."""
    gaps = site.known_gaps()
    return {
        mode: validate_rfr(
            site.frame,
            config=site.config(mode, hyperparameter_grid=TEST_GRID, cv_folds=3, features=REACHING),
            targets=["NEE"],
            qc_columns={"NEE": site.qc_column("NEE")},
            gaps=gaps,
        )
        for mode in ("RFR3", "RFR10")
    }


class TestFromValidationRuns:
    """The tables are built from real results, not only from fixtures."""

    def test_two_arms_land_in_one_table(self, arms):
        table = gap_length_table(list(arms.values()), site="synthetic")
        assert set(table["method"]) == {"RFR3", "RFR10"}
        assert set(table["site"]) == {"synthetic"}
        assert tuple(table.columns) == SENSITIVITY_TABLE_COLUMNS

    def test_every_placed_gap_class_is_reported_separately(self, arms):
        report = arms["RFR3"]
        table = gap_length_table(report, site="synthetic")
        placed = {gap.gap_class.value for gap in report.gaps}
        assert placed <= set(table["gap_class"])
        assert "all" in set(table["gap_class"])

    def test_the_numbers_are_the_ones_the_result_carries(self, arms):
        report = arms["RFR10"]
        table = gap_length_table(report, site="synthetic")
        result = report["NEE"]
        for gap_class in (None, GapClass.VERY_LONG):
            for subset in (MetricSubset.ALL, MetricSubset.NIGHTTIME):
                scores = result.metric(subset=subset, gap_class=gap_class)
                row = cell(
                    table,
                    gap_class="all" if gap_class is None else gap_class.value,
                    subset=subset.value,
                )
                assert row["n"] == scores.n
                if scores.r2 is None:
                    assert pd.isna(row["r2"])
                else:
                    assert row["r2"] == pytest.approx(scores.r2)

    def test_the_arms_can_be_read_side_by_side(self, arms):
        table = gap_length_table(list(arms.values()), site="synthetic")
        wide = gap_length_pivot(table, metric="r2", gap_class="very_long")
        assert list(wide.columns) == ["RFR3", "RFR10"]
        assert list(wide.index) == ["NEE"]

    def test_two_sites_of_one_arm_aggregate_to_a_median(self, arms):
        # The same result relabelled as two sites: the median of one value with
        # itself is that value, which is exactly what makes it a safe check that
        # the aggregation through a real result works at all.
        table = pd.concat(
            [
                gap_length_table(arms["RFR3"], site="A"),
                gap_length_table(arms["RFR3"], site="B"),
            ],
            ignore_index=True,
        )
        medians = median_across_sites(table)
        one = cell(gap_length_table(arms["RFR3"], site="A"), gap_class="all", subset="all")
        both = cell(medians, gap_class="all", subset="all")
        assert both["n_sites"] == 2
        assert both["r2"] == pytest.approx(one["r2"])
