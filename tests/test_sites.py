"""Multi-site and ecosystem-stratified reports. Covers Step 20A and acceptance test 39.

Five parts. The first checks Table S10 against ``docs/supplement_benchmarks.md``,
as ``test_benchmarks.py`` does for Table S3, and checks the scientific
constraints of Step 20A that can be stated as facts about the package: RFR3's
evidence is broader than RFR10's, and nothing here is a threshold.

The middle three build tables by hand, because what these functions get right or
wrong is metadata, grouping, pairing and a t-test, and five sites with chosen
values say that more plainly than a fitted model does.

The last two run the real thing: two synthetic sites validated separately and
reported together - the exit criterion, reports without a pooled model - and,
where a local copy of the journal supplement is available, the published tables
themselves, from which Table S10 is reproduced.
"""

from __future__ import annotations

import inspect
import os
import re
from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy
from scipy import stats

import rfrgapfill
from rfrgapfill import sites
from rfrgapfill.benchmarks import NEE_BENCHMARK_UNITS, NEE_MODEL_UNITS, compare_to_benchmarks
from rfrgapfill.config import FeatureConfig
from rfrgapfill.sensitivity import (
    SENSITIVITY_TABLE_COLUMNS,
    SensitivityError,
    gap_length_table,
    median_across_sites,
)
from rfrgapfill.sites import (
    IGBP_CLASSES,
    METHOD_DIFFERENCE_COLUMNS,
    POPULATION_SUMMARY_COLUMNS,
    PUBLISHED_SITE_TABLE_COLUMNS,
    PUBLISHED_WELCH_TABLE_COLUMNS,
    S10_COMPARISON_COLUMNS,
    SITE_EVIDENCE_COLUMNS,
    SITE_METADATA_COLUMNS,
    SITE_REPORT_COLUMNS,
    STRATIFIED_SUMMARY_COLUMNS,
    TABLE_S10_COMPARISONS,
    WELCH_TABLE_COLUMNS,
    PublishedWelchComparison,
    SiteReportError,
    SiteReportWarning,
    compare_to_table_s10,
    method_differences,
    published_welch_table,
    read_table_s2,
    read_table_s9,
    read_tables_s4_to_s6,
    site_level_evidence,
    site_metadata,
    site_population_summary,
    site_report,
    stratified_summary,
    welch_comparison,
)
from rfrgapfill.synthetic import synthetic_site
from rfrgapfill.uncertainty import bias_iqr
from rfrgapfill.validation import validate_rfr

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DOC = ROOT / "docs" / "supplement_benchmarks.md"

FLUXES = ("NEE", "H", "LE")
CHECK = "\N{CHECK MARK}"
CROSS = "\N{MULTIPLICATION SIGN}"

#: One grid point and three folds. Nothing here scores predictive quality.
TEST_GRID = {"n_estimators": (15,)}

#: The reaching strategy a long-gap run has to choose deliberately (A4).
REACHING = FeatureConfig(daily_statistic_strategy="rolling_available")


# ---------------------------------------------------------------------------
# Table S10 as documentation, and the constraints of Step 20A
# ---------------------------------------------------------------------------


def documented_s10() -> dict[tuple[str, str, str, str], tuple[float, float, float, bool]]:
    """Return Table S10 as ``docs/supplement_benchmarks.md`` records it."""
    text = BENCHMARK_DOC.read_text(encoding="utf-8")
    blocks = re.split(r"^## ", text, flags=re.MULTILINE)
    matching = [block for block in blocks if block.startswith("5. Statistical comparison")]
    assert len(matching) == 1, "the benchmark document has no section 5"
    metrics = {"R2": "r2", "slope": "slope", "RMSE": "rmse", "bias": "bias"}
    found: dict[tuple[str, str, str, str], tuple[float, float, float, bool]] = {}
    for line in matching[0].splitlines():
        if not line.startswith("|"):
            continue
        row = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(row) != 7 or row[0] not in FLUXES:
            continue
        method, baseline = row[2].split(" vs ")
        found[row[0], metrics[row[1]], method, baseline] = (
            float(row[3]),
            float(row[4]),
            float(row[5]),
            row[6] != "*",
        )
    assert len(found) == 24, f"expected 24 Table S10 cells, read {len(found)}"
    return found


class TestTableS10IsTraceable:
    """The constants and the benchmark document say the same thing."""

    def test_every_cell_matches_the_document(self):
        carried = {
            (r.target, r.metric, r.method, r.baseline): (
                r.mean_difference,
                r.ci_lower,
                r.ci_upper,
                r.significant,
            )
            for r in TABLE_S10_COMPARISONS
        }
        assert carried == documented_s10()

    def test_every_cell_names_its_source_and_its_population(self):
        for record in TABLE_S10_COMPARISONS:
            assert "Table S10" in record.source
            assert "94-site" in record.site_subset

    def test_the_two_comparisons_are_the_published_ones(self):
        pairs = {(r.method, r.baseline) for r in TABLE_S10_COMPARISONS}
        assert pairs == {("RFR3", "MDS"), ("RFR10", "RFR3")}

    def test_units_follow_the_caption(self):
        for record in TABLE_S10_COMPARISONS:
            if record.metric in ("r2", "slope"):
                assert record.units == "dimensionless"
            else:
                assert record.units == (NEE_BENCHMARK_UNITS if record.target == "NEE" else "W m-2")

    def test_the_asterisks_are_where_the_supplement_prints_them(self):
        not_significant = {
            (r.target, r.metric, r.method) for r in TABLE_S10_COMPARISONS if not r.significant
        }
        assert not_significant == {
            ("NEE", "slope", "RFR3"),
            ("NEE", "bias", "RFR3"),
            ("NEE", "bias", "RFR10"),
            ("NEE", "rmse", "RFR10"),
            ("H", "bias", "RFR10"),
        }

    def test_the_table_has_the_documented_columns(self):
        table = published_welch_table()
        assert tuple(table.columns) == PUBLISHED_WELCH_TABLE_COLUMNS
        assert len(table) == 24
        assert table["significant"].dtype == "boolean"

    def test_a_swapped_interval_is_refused(self):
        with pytest.raises(SiteReportError, match="swapped"):
            PublishedWelchComparison("NEE", "RFR3", "MDS", "r2", 0.07, 0.1, 0.0, True)

    def test_an_unknown_metric_is_refused(self):
        with pytest.raises(SiteReportError, match="unknown metric"):
            PublishedWelchComparison("NEE", "RFR3", "MDS", "mae", 0.07, 0.0, 0.1, True)


class TestTheEvidenceIsNotEquallyBroad:
    """RFR3 has broader site-level evidence than RFR10 (Step 20A)."""

    def evidence(self, target: str, method: str) -> pd.Series:
        table = site_level_evidence()
        assert tuple(table.columns) == SITE_EVIDENCE_COLUMNS
        rows = table.loc[(table["target"] == target) & (table["method"] == method)]
        assert len(rows) == 1
        return rows.iloc[0]

    def test_rfr3_nee_is_scored_at_all_194_sites(self):
        row = self.evidence("NEE", "RFR3")
        assert row["n_sites"] == 194
        assert "Table S9" in row["tables"]

    def test_rfr10_is_scored_at_the_94_site_subset_only(self):
        for target in FLUXES:
            row = self.evidence(target, "RFR10")
            assert row["n_sites"] == 94
            assert "94-site" in row["site_population"]
            assert "Table S9" not in row["tables"]

    def test_rfr3_has_the_broader_evidence_for_nee(self):
        assert self.evidence("NEE", "RFR3")["n_sites"] > self.evidence("NEE", "RFR10")["n_sites"]

    def test_h_and_le_rest_on_the_94_site_subset_for_every_method(self):
        for target in ("H", "LE"):
            for method in ("MDS", "RFR3", "RFR10"):
                assert self.evidence(target, method)["n_sites"] == 94


class TestNothingHereIsAThreshold:
    """No universal performance threshold, and no site required to improve."""

    def test_no_public_function_takes_a_threshold(self):
        # Tripwire for whoever adds a pass/fail rule: performance differs by
        # ecosystem, and Step 20A forbids a universal one.
        forbidden = ("threshold", "min_", "max_", "require", "pass", "fail", "tolerance")
        for name in sites.__all__:
            member = getattr(sites, name)
            if not inspect.isfunction(member):
                continue
            for parameter in inspect.signature(member).parameters:
                assert not any(word in parameter for word in forbidden), (name, parameter)

    def test_the_new_names_are_public(self):
        for name in (
            "site_metadata",
            "site_report",
            "stratified_summary",
            "method_differences",
            "welch_comparison",
            "compare_to_table_s10",
            "TABLE_S10_COMPARISONS",
        ):
            assert name in rfrgapfill.__all__
            assert getattr(rfrgapfill, name) is getattr(sites, name)


# ---------------------------------------------------------------------------
# Site metadata
# ---------------------------------------------------------------------------

#: Five sites with the headers Supplementary Table S2 prints, bar the site
#: column, which S2 leaves unlabelled and Table S9 calls ``ID``.
S2_LIKE = pd.DataFrame(
    {
        "ID": ["A", "B", "C", "D", "E"],
        "Latitude": ["45.10", "-33.46", "60.00", "10.50", "-5.00"],
        "Longitude": ["11.32", "-66.46", "24.30", "-60.00", "\N{MINUS SIGN}55.00"],
        "Continent": ["Europe", "South America", "Europe", "South America", "South America"],
        "IGBP": ["ENF", "ENF", "Wet", "GRA", "GRA"],
        "Koppen": ["Cfb", "BSh", "ET", "Aw", "Af"],
        "Start": ["2009-01-01", "2002-01-01", "2004-01-01", "2010-01-01", "2012-01-01"],
        "End": ["2011-12-31", "2012-12-31", "2014-12-31", "2014-12-31", "2014-12-31"],
        "Elevation": ["505", "105", "", "76", "40"],
        "System": ["O", "C", "O", "M", "O"],
        "Height ratio": ["2.00", "1.50", "3.95", "", "1.20"],
        "Included in 94  sites": [CHECK, CROSS, CHECK, CHECK, CROSS],
    }
)


class TestSiteMetadata:
    """Table S2 in, canonical metadata out."""

    def test_the_supplement_headers_are_understood(self):
        table = site_metadata(S2_LIKE)
        assert tuple(table.columns) == SITE_METADATA_COLUMNS
        assert list(table["site_id"]) == ["A", "B", "C", "D", "E"]
        assert list(table["instrument_system"]) == ["O", "C", "O", "M", "O"]
        assert table.loc[4, "longitude"] == pytest.approx(-55.0)

    def test_values_have_their_types(self):
        table = site_metadata(S2_LIKE)
        for column in ("latitude", "longitude", "elevation", "height_ratio"):
            assert table[column].dtype == "float64"
        assert table["start_date"].dtype == "datetime64[ns]"
        assert table["included_in_94_site_subset"].dtype == "boolean"
        assert list(table["included_in_94_site_subset"]) == [True, False, True, True, False]
        assert pd.isna(table.loc[2, "elevation"])

    def test_igbp_codes_are_normalised(self):
        # Table S2 spells one wetland "Wet".
        assert site_metadata(S2_LIKE).loc[2, "igbp"] == "WET"

    def test_fluxnet_headers_are_understood(self):
        table = site_metadata(
            pd.DataFrame(
                {"SITE_ID": ["US-Ha1"], "LOCATION_LAT": [42.54], "LOCATION_LONG": [-72.17],
                 "IGBP": ["DBF"]}
            )
        )  # fmt: skip
        assert table.loc[0, "site_id"] == "US-Ha1"
        assert table.loc[0, "latitude"] == pytest.approx(42.54)
        assert table.loc[0, "igbp"] == "DBF"
        assert pd.isna(table.loc[0, "koppen"])

    def test_records_are_accepted(self):
        table = site_metadata([{"site": "A", "igbp": "gra", "included_in_94_site_subset": True}])
        assert table.loc[0, "igbp"] == "GRA"
        assert bool(table.loc[0, "included_in_94_site_subset"])

    def test_other_columns_are_kept_after_the_canonical_ones(self):
        table = site_metadata(S2_LIKE.assign(tower_height=[1, 2, 3, 4, 5]))
        assert tuple(table.columns) == (*SITE_METADATA_COLUMNS, "tower_height")

    def test_it_is_idempotent(self):
        once = site_metadata(S2_LIKE)
        pd.testing.assert_frame_equal(site_metadata(once), once)

    def test_the_caller_is_not_mutated(self):
        before = S2_LIKE.copy()
        site_metadata(S2_LIKE)
        pd.testing.assert_frame_equal(S2_LIKE, before)

    @pytest.mark.parametrize(
        ("change", "message"),
        [
            ({"ID": ["A", "A", "C", "D", "E"]}, "more than once"),
            ({"ID": ["A", "", "C", "D", "E"]}, "no site label"),
            ({"IGBP": ["ENF", "ENF", "WET", "GRA", "TUNDRA"]}, "not an IGBP class"),
            ({"Latitude": ["45", "-33", "91", "10", "-5"]}, "outside"),
            ({"Longitude": ["1", "2", "x", "4", "5"]}, "not a number"),
            ({"System": ["O", "C", "O", "M", "laser"]}, "instrument system"),
            ({"Included in 94  sites": [CHECK, "maybe", CHECK, CHECK, CROSS]}, "94-site"),
            ({"End": ["2008-12-31", "2012-12-31", "2014-12-31", "2014-12-31", "2014-12-31"]},
             "ends before it starts"),
            ({"Start": ["soon", "2002-01-01", "2004-01-01", "2010-01-01", "2012-01-01"]},
             "not a date"),
        ],
    )  # fmt: skip
    def test_what_a_field_does_not_allow_is_refused(self, change, message):
        with pytest.raises(SiteReportError, match=message):
            site_metadata(S2_LIKE.assign(**change))

    def test_a_frame_without_a_site_column_is_refused(self):
        with pytest.raises(SiteReportError, match="site column"):
            site_metadata(S2_LIKE.drop(columns="ID"))

    def test_two_columns_for_one_field_are_refused(self):
        with pytest.raises(SiteReportError, match="both describe latitude"):
            site_metadata(S2_LIKE.assign(lat=1.0))

    def test_its_errors_are_sensitivity_errors(self):
        assert issubclass(SiteReportError, SensitivityError)

    def test_the_igbp_vocabulary_holds_the_supplements_eleven_classes(self):
        table_s1 = {"CRO", "CSH", "DBF", "EBF", "ENF", "GRA", "MF", "OSH", "SAV", "WET", "WSA"}
        assert table_s1 <= set(IGBP_CLASSES)


class TestPopulationSummary:
    """The shape of Table S1."""

    def test_sites_are_counted_per_class(self):
        table = site_population_summary(S2_LIKE, by="igbp")
        assert tuple(table.columns) == POPULATION_SUMMARY_COLUMNS
        assert dict(zip(table["stratum"], table["n_sites"], strict=True)) == {
            "ENF": 2,
            "GRA": 2,
            "WET": 1,
        }
        assert table["fraction"].sum() == pytest.approx(1.0)

    def test_sites_without_a_value_are_a_row_of_their_own(self):
        table = site_population_summary(S2_LIKE, by="elevation")
        assert pd.isna(table["stratum"].iloc[-1])
        assert table["n_sites"].iloc[-1] == 1
        assert table["fraction"].sum() == pytest.approx(1.0)

    def test_an_unknown_field_is_refused(self):
        with pytest.raises(SiteReportError, match="cannot count sites by"):
            site_population_summary(S2_LIKE, by="biome")


# ---------------------------------------------------------------------------
# Hand-built results
# ---------------------------------------------------------------------------


def scores(
    method: str,
    *,
    r2: float | None = 0.8,
    slope: float | None = 0.9,
    rmse: float | None = 2.5,
    bias: float | None = 0.0,
    target: str = "NEE",
    subsets: tuple[str, ...] = ("all",),
) -> pd.DataFrame:
    """Return the tidy table one arm of one site's run would produce."""
    return pd.DataFrame(
        [
            {
                "target": target,
                "method": method,
                "mode": method if method.startswith("RFR") else None,
                "gap_class": "all",
                "subset": subset,
                "n": 100,
                "n_offered": 105,
                "r2": r2,
                "slope": slope,
                "rmse": rmse,
                "bias": bias,
            }
            for subset in subsets
        ]
    )


#: Five sites whose RFR3 R2 lands on the quartile order statistics:
#: q1 = 0.6, median = 0.7, q3 = 0.8, IQR = 0.2.
FIVE = {"A": 0.5, "B": 0.6, "C": 0.7, "D": 0.8, "E": 1.1}


def five_sites() -> dict[str, pd.DataFrame]:
    return {site: scores("RFR3", r2=r2) for site, r2 in FIVE.items()}


def cell(table: pd.DataFrame, **where: object) -> pd.Series:
    """Return the single row of ``table`` matching ``where``."""
    selected = table
    for column, value in where.items():
        selected = selected.loc[selected[column].astype(str) == str(value)]
    assert len(selected) == 1, f"expected one row for {where}, found {len(selected)}"
    return selected.iloc[0]


class TestSiteReport:
    """Acceptance test 39: site ID and, where available, IGBP metadata."""

    def test_every_row_carries_its_site_and_its_metadata(self):
        report = site_report(five_sites(), S2_LIKE)
        assert tuple(report.columns) == SITE_REPORT_COLUMNS
        assert list(report["site"]) == list(FIVE)
        assert list(report["igbp"]) == ["ENF", "ENF", "WET", "GRA", "GRA"]
        assert list(report["included_in_94_site_subset"]) == [True, False, True, True, False]

    def test_the_scores_are_the_sites_own(self):
        report = site_report(five_sites(), S2_LIKE)
        assert list(report["r2"]) == list(FIVE.values())

    def test_without_metadata_the_columns_are_present_and_missing(self):
        report = site_report(five_sites())
        assert tuple(report.columns) == SITE_REPORT_COLUMNS
        assert bool(report["igbp"].isna().all())
        assert report["latitude"].dtype == "float64"

    def test_a_site_without_metadata_is_kept_and_named(self):
        with pytest.warns(SiteReportWarning, match="E"):
            report = site_report(five_sites(), S2_LIKE.iloc[:4])
        assert pd.isna(cell(report, site="E")["igbp"])

    def test_the_metadata_fields_can_be_chosen(self):
        report = site_report(five_sites(), S2_LIKE, columns=("continent", "igbp"))
        assert tuple(report.columns) == (
            "site",
            "continent",
            "igbp",
            *SENSITIVITY_TABLE_COLUMNS[1:],
        )

    def test_an_unknown_metadata_field_is_refused(self):
        with pytest.raises(SiteReportError, match="unknown metadata field"):
            site_report(five_sites(), S2_LIKE, columns=("biome",))

    def test_an_unlabelled_row_is_refused(self):
        with pytest.raises(SiteReportError, match="no site label"):
            site_report(scores("RFR3"))

    def test_one_site_contributing_a_cell_twice_is_refused(self):
        with pytest.raises(SiteReportError, match="twice"):
            site_report({"A": [scores("RFR3"), scores("RFR3", r2=0.1)]})

    def test_the_metadata_is_usable_by_bias_iqr(self):
        # Step 18A's IGBP stratification reads the same canonical table.
        table = bias_iqr(
            {site: scores("RFR3", bias=value) for site, value in FIVE.items()},
            igbp=site_metadata(S2_LIKE),
        )
        assert set(table["igbp"]) == {"all", "ENF", "GRA", "WET"}


class TestStratifiedSummary:
    """Grouped by target, method, gap class, day/night and IGBP."""

    def report(self, **overrides: pd.DataFrame) -> pd.DataFrame:
        return site_report({**five_sites(), **overrides}, S2_LIKE)

    def test_the_pooled_quartiles_are_those_of_the_per_site_values(self):
        row = cell(stratified_summary(self.report()), stratum="all", metric="r2")
        assert (row["q1"], row["median"], row["q3"], row["iqr"]) == pytest.approx(
            (0.6, 0.7, 0.8, 0.2)
        )
        assert row["n_sites"] == 5

    def test_each_ecosystem_is_summarised_separately(self):
        table = stratified_summary(self.report())
        enf = cell(table, stratum="ENF", metric="r2")
        assert (enf["q1"], enf["median"], enf["q3"]) == pytest.approx((0.525, 0.55, 0.575))
        assert enf["n_sites"] == 2

    def test_one_site_has_quartiles_but_no_spread(self):
        wet = cell(stratified_summary(self.report()), stratum="WET", metric="r2")
        assert wet["median"] == pytest.approx(0.7)
        assert pd.isna(wet["iqr"])

    def test_the_pooled_row_comes_first_then_the_strata_in_order(self):
        table = stratified_summary(self.report(), metrics=("r2",))
        assert list(table["stratum"]) == ["all", "ENF", "GRA", "WET"]
        assert tuple(table.columns) == STRATIFIED_SUMMARY_COLUMNS

    def test_a_site_without_a_class_is_pooled_only(self):
        metadata = S2_LIKE.assign(IGBP=["ENF", "ENF", "WET", "GRA", None])
        table = stratified_summary(site_report(five_sites(), metadata), metrics=("r2",))
        assert cell(table, stratum="all")["n_sites_offered"] == 5
        assert cell(table, stratum="GRA")["n_sites_offered"] == 1

    def test_sites_with_a_value_are_counted_per_metric(self):
        table = stratified_summary(self.report(A=scores("RFR3", r2=None)))
        assert cell(table, stratum="all", metric="r2")["n_sites"] == 4
        assert cell(table, stratum="all", metric="rmse")["n_sites"] == 5
        assert cell(table, stratum="all", metric="r2")["n_sites_offered"] == 5

    def test_day_night_and_methods_are_kept_apart(self):
        study = {
            site: [
                scores("RFR3", r2=r2, subsets=("all", "nighttime")),
                scores("MDS", r2=r2 - 0.1, subsets=("all", "nighttime")),
            ]
            for site, r2 in FIVE.items()
        }
        table = stratified_summary(site_report(study, S2_LIKE), by=None, metrics=("r2",))
        assert len(table) == 4
        assert cell(table, method="MDS", subset="nighttime")["median"] == pytest.approx(0.6)
        assert list(table["method"]) == ["MDS", "MDS", "RFR3", "RFR3"]

    def test_without_a_stratifier_only_pooled_rows_come_back(self):
        table = stratified_summary(self.report(), by=None)
        assert set(table["stratum"]) == {"all"}
        assert bool(table["stratifier"].isna().all())

    def test_any_metadata_column_can_stratify(self):
        table = stratified_summary(self.report(), by="included_in_94_site_subset", metrics=("r2",))
        assert list(table["stratum"]) == ["all", "False", "True"]
        assert cell(table, stratum="True")["n_sites"] == 3

    def test_a_missing_stratifier_column_is_refused(self):
        with pytest.raises(SiteReportError, match="site_report"):
            stratified_summary(gap_length_table(five_sites()))

    def test_a_stratum_labelled_all_is_refused(self):
        report = site_report(five_sites(), S2_LIKE.assign(Koppen="all"))
        with pytest.raises(SiteReportError, match="pooling every"):
            stratified_summary(report, by="koppen")

    def test_a_repeated_cell_is_refused(self):
        report = self.report()
        with pytest.raises(SiteReportError, match="twice"):
            stratified_summary(pd.concat([report, report.iloc[:1]], ignore_index=True))


class TestMethodDifferences:
    """Per-site changes, reported and never required."""

    def study(self) -> pd.DataFrame:
        return site_report(
            {
                # RFR3 better than MDS on every metric.
                "A": [
                    scores("MDS", r2=0.70, slope=1.10, rmse=3.0, bias=-0.5),
                    scores("RFR3", r2=0.80, slope=0.95, rmse=2.5, bias=0.3),
                ],
                # RFR3 worse than MDS on every metric.
                "B": [
                    scores("MDS", r2=0.80, slope=1.00, rmse=2.0, bias=0.0),
                    scores("RFR3", r2=0.75, slope=0.80, rmse=2.4, bias=-0.2),
                ],
                # MDS only: RFR3 was not run here.
                "C": [scores("MDS", r2=0.60)],
            },
            S2_LIKE,
        )

    def test_the_difference_is_method_minus_baseline(self):
        table = method_differences(self.study(), method="RFR3", baseline="MDS")
        assert cell(table, site="A", metric="r2")["difference"] == pytest.approx(0.10)
        assert cell(table, site="A", metric="bias")["difference"] == pytest.approx(0.8)

    def test_an_improvement_is_judged_by_the_metrics_own_direction(self):
        table = method_differences(self.study(), method="RFR3", baseline="MDS")
        # A larger bias difference is still an improvement: 0.3 is closer to 0
        # than -0.5. A slope of 0.95 improves on 1.10: closer to 1.
        for metric in ("r2", "slope", "rmse", "bias"):
            assert cell(table, site="A", metric=metric)["change"] == "improved"
            assert cell(table, site="B", metric=metric)["change"] == "worsened"

    def test_a_site_that_worsened_is_a_row_not_an_error(self):
        table = method_differences(self.study(), method="RFR3", baseline="MDS")
        assert set(table["change"].dropna()) == {"improved", "worsened"}

    def test_a_site_scored_by_one_method_keeps_its_value(self):
        row = cell(
            method_differences(self.study(), method="RFR3", baseline="MDS"), site="C", metric="r2"
        )
        assert pd.isna(row["method_value"])
        assert row["baseline_value"] == pytest.approx(0.60)
        assert pd.isna(row["difference"])
        assert pd.isna(row["change"])

    def test_an_equal_score_is_unchanged(self):
        report = site_report({"A": [scores("MDS"), scores("RFR3")]})
        table = method_differences(report, method="RFR3", baseline="MDS")
        assert set(table["change"]) == {"unchanged"}

    def test_the_metadata_follows_the_site(self):
        table = method_differences(self.study(), method="RFR3", baseline="MDS")
        assert tuple(table.columns) == (
            "site",
            *sites.SITE_REPORT_METADATA,
            *METHOD_DIFFERENCE_COLUMNS[1:],
        )
        assert cell(table, site="C", metric="r2")["igbp"] == "WET"

    def test_a_table_without_metadata_has_the_bare_columns(self):
        report = gap_length_table({"A": [scores("MDS"), scores("RFR3")]})
        table = method_differences(report, method="RFR3", baseline="MDS")
        assert tuple(table.columns) == METHOD_DIFFERENCE_COLUMNS

    def test_a_method_cannot_be_compared_with_itself(self):
        with pytest.raises(SiteReportError, match="itself"):
            method_differences(self.study(), method="RFR3", baseline="RFR3")

    def test_an_absent_method_is_named(self):
        with pytest.raises(SiteReportError, match="no RFR10 rows"):
            method_differences(self.study(), method="RFR10", baseline="RFR3")

    def test_two_values_for_one_cell_are_refused(self):
        two_modes = pd.concat(
            [scores("RFR3"), scores("RFR3").assign(mode="RFR10")], ignore_index=True
        )
        report = site_report({"A": [scores("MDS"), two_modes]})
        with pytest.raises(SiteReportError, match="more than once"):
            method_differences(report, method="RFR3", baseline="MDS")


#: Six sites scored by MDS, five of them by RFR3 too: a Welch test does not
#: need the samples to be the same size, or the same sites.
WELCH_MDS = {"S1": 0.70, "S2": 0.72, "S3": 0.65, "S4": 0.80, "S5": 0.60, "S6": 0.75}
WELCH_RFR3 = {"S1": 0.81, "S2": 0.77, "S3": 0.90, "S4": 0.68, "S5": 0.85}


def welch_study(igbp: dict[str, str] | None = None) -> pd.DataFrame:
    study = {site: [scores("MDS", r2=r2)] for site, r2 in WELCH_MDS.items()}
    for site, r2 in WELCH_RFR3.items():
        study[site].append(scores("RFR3", r2=r2))
    metadata = None if igbp is None else [{"site": s, "igbp": c} for s, c in igbp.items()]
    return site_report(study, metadata)


#: ``ttest_ind`` results carry ``df`` and ``confidence_interval()`` from scipy 1.11.
#: The package computes both itself and supports older scipy; only this oracle
#: needs the newer one, so the dependency floor stays where the package needs it.
SCIPY_REPORTS_THE_INTERVAL = tuple(int(part) for part in scipy.__version__.split(".")[:2]) >= (
    1,
    11,
)


class TestWelchComparison:
    """Table S10's statistic, checked against scipy."""

    def expected(self):
        return stats.ttest_ind(list(WELCH_RFR3.values()), list(WELCH_MDS.values()), equal_var=False)

    def test_it_is_welchs_test(self):
        row = cell(welch_comparison(welch_study(), method="RFR3", baseline="MDS"), metric="r2")
        expected = self.expected()
        assert row["t_statistic"] == pytest.approx(expected.statistic)
        assert row["p_value"] == pytest.approx(expected.pvalue)
        assert row["mean_difference"] == pytest.approx(
            np.mean(list(WELCH_RFR3.values())) - np.mean(list(WELCH_MDS.values()))
        )

    @pytest.mark.skipif(
        not SCIPY_REPORTS_THE_INTERVAL, reason="scipy < 1.11 reports no Welch df or interval"
    )
    def test_its_degrees_of_freedom_and_interval_are_welchs(self):
        row = cell(welch_comparison(welch_study(), method="RFR3", baseline="MDS"), metric="r2")
        expected = self.expected()
        interval = expected.confidence_interval(0.95)
        assert row["df"] == pytest.approx(expected.df)
        assert (row["ci_lower"], row["ci_upper"]) == pytest.approx((interval.low, interval.high))

    def test_the_samples_and_the_pairs_are_counted(self):
        row = cell(welch_comparison(welch_study(), method="RFR3", baseline="MDS"), metric="r2")
        assert (row["n_method"], row["n_baseline"], row["n_paired"]) == (5, 6, 5)
        # S1, S2, S3, S5 gained R2; S4 lost it. Reported, not required.
        assert (row["n_improved"], row["n_worsened"], row["n_unchanged"]) == (4, 1, 0)

    def test_a_lower_confidence_gives_a_narrower_interval(self):
        wide = cell(welch_comparison(welch_study(), method="RFR3", baseline="MDS"), metric="r2")
        narrow = cell(
            welch_comparison(welch_study(), method="RFR3", baseline="MDS", confidence=0.8),
            metric="r2",
        )
        assert narrow["ci_upper"] - narrow["ci_lower"] < wide["ci_upper"] - wide["ci_lower"]
        assert narrow["confidence"] == pytest.approx(0.8)

    def test_significance_is_reported_at_the_confidence_level(self):
        row = cell(welch_comparison(welch_study(), method="RFR3", baseline="MDS"), metric="r2")
        assert bool(row["significant"]) == bool(row["p_value"] < 0.05)

    def test_one_site_a_side_gives_means_but_no_test(self):
        report = site_report({"A": [scores("MDS", r2=0.5), scores("RFR3", r2=0.7)]})
        row = cell(welch_comparison(report, method="RFR3", baseline="MDS"), metric="r2")
        assert row["mean_difference"] == pytest.approx(0.2)
        for column in ("ci_lower", "ci_upper", "df", "t_statistic", "p_value"):
            assert pd.isna(row[column])
        assert pd.isna(row["significant"])

    def test_no_spread_on_either_side_gives_no_test(self):
        report = site_report(
            {site: [scores("MDS", r2=0.5), scores("RFR3", r2=0.7)] for site in "ABC"}
        )
        row = cell(welch_comparison(report, method="RFR3", baseline="MDS"), metric="r2")
        assert row["mean_difference"] == pytest.approx(0.2)
        assert pd.isna(row["p_value"])

    def test_it_can_be_repeated_within_each_ecosystem(self):
        igbp = {"S1": "ENF", "S2": "ENF", "S3": "ENF", "S4": "GRA", "S5": "GRA", "S6": "GRA"}
        table = welch_comparison(
            welch_study(igbp), method="RFR3", baseline="MDS", metrics=("r2",), by="igbp"
        )
        assert tuple(table.columns) == WELCH_TABLE_COLUMNS
        assert list(table["stratum"]) == ["all", "ENF", "GRA"]
        assert cell(table, stratum="GRA")["n_method"] == 2
        assert cell(table, stratum="GRA")["n_baseline"] == 3

    def test_a_confidence_outside_zero_and_one_is_refused(self):
        with pytest.raises(SiteReportError, match="confidence"):
            welch_comparison(welch_study(), method="RFR3", baseline="MDS", confidence=95)

    def test_a_missing_stratifier_is_refused(self):
        with pytest.raises(SiteReportError, match="site_report"):
            welch_comparison(
                gap_length_table({"A": scores("MDS")}), method="RFR3", baseline="MDS", by="igbp"
            )


class TestCompareToTableS10:
    """A reproduction beside the published Welch table."""

    def welch(self, **units: str) -> pd.DataFrame:
        return welch_comparison(
            site_report(
                {
                    site: [scores("MDS", r2=r2), scores("RFR3", r2=WELCH_RFR3[site])]
                    for site, r2 in WELCH_MDS.items()
                    if site in WELCH_RFR3
                },
                units=units or None,
            ),
            method="RFR3",
            baseline="MDS",
        )

    def test_dimensionless_metrics_are_differenced(self):
        comparison = compare_to_table_s10(self.welch())
        assert tuple(comparison.columns) == S10_COMPARISON_COLUMNS
        row = cell(comparison, metric="r2")
        assert bool(row["comparable"])
        assert row["published_mean_difference"] == pytest.approx(0.07)
        assert row["difference"] == pytest.approx(row["run_mean_difference"] - 0.07)

    def test_nee_rmse_in_model_units_is_refused_not_differenced(self):
        row = cell(compare_to_table_s10(self.welch()), metric="rmse")
        assert not bool(row["comparable"])
        assert row["run_units"] == NEE_MODEL_UNITS
        assert pd.isna(row["difference"])
        assert "A10" in row["note"]

    def test_nee_rmse_in_carbon_units_compares(self):
        row = cell(compare_to_table_s10(self.welch(NEE=NEE_BENCHMARK_UNITS)), metric="rmse")
        assert bool(row["comparable"])

    def test_only_the_pooled_rows_are_compared(self):
        igbp = {"S1": "ENF", "S2": "ENF", "S3": "ENF", "S4": "GRA", "S5": "GRA", "S6": "GRA"}
        welch = welch_comparison(welch_study(igbp), method="RFR3", baseline="MDS", by="igbp")
        assert len(compare_to_table_s10(welch)) == 4

    def test_a_comparison_the_table_does_not_make_is_refused(self):
        report = site_report({s: [scores("ORF3"), scores("RFR3", r2=0.1)] for s in "ABC"})
        with pytest.raises(SiteReportError, match="no row of this table is in Supplementary"):
            compare_to_table_s10(welch_comparison(report, method="RFR3", baseline="ORF3"))


# ---------------------------------------------------------------------------
# Two real sites, validated separately and reported together
# ---------------------------------------------------------------------------

TWO_SITES = {"SY-Nth": (45.0, 20180101), "SY-Sth": (-35.0, 7)}


@pytest.fixture(scope="module")
def two_runs():
    """One RFR3 NEE validation per synthetic site, each with its own model."""
    runs = {}
    for site_id, (latitude, seed) in TWO_SITES.items():
        site = synthetic_site(site_id=site_id, latitude=latitude, seed=seed)
        runs[site_id] = validate_rfr(
            site.frame,
            config=site.config(
                "RFR3", hyperparameter_grid=TEST_GRID, cv_folds=3, features=REACHING
            ),
            targets=["NEE"],
            qc_columns={"NEE": site.qc_column("NEE")},
            gaps=site.known_gaps(),
        )
    return runs


TWO_SITE_METADATA = pd.DataFrame(
    {"site_id": list(TWO_SITES), "latitude": [45.0, -35.0], "IGBP": ["ENF", "GRA"]}
)


class TestTwoSites:
    """The exit criterion: site- and ecosystem-stratified, nothing pooled."""

    def test_each_sites_rows_are_its_own_run(self, two_runs):
        report = site_report(two_runs, TWO_SITE_METADATA)
        for site_id, run in two_runs.items():
            own = run.to_frame()
            rows = report.loc[report["site"] == site_id].reset_index(drop=True)
            assert len(rows) == len(own)
            for metric in ("r2", "slope", "rmse", "bias"):
                np.testing.assert_allclose(
                    rows[metric].to_numpy(dtype=float),
                    pd.to_numeric(own[metric]).to_numpy(dtype=float),
                )

    def test_the_two_sites_were_not_one_model(self, two_runs):
        # Separately fitted forests score the same gap classes differently.
        report = site_report(two_runs, TWO_SITE_METADATA)
        pooled = report.loc[(report["gap_class"] == "all") & (report["subset"] == "all")]
        assert pooled["rmse"].nunique() == 2

    def test_the_summary_is_stratified_by_ecosystem(self, two_runs):
        table = stratified_summary(site_report(two_runs, TWO_SITE_METADATA), metrics=("rmse",))
        pooled = cell(table, gap_class="all", subset="all", stratum="all")
        assert pooled["n_sites"] == 2
        for igbp in ("ENF", "GRA"):
            assert cell(table, gap_class="all", subset="all", stratum=igbp)["n_sites"] == 1


# ---------------------------------------------------------------------------
# The published tables themselves (local copy of the supplement required)
# ---------------------------------------------------------------------------

SUPPLEMENT_PREFIX = "1-s2.0-S0168192321004639-"


def supplement(name: str) -> Path:
    """Return the path of one supplementary file, or skip where it is absent.

    The files are publisher material and git-ignored. They are looked for in
    ``$RFRGAPFILL_SUPPLEMENT_DIR``, then ``publication/``, then the repository
    root.
    """
    directories = [ROOT / "publication", ROOT]
    if os.environ.get("RFRGAPFILL_SUPPLEMENT_DIR"):
        directories.insert(0, Path(os.environ["RFRGAPFILL_SUPPLEMENT_DIR"]))
    for directory in directories:
        path = directory / f"{SUPPLEMENT_PREFIX}{name}"
        if path.exists():
            return path
    pytest.skip(f"supplementary file {name} is not available locally")


@pytest.fixture(scope="module")
def table_s2() -> pd.DataFrame:
    return read_table_s2(supplement("mmc3.docx"))


@pytest.fixture(scope="module")
def tables_s4_to_s6() -> pd.DataFrame:
    return read_tables_s4_to_s6(supplement("mmc5.docx"))


@pytest.fixture(scope="module")
def table_s9() -> pd.DataFrame:
    return read_table_s9(supplement("mmc8.docx"))


def s10_number(text: str) -> float:
    """Return one Table S10 number, written ``-0.24`` or ``8.17x10-02``."""
    plain = text.replace("*", "").replace("\N{MINUS SIGN}", "-").strip()
    match = re.fullmatch(r"(-?[\d.]+)(?:\s*\N{MULTIPLICATION SIGN}\s*10([-+]?\d+))?", plain)
    assert match, text
    return float(match[1]) * (10.0 ** int(match[2]) if match[2] else 1.0)


@pytest.mark.supplement
class TestTheSupplementItself:
    """The readers against the publisher's files, and what the files say."""

    def test_table_s2_is_the_194_sites(self, table_s2):
        assert len(table_s2) == 194
        assert int(table_s2["included_in_94_site_subset"].sum()) == 94
        assert set(table_s2["igbp"]) == {
            "CRO", "CSH", "DBF", "EBF", "ENF", "GRA", "MF", "OSH", "SAV", "WET", "WSA",
        }  # fmt: skip
        systems = site_population_summary(table_s2, by="instrument_system")
        assert dict(zip(systems["stratum"], systems["n_sites"], strict=True)) == {
            "C": 69,
            "M": 4,
            "O": 121,
        }

    def test_table_s1_counts_are_table_s2s_bar_one_site(self, table_s2):
        _, tables = sites._docx(supplement("mmc3.docx"))
        rows = tables[0]
        published: dict[str, dict[str, int]] = {}
        for header, counts in pairwise(rows):
            if (
                counts
                and counts[0] == "#"
                and header[0] in ("Continent", "IGBP land classification")
            ):
                labels = header[1:]
                published[header[0]] = {
                    label: int(count) for label, count in zip(labels, counts[1:], strict=False)
                    if label and count
                }  # fmt: skip
        continents = site_population_summary(table_s2, by="continent")
        derived = dict(zip(continents["stratum"], continents["n_sites"], strict=True))
        assert derived == {k: v for k, v in published["Continent"].items() if v}
        # S1 names classes "Cropland (CRO)" - and one without its code.
        igbp_s1 = {
            (re.search(r"\((\w+)\)", label)[1] if "(" in label else "ENF"): count
            for label, count in published["IGBP land classification"].items()
        }
        classes = site_population_summary(table_s2, by="igbp")
        igbp_s2 = dict(zip(classes["stratum"], classes["n_sites"], strict=True))
        # A publisher inconsistency, recorded in docs/supplement_benchmarks.md
        # section 8: S1 counts one site more as DBF and one fewer as EBF than
        # the per-site rows of S2 do. Every other class agrees.
        differing = {k: (igbp_s1[k], igbp_s2[k]) for k in igbp_s1 if igbp_s1[k] != igbp_s2[k]}
        assert differing == {"DBF": (23, 22), "EBF": (12, 13)}

    def test_tables_s4_to_s6_score_the_94_site_subset(self, table_s2, tables_s4_to_s6):
        table = tables_s4_to_s6
        assert tuple(table.columns) == PUBLISHED_SITE_TABLE_COLUMNS
        assert len(table) == 94 * 3 * 3 * 3  # sites x fluxes x methods x subsets
        included = set(table_s2.loc[table_s2["included_in_94_site_subset"], "site_id"])
        assert set(table["site"]) == included
        assert set(table["subset"]) == {"all", "daytime", "nighttime"}
        assert set(table["site_population"]) == {"94-site complete-analysis subset"}

    def test_table_s9_scores_nee_at_every_site(self, table_s2, table_s9):
        assert len(table_s9) == 194 * 2
        assert set(table_s9["site"]) == set(table_s2["site_id"])
        assert set(table_s9["method"]) == {"MDS", "RFR3"}
        assert set(table_s9["units"]) == {NEE_MODEL_UNITS}

    def test_s4_and_s9_print_the_same_nee_skill_and_different_biases(
        self, tables_s4_to_s6, table_s9
    ):
        # The finding recorded under ambiguity A10: at the 94 shared sites the
        # two tables agree on R2, slope and RMSE exactly - so the RMSE that S3
        # and S10 label g C m-2 d-1 is the one S9 labels umol m-2 s-1 - and
        # disagree on bias, so they are not one experiment printed twice.
        s4 = tables_s4_to_s6.query("target == 'NEE' and subset == 'all' and method != 'RFR10'")
        joined = s4.merge(table_s9, on=["site", "method"], suffixes=("_s4", "_s9"))
        assert len(joined) == 94 * 2
        for metric in ("r2", "slope", "rmse"):
            np.testing.assert_array_equal(joined[f"{metric}_s4"], joined[f"{metric}_s9"])
        assert int((joined["bias_s4"] != joined["bias_s9"]).sum()) >= 180

    def test_the_site_report_attaches_table_s2(self, table_s2, tables_s4_to_s6):
        report = site_report(tables_s4_to_s6, table_s2)
        assert bool(report["igbp"].notna().all())
        assert bool(report["included_in_94_site_subset"].all())

    def test_the_table_s10_constants_match_the_supplement(self):
        _, tables = sites._docx(supplement("mmc9.docx"))
        names = {"R2": "r2", "slope": "slope", "bias": "bias", "RMSE": "rmse"}
        found: dict[tuple[str, str, str], tuple[float, float, float, bool]] = {}
        target = None
        for row in tables[0]:
            filled = [value for value in row if value]
            if len(filled) == 1 and filled[0] in FLUXES:
                target = filled[0]
                continue
            if len(row) != 5 or row[0] not in names:
                continue
            for method, mean, interval in (("RFR3", row[1], row[2]), ("RFR10", row[3], row[4])):
                lower, upper = interval.replace("*", "").strip().strip("()").split(",")
                found[target, names[row[0]], method] = (
                    s10_number(mean),
                    s10_number(lower),
                    s10_number(upper),
                    "*" not in mean + interval,
                )
        assert len(found) == 24
        for record in TABLE_S10_COMPARISONS:
            mean, lower, upper, significant = found[record.target, record.metric, record.method]
            assert record.mean_difference == pytest.approx(mean)
            assert (record.ci_lower, record.ci_upper) == pytest.approx((lower, upper))
            assert record.significant is significant

    def test_table_s10_is_reproduced_from_table_s4(self, tables_s4_to_s6):
        report = site_report(tables_s4_to_s6.query("subset == 'all'"))
        welch = pd.concat(
            [
                welch_comparison(report, method="RFR3", baseline="MDS"),
                welch_comparison(report, method="RFR10", baseline="RFR3"),
            ],
            ignore_index=True,
        )
        comparison = compare_to_table_s10(welch)
        assert len(comparison) == 24
        assert bool(comparison["comparable"].all())
        published = comparison["published_mean_difference"].abs()
        # Both sides are rounded to two decimals, and S10 was computed from
        # unrounded per-site values: agreement to 0.05 (plus 1%) is what the
        # printed precision allows. The intervals are printed to one decimal in
        # half their cells, so each end is held to 10% of the published
        # half-width on top of that.
        assert bool((comparison["difference"].abs() <= 0.05 + 0.01 * published).all())
        half_width = (comparison["published_ci_upper"] - comparison["published_ci_lower"]) / 2
        for end in ("lower", "upper"):
            gap = (comparison[f"run_ci_{end}"] - comparison[f"published_ci_{end}"]).abs()
            assert bool((gap <= 0.05 + 0.1 * half_width).all()), end
        # The one disagreement: S10 marks the NEE RMSE gain of RFR3 over MDS
        # significant, while its own interval (-0.9, 0.1) crosses zero. The
        # reproduction agrees with the interval (p ~ 0.07).
        disagreeing = comparison.loc[~comparison["significance_agrees"].astype(bool)]
        assert [tuple(row) for row in disagreeing[["target", "method", "metric"]].to_numpy()] == [
            ("NEE", "RFR3", "rmse")
        ]

    def test_table_s3_medians_are_the_medians_of_tables_s4_to_s6(self, tables_s4_to_s6):
        table = tables_s4_to_s6.loc[:, list(SENSITIVITY_TABLE_COLUMNS)]
        comparison = compare_to_benchmarks(median_across_sites(table))
        assert bool(comparison["comparable"].all())
        # R2 and slope agree to the rounding of a median of 94 two-decimal
        # values. RMSE and bias do not always (docs/supplement_benchmarks.md
        # section 8): the largest gap is LE MDS nighttime RMSE, printed 21.93
        # in S3 against a median of 21.39 in S6.
        shape = comparison.loc[comparison["metric"].isin(["r2", "slope"])]
        assert bool((shape["difference"].abs() <= 0.0051).all())
        rmse = comparison.loc[comparison["metric"] == "rmse"].set_index(
            ["target", "method", "subset"]
        )
        assert rmse.loc[("LE", "MDS", "nighttime"), "run"] == pytest.approx(21.39)
        assert rmse.loc[("LE", "MDS", "nighttime"), "published"] == pytest.approx(21.93)

    def test_rfr3_does_not_improve_at_every_site(self, table_s9):
        # The reason no site is required to improve: in the paper's own
        # 194-site table, RFR3 loses NEE R2 to MDS at some sites.
        report = site_report(table_s9)
        changes = method_differences(report, method="RFR3", baseline="MDS", metrics=("r2",))
        counts = changes["change"].value_counts()
        assert counts["improved"] > counts.get("worsened", 0) > 0

    def test_ecosystems_differ(self, table_s2, tables_s4_to_s6):
        report = site_report(tables_s4_to_s6.query("subset == 'all'"), table_s2)
        table = stratified_summary(report, metrics=("r2",)).query(
            "target == 'NEE' and method == 'RFR3' and stratum != 'all'"
        )
        # Why no threshold is universal: the median NEE R2 of RFR3 spans a
        # wide range across the IGBP classes of the 94-site subset.
        assert table["median"].max() - table["median"].min() > 0.2
