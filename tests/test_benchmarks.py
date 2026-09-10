"""Published benchmarks and the comparison against them. Covers Step 18.

The first class is the one that matters most: it re-reads
``docs/supplement_benchmarks.md`` and checks every number in
:data:`PUBLISHED_BENCHMARKS` against the table it was transcribed from. The rule
the project works under is that no benchmark number appears in code without a
traceable source, and a test that only compared the constants against themselves
would not enforce it - editing one side would go unnoticed until a reproduction
run quietly compared against the wrong medians.

The rest covers ambiguity A10: Table S3 reports NEE ``RMSE`` and ``bias`` in
``g C m-2 d-1`` and this package models NEE in ``umol m-2 s-1``, so the
comparison must refuse those two cells until the caller converts, and must never
convert on its own.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pytest

from rfrgapfill.benchmarks import (
    BENCHMARK_TABLE_COLUMNS,
    COMPARISON_TABLE_COLUMNS,
    NEE_BENCHMARK_UNITS,
    NEE_MODEL_UNITS,
    NEE_RATE_TO_G_C_PER_DAY,
    PUBLISHED_BENCHMARKS,
    BenchmarkError,
    PublishedBenchmark,
    benchmark_table,
    compare_to_benchmarks,
    convert_nee_to_carbon_units,
)
from rfrgapfill.schema import ConfigError
from rfrgapfill.sensitivity import gap_length_table

BENCHMARK_DOC = Path(__file__).resolve().parents[1] / "docs" / "supplement_benchmarks.md"

FLUXES = ("NEE", "H", "LE")
METHODS = ("MDS", "RFR3", "RFR10")


# ---------------------------------------------------------------------------
# Reading the source document
# ---------------------------------------------------------------------------


def section(title: str) -> str:
    """Return the text of one numbered section of the benchmark document."""
    text = BENCHMARK_DOC.read_text(encoding="utf-8")
    blocks = re.split(r"^## ", text, flags=re.MULTILINE)
    for block in blocks:
        if block.startswith(title):
            return block
    raise AssertionError(f"{BENCHMARK_DOC.name} has no section {title!r}")


def cells(line: str) -> list[str]:
    """Return the cells of one markdown table row."""
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def number(text: str) -> float:
    """Return one tabulated value."""
    return float(text)


def documented_diel() -> dict[tuple[str, str], tuple[float, float, float, float]]:
    """Return the diel medians as ``docs/supplement_benchmarks.md`` records them."""
    found: dict[tuple[str, str], tuple[float, float, float, float]] = {}
    for line in section("2. Diel median performance").splitlines():
        if not line.startswith("|"):
            continue
        row = cells(line)
        if len(row) == 6 and row[0] in FLUXES and row[1] in METHODS:
            found[row[0], row[1]] = (
                number(row[2]),
                number(row[3]),
                number(row[4]),
                number(row[5]),
            )
    assert len(found) == 9, f"expected 9 diel rows, read {len(found)}"
    return found


def documented_very_long() -> dict[tuple[str, str], tuple[float, float, float, float]]:
    """Return the very-long-gap medians as the document records them."""
    found: dict[tuple[str, str], tuple[float, float, float, float]] = {}
    flux: str | None = None
    for line in section("3. Very-long-gap medians").splitlines():
        if line.startswith("### "):
            flux = line.removeprefix("### ").strip()
            continue
        if not line.startswith("|") or flux not in FLUXES:
            continue
        row = cells(line)
        if len(row) == 5 and row[0] in METHODS:
            assert flux is not None
            found[flux, row[0]] = (
                number(row[1]),
                number(row[2]),
                number(row[3]),
                number(row[4]),
            )
    assert len(found) == 9, f"expected 9 very-long rows, read {len(found)}"
    return found


def documented_nighttime() -> dict[tuple[str, str], tuple[float, float, float]]:
    """Return the nighttime medians as the document records them."""
    found: dict[tuple[str, str], tuple[float, float, float]] = {}
    flux: str | None = None
    pattern = re.compile(
        r"^(?P<method>MDS|RFR3|RFR10)\s+R2=(?P<r2>[-\d.]+),\s*"
        r"slope=(?P<slope>[-\d.]+),\s*RMSE=(?P<rmse>[-\d.]+)"
    )
    for line in section("4. Nighttime behaviour").splitlines():
        stripped = line.strip()
        if stripped.endswith("nighttime:"):
            flux = stripped.split()[0]
            continue
        match = pattern.match(stripped)
        if match and flux in FLUXES:
            assert flux is not None
            found[flux, match["method"]] = (
                number(match["r2"]),
                number(match["slope"]),
                number(match["rmse"]),
            )
    assert len(found) == 6, f"expected 6 nighttime rows, read {len(found)}"
    return found


def published(target: str, method: str, gap_class: str, subset: str) -> PublishedBenchmark:
    """Return the one published record matching the four keys."""
    matches = [
        benchmark
        for benchmark in PUBLISHED_BENCHMARKS
        if (benchmark.target, benchmark.method, benchmark.gap_class, benchmark.subset)
        == (target, method, gap_class, subset)
    ]
    assert len(matches) == 1, f"expected one record for {target}/{method}, found {len(matches)}"
    return matches[0]


# ---------------------------------------------------------------------------
# Traceability
# ---------------------------------------------------------------------------


class TestEveryNumberIsTraceable:
    """The constants and ``docs/supplement_benchmarks.md`` say the same thing."""

    def test_the_diel_medians_match_the_document(self):
        for (target, method), values in documented_diel().items():
            record = published(target, method, "all", "all")
            assert (record.r2, record.slope, record.rmse, record.bias) == values

    def test_the_very_long_gap_medians_match_the_document(self):
        for (target, method), values in documented_very_long().items():
            record = published(target, method, "very_long", "all")
            assert (record.r2, record.slope, record.rmse, record.bias) == values

    def test_the_nighttime_medians_match_the_document(self):
        for (target, method), values in documented_nighttime().items():
            record = published(target, method, "all", "nighttime")
            assert (record.r2, record.slope, record.rmse) == values

    def test_nothing_extra_is_tabulated(self):
        # 9 diel + 9 very-long + 6 nighttime, and no fourth block that arrived
        # without a source.
        assert len(PUBLISHED_BENCHMARKS) == 24

    def test_every_record_names_its_source_and_its_population(self):
        for record in PUBLISHED_BENCHMARKS:
            assert "Table S3" in record.source
            assert "94-site" in record.site_subset

    def test_the_supplement_quotes_no_nighttime_bias(self):
        # Absent, not zero: `docs/supplement_benchmarks.md` section 4 tabulates
        # R2, slope and RMSE only.
        for record in PUBLISHED_BENCHMARKS:
            if record.subset == "nighttime":
                assert record.bias is None

    def test_nee_is_published_in_carbon_units_and_the_others_in_watts(self):
        for record in PUBLISHED_BENCHMARKS:
            expected = NEE_BENCHMARK_UNITS if record.target == "NEE" else "W m-2"
            assert record.units == expected


class TestPublishedRecord:
    """The record refuses a transcription that does not fit the vocabulary."""

    def test_an_unknown_subset_is_refused(self):
        with pytest.raises(BenchmarkError, match="unknown subset"):
            PublishedBenchmark(
                target="NEE",
                method="RFR3",
                gap_class="all",
                subset="dusk",
                r2=0.5,
                slope=0.5,
                rmse=1.0,
                bias=0.0,
                units=NEE_BENCHMARK_UNITS,
            )

    def test_an_unknown_gap_class_is_refused(self):
        with pytest.raises(ConfigError):
            PublishedBenchmark(
                target="NEE",
                method="RFR3",
                gap_class="fortnight",
                subset="all",
                r2=0.5,
                slope=0.5,
                rmse=1.0,
                bias=0.0,
                units=NEE_BENCHMARK_UNITS,
            )

    def test_a_duration_alias_resolves_to_the_class_it_names(self):
        record = PublishedBenchmark(
            target="NEE",
            method="RFR3",
            gap_class="30d",
            subset="all",
            r2=0.5,
            slope=0.5,
            rmse=1.0,
            bias=0.0,
            units=NEE_BENCHMARK_UNITS,
        )
        assert record.gap_class == "very_long"


class TestBenchmarkTable:
    """The published values, tidy."""

    def test_columns_are_the_documented_order(self):
        assert tuple(benchmark_table().columns) == BENCHMARK_TABLE_COLUMNS

    def test_every_record_appears(self):
        assert len(benchmark_table()) == len(PUBLISHED_BENCHMARKS)

    def test_it_can_be_narrowed_to_one_block(self):
        table = benchmark_table(gap_classes=["very_long"], subsets=["all"])
        assert len(table) == 9
        assert set(table["target"]) == set(FLUXES)

    def test_it_pivots_like_a_reproduction_table(self):
        from rfrgapfill.sensitivity import gap_length_pivot

        wide = gap_length_pivot(benchmark_table(), metric="r2", gap_class="all", subset="all")
        assert list(wide.columns) == ["MDS", "RFR3", "RFR10"]
        assert wide.loc["NEE", "RFR10"] == pytest.approx(0.84)


# ---------------------------------------------------------------------------
# The unit conversion of ambiguity A10
# ---------------------------------------------------------------------------


def run_table(**changes: object) -> pd.DataFrame:
    """Return a one-site run table to compare against the published medians."""
    defaults: dict[str, object] = {
        "target": "NEE",
        "method": "RFR3",
        "mode": "RFR3",
        "gap_class": "all",
        "subset": "all",
        "n": 100,
        "n_offered": 105,
        "r2": 0.78,
        "slope": 0.79,
        "rmse": 2.5,
        "bias": -0.01,
    }
    defaults.update(changes)
    return gap_length_table(pd.DataFrame([defaults]), site="US-Ha1")


class TestUnitConversion:
    """``umol m-2 s-1`` to ``g C m-2 d-1``, explicitly (A10)."""

    def test_the_factor_is_the_documented_rate_conversion(self):
        assert pytest.approx(12.011e-6 * 86400.0) == NEE_RATE_TO_G_C_PER_DAY
        assert pytest.approx(1.0377504) == NEE_RATE_TO_G_C_PER_DAY

    def test_rmse_and_bias_are_scaled_and_relabelled(self):
        converted = convert_nee_to_carbon_units(run_table())
        row = converted.iloc[0]
        assert row["rmse"] == pytest.approx(2.5 * NEE_RATE_TO_G_C_PER_DAY)
        assert row["bias"] == pytest.approx(-0.01 * NEE_RATE_TO_G_C_PER_DAY)
        assert row["units"] == NEE_BENCHMARK_UNITS

    def test_the_dimensionless_metrics_are_untouched(self):
        # Both are invariant under a common rescaling of measured and filled, so
        # converting them would be wrong rather than merely unnecessary.
        converted = convert_nee_to_carbon_units(run_table())
        assert converted.iloc[0]["r2"] == pytest.approx(0.78)
        assert converted.iloc[0]["slope"] == pytest.approx(0.79)

    def test_the_input_frame_is_not_modified(self):
        table = run_table()
        before = table["rmse"].to_list()
        convert_nee_to_carbon_units(table)
        assert table["rmse"].to_list() == before

    def test_it_is_safe_to_apply_twice(self):
        once = convert_nee_to_carbon_units(run_table())
        twice = convert_nee_to_carbon_units(once)
        assert twice.iloc[0]["rmse"] == pytest.approx(once.iloc[0]["rmse"])

    def test_other_fluxes_are_left_alone(self):
        table = run_table(target="H", rmse=39.0)
        converted = convert_nee_to_carbon_units(table)
        assert converted.iloc[0]["rmse"] == pytest.approx(39.0)
        assert converted.iloc[0]["units"] == "W m-2"

    def test_unknown_units_are_an_error_rather_than_an_assumption(self):
        table = gap_length_table(
            pd.DataFrame([dict(run_table().iloc[0])]).drop(columns=["units"]),
            site="US-Ha1",
            units={"NEE": "mg m-2 s-1"},
        )
        with pytest.raises(BenchmarkError, match="cannot convert NEE"):
            convert_nee_to_carbon_units(table)


# ---------------------------------------------------------------------------
# The comparison
# ---------------------------------------------------------------------------


def comparison_cell(comparison: pd.DataFrame, metric: str) -> pd.Series:
    """Return the one comparison row for ``metric``."""
    rows = comparison.loc[comparison["metric"] == metric]
    assert len(rows) == 1, f"expected one {metric} row, found {len(rows)}"
    return rows.iloc[0]


class TestComparison:
    """A reproduction run beside the published medians."""

    def test_columns_are_the_documented_order(self):
        assert tuple(compare_to_benchmarks(run_table()).columns) == COMPARISON_TABLE_COLUMNS

    def test_the_dimensionless_metrics_compare_immediately(self):
        comparison = compare_to_benchmarks(run_table())
        r2 = comparison_cell(comparison, "r2")
        assert bool(r2["comparable"])
        assert r2["published"] == pytest.approx(0.78)
        assert r2["difference"] == pytest.approx(0.0)

    def test_the_difference_is_the_run_minus_the_published_value(self):
        comparison = compare_to_benchmarks(run_table(r2=0.83))
        assert comparison_cell(comparison, "r2")["difference"] == pytest.approx(0.05)

    def test_nee_rmse_is_refused_until_it_is_converted(self):
        # Ambiguity A10: the run is in molar units and Table S3 is in daily
        # carbon units, and comparing the two numbers would be meaningless.
        rmse = comparison_cell(compare_to_benchmarks(run_table()), "rmse")
        assert not bool(rmse["comparable"])
        assert pd.isna(rmse["difference"])
        assert "A10" in str(rmse["note"])
        assert "convert_nee_to_carbon_units" in str(rmse["note"])

    def test_both_sides_units_are_reported_rather_than_one(self):
        rmse = comparison_cell(compare_to_benchmarks(run_table()), "rmse")
        assert rmse["run_units"] == NEE_MODEL_UNITS
        assert rmse["published_units"] == NEE_BENCHMARK_UNITS

    def test_nee_rmse_compares_once_it_has_been_converted(self):
        comparison = compare_to_benchmarks(convert_nee_to_carbon_units(run_table()))
        rmse = comparison_cell(comparison, "rmse")
        assert bool(rmse["comparable"])
        assert rmse["run"] == pytest.approx(2.5 * NEE_RATE_TO_G_C_PER_DAY)
        assert rmse["difference"] == pytest.approx(2.5 * NEE_RATE_TO_G_C_PER_DAY - 2.63)

    def test_an_energy_flux_compares_without_any_conversion(self):
        comparison = compare_to_benchmarks(run_table(target="H", rmse=42.0))
        rmse = comparison_cell(comparison, "rmse")
        assert bool(rmse["comparable"])
        assert rmse["difference"] == pytest.approx(42.0 - 39.37)

    def test_a_metric_the_supplement_does_not_tabulate_says_so(self):
        comparison = compare_to_benchmarks(run_table(target="H", subset="nighttime"))
        bias = comparison_cell(comparison, "bias")
        assert pd.isna(bias["published"])
        assert pd.isna(bias["difference"])
        assert "does not tabulate" in str(bias["note"])

    def test_a_metric_the_run_left_undefined_says_so(self):
        comparison = compare_to_benchmarks(run_table(r2=None))
        r2 = comparison_cell(comparison, "r2")
        assert pd.isna(r2["run"])
        assert "undefined" in str(r2["note"])

    def test_the_population_of_the_published_value_travels_with_it(self):
        # A single site compared against a 94-site median is a single draw
        # against a distribution, and the row says so.
        comparison = compare_to_benchmarks(run_table())
        assert set(comparison["site_subset"]) == {"94-site complete-analysis subset"}

    def test_only_the_requested_metrics_are_compared(self):
        comparison = compare_to_benchmarks(run_table(), metrics=["r2"])
        assert set(comparison["metric"]) == {"r2"}

    def test_an_unknown_metric_is_refused(self):
        with pytest.raises(BenchmarkError, match="unknown metric"):
            compare_to_benchmarks(run_table(), metrics=["mape"])

    def test_a_multi_site_table_is_refused_rather_than_averaged(self):
        table = pd.concat(
            [run_table(r2=0.70), run_table(r2=0.90).assign(site="FI-Hyy")], ignore_index=True
        )
        with pytest.raises(BenchmarkError, match="median_across_sites"):
            compare_to_benchmarks(table)

    def test_an_orf_arm_has_nothing_published_to_compare_against(self):
        # Supplementary Figure S1 is a figure; there are no ORF numbers to
        # transcribe, and inventing a comparison would be worse than refusing.
        with pytest.raises(BenchmarkError, match="no cell of this table"):
            compare_to_benchmarks(run_table(method="ORF3", mode="RFR3"))

    def test_an_empty_table_is_refused(self):
        with pytest.raises(BenchmarkError, match="empty"):
            compare_to_benchmarks(gap_length_table([]))
