"""Figures for the gap-length comparison. Covers Step 18's plotting helpers.

Step 18 puts the plots last, after the tables are correct, and these tests keep
them there: every assertion below reads a number back off an artist and compares
it with the table the figure was drawn from. A figure that disagreed with its own
table would pass no test here.

``matplotlib`` is an optional dependency, so the drawing tests skip without it -
but the test that the absence is *reported properly* runs either way, because
that error message is what a user without matplotlib actually sees.
"""

from __future__ import annotations

import builtins

import pandas as pd
import pytest

from rfrgapfill.benchmarks import compare_to_benchmarks, convert_nee_to_carbon_units
from rfrgapfill.plotting import (
    PlottingError,
    plot_benchmark_comparison,
    plot_gap_length_grid,
    plot_gap_length_sensitivity,
)
from rfrgapfill.sensitivity import gap_length_table

try:  # matplotlib is in the `notebooks` extra, never in the core requirements
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot

    HAVE_MATPLOTLIB = True
except ModuleNotFoundError:  # pragma: no cover - depends on the environment
    HAVE_MATPLOTLIB = False

needs_matplotlib = pytest.mark.skipif(
    not HAVE_MATPLOTLIB, reason="matplotlib is an optional dependency of this package"
)

#: R2 by gap class for each arm, decreasing with duration as the paper's does.
SCORES = {
    ("NEE", "RFR3"): {"all": 0.78, "short": 0.80, "long": 0.76, "very_long": 0.74},
    ("NEE", "RFR10"): {"all": 0.84, "short": 0.86, "long": 0.80, "very_long": 0.74},
    ("H", "RFR3"): {"all": 0.78, "short": 0.81, "long": 0.77, "very_long": 0.76},
    ("H", "RFR10"): {"all": 0.90, "short": 0.92, "long": 0.90, "very_long": 0.89},
}


@pytest.fixture(autouse=True)
def close_figures():
    """Close every figure a test opened, so a run never leaks them."""
    yield
    if HAVE_MATPLOTLIB:
        pyplot.close("all")


@pytest.fixture
def table() -> pd.DataFrame:
    """Two arms over two fluxes, scored by gap class."""
    rows = [
        {
            "target": target,
            "method": method,
            "mode": method,
            "gap_class": gap_class,
            "subset": subset,
            "n": 100,
            "n_offered": 105,
            "r2": r2 if subset == "all" else r2 - 0.10,
            "slope": 0.9,
            "rmse": 2.5 if target == "NEE" else 39.0,
            "bias": -0.01,
        }
        for (target, method), by_class in SCORES.items()
        for gap_class, r2 in by_class.items()
        for subset in ("all", "daytime", "nighttime")
    ]
    return gap_length_table(pd.DataFrame(rows), site="US-Ha1")


# ---------------------------------------------------------------------------
# The optional dependency
# ---------------------------------------------------------------------------


def test_a_missing_matplotlib_names_the_extra_to_install(monkeypatch, table):
    real_import = builtins.__import__

    def without_matplotlib(name, *args, **kwargs):
        if name == "matplotlib" or name.startswith("matplotlib."):
            raise ModuleNotFoundError(f"No module named {name!r}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_matplotlib)
    with pytest.raises(PlottingError, match=r"rfr-gapfill\[notebooks\]"):
        plot_gap_length_sensitivity(table, target="NEE")


# ---------------------------------------------------------------------------
# The sensitivity figure
# ---------------------------------------------------------------------------


@needs_matplotlib
class TestGapLengthSensitivity:
    """One metric against gap duration, one line per method."""

    def test_the_x_axis_is_the_durations_in_order(self, table):
        axes = plot_gap_length_sensitivity(table, target="NEE")
        labels = [text.get_text() for text in axes.get_xticklabels()]
        assert labels == ["short\n(24 h)", "long\n(7 d)", "very long\n(30 d)"]

    def test_the_pooled_row_is_not_a_duration(self, table):
        axes = plot_gap_length_sensitivity(table, target="NEE")
        assert len(axes.get_lines()[0].get_xdata()) == 3

    def test_the_pooled_row_can_be_asked_for(self, table):
        axes = plot_gap_length_sensitivity(table, target="NEE", include_pooled=True)
        assert len(axes.get_lines()[0].get_xdata()) == 4

    def test_one_line_per_method_in_published_order(self, table):
        axes = plot_gap_length_sensitivity(table, target="NEE")
        assert [line.get_label() for line in axes.get_lines()] == ["RFR3", "RFR10"]

    def test_the_drawn_values_are_the_tables_values(self, table):
        axes = plot_gap_length_sensitivity(table, target="H")
        drawn = dict(zip(["RFR3", "RFR10"], axes.get_lines(), strict=True))
        assert list(drawn["RFR10"].get_ydata()) == pytest.approx([0.92, 0.90, 0.89])

    def test_the_subset_is_the_one_that_was_asked_for(self, table):
        axes = plot_gap_length_sensitivity(table, target="NEE", subset="nighttime")
        values = list(axes.get_lines()[0].get_ydata())
        assert values == pytest.approx([0.80 - 0.10, 0.76 - 0.10, 0.74 - 0.10])

    def test_the_axis_label_carries_the_flux_units(self, table):
        one_flux = table.loc[table["target"] == "H"]
        axes = plot_gap_length_sensitivity(one_flux, metric="rmse")
        assert "W m-2" in axes.get_ylabel()

    def test_several_targets_without_a_choice_are_refused(self, table):
        with pytest.raises(PlottingError, match="pass target="):
            plot_gap_length_sensitivity(table)

    def test_a_target_the_table_does_not_cover_is_refused(self, table):
        with pytest.raises(PlottingError, match="no rows for LE"):
            plot_gap_length_sensitivity(table, target="LE")

    def test_an_unknown_metric_is_refused(self, table):
        with pytest.raises(PlottingError, match="unknown metric"):
            plot_gap_length_sensitivity(table, target="NEE", metric="mape")

    def test_it_draws_on_an_axes_it_is_given(self, table):
        _, axes = pyplot.subplots()
        assert plot_gap_length_sensitivity(table, target="NEE", ax=axes) is axes


@needs_matplotlib
class TestGapLengthGrid:
    """The same figure for every flux, side by side."""

    def test_one_panel_per_target(self, table):
        figure = plot_gap_length_grid(table)
        assert len(figure.axes) == 2
        assert [panel.get_title() for panel in figure.axes] == ["H", "NEE"]

    def test_the_panels_appear_in_the_order_asked_for(self, table):
        figure = plot_gap_length_grid(table, targets=["NEE", "H"])
        assert [panel.get_title() for panel in figure.axes] == ["NEE", "H"]

    def test_the_panels_share_a_y_axis_so_they_are_comparable(self, table):
        figure = plot_gap_length_grid(table)
        first, second = figure.axes
        assert first.get_ylim() == second.get_ylim()

    def test_an_absent_target_is_refused(self, table):
        with pytest.raises(PlottingError, match="no rows for LE"):
            plot_gap_length_grid(table, targets=["LE"])


# ---------------------------------------------------------------------------
# The benchmark figure
# ---------------------------------------------------------------------------


@needs_matplotlib
class TestBenchmarkComparison:
    """A run against the published medians, one dumbbell per cell."""

    @staticmethod
    def comparison(table: pd.DataFrame) -> pd.DataFrame:
        return compare_to_benchmarks(convert_nee_to_carbon_units(table))

    def test_one_dumbbell_per_comparable_cell(self, table):
        comparison = self.comparison(table)
        drawn = plot_benchmark_comparison(comparison, metric="r2")
        expected = comparison.loc[
            (comparison["metric"] == "r2") & comparison["comparable"].astype(bool)
        ]
        assert len(drawn.get_yticklabels()) == len(expected)

    def test_the_two_scatters_are_the_published_and_the_run_values(self, table):
        comparison = self.comparison(table)
        axes = plot_benchmark_comparison(comparison, metric="r2")
        published, run = (collection.get_offsets()[:, 0] for collection in axes.collections)
        expected = comparison.loc[
            (comparison["metric"] == "r2") & comparison["comparable"].astype(bool)
        ]
        assert list(published) == pytest.approx(expected["published"].to_list())
        assert list(run) == pytest.approx(expected["run"].to_list())

    def test_an_incomparable_metric_is_not_drawn_at_all(self, table):
        # NEE RMSE before conversion is ambiguity A10; a figure has nowhere to
        # put the caveat that goes with it.
        unconverted = compare_to_benchmarks(table.loc[table["target"] == "NEE"])
        with pytest.raises(PlottingError, match="no comparable"):
            plot_benchmark_comparison(unconverted, metric="rmse")

    def test_a_frame_that_is_not_a_comparison_is_refused(self, table):
        with pytest.raises(PlottingError, match="compare_to_benchmarks"):
            plot_benchmark_comparison(table, metric="r2")
