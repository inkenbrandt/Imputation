"""Figures for the gap-length comparison (Step 18, after the tables).

Deliberately the thinnest layer in the package. Every helper takes a table from
:mod:`rfrgapfill.sensitivity` or :mod:`rfrgapfill.benchmarks` and draws it -
none of them takes a validation result, computes a metric, or aggregates
anything. A figure here can therefore be wrong about layout but never about
numbers: whatever it shows, the table beside it shows too.

``matplotlib`` is **not** a dependency of this package. It is imported when a
plot is actually drawn, and its absence is reported as a
:class:`PlottingError` naming the extra to install::

    pip install "rfr-gapfill[notebooks]"

so that importing :mod:`rfrgapfill` on a headless processing machine never
requires a plotting stack it will not use.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Final, cast

import pandas as pd

from rfrgapfill.sensitivity import (
    GAP_CLASS_ORDER,
    METHOD_ORDER,
    METRIC_COLUMNS,
    POOLED_GAP_CLASS,
    SensitivityError,
    gap_length_pivot,
)

if TYPE_CHECKING:  # pragma: no cover - import-time typing only
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

__all__ = [
    "METRIC_LABELS",
    "PlottingError",
    "plot_benchmark_comparison",
    "plot_gap_length_grid",
    "plot_gap_length_sensitivity",
]


class PlottingError(RuntimeError):
    """A figure could not be drawn from what was supplied."""


#: Axis labels for the four core metrics.
METRIC_LABELS: Final[dict[str, str]] = {
    "r2": "$R^2$",
    "slope": "slope (filled on measured)",
    "rmse": "RMSE",
    "bias": "bias",
}

#: The x positions of a sensitivity plot: durations, never the pooled row.
_DURATION_CLASSES: Final[tuple[str, ...]] = tuple(
    gap_class for gap_class in GAP_CLASS_ORDER if gap_class != POOLED_GAP_CLASS
)

#: Duration labels, so an axis reads "24 h" rather than "short".
_DURATION_LABELS: Final[dict[str, str]] = {
    "short": "short\n(24 h)",
    "long": "long\n(7 d)",
    "very_long": "very long\n(30 d)",
}


def _pyplot() -> Any:
    """Return ``matplotlib.pyplot``, or say how to install it."""
    try:
        from matplotlib import pyplot
    except ModuleNotFoundError as error:  # pragma: no cover - depends on the env
        raise PlottingError(
            "matplotlib is required to draw this figure and is not installed; it is "
            'an optional dependency of this package: pip install "rfr-gapfill[notebooks]". '
            "The tables these helpers draw need no plotting stack at all."
        ) from error
    return pyplot


def _axes(ax: Axes | None, **kwargs: Any) -> Axes:
    """Return ``ax``, or a new one on a new figure."""
    if ax is not None:
        return ax
    pyplot = _pyplot()
    _, created = pyplot.subplots(**kwargs)
    axes: Axes = created
    return axes


def plot_gap_length_sensitivity(
    table: pd.DataFrame,
    *,
    target: str | None = None,
    metric: str = "r2",
    subset: str = "all",
    ax: Axes | None = None,
    include_pooled: bool = False,
) -> Axes:
    """Plot one metric against gap duration, one line per method.

    The figure the paper's central claim is about: whether skill holds up as the
    artificial gap grows from 24 hours to 30 days. The x axis is the duration
    classes in order - the pooled ``all`` row is not a duration and is left out
    unless asked for - and each method is a line across them.

    :param table: a :func:`~rfrgapfill.sensitivity.gap_length_table` or
        :func:`~rfrgapfill.sensitivity.median_across_sites` frame, or the
        published :func:`~rfrgapfill.benchmarks.benchmark_table`.
    :param target: the flux to draw. Optional when the table holds exactly one.
    :param metric: one of :data:`~rfrgapfill.sensitivity.METRIC_COLUMNS`.
    :param subset: the day/night subset to draw.
    :param ax: an existing axes to draw on; a new figure is made otherwise.
    :param include_pooled: also plot the pooled ``all`` row, at the right.
    :returns: the axes drawn on.
    :raises PlottingError: if the table holds several targets and none was
        chosen, or if the selection is empty.
    """
    chosen = _single_target(table, target)
    wide = _duration_pivot(
        table, target=chosen, metric=metric, subset=subset, include_pooled=include_pooled
    )
    axes = _axes(ax)
    for method in wide.columns:
        series = wide[method]
        axes.plot(
            range(len(series)),
            series.to_numpy(dtype="float64"),
            marker="o",
            label=str(method),
        )
    axes.set_xticks(range(len(wide.index)))
    axes.set_xticklabels([_DURATION_LABELS.get(str(label), str(label)) for label in wide.index])
    axes.set_xlabel("artificial gap class")
    axes.set_ylabel(_metric_label(table, metric))
    axes.set_title(f"{chosen} - {METRIC_LABELS.get(metric, metric)} by gap length ({subset})")
    axes.legend(title="method")
    return axes


def plot_gap_length_grid(
    table: pd.DataFrame,
    *,
    targets: Sequence[str] | None = None,
    metric: str = "r2",
    subset: str = "all",
    include_pooled: bool = False,
    figsize: tuple[float, float] | None = None,
) -> Figure:
    """Draw :func:`plot_gap_length_sensitivity` for every target, side by side.

    One panel per flux, sharing the y axis so the panels are actually comparable
    - which is the whole point of putting NEE, H and LE on one figure.

    :param table: as :func:`plot_gap_length_sensitivity`.
    :param targets: the fluxes to draw, in order; every target in the table
        otherwise.
    :param metric: one of :data:`~rfrgapfill.sensitivity.METRIC_COLUMNS`.
    :param subset: the day/night subset to draw.
    :param include_pooled: also plot the pooled ``all`` row, at the right.
    :param figsize: figure size in inches; scaled to the panel count otherwise.
    :returns: the figure.
    :raises PlottingError: if the table holds no target to draw.
    """
    wanted = _targets(table, targets)
    pyplot = _pyplot()
    figure, axes = pyplot.subplots(
        1,
        len(wanted),
        figsize=figsize or (4.0 * len(wanted), 3.5),
        sharey=True,
        squeeze=False,
    )
    for panel, target in zip(axes[0], wanted, strict=True):
        plot_gap_length_sensitivity(
            table,
            target=target,
            metric=metric,
            subset=subset,
            ax=panel,
            include_pooled=include_pooled,
        )
        panel.set_title(str(target))
        panel.get_legend().remove()
    axes[0][0].set_ylabel(_metric_label(table, metric))
    handles, labels = axes[0][0].get_legend_handles_labels()
    figure.legend(handles, labels, title="method", loc="center right")
    figure.suptitle(f"{METRIC_LABELS.get(metric, metric)} by gap length ({subset})")
    result: Figure = figure
    return result


def plot_benchmark_comparison(
    comparison: pd.DataFrame,
    *,
    metric: str = "r2",
    ax: Axes | None = None,
) -> Axes:
    """Plot a reproduction run against the published medians, cell by cell.

    A dumbbell per comparable cell: the published median and the run's value on
    one line, so the distance between them *is* the difference. Cells the
    comparison marked incomparable - an unconverted NEE ``RMSE``, ambiguity A10 -
    are not drawn, because a figure has nowhere to put the caveat that goes with
    them.

    :param comparison: a :func:`~rfrgapfill.benchmarks.compare_to_benchmarks`
        frame.
    :param metric: the metric to draw.
    :param ax: an existing axes to draw on; a new figure is made otherwise.
    :returns: the axes drawn on.
    :raises PlottingError: if no comparable cell of ``metric`` is present.
    """
    for column in ("metric", "comparable", "run", "published"):
        if column not in comparison.columns:
            raise PlottingError(
                f"plot_benchmark_comparison needs a {column!r} column; pass the frame "
                "compare_to_benchmarks() returned"
            )
    rows = cast(
        "pd.DataFrame",
        comparison.loc[
            (comparison["metric"].astype(str) == str(metric))
            & comparison["comparable"].astype(bool)
            & comparison["run"].notna()
            & comparison["published"].notna()
        ],
    )
    if rows.empty:
        raise PlottingError(
            f"no comparable {metric!r} cell to draw. Metrics in flux units need the "
            "run and the supplement to be in the same units (A10); the 'note' column "
            "says what each cell was missing"
        )
    labels = [
        f"{record['target']} {record['method']} ({record['gap_class']}/{record['subset']})"
        for _, record in rows.iterrows()
    ]
    positions = range(len(labels))
    axes = _axes(ax, figsize=(6.0, 0.4 * len(labels) + 2.0))
    published = rows["published"].to_numpy(dtype="float64")
    run = rows["run"].to_numpy(dtype="float64")
    for position, low, high in zip(positions, published, run, strict=True):
        axes.plot([low, high], [position, position], color="0.7", zorder=1)
    axes.scatter(published, list(positions), label="published", zorder=2)
    axes.scatter(run, list(positions), label="this run", zorder=2)
    axes.set_yticks(list(positions))
    axes.set_yticklabels(labels)
    axes.invert_yaxis()
    axes.set_xlabel(_comparison_label(rows, metric))
    axes.set_title(f"{METRIC_LABELS.get(metric, metric)}: run against published medians")
    axes.legend()
    return axes


# ---------------------------------------------------------------------------
# Selection helpers
# ---------------------------------------------------------------------------


def _targets(table: pd.DataFrame, targets: Sequence[str] | None) -> tuple[str, ...]:
    """Return the targets to draw, in the order they were asked for."""
    if "target" not in getattr(table, "columns", ()):
        raise PlottingError("the table carries no 'target' column")
    present = list(dict.fromkeys(str(value) for value in table["target"]))
    if targets is None:
        if not present:
            raise PlottingError("the table holds no target to draw")
        return tuple(present)
    wanted = [str(target) for target in targets]
    unknown = [target for target in wanted if target not in present]
    if unknown:
        raise PlottingError(
            f"the table has no rows for {', '.join(unknown)}; it covers "
            f"{', '.join(present) or 'nothing'}"
        )
    return tuple(wanted)


def _single_target(table: pd.DataFrame, target: str | None) -> str:
    """Return the one target to draw, or say which ones are on offer."""
    present = _targets(table, None if target is None else [target])
    if len(present) > 1:
        raise PlottingError(
            f"the table covers {', '.join(present)}; pass target= to choose one, or "
            "plot_gap_length_grid() to draw them all"
        )
    return present[0]


def _duration_pivot(
    table: pd.DataFrame,
    *,
    target: str,
    metric: str,
    subset: str,
    include_pooled: bool,
) -> pd.DataFrame:
    """Return gap classes by method for one target, in duration order."""
    if metric not in METRIC_COLUMNS:
        raise PlottingError(
            f"unknown metric {metric!r}; expected one of {', '.join(METRIC_COLUMNS)}"
        )
    try:
        wide = gap_length_pivot(table, metric=metric, subset=subset, targets=[target])
    except SensitivityError as error:
        raise PlottingError(str(error)) from error
    # The pivot is indexed by (target, gap_class); one target leaves the classes.
    wide = wide.droplevel("target")
    classes = [
        gap_class
        for gap_class in GAP_CLASS_ORDER
        if gap_class in wide.index and (include_pooled or gap_class != POOLED_GAP_CLASS)
    ]
    if not classes:
        raise PlottingError(
            f"{target} has no gap-class rows to draw for metric={metric!r}, subset={subset!r}"
        )
    drawn: pd.DataFrame = cast("pd.DataFrame", wide.loc[classes])
    kept = [method for method in METHOD_ORDER if method in drawn.columns]
    return cast("pd.DataFrame", drawn.loc[:, kept])


def _metric_label(table: pd.DataFrame, metric: str) -> str:
    """Return the axis label for ``metric``, with units where the table gives one."""
    label = METRIC_LABELS.get(metric, metric)
    if metric in {"rmse", "bias"} and "units" in getattr(table, "columns", ()):
        units = sorted({str(value) for value in table["units"] if value is not None})
        if len(units) == 1:
            return f"{label} ({units[0]})"
    return label


def _comparison_label(rows: pd.DataFrame, metric: str) -> str:
    """Return the axis label for a comparison figure, with its units."""
    label = METRIC_LABELS.get(metric, metric)
    if "published_units" in rows.columns:
        units = sorted({str(value) for value in rows["published_units"] if value is not None})
        if len(units) == 1 and units[0] != "dimensionless":
            return f"{label} ({units[0]})"
    return label
