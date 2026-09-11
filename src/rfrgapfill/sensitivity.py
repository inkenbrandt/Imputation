"""Gap-length sensitivity reporting (``docs/method_spec.md`` section 6.3; Step 18).

The layer above :mod:`rfrgapfill.validation`. A validation run already scores
itself per gap class - that is section 6.3 - and
:meth:`~rfrgapfill.validation.ValidationReport.to_frame` already hands back the
tidy table for *one* run. What the paper's central comparison needs is the table
for *many*: RFR3 against RFR10 against the ORF benchmark, across the sites of a
reproduction study, laid out so that the question "does skill hold up as the gap
grows" is one row selection away::

    table = gap_length_table({"US-Ha1": [rfr3, rfr10], "FI-Hyy": [...]})
    medians = median_across_sites(table)
    gap_length_pivot(medians, metric="r2", gap_class="very_long")

Nothing here computes a metric. Every number in these tables was produced by
:mod:`rfrgapfill.metrics` inside a validation run; this module only labels,
concatenates, aggregates and reshapes, which is why it can equally be handed
tidy frames read back from disk rather than live results. A 94-site reproduction
does not hold 94 fitted forests in memory, and it should not have to.

Three properties the tables are built to have:

* **the gap classes are in duration order, always.** ``short``, ``long``,
  ``very_long`` - never alphabetical, which would put ``long`` before ``short``
  and make a sensitivity plot read backwards. The pooled row, over every class at
  once, is labelled ``all`` and sorts first.
* **undefined is missing, not zero.** :mod:`rfrgapfill.metrics` returns ``None``
  for a metric that has no value on its subset; in a frame that is ``NaN``, and
  the metric columns are always float, so a class that scored nothing anywhere
  cannot arrive as an object column that a later median chokes on.
* **units travel with the numbers.** ``r2`` and ``slope`` are dimensionless, but
  ``rmse`` and ``bias`` are in the units of the flux, and the published NEE
  benchmarks are not in the units this package models NEE in (ambiguity A10).
  The ``units`` column is what lets :mod:`rfrgapfill.benchmarks` refuse a
  comparison instead of silently making one.

Plotting helpers live in :mod:`rfrgapfill.plotting` and consume these tables
rather than validation results, so a figure cannot disagree with the table
printed beside it.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import Any, Final, cast

import pandas as pd

from rfrgapfill.validation import METRIC_TABLE_COLUMNS, TargetValidation, ValidationReport

__all__ = [
    "DEFAULT_TARGET_UNITS",
    "GAP_CLASS_ORDER",
    "METHOD_ORDER",
    "METRIC_COLUMNS",
    "POOLED_GAP_CLASS",
    "SENSITIVITY_TABLE_COLUMNS",
    "SITE_MEDIAN_TABLE_COLUMNS",
    "SUBSET_ORDER",
    "SensitivityError",
    "gap_length_pivot",
    "gap_length_table",
    "median_across_sites",
]


class SensitivityError(ValueError):
    """A reporting table could not be built from what was supplied."""


# ---------------------------------------------------------------------------
# Vocabulary and column order
# ---------------------------------------------------------------------------

#: The label a row carries when it pools every gap class rather than naming one.
POOLED_GAP_CLASS: Final = "all"

#: Gap classes in duration order, pooled row first (method_spec.md 4.1).
GAP_CLASS_ORDER: Final[tuple[str, ...]] = (POOLED_GAP_CLASS, "short", "long", "very_long")

#: Observation subsets in reporting order (method_spec.md 6.2).
SUBSET_ORDER: Final[tuple[str, ...]] = ("all", "daytime", "nighttime")

#: Methods in the order Supplementary Table S3 lists them, benchmark arms last.
METHOD_ORDER: Final[tuple[str, ...]] = (
    "MDS",
    "RFR3",
    "RFR10",
    "ORF3",
    "ORF10",
    # Historical-compatibility arms (feature_mode="legacy_fluxlib"), after the
    # arms the paper reports, so they read as a comparison rather than a result.
    "RFR3-legacy",
    "RFR10-legacy",
)

#: The four core metrics, as column names (method_spec.md 6.1).
METRIC_COLUMNS: Final[tuple[str, ...]] = ("r2", "slope", "rmse", "bias")

#: Column order of :func:`gap_length_table`.
SENSITIVITY_TABLE_COLUMNS: Final[tuple[str, ...]] = (
    "site",
    "target",
    "method",
    "mode",
    "gap_class",
    "subset",
    "units",
    "n",
    "n_offered",
    "r2",
    "slope",
    "rmse",
    "bias",
)

#: Column order of :func:`median_across_sites`.
SITE_MEDIAN_TABLE_COLUMNS: Final[tuple[str, ...]] = (
    "target",
    "method",
    "mode",
    "gap_class",
    "subset",
    "units",
    "n_sites",
    "n",
    "n_offered",
    "r2",
    "slope",
    "rmse",
    "bias",
)

#: Units of ``rmse`` and ``bias`` for the paper's three fluxes, as this package
#: models them. NEE is a molar flux here and a daily carbon flux in Table S3,
#: which is ambiguity A10 and the reason this mapping is written down at all.
DEFAULT_TARGET_UNITS: Final[Mapping[str, str]] = {
    "NEE": "umol m-2 s-1",
    "H": "W m-2",
    "LE": "W m-2",
}

#: Columns a tidy frame must already carry to be accepted in place of a result.
_REQUIRED_FRAME_COLUMNS: Final[frozenset[str]] = frozenset(
    {"target", "method", "gap_class", "subset", *METRIC_COLUMNS}
)

#: Integer-valued columns, kept nullable so a table read back from CSV that lost
#: a row count is still a valid table rather than a float count of half hours.
_COUNT_COLUMNS: Final[tuple[str, ...]] = ("n", "n_offered", "n_sites")

#: What the table builders accept in place of a live validation result.
Reportish = ValidationReport | TargetValidation | pd.DataFrame


# ---------------------------------------------------------------------------
# The tidy table
# ---------------------------------------------------------------------------


def gap_length_table(
    reports: Reportish | Iterable[Any] | Mapping[str, Any],
    *,
    site: str | None = None,
    units: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Return one tidy metric table covering every result supplied.

    The Step 18 table: one row per site, target, method, gap class and day/night
    subset, carrying the four core metrics and the row counts behind them. The
    pooled ``gap_class="all"`` rows of each result are kept, because the paper's
    diel medians are exactly those rows and dropping them would put the published
    comparison out of reach.

    :param reports: a :class:`~rfrgapfill.validation.ValidationReport`, a
        :class:`~rfrgapfill.validation.TargetValidation`, an already tidy
        :class:`~pandas.DataFrame`, any iterable of those, or a mapping from site
        label to any of them. A mapping is how a multi-site reproduction labels
        its runs; a frame is how it aggregates runs it no longer holds in memory.
    :param site: one site label for every result supplied. Mutually exclusive
        with a mapping, which carries its own labels.
    :param units: units of ``rmse`` and ``bias`` per target, layered over
        :data:`DEFAULT_TARGET_UNITS`. Pass this when a flux is not in the units
        this package models it in - NEE already converted to ``g C m-2 d-1``,
        say - so that a later benchmark comparison compares like with like.
    :returns: a frame with :data:`SENSITIVITY_TABLE_COLUMNS`, sorted by site,
        target, method, gap-class duration and subset.
    :raises SensitivityError: if something other than a result or a tidy frame is
        supplied, or if a supplied frame is missing a required column.
    """
    if site is not None and isinstance(reports, Mapping):
        raise SensitivityError(
            "pass site= or a mapping of site labels, not both: the mapping keys "
            "already label every result"
        )
    unit_map = {**DEFAULT_TARGET_UNITS, **(dict(units) if units is not None else {})}
    parts = [_labelled_frame(result, label, unit_map) for label, result in _walk(reports, site)]
    if not parts:
        return _empty(SENSITIVITY_TABLE_COLUMNS)
    table: pd.DataFrame = pd.concat(parts, ignore_index=True)
    return _ordered(_typed(table, SENSITIVITY_TABLE_COLUMNS))


def _walk(
    reports: Reportish | Iterable[Any] | Mapping[str, Any],
    site: str | None,
) -> Iterator[tuple[str | None, Reportish]]:
    """Yield every supplied result with the site label that applies to it."""
    if isinstance(reports, Mapping):
        for label, value in reports.items():
            for result in _flatten(value):
                yield str(label), result
        return
    for result in _flatten(reports):
        yield (None if site is None else str(site)), result


def _flatten(value: object) -> Iterator[Reportish]:
    """Yield the individual results inside ``value``, however they were nested."""
    # `ValidationReport` is itself iterable - over its targets - so it has to be
    # recognised before the generic iterable branch, or a report would be taken
    # apart into results that no longer know they shared a gap manifest.
    if isinstance(value, ValidationReport | TargetValidation | pd.DataFrame):
        yield value
        return
    if isinstance(value, Mapping):
        raise SensitivityError(
            "nested mappings are not accepted: pass one mapping of site label to "
            "result(s), or an iterable of results"
        )
    if isinstance(value, Iterable) and not isinstance(value, str | bytes):
        for item in value:
            yield from _flatten(item)
        return
    raise SensitivityError(
        "expected a ValidationReport, a TargetValidation, a tidy DataFrame or an "
        f"iterable of those, got {type(value).__name__}"
    )


def _labelled_frame(
    result: Reportish,
    site: str | None,
    unit_map: Mapping[str, str],
) -> pd.DataFrame:
    """Return one result as a tidy frame carrying its site label and its units."""
    if isinstance(result, pd.DataFrame):
        missing = sorted(_REQUIRED_FRAME_COLUMNS - set(result.columns))
        if missing:
            raise SensitivityError(
                f"tidy frame is missing required column(s): {', '.join(missing)}; a "
                f"validation table carries {', '.join(METRIC_TABLE_COLUMNS)}"
            )
        frame = result.copy()
    else:
        frame = result.to_frame()
    if site is not None or "site" not in frame.columns:
        # An explicit label wins over whatever the frame carried: the caller
        # naming the site now knows something the saved table did not.
        frame["site"] = site
    if "units" not in frame.columns:
        frame["units"] = [unit_map.get(str(target)) for target in frame["target"]]
    else:
        frame["units"] = [
            unit_map.get(str(target)) if _is_missing(existing) else existing
            for target, existing in zip(frame["target"], frame["units"], strict=True)
        ]
    for column in SENSITIVITY_TABLE_COLUMNS:
        if column not in frame.columns:
            frame[column] = None
    selected: pd.DataFrame = frame.loc[:, list(SENSITIVITY_TABLE_COLUMNS)]
    return selected


def _is_missing(value: object) -> bool:
    """Whether ``value`` is one of the several ways pandas spells "absent"."""
    # Which one arrives depends on the dtype a frame happened to be built with -
    # a table read back from CSV and one built in memory do not agree - so all
    # of them are recognised rather than the float `NaN` alone.
    if value is None or value is pd.NA or value is pd.NaT:
        return True
    return isinstance(value, float) and bool(pd.isna(value))


def _typed(table: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Return ``table`` with metric columns float and count columns nullable int."""
    typed: pd.DataFrame = cast("pd.DataFrame", table.loc[:, list(columns)]).copy()
    for column in METRIC_COLUMNS:
        if column in typed.columns:
            # `to_numeric` rather than a cast: a column of nothing but `None` -
            # every subset undefined - arrives as object dtype, and a median over
            # that raises rather than reporting "no value".
            typed[column] = pd.to_numeric(typed[column], errors="coerce").astype("float64")
    for column in _COUNT_COLUMNS:
        if column in typed.columns:
            typed[column] = pd.to_numeric(typed[column], errors="coerce").astype("Int64")
    return typed


def _rank(values: Iterable[object], order: Sequence[str]) -> list[float]:
    """Return sort positions for ``values``; unknown labels sort last, stably."""
    positions = {label: index for index, label in enumerate(order)}
    return [float(positions.get(str(value), len(order))) for value in values]


def _ordered(table: pd.DataFrame) -> pd.DataFrame:
    """Return ``table`` in reporting order, gap classes by duration."""
    if table.empty:
        return cast("pd.DataFrame", table.reset_index(drop=True))
    keys = pd.DataFrame(index=table.index)
    keys["site"] = (
        ["" if _is_missing(value) else str(value) for value in table["site"]]
        if "site" in table.columns
        else ""
    )
    keys["target"] = [str(value) for value in table["target"]]
    keys["method_rank"] = _rank(table["method"], METHOD_ORDER)
    keys["method"] = [str(value) for value in table["method"]]
    keys["gap_class"] = _rank(table["gap_class"], GAP_CLASS_ORDER)
    keys["subset"] = _rank(table["subset"], SUBSET_ORDER)
    order = keys.sort_values(list(keys.columns), kind="stable").index
    ordered: pd.DataFrame = table.loc[order].reset_index(drop=True)
    return ordered


def _empty(columns: Sequence[str]) -> pd.DataFrame:
    """Return an empty table with the right columns and the right dtypes."""
    data = {
        column: pd.Series(
            dtype=(
                "float64"
                if column in METRIC_COLUMNS
                else "Int64"
                if column in _COUNT_COLUMNS
                else "object"
            )
        )
        for column in columns
    }
    frame: pd.DataFrame = pd.DataFrame(data)
    return frame


# ---------------------------------------------------------------------------
# Across sites
# ---------------------------------------------------------------------------


def median_across_sites(table: pd.DataFrame) -> pd.DataFrame:
    """Return the median of each metric across sites, per target and gap class.

    The aggregation Supplementary Table S3 reports: a median over the sites of a
    reproduction study, taken separately for every target, method, gap class and
    day/night subset. Quantiles use linear interpolation between order
    statistics, the convention ambiguity A11 settles for every quantile this
    package reports.

    Sites whose value is undefined are skipped rather than counted as zero, so a
    median rests on the sites that had one; ``n_sites`` is how many sites
    contributed a *row* to the group, and a group where every site was undefined
    comes back missing rather than as a number.

    Units are part of the grouping. Two sites reporting NEE in different units
    therefore appear as two rows rather than one meaningless median.

    :param table: a :func:`gap_length_table` frame with a populated ``site``
        column.
    :returns: a frame with :data:`SITE_MEDIAN_TABLE_COLUMNS`.
    :raises SensitivityError: if a site label is missing, or if one site
        contributed the same cell twice - two runs of one arm at one site would
        otherwise weight that site twice in a median across sites.
    """
    _require_columns(table, SENSITIVITY_TABLE_COLUMNS, what="median_across_sites")
    if table.empty:
        return _empty(SITE_MEDIAN_TABLE_COLUMNS)
    unlabelled = int(sum(1 for value in table["site"] if _is_missing(value)))
    if unlabelled:
        raise SensitivityError(
            f"{unlabelled} row(s) carry no site label, so a median across sites is "
            "undefined; label the runs with gap_length_table(site=...) or with a "
            "mapping of site label to result"
        )
    keys = ["site", "target", "method", "mode", "gap_class", "subset", "units"]
    duplicated = table.duplicated(subset=keys, keep=False)
    if bool(duplicated.any()):
        offenders = table.loc[duplicated, keys].drop_duplicates()
        shown = "; ".join(
            " ".join(str(value) for value in row) for row in offenders.head(3).to_numpy()
        )
        raise SensitivityError(
            f"{int(duplicated.sum())} row(s) repeat a site/target/method/gap-class/subset "
            f"cell, which would weight those sites twice: {shown}"
            + (" ..." if len(offenders) > 3 else "")
        )
    group_keys = [key for key in keys if key != "site"]
    # `dropna=False`: a target whose units are unknown is still a group, and
    # pandas would otherwise drop those rows out of the aggregation entirely.
    grouped = table.groupby(group_keys, dropna=False, sort=False)
    aggregated: pd.DataFrame = grouped.agg(
        n_sites=("site", "nunique"),
        n=("n", "sum"),
        n_offered=("n_offered", "sum"),
        r2=("r2", "median"),
        slope=("slope", "median"),
        rmse=("rmse", "median"),
        bias=("bias", "median"),
    ).reset_index()
    return _ordered(_typed(aggregated, SITE_MEDIAN_TABLE_COLUMNS))


# ---------------------------------------------------------------------------
# The readable view
# ---------------------------------------------------------------------------


def gap_length_pivot(
    table: pd.DataFrame,
    *,
    metric: str = "r2",
    subset: str | None = "all",
    gap_class: str | None = None,
    targets: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Return one metric as a target-by-method table, for reading.

    The shape the paper's comparison is quoted in - fluxes down the side, methods
    across the top - and the shape Step 18's benchmark blocks are written in::

        gap_length_pivot(medians, metric="r2", gap_class="all")
        method   MDS   RFR3  RFR10
        NEE     0.72   0.78   0.84

    With ``gap_class=None`` the classes stay on the index beneath the target,
    which is the sensitivity view itself: one block per flux, one row per
    duration.

    :param table: a :func:`gap_length_table` or :func:`median_across_sites`
        frame, or any frame carrying the same key and metric columns - the
        published table of :func:`rfrgapfill.benchmarks.benchmark_table`
        included.
    :param metric: one of :data:`METRIC_COLUMNS`.
    :param subset: the day/night subset to show; ``None`` keeps every subset and
        puts it on the index.
    :param gap_class: the one class to show, or ``None`` for every class.
    :param targets: the targets to keep, in the order they should appear.
    :returns: a frame indexed by target - and by gap class and subset, where
        those were not fixed - with one column per method.
    :raises SensitivityError: on an unknown metric or an empty selection, or if
        the selection leaves two rows in one cell, which means the table mixes
        sites or runs and needs :func:`median_across_sites` first.
    """
    if metric not in METRIC_COLUMNS:
        raise SensitivityError(
            f"unknown metric {metric!r}; expected one of {', '.join(METRIC_COLUMNS)}"
        )
    _require_columns(table, ("target", "method", "gap_class", "subset", metric), what="pivot")
    selected = table
    if subset is not None:
        selected = cast("pd.DataFrame", selected.loc[selected["subset"].astype(str) == str(subset)])
    if gap_class is not None:
        selected = cast(
            "pd.DataFrame", selected.loc[selected["gap_class"].astype(str) == str(gap_class)]
        )
    if targets is not None:
        wanted = [str(target) for target in targets]
        selected = cast("pd.DataFrame", selected.loc[selected["target"].astype(str).isin(wanted)])
    if selected.empty:
        raise SensitivityError(
            f"nothing to show for metric={metric!r}, subset={subset!r}, "
            f"gap_class={gap_class!r}"
            + (f", targets={list(targets)!r}" if targets is not None else "")
        )
    index = ["target"]
    if gap_class is None:
        index.append("gap_class")
    if subset is None:
        index.append("subset")
    # `pivot`, never `pivot_table`: two rows in one cell mean a mixed table, and
    # averaging them silently is exactly the error this should surface.
    try:
        wide: pd.DataFrame = selected.pivot(  # noqa: PD010 - duplicates must raise
            index=index, columns="method", values=metric
        )
    except ValueError as error:
        raise SensitivityError(
            "two or more rows fall in the same target/method cell; aggregate a "
            "multi-site table with median_across_sites() first"
        ) from error
    wide = cast("pd.DataFrame", wide.loc[:, _method_order(wide.columns)])
    wide = wide.reindex(_pivot_index_order(wide, targets))
    wide.columns.name = None
    return wide


def _method_order(methods: Iterable[object]) -> list[str]:
    """Return ``methods`` in published order; unknown ones alphabetical, last."""
    names = [str(method) for method in methods]
    return sorted(names, key=lambda name: (_rank([name], METHOD_ORDER)[0], name))


def _pivot_index_order(wide: pd.DataFrame, targets: Sequence[str] | None) -> pd.Index:
    """Return the pivot's index in reporting order: targets, then duration."""
    frame = wide.index.to_frame(index=False)
    target_order = (
        {str(target): position for position, target in enumerate(targets)}
        if targets is not None
        else {}
    )
    keys = pd.DataFrame(index=frame.index)
    keys["target_rank"] = [
        float(target_order.get(str(value), len(target_order))) for value in frame["target"]
    ]
    keys["target"] = [str(value) for value in frame["target"]]
    if "gap_class" in frame.columns:
        keys["gap_class"] = _rank(frame["gap_class"], GAP_CLASS_ORDER)
    if "subset" in frame.columns:
        keys["subset"] = _rank(frame["subset"], SUBSET_ORDER)
    order = keys.sort_values(list(keys.columns), kind="stable").index
    return cast("pd.Index[Any]", wide.index[order])


def _require_columns(table: object, columns: Iterable[str], *, what: str) -> None:
    """Raise unless ``table`` is a frame carrying every column ``what`` needs."""
    if not isinstance(table, pd.DataFrame):
        raise SensitivityError(f"{what} needs a DataFrame, got {type(table).__name__}")
    missing = [column for column in columns if column not in table.columns]
    if missing:
        raise SensitivityError(
            f"{what} needs column(s) {', '.join(missing)}; the frame carries "
            f"{', '.join(str(column) for column in table.columns) or 'nothing'}"
        )
