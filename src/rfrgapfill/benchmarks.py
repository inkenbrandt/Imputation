"""Published reproduction benchmarks and the comparison against them (Step 18).

Supplementary Table S3 as data. Every number in :data:`PUBLISHED_BENCHMARKS` is
transcribed from ``docs/supplement_benchmarks.md``, carries the supplement it
came from, and names the site population it is a median over. Nothing here is
computed and nothing here is a threshold:

* these are **reproduction benchmarks**, valid for a run over matching FLUXNET
  inputs with matching preprocessing. A new station is never required to match
  them, and no function in this module asserts anything;
* the published values are **medians across the 94-site complete-analysis
  subset**, so the quantity a reproduction compares against them is the median
  across its own sites - :func:`~rfrgapfill.sensitivity.median_across_sites` -
  and a single site compared against them is a single draw against a
  distribution. :func:`compare_to_benchmarks` says which population each row
  came from rather than assuming the caller remembers;
* ``MDS`` appears here and nowhere else in the package. It is an external
  reference method (``method_spec.md`` section 1), implemented by no code of
  ours, and it is included because the comparison the paper is *about* is RFR
  against MDS.

**Units, ambiguity A10.** Table S3 reports NEE ``RMSE`` and ``bias`` in
``g C m-2 d-1`` while this package models NEE in ``umol m-2 s-1``, and the paper
does not spell out how it aggregated half-hourly residuals into daily carbon
units. So the conversion is explicit and refusable, never automatic:
:func:`compare_to_benchmarks` marks an NEE ``RMSE`` or ``bias`` row
``comparable=False`` until the caller has converted with
:func:`convert_nee_to_carbon_units`, whose factor converts a *rate* and is not a
claim about the paper's aggregation. ``R2`` and ``slope`` are dimensionless and
are unaffected either way, so they compare from the start.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, cast

import pandas as pd

from rfrgapfill.schema import FrozenRecord, GapClass
from rfrgapfill.sensitivity import (
    GAP_CLASS_ORDER,
    METHOD_ORDER,
    METRIC_COLUMNS,
    POOLED_GAP_CLASS,
    SUBSET_ORDER,
    SensitivityError,
    gap_length_table,
)

__all__ = [
    "BENCHMARK_TABLE_COLUMNS",
    "CARBON_MOLAR_MASS_G",
    "COMPARISON_TABLE_COLUMNS",
    "DIMENSIONLESS",
    "ENERGY_FLUX_UNITS",
    "NEE_BENCHMARK_UNITS",
    "NEE_MODEL_UNITS",
    "NEE_RATE_TO_G_C_PER_DAY",
    "PUBLISHED_BENCHMARKS",
    "SITE_SUBSET_94",
    "TABLE_S3",
    "BenchmarkError",
    "PublishedBenchmark",
    "benchmark_table",
    "compare_to_benchmarks",
    "convert_nee_to_carbon_units",
]


class BenchmarkError(ValueError):
    """A comparison against the published benchmarks could not be made."""


# ---------------------------------------------------------------------------
# Units (ambiguity A10)
# ---------------------------------------------------------------------------

#: Units of ``rmse`` and ``bias`` for NEE in Supplementary Table S3.
NEE_BENCHMARK_UNITS: Final = "g C m-2 d-1"

#: Units this package models NEE in, and reports its metrics in by default.
NEE_MODEL_UNITS: Final = "umol m-2 s-1"

#: Units of ``rmse`` and ``bias`` for H and LE, published and modelled alike.
ENERGY_FLUX_UNITS: Final = "W m-2"

#: What ``r2`` and ``slope`` are in, in both directions of any comparison.
DIMENSIONLESS: Final = "dimensionless"

#: Molar mass of carbon, g mol-1.
CARBON_MOLAR_MASS_G: Final = 12.011

#: Converts a CO2 flux *rate* from ``umol m-2 s-1`` to ``g C m-2 d-1``:
#: ``12.011 g mol-1 x 1e-6 mol umol-1 x 86400 s d-1``. This is a unit conversion
#: of a rate, not an aggregation of half hours into days, and the distinction is
#: ambiguity A10 - see :func:`convert_nee_to_carbon_units`.
NEE_RATE_TO_G_C_PER_DAY: Final = CARBON_MOLAR_MASS_G * 1e-6 * 86400.0

#: The site population every value in :data:`PUBLISHED_BENCHMARKS` is a median of.
SITE_SUBSET_94: Final = "94-site complete-analysis subset"

#: Where those medians are tabulated.
TABLE_S3: Final = "Supplementary Table S3 (mmc4.docx)"


# ---------------------------------------------------------------------------
# One published number
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PublishedBenchmark(FrozenRecord):
    """One published median: a target, a method, a gap class and a subset.

    A field is ``None`` where the supplement does not tabulate that metric for
    that cell - the nighttime medians of section 4 of
    ``docs/supplement_benchmarks.md`` quote no bias - rather than zero, matching
    :mod:`rfrgapfill.metrics`: an absent number is absent, not a value.
    """

    #: The flux: ``NEE``, ``H`` or ``LE``.
    target: str
    #: The method the median describes: ``MDS``, ``RFR3`` or ``RFR10``.
    method: str
    #: The gap class, or ``"all"`` where the median pools every class.
    gap_class: str
    #: The day/night subset: ``all``, ``daytime`` or ``nighttime``.
    subset: str
    #: Median coefficient of determination across the site population.
    r2: float | None
    #: Median regression slope of filled on measured.
    slope: float | None
    #: Median root mean squared error, in :attr:`units`.
    rmse: float | None
    #: Median bias, in :attr:`units`.
    bias: float | None
    #: Units of :attr:`rmse` and :attr:`bias`.
    units: str
    #: The supplementary table the values are transcribed from.
    source: str = TABLE_S3
    #: The site population the median was taken over.
    site_subset: str = SITE_SUBSET_94

    def __post_init__(self) -> None:
        if self.gap_class != POOLED_GAP_CLASS:
            # Validates the spelling against the package's own vocabulary, so a
            # transcription typo fails at import rather than joining nothing.
            object.__setattr__(self, "gap_class", GapClass.coerce(self.gap_class).value)
        if self.subset not in SUBSET_ORDER:
            raise BenchmarkError(
                f"unknown subset {self.subset!r}; expected one of {', '.join(SUBSET_ORDER)}"
            )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable record of this published value."""
        return {
            "target": self.target,
            "method": self.method,
            "gap_class": self.gap_class,
            "subset": self.subset,
            "r2": self.r2,
            "slope": self.slope,
            "rmse": self.rmse,
            "bias": self.bias,
            "units": self.units,
            "source": self.source,
            "site_subset": self.site_subset,
        }


def _s3(
    target: str,
    method: str,
    gap_class: str,
    subset: str,
    r2: float | None,
    slope: float | None,
    rmse: float | None,
    bias: float | None,
) -> PublishedBenchmark:
    """Build one Table S3 row, taking its units from the target."""
    units = NEE_BENCHMARK_UNITS if target == "NEE" else ENERGY_FLUX_UNITS
    return PublishedBenchmark(
        target=target,
        method=method,
        gap_class=gap_class,
        subset=subset,
        r2=r2,
        slope=slope,
        rmse=rmse,
        bias=bias,
        units=units,
    )


#: Supplementary Table S3, as transcribed in ``docs/supplement_benchmarks.md``.
#:
#: Three blocks: the diel medians over every gap class (section 2 of that
#: document), the very-long-gap medians (section 3), and the nighttime medians
#: that make weak nighttime skill impossible to hide (section 4). The values are
#: **provisional** until they are checked against the supplement files
#: themselves, which are publisher material and not committed here; tests that
#: depend on them carry the ``supplement`` marker.
PUBLISHED_BENCHMARKS: Final[tuple[PublishedBenchmark, ...]] = (
    # -- diel medians, every gap class pooled, every observation ------------
    _s3("NEE", "MDS", "all", "all", 0.72, 0.79, 2.98, 0.00),
    _s3("NEE", "RFR3", "all", "all", 0.78, 0.79, 2.63, -0.01),
    _s3("NEE", "RFR10", "all", "all", 0.84, 0.84, 2.42, 0.02),
    _s3("H", "MDS", "all", "all", 0.67, 0.72, 50.33, -1.12),
    _s3("H", "RFR3", "all", "all", 0.78, 0.78, 39.37, 0.26),
    _s3("H", "RFR10", "all", "all", 0.90, 0.90, 26.80, -0.13),
    _s3("LE", "MDS", "all", "all", 0.63, 0.70, 39.84, -2.36),
    _s3("LE", "RFR3", "all", "all", 0.75, 0.76, 33.50, -0.11),
    _s3("LE", "RFR10", "all", "all", 0.85, 0.85, 25.51, -0.46),
    # -- very-long-gap medians (30-day class) -------------------------------
    _s3("NEE", "MDS", "very_long", "all", 0.59, 0.85, 2.99, -0.48),
    _s3("NEE", "RFR3", "very_long", "all", 0.74, 0.97, 2.61, -0.14),
    _s3("NEE", "RFR10", "very_long", "all", 0.74, 0.97, 2.48, -0.04),
    _s3("H", "MDS", "very_long", "all", 0.62, 0.88, 55.00, -9.41),
    _s3("H", "RFR3", "very_long", "all", 0.76, 0.99, 40.11, 0.66),
    _s3("H", "RFR10", "very_long", "all", 0.89, 1.00, 26.80, -0.20),
    _s3("LE", "MDS", "very_long", "all", 0.57, 0.85, 45.45, -20.24),
    _s3("LE", "RFR3", "very_long", "all", 0.74, 0.98, 34.59, -0.35),
    _s3("LE", "RFR10", "very_long", "all", 0.83, 0.99, 25.49, -3.66),
    # -- nighttime medians; the supplement quotes no bias for these ---------
    _s3("H", "MDS", "all", "nighttime", 0.03, 0.16, 29.88, None),
    _s3("H", "RFR3", "all", "nighttime", 0.16, 0.20, 19.46, None),
    _s3("H", "RFR10", "all", "nighttime", 0.49, 0.47, 14.24, None),
    _s3("LE", "MDS", "all", "nighttime", 0.02, 0.17, 21.93, None),
    _s3("LE", "RFR3", "all", "nighttime", 0.10, 0.15, 12.75, None),
    _s3("LE", "RFR10", "all", "nighttime", 0.27, 0.32, 9.79, None),
)

#: Column order of :func:`benchmark_table`.
BENCHMARK_TABLE_COLUMNS: Final[tuple[str, ...]] = (
    "target",
    "method",
    "gap_class",
    "subset",
    "units",
    *METRIC_COLUMNS,
    "site_subset",
    "source",
)

#: Column order of :func:`compare_to_benchmarks`.
COMPARISON_TABLE_COLUMNS: Final[tuple[str, ...]] = (
    "target",
    "method",
    "gap_class",
    "subset",
    "metric",
    "run",
    "published",
    "difference",
    "run_units",
    "published_units",
    "comparable",
    "note",
    "site_subset",
    "source",
)

#: The key a run row and a published row are matched on.
_JOIN_KEYS: Final[tuple[str, ...]] = ("target", "method", "gap_class", "subset")

#: Metrics that carry the flux's units; the other two are dimensionless.
_DIMENSIONAL_METRICS: Final[frozenset[str]] = frozenset({"rmse", "bias"})


# ---------------------------------------------------------------------------
# The published table
# ---------------------------------------------------------------------------


def benchmark_table(
    *,
    targets: Sequence[str] | None = None,
    methods: Sequence[str] | None = None,
    gap_classes: Sequence[str] | None = None,
    subsets: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Return :data:`PUBLISHED_BENCHMARKS` as a tidy frame.

    The same key columns as :func:`~rfrgapfill.sensitivity.gap_length_table`, so
    the published medians can be pivoted by
    :func:`~rfrgapfill.sensitivity.gap_length_pivot` exactly as a reproduction's
    own medians are, and the two printed side by side::

        gap_length_pivot(benchmark_table(), metric="r2", gap_class="all")

    :param targets: keep only these fluxes.
    :param methods: keep only these methods.
    :param gap_classes: keep only these gap classes.
    :param subsets: keep only these day/night subsets.
    :returns: a frame with :data:`BENCHMARK_TABLE_COLUMNS`.
    """
    rows = [benchmark.to_dict() for benchmark in PUBLISHED_BENCHMARKS]
    table: pd.DataFrame = pd.DataFrame(rows, columns=list(BENCHMARK_TABLE_COLUMNS))
    for column, wanted in (
        ("target", targets),
        ("method", methods),
        ("gap_class", gap_classes),
        ("subset", subsets),
    ):
        if wanted is not None:
            table = cast(
                "pd.DataFrame", table.loc[table[column].isin([str(value) for value in wanted])]
            )
    return cast("pd.DataFrame", table.reset_index(drop=True))


# ---------------------------------------------------------------------------
# The unit conversion of ambiguity A10
# ---------------------------------------------------------------------------


def convert_nee_to_carbon_units(
    table: pd.DataFrame,
    *,
    target: str = "NEE",
    factor: float = NEE_RATE_TO_G_C_PER_DAY,
) -> pd.DataFrame:
    """Return ``table`` with NEE ``rmse`` and ``bias`` in ``g C m-2 d-1``.

    Multiplies the two dimensional metrics of every ``target`` row by ``factor``
    and rewrites their ``units``. ``r2`` and ``slope`` are untouched: both are
    invariant under a common rescaling of measured and filled values, so
    converting them would be wrong rather than merely unnecessary.

    **What this is, and is not (A10).** ``factor`` converts a flux *rate*:
    ``1 umol CO2 m-2 s-1`` sustained for a day is ``12.011e-6 x 86400`` grams of
    carbon per square metre, so a mean half-hourly error in molar units becomes
    the daily carbon flux that error corresponds to. What it does **not** do is
    reproduce an aggregation of half hours into daily totals before scoring, and
    the paper does not say which of the two produced Table S3. The resulting
    comparison is therefore documented and explicit, never "paper exact".

    Rows already in :data:`NEE_BENCHMARK_UNITS` are left alone, so the function
    is safe to apply twice; a row whose units are unknown is an error rather than
    an assumption.

    :param table: any tidy table carrying ``target``, ``units``, ``rmse`` and
        ``bias`` - a :func:`~rfrgapfill.sensitivity.gap_length_table` or a
        :func:`~rfrgapfill.sensitivity.median_across_sites` frame.
    :param target: the flux to convert. The paper's carbon target is ``NEE``.
    :param factor: the rate conversion, exposed so a differently defined carbon
        unit can be documented by the caller rather than hidden here.
    :returns: a converted copy; the input frame is never modified.
    :raises BenchmarkError: if a row of ``target`` carries no units, or units
        this conversion does not start from.
    """
    for column in ("target", "units", *_DIMENSIONAL_METRICS):
        if column not in table.columns:
            raise BenchmarkError(
                f"convert_nee_to_carbon_units needs a {column!r} column; the frame "
                f"carries {', '.join(str(name) for name in table.columns) or 'nothing'}"
            )
    converted: pd.DataFrame = table.copy()
    rows = converted["target"].astype(str) == str(target)
    if not bool(rows.any()):
        return converted
    units = [str(value) for value in cast("pd.Series[Any]", converted.loc[rows, "units"])]
    unknown = sorted({unit for unit in units if unit not in {NEE_MODEL_UNITS, NEE_BENCHMARK_UNITS}})
    if unknown:
        raise BenchmarkError(
            f"cannot convert {target} from {', '.join(unknown)} to {NEE_BENCHMARK_UNITS}: "
            f"this conversion starts from {NEE_MODEL_UNITS}. State the units the run "
            "reported with gap_length_table(units=...)"
        )
    to_convert = rows & (converted["units"].astype(str) == NEE_MODEL_UNITS)
    for column in _DIMENSIONAL_METRICS:
        converted.loc[to_convert, column] = pd.to_numeric(
            converted.loc[to_convert, column], errors="coerce"
        ) * float(factor)
    converted.loc[to_convert, "units"] = NEE_BENCHMARK_UNITS
    return converted


# ---------------------------------------------------------------------------
# The comparison
# ---------------------------------------------------------------------------


def compare_to_benchmarks(
    table: pd.DataFrame,
    *,
    metrics: Sequence[str] = METRIC_COLUMNS,
) -> pd.DataFrame:
    """Return a reproduction run beside the published medians, metric by metric.

    Long form - one row per target, method, gap class, subset and metric - because
    comparability is a property of the *metric*, not of the cell: an NEE ``R2``
    compares against Table S3 while the NEE ``RMSE`` beside it does not until it
    has been converted (A10).

    Only cells the supplement tabulates come back. There is no published ORF
    benchmark - Supplementary Figure S1 is a figure, with no numbers to
    transcribe - so an ORF arm compares against nothing here and belongs in the
    paired RFR-versus-ORF comparison of ``method_spec.md`` section 3.6 instead.

    :param table: a table whose cells are unique on target, method, gap class and
        subset: a single run's :func:`~rfrgapfill.sensitivity.gap_length_table`,
        or - for a multi-site reproduction, which is what the published medians
        are - a :func:`~rfrgapfill.sensitivity.median_across_sites` frame.
    :param metrics: the metrics to compare; the default is all four.
    :returns: a frame with :data:`COMPARISON_TABLE_COLUMNS`. ``difference`` is
        ``run - published`` where both exist and the units agree, and missing
        otherwise, with ``note`` saying which of the two it was.
    :raises BenchmarkError: if the table repeats a cell, if a requested metric is
        unknown, or if no cell of the table is tabulated in the supplement.
    """
    unknown_metrics = [metric for metric in metrics if metric not in METRIC_COLUMNS]
    if unknown_metrics:
        raise BenchmarkError(
            f"unknown metric(s) {', '.join(unknown_metrics)}; expected {', '.join(METRIC_COLUMNS)}"
        )
    run = _prepared(table)
    published = benchmark_table()
    merged = run.merge(published, on=list(_JOIN_KEYS), how="inner", suffixes=("_run", "_published"))
    if merged.empty:
        raise BenchmarkError(
            "no cell of this table is tabulated in the supplement. It covers "
            f"{_describe(run)}; the published medians cover {_describe(published)}"
        )
    rows = [
        _comparison_row(record, metric) for _, record in merged.iterrows() for metric in metrics
    ]
    comparison: pd.DataFrame = pd.DataFrame(rows, columns=list(COMPARISON_TABLE_COLUMNS))
    comparison["run"] = pd.to_numeric(comparison["run"], errors="coerce").astype("float64")
    comparison["published"] = pd.to_numeric(comparison["published"], errors="coerce").astype(
        "float64"
    )
    comparison["difference"] = pd.to_numeric(comparison["difference"], errors="coerce").astype(
        "float64"
    )
    return _sorted(comparison, metrics)


def _prepared(table: pd.DataFrame) -> pd.DataFrame:
    """Return ``table`` validated as one comparable run, one row per cell."""
    if not isinstance(table, pd.DataFrame):
        raise BenchmarkError(f"compare_to_benchmarks needs a DataFrame, got {type(table).__name__}")
    try:
        # Routing through the table builder rather than reimplementing its
        # checks: whatever it accepts as a tidy table is what compares here, and
        # a frame that has already been through it passes through unchanged.
        prepared = gap_length_table(table)
    except SensitivityError as error:
        raise BenchmarkError(str(error)) from error
    if prepared.empty:
        raise BenchmarkError("the table is empty; there is nothing to compare")
    duplicated = prepared.duplicated(subset=list(_JOIN_KEYS), keep=False)
    if bool(duplicated.any()):
        sites = sorted(
            {str(value) for value in prepared.loc[duplicated, "site"] if value is not None}
        )
        raise BenchmarkError(
            f"{int(duplicated.sum())} row(s) repeat a target/method/gap-class/subset cell, "
            "so there is no single value to compare"
            + (
                f" (sites {', '.join(sites)}): the published values are medians across "
                "sites, so aggregate with median_across_sites() first"
                if sites
                else "; aggregate with median_across_sites() first"
            )
        )
    return prepared


def _comparison_row(record: Mapping[str, Any], metric: str) -> dict[str, Any]:
    """Return one long-form comparison row for ``metric`` of one matched cell."""
    dimensional = metric in _DIMENSIONAL_METRICS
    run_units = _clean(record.get("units_run")) if dimensional else DIMENSIONLESS
    published_units = _clean(record.get("units_published")) if dimensional else DIMENSIONLESS
    comparable = True
    note: str | None = None
    if run_units != published_units:
        # Both sides are reported rather than one: the run's number and the
        # published number are in different units here, and a single `units`
        # cell would have to misdescribe one of them.
        comparable = False
        note = (
            f"run reports {metric} in {run_units or 'unstated units'}; the "
            f"supplement reports {published_units} (ambiguity A10)"
        )
        if run_units == NEE_MODEL_UNITS:
            note += " - convert with convert_nee_to_carbon_units()"
    run_value = _number(record.get(f"{metric}_run"))
    published_value = _number(record.get(f"{metric}_published"))
    if comparable and published_value is None:
        note = "the supplement does not tabulate this metric for this cell"
    elif comparable and run_value is None:
        note = "the run left this metric undefined on this subset"
    difference = (
        run_value - published_value
        if comparable and run_value is not None and published_value is not None
        else None
    )
    return {
        "target": record["target"],
        "method": record["method"],
        "gap_class": record["gap_class"],
        "subset": record["subset"],
        "metric": metric,
        "run": run_value,
        "published": published_value,
        "difference": difference,
        "run_units": run_units,
        "published_units": published_units,
        "comparable": comparable,
        "note": note,
        "site_subset": record["site_subset"],
        "source": record["source"],
    }


def _number(value: object) -> float | None:
    """Return ``value`` as a float, or ``None`` where it is missing."""
    if value is None:
        return None
    number = pd.to_numeric(value, errors="coerce")
    return None if pd.isna(number) else float(number)


def _clean(units: object) -> str | None:
    """Return ``units`` as a stripped string, or ``None`` where it is missing."""
    if units is None or units is pd.NA or (isinstance(units, float) and pd.isna(units)):
        return None
    text = str(units).strip()
    return text or None


def _describe(table: pd.DataFrame) -> str:
    """Return a short account of which cells a table covers, for an error."""
    parts = [
        f"{column}={', '.join(sorted({str(value) for value in table[column]}))}"
        for column in _JOIN_KEYS
        if column in table.columns
    ]
    return "; ".join(parts) or "nothing"


def _sorted(comparison: pd.DataFrame, metrics: Iterable[str]) -> pd.DataFrame:
    """Return the comparison in reporting order: duration, then requested metric."""
    keys = pd.DataFrame(index=comparison.index)
    keys["target"] = [str(value) for value in comparison["target"]]
    keys["method"] = _positions(comparison["method"], METHOD_ORDER)
    keys["gap_class"] = _positions(comparison["gap_class"], GAP_CLASS_ORDER)
    keys["subset"] = _positions(comparison["subset"], SUBSET_ORDER)
    keys["metric"] = _positions(comparison["metric"], tuple(str(metric) for metric in metrics))
    order = keys.sort_values(list(keys.columns), kind="stable").index
    result: pd.DataFrame = comparison.loc[order].reset_index(drop=True)
    return result


def _positions(values: Iterable[object], order: Sequence[str]) -> list[float]:
    """Return sort positions for ``values``; unknown labels sort last, stably."""
    known = {label: index for index, label in enumerate(order)}
    return [float(known.get(str(value), len(order))) for value in values]
