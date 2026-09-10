"""Bias-spread uncertainty diagnostics (``docs/method_spec.md`` section 6.3; Step 18A).

Supplementary Table S8 and its companion figure describe how *uncertain* a
gap-filled bias is as the gap grows, not only how large it is: the spread of bias
rises steeply with gap length for MDS and more gradually for RFR. This module
reports that spread and carries the published figures, and it is careful about
what it does not do.

Two populations, never mixed
----------------------------

A bias interquartile range is a spread *over something*, and this package
reports it over two different things:

* **across sites** - :func:`bias_iqr`. One bias per site, taken from that site's
  cell of :func:`~rfrgapfill.sensitivity.gap_length_table`, and the IQR across
  the sites of a reproduction study. This is the analogue of the Table S8
  numerator, and it is the only one of the two that can be stratified by IGBP
  class, because IGBP describes a site;
* **across the gaps of one site** -
  :meth:`~rfrgapfill.validation.TargetValidation.bias_spread_frame`. One bias
  per placed interval, the IQR across the intervals of one class. Computed inside
  the validation run, because the per-gap biases do not survive into the tidy
  metric table.

That Table S8 spreads bias across sites rather than across gaps is an inference
from the supplement's cross-site framing, not a statement in it, and it is part
of ambiguity A8.

What is deliberately absent (A8)
--------------------------------

Table S8 reports a *normalized* quantity - bias IQR divided by a flux confidence
interval, or by a joint flux-uncertainty confidence interval - and the
denominator has not been reconstructed from the supplementary methods. So:

* no function here, or anywhere in the package, computes that ratio;
* the published very-long-gap ranges are carried as data in
  :data:`TABLE_S8_RANGES`, every record marked :data:`EXPERIMENTAL_STATUS`, so
  they document the paper without posing as a reproduction target;
* they are kept out of :data:`rfrgapfill.benchmarks.PUBLISHED_BENCHMARKS`, so
  :func:`~rfrgapfill.benchmarks.compare_to_benchmarks` can never difference a
  run against a number this package has no way to produce.

A ratio added later must be named and labelled experimental until it has been
reproduced against Table S8.

Like :mod:`rfrgapfill.sensitivity`, nothing here computes a metric: every bias
it spreads came out of :mod:`rfrgapfill.metrics` inside a validation run.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, cast

import numpy as np
import pandas as pd

from rfrgapfill.schema import FrozenRecord, GapClass
from rfrgapfill.sensitivity import (
    GAP_CLASS_ORDER,
    METHOD_ORDER,
    SUBSET_ORDER,
    Reportish,
    SensitivityError,
    gap_length_table,
)

__all__ = [
    "BIAS_IQR_TABLE_COLUMNS",
    "EXPERIMENTAL_STATUS",
    "NORMALIZED_UNCERTAINTY_CAVEAT",
    "NORMALIZED_UNCERTAINTY_QUANTITY",
    "POOLED_IGBP",
    "TABLE_S8",
    "TABLE_S8_RANGES",
    "UNCERTAINTY_RANGE_TABLE_COLUMNS",
    "PublishedUncertaintyRange",
    "UncertaintyError",
    "bias_iqr",
    "published_uncertainty_table",
]


class UncertaintyError(SensitivityError):
    """An uncertainty table could not be built from what was supplied.

    A :class:`~rfrgapfill.sensitivity.SensitivityError`, because the tables here
    are built on the sensitivity tables and fail for the same kinds of reason.
    """


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

#: The ``igbp`` label of a row that pools every ecosystem class.
POOLED_IGBP: Final = "all"

#: Column order of :func:`bias_iqr`.
BIAS_IQR_TABLE_COLUMNS: Final[tuple[str, ...]] = (
    "target",
    "method",
    "mode",
    "gap_class",
    "subset",
    "igbp",
    "units",
    "n_sites",
    "n_sites_offered",
    "bias_q1",
    "bias_median",
    "bias_q3",
    "bias_iqr",
)

#: The cell one site contributes one bias to.
_CELL_KEYS: Final[tuple[str, ...]] = (
    "site",
    "target",
    "method",
    "mode",
    "gap_class",
    "subset",
    "units",
)

#: What the across-site spread is grouped by.
_GROUP_KEYS: Final[tuple[str, ...]] = (*_CELL_KEYS[1:], "igbp")

_QUANTILE_COLUMNS: Final[tuple[str, ...]] = ("bias_q1", "bias_median", "bias_q3", "bias_iqr")
_COUNT_COLUMNS: Final[tuple[str, ...]] = ("n_sites", "n_sites_offered")

#: Column names a site-metadata frame may use for the site and its IGBP class.
_SITE_COLUMNS: Final[tuple[str, ...]] = ("site", "site_id")
_IGBP_COLUMNS: Final[tuple[str, ...]] = ("IGBP", "igbp")


# ---------------------------------------------------------------------------
# Bias IQR across sites
# ---------------------------------------------------------------------------


def bias_iqr(
    results: Reportish | Iterable[Any] | Mapping[str, Any],
    *,
    igbp: Mapping[str, object] | pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return the spread of bias across sites, per target, method and gap class.

    The Step 18A table and the analogue of the Supplementary Table S8 numerator:
    for every target, method, gap class and day/night subset, the quartiles of
    the per-site biases and their interquartile range. A reproduction whose fill
    is stable as the gap grows shows an IQR that grows slowly; the paper reports
    it growing steeply for MDS.

    Rows pooling every ecosystem carry ``igbp="all"`` and are always present.
    When ``igbp`` supplies site metadata, the same spread is reported again for
    each IGBP class; a site with no class in the metadata contributes to the
    pooled rows only, since putting it in a class would invent its ecosystem.

    Quartiles use linear interpolation between order statistics, the convention
    ambiguity A11 settles for every quantile this package reports. A site whose
    bias is undefined for a cell is skipped rather than counted as zero:
    ``n_sites`` counts the sites that had a bias, ``n_sites_offered`` every site
    that contributed a row. One site has quartiles but no spread, so its
    ``bias_iqr`` is missing; none has neither.

    Units are part of the grouping - bias carries the units of the flux, and two
    sites reporting NEE in different units are two rows, never one spread.

    :param results: anything :func:`~rfrgapfill.sensitivity.gap_length_table`
        accepts, with every row labelled by site - a mapping of site label to
        results, or a table built from one.
    :param igbp: the IGBP class of each site, as a mapping of site label to
        class, or as a site-metadata frame (Supplementary Tables S1-S2) with a
        ``site`` or ``site_id`` column and an ``IGBP`` or ``igbp`` column.
        Missing classes are treated as unavailable.
    :returns: a frame with :data:`BIAS_IQR_TABLE_COLUMNS`, in reporting order:
        gap classes by duration and the pooled ecosystem row first.
    :raises SensitivityError: if a row carries no site label, if one site
        contributes a cell twice, or if the metadata is unusable.
    """
    sites = gap_length_table(results)
    if sites.empty:
        return _empty()
    _require_one_value_per_site(sites)
    classes = _igbp_by_site(igbp)

    parts = [sites.assign(igbp=POOLED_IGBP)]
    if classes:
        stratified = sites.assign(igbp=[classes.get(str(site)) for site in sites["site"]])
        parts.append(cast("pd.DataFrame", stratified.loc[stratified["igbp"].notna()]))
    frame: pd.DataFrame = pd.concat(parts, ignore_index=True)

    # `dropna=False`: a target whose units are unknown is still a group.
    grouped = frame.groupby(list(_GROUP_KEYS), dropna=False, sort=False)
    rows = [_spread(cast("tuple[Any, ...]", key), group) for key, group in grouped]
    table: pd.DataFrame = pd.DataFrame(rows, columns=list(BIAS_IQR_TABLE_COLUMNS))
    return _ordered(_typed(table))


def _require_one_value_per_site(sites: pd.DataFrame) -> None:
    """Raise unless every row names its site and no site repeats a cell."""
    unlabelled = int(sites["site"].isna().sum())
    if unlabelled:
        raise UncertaintyError(
            f"{unlabelled} row(s) carry no site label, so a spread across sites is "
            "undefined; label the runs with a mapping of site label to result. The "
            "spread across the gaps of one site is TargetValidation.bias_spread_frame()"
        )
    duplicated = sites.duplicated(subset=list(_CELL_KEYS), keep=False)
    if bool(duplicated.any()):
        offenders = sites.loc[duplicated, list(_CELL_KEYS)].drop_duplicates()
        shown = "; ".join(
            " ".join(str(value) for value in row) for row in offenders.head(3).to_numpy()
        )
        raise UncertaintyError(
            f"{int(duplicated.sum())} row(s) repeat a site/target/method/gap-class/subset "
            f"cell, which would count those sites twice in the spread: {shown}"
            + (" ..." if len(offenders) > 3 else "")
        )


def _igbp_by_site(igbp: Mapping[str, object] | pd.DataFrame | None) -> dict[str, str]:
    """Return the IGBP class of every site that has one."""
    if igbp is None:
        return {}
    pairs: Iterable[tuple[object, object]]
    if isinstance(igbp, pd.DataFrame):
        site_column = next((name for name in _SITE_COLUMNS if name in igbp.columns), None)
        class_column = next((name for name in _IGBP_COLUMNS if name in igbp.columns), None)
        if site_column is None or class_column is None:
            raise UncertaintyError(
                "site metadata needs a site column (site or site_id) and an IGBP column "
                f"(IGBP or igbp); the frame carries "
                f"{', '.join(str(name) for name in igbp.columns) or 'nothing'}"
            )
        pairs = zip(igbp[site_column], igbp[class_column], strict=True)
    elif isinstance(igbp, Mapping):
        pairs = igbp.items()
    else:
        raise UncertaintyError(
            "igbp must be a mapping of site label to IGBP class or a site-metadata "
            f"DataFrame, got {type(igbp).__name__}"
        )
    classes: dict[str, str] = {}
    for site, label in pairs:
        if label is None or bool(pd.isna(cast("Any", label))):
            continue
        text = str(label).strip()
        if not text:
            continue
        if text == POOLED_IGBP:
            raise UncertaintyError(
                f"site {site!r} is given the IGBP class {POOLED_IGBP!r}, which is the "
                "label of the rows pooling every class"
            )
        name = str(site)
        if classes.get(name, text) != text:
            raise UncertaintyError(
                f"site {name!r} is given two IGBP classes, {classes[name]!r} and {text!r}"
            )
        classes[name] = text
    return classes


def _spread(key: tuple[Any, ...], group: pd.DataFrame) -> dict[str, Any]:
    """Return one row of the across-site table for one group of site cells."""
    values = pd.to_numeric(group["bias"], errors="coerce").to_numpy(dtype=float)
    defined = values[np.isfinite(values)]
    q1: float | None = None
    median: float | None = None
    q3: float | None = None
    iqr: float | None = None
    if defined.size:
        # Linear interpolation between order statistics (A11).
        q1, median, q3 = (float(value) for value in np.percentile(defined, (25.0, 50.0, 75.0)))
        if defined.size > 1:
            iqr = q3 - q1
    return {
        **dict(zip(_GROUP_KEYS, key, strict=True)),
        "n_sites": int(defined.size),
        "n_sites_offered": int(group["site"].nunique()),
        "bias_q1": q1,
        "bias_median": median,
        "bias_q3": q3,
        "bias_iqr": iqr,
    }


def _typed(table: pd.DataFrame) -> pd.DataFrame:
    """Return ``table`` with quartile columns float and count columns nullable int."""
    typed: pd.DataFrame = table.copy()
    for column in _QUANTILE_COLUMNS:
        typed[column] = pd.to_numeric(typed[column], errors="coerce").astype("float64")
    for column in _COUNT_COLUMNS:
        typed[column] = pd.to_numeric(typed[column], errors="coerce").astype("Int64")
    return typed


def _positions(values: Iterable[object], order: Sequence[str]) -> list[float]:
    """Return sort positions for ``values``; unknown labels sort last, stably."""
    known = {label: index for index, label in enumerate(order)}
    return [float(known.get(str(value), len(order))) for value in values]


def _ordered(table: pd.DataFrame) -> pd.DataFrame:
    """Return ``table`` in reporting order, gap classes by duration."""
    keys = pd.DataFrame(index=table.index)
    keys["target"] = [str(value) for value in table["target"]]
    keys["method_rank"] = _positions(table["method"], METHOD_ORDER)
    keys["method"] = [str(value) for value in table["method"]]
    keys["gap_class"] = _positions(table["gap_class"], GAP_CLASS_ORDER)
    keys["subset"] = _positions(table["subset"], SUBSET_ORDER)
    keys["igbp_rank"] = [0.0 if value == POOLED_IGBP else 1.0 for value in table["igbp"]]
    keys["igbp"] = [str(value) for value in table["igbp"]]
    keys["units"] = [str(value) for value in table["units"]]
    order = keys.sort_values(list(keys.columns), kind="stable").index
    ordered: pd.DataFrame = table.loc[order].reset_index(drop=True)
    return ordered


def _empty() -> pd.DataFrame:
    """Return an empty :func:`bias_iqr` table with the right dtypes."""
    return _typed(pd.DataFrame(columns=list(BIAS_IQR_TABLE_COLUMNS)))


# ---------------------------------------------------------------------------
# Supplementary Table S8, as documentation (A8)
# ---------------------------------------------------------------------------

#: Where the normalized ranges are tabulated.
TABLE_S8: Final = "Supplementary Table S8 (mmc6.docx)"

#: What the published ranges are ranges of.
NORMALIZED_UNCERTAINTY_QUANTITY: Final = "bias IQR / joint flux-uncertainty CI"

#: The status every published normalized range carries, and the only one it can.
EXPERIMENTAL_STATUS: Final = "experimental"

#: Why the published ranges are documentation rather than a reproduction target.
NORMALIZED_UNCERTAINTY_CAVEAT: Final = (
    "denominator (flux CI / joint flux-uncertainty CI) not reconstructed from the "
    "supplementary methods; this package does not compute the ratio and no run can be "
    "compared against it (ambiguity A8)"
)

#: Column order of :func:`published_uncertainty_table`.
UNCERTAINTY_RANGE_TABLE_COLUMNS: Final[tuple[str, ...]] = (
    "target",
    "method",
    "gap_class",
    "quantity",
    "lower",
    "upper",
    "units",
    "status",
    "caveat",
    "source",
)


@dataclass(frozen=True)
class PublishedUncertaintyRange(FrozenRecord):
    """One published normalized-uncertainty interval from Table S8.

    The interval the supplement reports for bias IQR divided by the joint
    flux-uncertainty confidence interval - a dimensionless ratio - for one flux,
    one method and one gap class.

    The record has no status field to set. Every published range is
    :data:`EXPERIMENTAL_STATUS` for as long as the denominator is unverified,
    and that is a property of the package's knowledge, not of any one number.
    """

    #: The flux: ``NEE``, ``H`` or ``LE``.
    target: str
    #: ``MDS``, ``RFR3`` or ``RFR10``.
    method: str
    #: The gap class the range describes.
    gap_class: str
    #: Lower end of the published interval.
    lower: float
    #: Upper end of the published interval.
    upper: float
    #: The supplementary table the values are transcribed from.
    source: str = TABLE_S8

    def __post_init__(self) -> None:
        object.__setattr__(self, "gap_class", GapClass.coerce(self.gap_class).value)
        if self.lower > self.upper:
            raise UncertaintyError(
                f"{self.target}/{self.method}: published interval runs from {self.lower} "
                f"down to {self.upper}; a transcription has swapped its ends"
            )

    @property
    def status(self) -> str:
        """Always :data:`EXPERIMENTAL_STATUS` while ambiguity A8 stands."""
        return EXPERIMENTAL_STATUS

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable record of this published range."""
        return {
            "target": self.target,
            "method": self.method,
            "gap_class": self.gap_class,
            "quantity": NORMALIZED_UNCERTAINTY_QUANTITY,
            "lower": self.lower,
            "upper": self.upper,
            "units": "dimensionless",
            "status": self.status,
            "caveat": NORMALIZED_UNCERTAINTY_CAVEAT,
            "source": self.source,
        }


def _s8(target: str, method: str, lower: float, upper: float) -> PublishedUncertaintyRange:
    """Build one very-long-gap Table S8 range."""
    return PublishedUncertaintyRange(
        target=target, method=method, gap_class="very_long", lower=lower, upper=upper
    )


#: Supplementary Table S8's very-long-gap ranges, as transcribed in section 7 of
#: ``docs/supplement_benchmarks.md``. Provisional until checked against the
#: supplement file itself, and experimental regardless (A8).
TABLE_S8_RANGES: Final[tuple[PublishedUncertaintyRange, ...]] = (
    _s8("NEE", "MDS", 5.21, 5.34),
    _s8("NEE", "RFR3", 2.90, 2.97),
    _s8("NEE", "RFR10", 0.87, 0.89),
    _s8("H", "MDS", 5.37, 5.63),
    _s8("H", "RFR3", 2.46, 2.57),
    _s8("H", "RFR10", 1.65, 1.72),
    _s8("LE", "MDS", 7.90, 8.93),
    _s8("LE", "RFR3", 2.25, 2.55),
    _s8("LE", "RFR10", 1.79, 2.02),
)


def published_uncertainty_table() -> pd.DataFrame:
    """Return :data:`TABLE_S8_RANGES` as a tidy frame, caveat included on every row.

    For reading and for plotting beside a reproduction's own :func:`bias_iqr`
    table - not for comparison: the two are different quantities until the
    Table S8 denominator is reconstructed, and every row says so.
    """
    rows = [record.to_dict() for record in TABLE_S8_RANGES]
    table: pd.DataFrame = pd.DataFrame(rows, columns=list(UNCERTAINTY_RANGE_TABLE_COLUMNS))
    return table
