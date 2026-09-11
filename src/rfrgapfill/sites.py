"""Multi-site and ecosystem-stratified reproduction reports (``method_spec.md`` 6.6; Step 20A).

The paper validated RFR one site at a time and then described the sites
together: Supplementary Tables S1-S2 say which sites they were, Tables S4-S6
score every site of the 94-site subset, Table S9 scores NEE at all 194, and
Table S10 tests the method differences across sites. This module is that second
half for a reproduction study - site-level results put beside what is known about
each site - and it never touches the first half:

* **no model is fitted here, and none is pooled.** Every number arrives from a
  per-site validation run, or from a published per-site table, through
  :func:`~rfrgapfill.sensitivity.gap_length_table`. Stratifying by ecosystem is a
  grouping of *scores*, never of training data;
* **no site is required to improve.** :func:`method_differences` reports each
  site's change and :func:`welch_comparison` counts the sites that improved,
  worsened or did not move. Nothing here asserts a direction: a site where RFR
  does worse than MDS is a row, not an error;
* **no threshold is universal.** No argument anywhere in this module turns a
  metric into a pass or a fail. Performance differs by ecosystem, and
  :func:`stratified_summary` is how that difference is shown rather than averaged
  away.

The evidence is not equally broad for every method, and the tables say so. RFR3
has site-level NEE results at all 194 sites (Table S9); RFR10 has them only at
the 94-site subset (Tables S4-S6), because complete QC and RFR10 driver
availability were more restrictive. :func:`site_level_evidence` records the
population behind every target and method, and the readers label every row with
the population it came from.

Published tables as data
------------------------

The supplementary files are publisher material and are not shipped with the
package. :func:`read_table_s2`, :func:`read_tables_s4_to_s6` and
:func:`read_table_s9` parse a local copy into the tidy shape a reproduction
produces, so published and reproduced sites are stratified, differenced and
tested by the same functions. Table S10 is small, and is carried as
:data:`TABLE_S10_COMPARISONS`, traceable to ``docs/supplement_benchmarks.md``.

Table S10 applies Welch's test to the per-site values of two methods as two
independent samples, and :func:`welch_comparison` reproduces exactly that. The
samples are in fact paired by site; the paired view is
:func:`method_differences`, and the two are reported side by side rather than
one silently substituted for the other.
"""

from __future__ import annotations

import math
import re
import unicodedata
import warnings
import zipfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, cast
from xml.etree import ElementTree

import numpy as np
import pandas as pd
from scipy import stats

from rfrgapfill.benchmarks import (
    DIMENSIONLESS,
    ENERGY_FLUX_UNITS,
    NEE_BENCHMARK_UNITS,
    NEE_MODEL_UNITS,
    SITE_SUBSET_94,
)
from rfrgapfill.schema import FrozenRecord
from rfrgapfill.sensitivity import (
    GAP_CLASS_ORDER,
    METHOD_ORDER,
    METRIC_COLUMNS,
    POOLED_GAP_CLASS,
    SENSITIVITY_TABLE_COLUMNS,
    SUBSET_ORDER,
    Reportish,
    SensitivityError,
    _is_missing,
    _rank,
    gap_length_table,
)
from rfrgapfill.uncertainty import POOLED_IGBP

__all__ = [
    "IGBP_CLASSES",
    "IMPROVEMENT_DIRECTION",
    "INSTRUMENT_SYSTEMS",
    "METHOD_DIFFERENCE_COLUMNS",
    "POOLED_STRATUM",
    "POPULATION_SUMMARY_COLUMNS",
    "PUBLISHED_SITE_TABLES",
    "PUBLISHED_SITE_TABLE_COLUMNS",
    "PUBLISHED_WELCH_TABLE_COLUMNS",
    "S10_COMPARISON_COLUMNS",
    "SITE_EVIDENCE_COLUMNS",
    "SITE_METADATA_COLUMNS",
    "SITE_REPORT_COLUMNS",
    "SITE_REPORT_METADATA",
    "SITE_SET_194",
    "STRATIFIED_SUMMARY_COLUMNS",
    "TABLE_S10",
    "TABLE_S10_COMPARISONS",
    "WELCH_TABLE_COLUMNS",
    "PublishedSiteTable",
    "PublishedWelchComparison",
    "SiteReportError",
    "SiteReportWarning",
    "compare_to_table_s10",
    "method_differences",
    "published_welch_table",
    "read_table_s2",
    "read_table_s9",
    "read_tables_s4_to_s6",
    "site_level_evidence",
    "site_metadata",
    "site_population_summary",
    "site_report",
    "stratified_summary",
    "welch_comparison",
]


class SiteReportError(SensitivityError):
    """A multi-site report could not be built from what was supplied.

    A :class:`~rfrgapfill.sensitivity.SensitivityError`, because every table here
    is built on the sensitivity table and fails for the same kinds of reason.
    """


class SiteReportWarning(UserWarning):
    """A multi-site report was built, but part of what it describes is missing."""


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

#: The ``stratum`` label of a row that pools every stratum, as in
#: :func:`~rfrgapfill.uncertainty.bias_iqr`.
POOLED_STRATUM: Final = POOLED_IGBP

#: The IGBP land-cover classes FLUXNET2015 assigns, by code. Table S1 uses 11 of
#: them; the rest are listed so a reproduction beyond the paper's sites is not
#: refused for describing its ecosystem correctly.
IGBP_CLASSES: Final[Mapping[str, str]] = {
    "ENF": "Evergreen Needleleaf Forests",
    "EBF": "Evergreen Broadleaf Forests",
    "DNF": "Deciduous Needleleaf Forests",
    "DBF": "Deciduous Broadleaf Forests",
    "MF": "Mixed Forests",
    "CSH": "Closed Shrublands",
    "OSH": "Open Shrublands",
    "WSA": "Woody Savannas",
    "SAV": "Savannas",
    "GRA": "Grasslands",
    "WET": "Permanent Wetlands",
    "CRO": "Croplands",
    "URB": "Urban and Built-up Lands",
    "CVM": "Cropland/Natural Vegetation Mosaics",
    "SNO": "Snow and Ice",
    "BSV": "Barren or Sparsely Vegetated",
    "WAT": "Water Bodies",
}

#: Gas-analyser configurations, by the code Table S2 uses for them.
INSTRUMENT_SYSTEMS: Final[Mapping[str, str]] = {
    "O": "open-path",
    "C": "closed-path",
    "M": "mixed open- and closed-path",
}

#: Every site-metadata field Supplementary Table S2 carries, in canonical form.
SITE_METADATA_COLUMNS: Final[tuple[str, ...]] = (
    "site_id",
    "latitude",
    "longitude",
    "continent",
    "country",
    "igbp",
    "koppen",
    "start_date",
    "end_date",
    "elevation",
    "instrument_system",
    "height_ratio",
    "included_in_94_site_subset",
)

#: The metadata :func:`site_report` attaches by default - the Step 20A list.
SITE_REPORT_METADATA: Final[tuple[str, ...]] = (
    "latitude",
    "longitude",
    "igbp",
    "koppen",
    "instrument_system",
    "included_in_94_site_subset",
)

#: Column order of :func:`site_report` with the default metadata.
SITE_REPORT_COLUMNS: Final[tuple[str, ...]] = (
    "site",
    *SITE_REPORT_METADATA,
    *SENSITIVITY_TABLE_COLUMNS[1:],
)

#: Column order of :func:`site_population_summary`.
POPULATION_SUMMARY_COLUMNS: Final[tuple[str, ...]] = (
    "stratifier",
    "stratum",
    "n_sites",
    "fraction",
)

#: Column order of :func:`stratified_summary`.
STRATIFIED_SUMMARY_COLUMNS: Final[tuple[str, ...]] = (
    "target",
    "method",
    "mode",
    "gap_class",
    "subset",
    "stratifier",
    "stratum",
    "units",
    "metric",
    "n_sites",
    "n_sites_offered",
    "q1",
    "median",
    "q3",
    "iqr",
)

#: Column order of :func:`method_differences`, before the site metadata the
#: report carries is inserted after ``site``.
METHOD_DIFFERENCE_COLUMNS: Final[tuple[str, ...]] = (
    "site",
    "target",
    "gap_class",
    "subset",
    "units",
    "method",
    "baseline",
    "metric",
    "method_value",
    "baseline_value",
    "difference",
    "change",
)

#: Column order of :func:`welch_comparison`.
WELCH_TABLE_COLUMNS: Final[tuple[str, ...]] = (
    "target",
    "method",
    "baseline",
    "gap_class",
    "subset",
    "stratifier",
    "stratum",
    "units",
    "metric",
    "n_method",
    "n_baseline",
    "mean_method",
    "mean_baseline",
    "mean_difference",
    "ci_lower",
    "ci_upper",
    "confidence",
    "df",
    "t_statistic",
    "p_value",
    "significant",
    "n_paired",
    "n_improved",
    "n_worsened",
    "n_unchanged",
)

#: What counts as an improvement for each metric, in :func:`method_differences`.
#: A package convention for *counting* sites, not a statement in the paper -
#: Table S10 reports raw differences - and never a pass/fail rule.
IMPROVEMENT_DIRECTION: Final[Mapping[str, str]] = {
    "r2": "higher",
    "slope": "closer to 1",
    "rmse": "lower",
    "bias": "closer to 0",
}

#: Scores that grow as a metric improves, in the direction above.
_SCORES: Final[Mapping[str, Callable[[float], float]]] = {
    "r2": lambda value: value,
    "slope": lambda value: -abs(value - 1.0),
    "rmse": lambda value: -value,
    "bias": lambda value: -abs(value),
}

#: The cell one site contributes one value to.
_CELL_KEYS: Final[tuple[str, ...]] = (
    "site",
    "target",
    "method",
    "mode",
    "gap_class",
    "subset",
    "units",
)

#: The cell two methods are paired on at one site.
_PAIR_KEYS: Final[tuple[str, ...]] = ("site", "target", "gap_class", "subset", "units")

#: Sort orders for the labelled columns of every table here.
_ORDERS: Final[Mapping[str, Sequence[str]]] = {
    "method": METHOD_ORDER,
    "baseline": METHOD_ORDER,
    "gap_class": GAP_CLASS_ORDER,
    "subset": SUBSET_ORDER,
    "metric": METRIC_COLUMNS,
}

_MINUS: Final = "\N{MINUS SIGN}"


# ---------------------------------------------------------------------------
# Site metadata (Supplementary Tables S1-S2)
# ---------------------------------------------------------------------------

#: Header spellings accepted for each metadata field, after :func:`_column_key`
#: folds case, accents and punctuation: Table S2's own headers, FLUXNET's
#: ``SITE_ID`` / ``LOCATION_LAT`` style, and the canonical names themselves.
_METADATA_ALIASES: Final[Mapping[str, str]] = {
    "site_id": "site_id",
    "site": "site_id",
    "siteid": "site_id",
    "id": "site_id",
    "latitude": "latitude",
    "lat": "latitude",
    "location_lat": "latitude",
    "longitude": "longitude",
    "lon": "longitude",
    "long": "longitude",
    "location_long": "longitude",
    "continent": "continent",
    "country": "country",
    "igbp": "igbp",
    "koppen": "koppen",
    "koppen_climate": "koppen",
    "koppen_climate_class": "koppen",
    "start": "start_date",
    "start_date": "start_date",
    "end": "end_date",
    "end_date": "end_date",
    "elevation": "elevation",
    "location_elev": "elevation",
    "system": "instrument_system",
    "instrument_system": "instrument_system",
    "height_ratio": "height_ratio",
    "instrument_to_canopy_height_ratio": "height_ratio",
    "included_in_94_sites": "included_in_94_site_subset",
    "included_in_94_site_subset": "included_in_94_site_subset",
}

_SYSTEM_ALIASES: Final[Mapping[str, str]] = {
    "o": "O",
    "open": "O",
    "open_path": "O",
    "c": "C",
    "closed": "C",
    "closed_path": "C",
    "m": "M",
    "mixed": "M",
}

#: Table S2 marks membership of the 94-site subset with a check or a cross.
_INCLUDED_MARKS: Final[Mapping[str, bool]] = {
    "\N{CHECK MARK}": True,
    "\N{HEAVY CHECK MARK}": True,
    "true": True,
    "yes": True,
    "y": True,
    "1": True,
    "\N{MULTIPLICATION SIGN}": False,
    "\N{BALLOT X}": False,
    "\N{HEAVY BALLOT X}": False,
    "x": False,
    "false": False,
    "no": False,
    "n": False,
    "0": False,
}

_FLOAT_METADATA: Final[tuple[str, ...]] = ("latitude", "longitude", "elevation", "height_ratio")
_DATE_METADATA: Final[tuple[str, ...]] = ("start_date", "end_date")


def site_metadata(metadata: pd.DataFrame | Iterable[Mapping[str, object]]) -> pd.DataFrame:
    """Return site metadata in canonical form, validated.

    Accepts Supplementary Table S2 as printed (``Latitude``, ``IGBP``,
    ``System``, ``Included in 94 sites`` with its check and cross marks),
    FLUXNET-style headers (``SITE_ID``, ``LOCATION_LAT``, ...) or the canonical
    names of :data:`SITE_METADATA_COLUMNS`, and returns those canonical columns
    - every one, missing where the input had no such field - followed by any
    other column the input carried, unchanged.

    Values are normalised rather than guessed: IGBP codes are upper-cased (Table
    S2 spells one wetland ``Wet``) and must be a class of :data:`IGBP_CLASSES`;
    the instrument system is one of the codes of :data:`INSTRUMENT_SYSTEMS`;
    membership of the 94-site subset is a nullable boolean. Applying the function
    twice gives the same table.

    :param metadata: a frame, or an iterable of per-site mappings.
    :returns: one row per site, in input order.
    :raises SiteReportError: if there is no site column, a site is missing or
        repeated, two input columns describe one field, or a value is not what
        its field allows - a latitude beyond 90 degrees, an unknown IGBP class,
        an end date before a start date.
    """
    if isinstance(metadata, pd.DataFrame):
        source = metadata
    elif isinstance(metadata, Iterable) and not isinstance(metadata, str | bytes | Mapping):
        source = pd.DataFrame(list(metadata))
    else:
        raise SiteReportError(
            "site metadata must be a DataFrame or an iterable of per-site mappings, got "
            f"{type(metadata).__name__}"
        )
    renames: dict[Any, str] = {}
    for column in source.columns:
        canonical = _METADATA_ALIASES.get(_column_key(column))
        if canonical is None:
            continue
        clash = next((name for name, field in renames.items() if field == canonical), None)
        if clash is not None:
            raise SiteReportError(
                f"columns {clash!r} and {column!r} both describe {canonical}; keep one"
            )
        renames[column] = canonical
    if "site_id" not in renames.values():
        raise SiteReportError(
            "site metadata needs a site column (site_id, site or SITE_ID); the frame carries "
            f"{', '.join(str(name) for name in source.columns) or 'nothing'}"
        )
    frame = source.rename(columns=renames)
    rows = len(frame)

    def values(column: str) -> list[Any]:
        return frame[column].tolist() if column in frame.columns else [None] * rows

    sites = _site_ids(values("site_id"))
    data: dict[str, Any] = {"site_id": pd.Series(sites, dtype=object)}
    for column, bounds in (
        ("latitude", (-90.0, 90.0)),
        ("longitude", (-180.0, 180.0)),
        ("elevation", None),
        ("height_ratio", None),
    ):
        data[column] = pd.Series(_numbers(values(column), sites, column, bounds), dtype="float64")
    for column in ("continent", "country", "koppen"):
        data[column] = pd.Series([_text(value) for value in values(column)], dtype=object)
    data["igbp"] = pd.Series(_igbp(values("igbp"), sites), dtype=object)
    data["instrument_system"] = pd.Series(
        _systems(values("instrument_system"), sites), dtype=object
    )
    for column in _DATE_METADATA:
        data[column] = pd.Series(_dates(values(column), sites, column), dtype="datetime64[ns]")
    data["included_in_94_site_subset"] = pd.array(
        _included(values("included_in_94_site_subset"), sites), dtype="boolean"
    )
    table: pd.DataFrame = pd.DataFrame(data).loc[:, list(SITE_METADATA_COLUMNS)]
    late = table["end_date"] < table["start_date"]
    if bool(late.any()):
        offenders = [str(site) for site, flag in zip(table["site_id"], late, strict=True) if flag]
        shown = ", ".join(offenders[:3])
        raise SiteReportError(f"site(s) {shown}: the record ends before it starts")
    for extra in (column for column in frame.columns if column not in SITE_METADATA_COLUMNS):
        table[extra] = frame[extra].to_numpy()
    return table


def _column_key(name: object) -> str:
    """Fold a header to lower-case ASCII words joined by underscores."""
    decomposed = unicodedata.normalize("NFKD", str(name))
    plain = "".join(char for char in decomposed if not unicodedata.combining(char))
    return re.sub(r"[^0-9a-z]+", "_", plain.lower()).strip("_")


def _blank(value: object) -> bool:
    """Whether a metadata cell is empty, however it was spelled."""
    return _is_missing(value) or (isinstance(value, str) and not value.strip())


def _text(value: object) -> str | None:
    """Return a stripped label, or ``None`` for an empty cell."""
    return None if _blank(value) else str(value).strip()


def _site_ids(values: Sequence[object]) -> list[str]:
    """Return the site labels, refusing a missing or a repeated one."""
    labels = [_text(value) for value in values]
    empty = sum(1 for label in labels if label is None)
    if empty:
        raise SiteReportError(f"{empty} metadata row(s) carry no site label")
    sites = cast("list[str]", labels)
    repeated = sorted({site for site in sites if sites.count(site) > 1})
    if repeated:
        raise SiteReportError(
            f"site(s) {', '.join(repeated[:5])} appear more than once in the metadata; one "
            "row per site is what lets a report attach it without guessing"
        )
    return sites


def _numbers(
    values: Sequence[object],
    sites: Sequence[str],
    column: str,
    bounds: tuple[float, float] | None,
) -> list[float]:
    """Return ``values`` as floats, missing as ``NaN``, refusing anything else."""
    numbers: list[float] = []
    for site, value in zip(sites, values, strict=True):
        if _blank(value):
            numbers.append(math.nan)
            continue
        try:
            number = float(str(value).strip().replace(_MINUS, "-"))
        except ValueError:
            raise SiteReportError(f"site {site}: {column} {value!r} is not a number") from None
        if math.isnan(number):
            numbers.append(number)
            continue
        if bounds is not None and not bounds[0] <= number <= bounds[1]:
            raise SiteReportError(
                f"site {site}: {column} {number} is outside [{bounds[0]}, {bounds[1]}]"
            )
        numbers.append(number)
    return numbers


def _igbp(values: Sequence[object], sites: Sequence[str]) -> list[str | None]:
    """Return IGBP codes upper-cased, refusing a class that does not exist."""
    codes: list[str | None] = []
    for site, value in zip(sites, values, strict=True):
        label = _text(value)
        if label is None:
            codes.append(None)
            continue
        code = label.upper()
        if code not in IGBP_CLASSES:
            raise SiteReportError(
                f"site {site}: {label!r} is not an IGBP class; expected one of "
                f"{', '.join(IGBP_CLASSES)}"
            )
        codes.append(code)
    return codes


def _systems(values: Sequence[object], sites: Sequence[str]) -> list[str | None]:
    """Return instrument-system codes, refusing one Table S2 does not define."""
    codes: list[str | None] = []
    for site, value in zip(sites, values, strict=True):
        label = _text(value)
        if label is None:
            codes.append(None)
            continue
        code = _SYSTEM_ALIASES.get(_column_key(label))
        if code is None:
            raise SiteReportError(
                f"site {site}: instrument system {label!r} is not one of "
                + ", ".join(f"{key} ({name})" for key, name in INSTRUMENT_SYSTEMS.items())
            )
        codes.append(code)
    return codes


def _dates(values: Sequence[object], sites: Sequence[str], column: str) -> list[Any]:
    """Return ``values`` as timestamps, missing as ``NaT``, refusing anything else."""
    dates: list[Any] = []
    for site, value in zip(sites, values, strict=True):
        if _blank(value):
            dates.append(pd.NaT)
            continue
        try:
            dates.append(pd.Timestamp(cast("Any", value)))
        except (TypeError, ValueError):
            raise SiteReportError(f"site {site}: {column} {value!r} is not a date") from None
    return dates


def _included(values: Sequence[object], sites: Sequence[str]) -> list[bool | None]:
    """Return membership of the 94-site subset, from any of its spellings."""
    flags: list[bool | None] = []
    for site, value in zip(sites, values, strict=True):
        if _blank(value):
            flags.append(None)
        elif isinstance(value, bool | np.bool_) or (
            isinstance(value, int | float) and value in (0, 1)
        ):
            flags.append(bool(value))
        else:
            flag = _INCLUDED_MARKS.get(str(value).strip().lower())
            if flag is None:
                raise SiteReportError(
                    f"site {site}: {value!r} does not say whether the site is in the 94-site "
                    "subset; use a check or cross mark, true/false, yes/no or 1/0"
                )
            flags.append(flag)
    return flags


def _empty_metadata() -> pd.DataFrame:
    """Return a metadata table with no sites and the canonical dtypes."""
    return site_metadata(pd.DataFrame({"site_id": pd.Series(dtype=object)}))


def site_population_summary(
    metadata: pd.DataFrame | Iterable[Mapping[str, object]],
    *,
    by: str = "igbp",
) -> pd.DataFrame:
    """Return how many sites fall in each class of one metadata field.

    The shape of Supplementary Table S1: a count, and the fraction of every site
    in the metadata, per continent, IGBP class, Koppen class or any other field.
    Sites with no value for the field are counted in a row of their own, with a
    missing ``stratum``, so the fractions always sum to one.

    :param metadata: anything :func:`site_metadata` accepts.
    :param by: the field to count by.
    :returns: a frame with :data:`POPULATION_SUMMARY_COLUMNS`, classes in sorted
        order and the unclassified row last.
    :raises SiteReportError: if ``by`` is not a field of the metadata.
    """
    table = site_metadata(metadata)
    if by not in table.columns or by == "site_id":
        raise SiteReportError(
            f"cannot count sites by {by!r}; the metadata carries "
            f"{', '.join(str(name) for name in table.columns if name != 'site_id')}"
        )
    counts: dict[str | None, int] = {}
    for label in (_stratum_label(value) for value in table[by]):
        counts[label] = counts.get(label, 0) + 1
    total = len(table)
    rows = [
        {"stratifier": by, "stratum": label, "n_sites": count, "fraction": count / total}
        for label, count in sorted(
            counts.items(), key=lambda item: (item[0] is None, item[0] or "")
        )
    ]
    summary: pd.DataFrame = pd.DataFrame(rows, columns=list(POPULATION_SUMMARY_COLUMNS))
    summary["n_sites"] = summary["n_sites"].astype("Int64")
    summary["fraction"] = summary["fraction"].astype("float64")
    return summary


# ---------------------------------------------------------------------------
# Published site-level tables, read from a local copy of the supplement
# ---------------------------------------------------------------------------

#: Where the site metadata is tabulated.
TABLE_S2: Final = "Supplementary Table S2 (mmc3.docx)"

#: The site population of Table S9.
SITE_SET_194: Final = "194-site FLUXNET2015 set"

#: Column order of the published per-site tables: a
#: :func:`~rfrgapfill.sensitivity.gap_length_table`, labelled with its source.
PUBLISHED_SITE_TABLE_COLUMNS: Final[tuple[str, ...]] = (
    *SENSITIVITY_TABLE_COLUMNS,
    "site_population",
    "source",
)


@dataclass(frozen=True)
class PublishedSiteTable(FrozenRecord):
    """One supplementary table that scores methods site by site."""

    #: The table, e.g. ``"Supplementary Table S9"``.
    table: str
    #: The supplementary file it is printed in.
    file: str
    #: The site population it covers.
    site_population: str
    #: How many sites it scores.
    n_sites: int
    #: The fluxes it scores.
    targets: tuple[str, ...]
    #: The methods it scores.
    methods: tuple[str, ...]
    #: The day/night subset it scores.
    subset: str

    @property
    def source(self) -> str:
        """The table and its file, as the ``source`` column spells it."""
        return f"{self.table} ({self.file})"


#: The site-level tables of the supplement, and what each one covers.
PUBLISHED_SITE_TABLES: Final[tuple[PublishedSiteTable, ...]] = (
    PublishedSiteTable(
        "Supplementary Table S4", "mmc5.docx", SITE_SUBSET_94, 94,
        ("NEE", "H", "LE"), ("MDS", "RFR3", "RFR10"), "all",
    ),
    PublishedSiteTable(
        "Supplementary Table S5", "mmc5.docx", SITE_SUBSET_94, 94,
        ("NEE", "H", "LE"), ("MDS", "RFR3", "RFR10"), "daytime",
    ),
    PublishedSiteTable(
        "Supplementary Table S6", "mmc5.docx", SITE_SUBSET_94, 94,
        ("NEE", "H", "LE"), ("MDS", "RFR3", "RFR10"), "nighttime",
    ),
    PublishedSiteTable(
        "Supplementary Table S9", "mmc8.docx", SITE_SET_194, 194,
        ("NEE",), ("MDS", "RFR3"), "all",
    ),
)  # fmt: skip

#: Column order of :func:`site_level_evidence`.
SITE_EVIDENCE_COLUMNS: Final[tuple[str, ...]] = (
    "target",
    "method",
    "n_sites",
    "site_population",
    "tables",
)


def site_level_evidence() -> pd.DataFrame:
    """Return how many sites each target and method was validated at, site by site.

    The constraint Step 20A states, as a table: RFR3 NEE is scored at all 194
    sites (Table S9), RFR10 at the 94-site subset only (Tables S4-S6). A
    reproduction that runs RFR3 at a site where RFR10 cannot be run is using the
    evidence the paper has, not exceeding it.

    :returns: a frame with :data:`SITE_EVIDENCE_COLUMNS`; ``n_sites`` is the
        widest population any table covers, and ``tables`` lists every table.
    """
    widest: dict[tuple[str, str], PublishedSiteTable] = {}
    tables: dict[tuple[str, str], list[str]] = {}
    for record in PUBLISHED_SITE_TABLES:
        for target in record.targets:
            for method in record.methods:
                key = (target, method)
                tables.setdefault(key, []).append(record.table.removeprefix("Supplementary "))
                if key not in widest or record.n_sites > widest[key].n_sites:
                    widest[key] = record
    rows = [
        {
            "target": target,
            "method": method,
            "n_sites": record.n_sites,
            "site_population": record.site_population,
            "tables": ", ".join(tables[target, method]),
        }
        for (target, method), record in widest.items()
    ]
    evidence = pd.DataFrame(rows, columns=list(SITE_EVIDENCE_COLUMNS))
    evidence["n_sites"] = evidence["n_sites"].astype("Int64")
    return _in_reporting_order(evidence, ("target", "method"))


_W: Final = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def _docx(path: str | Path) -> tuple[str, list[list[list[str]]]]:
    """Return the text of a Word document and its tables, as rows of cell text."""
    location = Path(path)
    try:
        with zipfile.ZipFile(location) as archive:
            document = archive.read("word/document.xml")
    except FileNotFoundError:
        raise SiteReportError(
            f"{location} does not exist; the supplementary files are publisher material and "
            "are not shipped with this package"
        ) from None
    except (zipfile.BadZipFile, KeyError) as error:
        raise SiteReportError(f"{location} is not a Word document") from error
    root = ElementTree.fromstring(document)
    tables = [
        [
            [
                " ".join(
                    " ".join(
                        "".join(run.text or "" for run in paragraph.iter(f"{_W}t"))
                        for paragraph in cell.iter(f"{_W}p")
                    ).split()
                )
                for cell in row.findall(f"{_W}tc")
            ]
            for row in table.findall(f"{_W}tr")
        ]
        for table in root.iter(f"{_W}tbl")
    ]
    return " ".join("".join(root.itertext()).split()), tables


def _require_caption(text: str, caption: str, path: str | Path) -> None:
    """Raise unless the document names ``caption``."""
    if caption not in text:
        raise SiteReportError(f"{Path(path).name} does not contain Supplementary {caption}")


def read_table_s2(path: str | Path) -> pd.DataFrame:
    """Return Supplementary Table S2 - the 194 sites - as canonical site metadata.

    :param path: a local copy of ``mmc3.docx``.
    :returns: :func:`site_metadata` of the table, one row per site, with
        ``attrs["source"]`` naming the table.
    :raises SiteReportError: if the file is not the supplement's Table S2.
    """
    text, tables = _docx(path)
    _require_caption(text, "Table S2", path)
    rows = next((rows for rows in tables if rows and {"Latitude", "IGBP"} <= set(rows[0])), None)
    if rows is None:
        raise SiteReportError(f"{Path(path).name}: no table with Latitude and IGBP columns")
    # Table S2 leaves its first header, over the site codes, empty.
    header = ["site_id" if index == 0 and not name else name for index, name in enumerate(rows[0])]
    uneven = [number for number, row in enumerate(rows[1:], start=2) if len(row) != len(header)]
    if uneven:
        raise SiteReportError(
            f"{Path(path).name}: Table S2 row(s) {uneven[:5]} do not have {len(header)} cells"
        )
    metadata = site_metadata(pd.DataFrame(rows[1:], columns=header))
    metadata.attrs["source"] = TABLE_S2
    return metadata


def read_tables_s4_to_s6(
    path: str | Path,
    *,
    nee_units: str = NEE_BENCHMARK_UNITS,
) -> pd.DataFrame:
    """Return Supplementary Tables S4-S6 - the 94 sites scored - as one tidy table.

    One row per site, flux, method and day/night subset (S4 diel, S5 daytime, S6
    nighttime), in the shape of :func:`~rfrgapfill.sensitivity.gap_length_table`
    plus ``site_population`` and ``source``. ``gap_class`` is ``all``: the
    tables pool every gap class. Row counts are not published and are missing.

    **NEE units (A10).** The file states no units. Table S3, whose medians these
    are, and Table S10 give NEE RMSE and bias in ``g C m-2 d-1``, so that is the
    default and the published medians compare against these rows directly. But
    Table S9 prints the *same* NEE RMSE values for the same 94 sites in
    ``umol m-2 s-1``; ``nee_units`` is how a caller who reads the conflict the
    other way says so.

    :param path: a local copy of ``mmc5.docx``.
    :param nee_units: the units NEE ``rmse`` and ``bias`` are labelled with.
    :raises SiteReportError: if the file does not have the layout of Tables S4-S6.
    """
    text, tables = _docx(path)
    captions = [text.find(f"Table {name}") for name in ("S4", "S5", "S6")]
    if min(captions) < 0 or captions != sorted(captions) or len(tables) != 3:
        raise SiteReportError(
            f"{Path(path).name} does not have the layout of Supplementary Tables S4-S6: three "
            f"captioned tables, in order; found {len(tables)} table(s)"
        )
    parts = [
        _published_site_rows(
            rows,
            record=record,
            units={"NEE": nee_units, "H": ENERGY_FLUX_UNITS, "LE": ENERGY_FLUX_UNITS},
        )
        for rows, record in zip(tables, PUBLISHED_SITE_TABLES[:3], strict=True)
    ]
    return _published_frame(pd.concat(parts, ignore_index=True))


def read_table_s9(path: str | Path, *, nee_units: str = NEE_MODEL_UNITS) -> pd.DataFrame:
    """Return Supplementary Table S9 - NEE at 194 sites, MDS and RFR3 - as a tidy table.

    The broader evidence for RFR3: every site of Table S2, not only the 94-site
    subset. The table's caption gives RMSE and bias in ``umol m-2 s-1``, which is
    the default ``nee_units``.

    :param path: a local copy of ``mmc8.docx``.
    :param nee_units: the units NEE ``rmse`` and ``bias`` are labelled with.
    :raises SiteReportError: if the file does not have the layout of Table S9.
    """
    text, tables = _docx(path)
    _require_caption(text, "Table S9", path)
    if len(tables) != 1:
        raise SiteReportError(f"{Path(path).name}: expected one table, found {len(tables)}")
    frame = _published_site_rows(
        tables[0], record=PUBLISHED_SITE_TABLES[3], units={"NEE": nee_units}
    )
    return _published_frame(frame)


def _published_site_rows(
    rows: Sequence[Sequence[str]],
    *,
    record: PublishedSiteTable,
    units: Mapping[str, str],
) -> pd.DataFrame:
    """Return one published per-site table as tidy rows.

    The layout the supplement uses throughout: a row naming the methods, a row
    naming the four metrics under each, then one row per site - with a row
    carrying nothing but a flux name opening each flux's block where the table
    covers more than one.
    """
    where = record.table.removeprefix("Supplementary ")
    methods = record.methods
    if len(rows) < 3:
        raise SiteReportError(f"{where} has no site rows")
    named = [cell for cell in rows[0] if cell]
    if named != list(methods):
        raise SiteReportError(f"{where}: expected method columns {list(methods)}, found {named}")
    metrics = [cell.lower() for cell in rows[1][1:] if cell]
    if metrics != list(METRIC_COLUMNS) * len(methods):
        raise SiteReportError(f"{where}: expected R2, Slope, RMSE and Bias under every method")
    width = 1 + len(METRIC_COLUMNS) * len(methods)
    target = record.targets[0] if len(record.targets) == 1 else None
    records: list[dict[str, Any]] = []
    for number, row in enumerate(rows[2:], start=3):
        filled = [cell for cell in row if cell]
        if not filled:
            continue
        if len(filled) == 1 and filled[0] in record.targets:
            target = filled[0]
            continue
        if target is None:
            raise SiteReportError(f"{where} row {number}: a site row before any flux heading")
        if len(row) != width:
            raise SiteReportError(f"{where} row {number}: expected {width} cells, found {len(row)}")
        scores = [_published_number(cell, where=f"{where} row {number}") for cell in row[1:]]
        for position, method in enumerate(methods):
            block = scores[position * len(METRIC_COLUMNS) : (position + 1) * len(METRIC_COLUMNS)]
            records.append(
                {
                    "site": row[0],
                    "target": target,
                    "method": method,
                    "mode": method if method.startswith("RFR") else None,
                    "gap_class": POOLED_GAP_CLASS,
                    "subset": record.subset,
                    "units": units.get(target),
                    "n": None,
                    "n_offered": None,
                    **dict(zip(METRIC_COLUMNS, block, strict=True)),
                    "site_population": record.site_population,
                    "source": record.source,
                }
            )
    frame: pd.DataFrame = pd.DataFrame(records, columns=list(PUBLISHED_SITE_TABLE_COLUMNS))
    repeated = frame.duplicated(subset=["site", "target", "method"], keep=False)
    if bool(repeated.any()):
        shown = ", ".join(sorted(set(frame.loc[repeated, "site"].astype(str)))[:5])
        raise SiteReportError(f"{where} scores site(s) {shown} twice")
    return frame


def _published_number(cell: str, *, where: str) -> float | None:
    """Return one printed score, or ``None`` for an empty cell."""
    text = cell.strip().replace(_MINUS, "-")
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        raise SiteReportError(f"{where}: {cell!r} is not a number") from None


def _published_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a published per-site table with the dtypes of a tidy table."""
    typed = _as_float(frame, METRIC_COLUMNS)
    typed = _as_count(typed, ("n", "n_offered"))
    return _in_reporting_order(typed, ("site", "target", "method", "subset"))


# ---------------------------------------------------------------------------
# Site-level reports
# ---------------------------------------------------------------------------


def site_report(
    results: Reportish | Iterable[Any] | Mapping[str, Any],
    metadata: pd.DataFrame | Iterable[Mapping[str, object]] | None = None,
    *,
    units: Mapping[str, str] | None = None,
    columns: Sequence[str] = SITE_REPORT_METADATA,
) -> pd.DataFrame:
    """Return every site's scores beside what is known about the site.

    The Step 20A table and acceptance test 39: the tidy table of
    :func:`~rfrgapfill.sensitivity.gap_length_table` - one row per site, target,
    method, gap class and day/night subset - with the site's metadata attached.
    Each row is the score of one site's own model; nothing is pooled.

    A site absent from the metadata keeps its rows, with the metadata missing,
    and is named in a :class:`SiteReportWarning`: it is still a site of the
    study, and still enters every pooled summary.

    :param results: anything :func:`~rfrgapfill.sensitivity.gap_length_table`
        accepts, with every row labelled by site - a mapping of site label to
        results, a site-labelled tidy table, or a published per-site table from
        :func:`read_tables_s4_to_s6` or :func:`read_table_s9`.
    :param metadata: anything :func:`site_metadata` accepts; Table S2 from
        :func:`read_table_s2` included.
    :param units: passed to :func:`~rfrgapfill.sensitivity.gap_length_table`.
    :param columns: the metadata fields to attach, from
        :data:`SITE_METADATA_COLUMNS`.
    :returns: a frame with ``site``, the ``columns``, then the rest of
        :data:`~rfrgapfill.sensitivity.SENSITIVITY_TABLE_COLUMNS`
        (:data:`SITE_REPORT_COLUMNS` by default).
    :raises SiteReportError: if a row carries no site label, one site repeats a
        cell, or an unknown metadata field is requested.
    """
    wanted = list(columns)
    unknown = [name for name in wanted if name not in SITE_METADATA_COLUMNS or name == "site_id"]
    if unknown:
        raise SiteReportError(
            f"unknown metadata field(s) {', '.join(unknown)}; expected any of "
            f"{', '.join(SITE_METADATA_COLUMNS[1:])}"
        )
    table = gap_length_table(results, units=units)
    _require_sites(table, what="a multi-site report")
    _require_unique_cells(table)
    table["site"] = table["site"].astype(str)
    known = _empty_metadata() if metadata is None else site_metadata(metadata)
    if metadata is not None:
        absent = sorted(set(table["site"]) - set(known["site_id"]))
        if absent:
            warnings.warn(
                SiteReportWarning(
                    f"{len(absent)} site(s) have no metadata row and are reported without "
                    f"it: {', '.join(absent[:5])}" + (" ..." if len(absent) > 5 else "")
                ),
                stacklevel=2,
            )
    attached = cast("pd.DataFrame", known.loc[:, ["site_id", *wanted]]).rename(
        columns={"site_id": "site"}
    )
    report = table.merge(attached, on="site", how="left", validate="many_to_one")
    ordered: pd.DataFrame = report.loc[:, ["site", *wanted, *SENSITIVITY_TABLE_COLUMNS[1:]]]
    return cast("pd.DataFrame", ordered.reset_index(drop=True))


def stratified_summary(
    report: pd.DataFrame,
    *,
    by: str | None = "igbp",
    metrics: Sequence[str] = METRIC_COLUMNS,
) -> pd.DataFrame:
    """Return the across-site distribution of each metric, pooled and per stratum.

    For every target, method, gap class, day/night subset and stratum - IGBP
    class by default - the quartiles of the per-site values of each metric.
    Rows pooling every stratum carry ``stratum="all"`` and are always present; a
    site with no value for ``by`` contributes to them only.

    The table is long - one row per metric - because how many sites *have* a
    value differs by metric: a site with constant measurements has an RMSE but
    no R2. ``n_sites`` counts the sites with a value, ``n_sites_offered`` every
    site in the stratum. Quartiles use linear interpolation (A11); one site has
    quartiles but no spread.

    No stratum is compared against a threshold, and none against another. The
    point of the table is that ecosystems differ.

    :param report: a :func:`site_report` frame, or any site-labelled tidy table
        carrying the ``by`` column.
    :param by: the site-metadata column to stratify by; ``None`` for the pooled
        rows only.
    :param metrics: the metrics to summarise.
    :returns: a frame with :data:`STRATIFIED_SUMMARY_COLUMNS`.
    :raises SiteReportError: if ``by`` is not a column of the report, if a site
        repeats a cell, or if a stratum is itself labelled ``all``.
    """
    _require_metrics(metrics)
    frame = _require_frame(report, (*SENSITIVITY_TABLE_COLUMNS[:7], *metrics), what="a summary")
    if by is not None and by not in frame.columns:
        raise SiteReportError(
            f"the report has no {by!r} column to stratify by; attach site metadata with "
            "site_report(results, metadata)"
        )
    _require_sites(frame, what="a summary across sites")
    _require_unique_cells(frame)
    strata = _strata(frame, by)
    keys = ["target", "method", "mode", "gap_class", "subset", "stratifier", "stratum", "units"]
    rows: list[dict[str, Any]] = []
    for key, group in strata.groupby(keys, dropna=False, sort=False):
        labels = dict(zip(keys, cast("tuple[Any, ...]", key), strict=True))
        offered = int(group["site"].nunique())
        for metric in metrics:
            values = pd.to_numeric(group[metric], errors="coerce").to_numpy(dtype=float)
            defined = values[np.isfinite(values)]
            q1 = median = q3 = iqr = None
            if defined.size:
                q1, median, q3 = (float(value) for value in np.percentile(defined, (25, 50, 75)))
                if defined.size > 1:
                    iqr = q3 - q1
            rows.append(
                {
                    **labels,
                    "metric": metric,
                    "n_sites": int(defined.size),
                    "n_sites_offered": offered,
                    "q1": q1,
                    "median": median,
                    "q3": q3,
                    "iqr": iqr,
                }
            )
    summary = pd.DataFrame(rows, columns=list(STRATIFIED_SUMMARY_COLUMNS))
    summary = _as_float(summary, ("q1", "median", "q3", "iqr"))
    summary = _as_count(summary, ("n_sites", "n_sites_offered"))
    return _in_reporting_order(
        summary, ("target", "method", "gap_class", "subset", "stratum", "units", "metric")
    )


def method_differences(
    report: pd.DataFrame,
    *,
    method: str,
    baseline: str,
    metrics: Sequence[str] = METRIC_COLUMNS,
) -> pd.DataFrame:
    """Return, site by site, how one method's scores differ from a baseline's.

    ``difference`` is ``method - baseline``, the sign Table S10 uses: a positive
    R2 difference is a gain. ``change`` says whether that difference is an
    improvement in the sense of :data:`IMPROVEMENT_DIRECTION` - ``improved``,
    ``worsened`` or ``unchanged`` - so a site where the new method does worse is
    reported as such rather than hidden in a mean. No site is required to
    improve.

    Every site scored by either method has a row. A site scored by only one -
    RFR3 at a site where RFR10 could not be run, say - keeps its one value, with
    the difference and the change missing.

    :param report: a :func:`site_report` frame, or any site-labelled tidy table.
    :param method: the method whose change is reported, e.g. ``"RFR3"``.
    :param baseline: the method it is compared with, e.g. ``"MDS"``.
    :param metrics: the metrics to difference.
    :returns: a frame with :data:`METHOD_DIFFERENCE_COLUMNS`, with any site
        metadata the report carries inserted after ``site``.
    :raises SiteReportError: if the two methods are the same or either is
        absent, if a row is unlabelled, or if one method scores one site's cell
        twice.
    """
    _require_metrics(metrics)
    frame = _require_frame(report, (*_PAIR_KEYS, "method", *metrics), what="method differences")
    if str(method) == str(baseline):
        raise SiteReportError(f"{method} cannot be compared with itself")
    present = sorted({str(value) for value in frame["method"]})
    for name in (method, baseline):
        if str(name) not in present:
            raise SiteReportError(
                f"the report has no {name} rows; it carries {', '.join(present) or 'nothing'}"
            )
    _require_sites(frame, what="a comparison across sites")
    frame = frame.assign(site=frame["site"].astype(str))
    arms: list[pd.DataFrame] = []
    for name in (method, baseline):
        rows = cast(
            "pd.DataFrame",
            frame.loc[frame["method"].astype(str) == str(name), [*_PAIR_KEYS, *metrics]],
        )
        repeated = rows.duplicated(subset=list(_PAIR_KEYS), keep=False)
        if bool(repeated.any()):
            shown = "; ".join(
                " ".join(str(value) for value in row)
                for row in rows.loc[repeated, list(_PAIR_KEYS)].drop_duplicates().head(3).to_numpy()
            )
            raise SiteReportError(
                f"{name} scores one site/target/gap-class/subset cell more than once - two "
                f"modes or two runs - so there is no single value to difference: {shown}"
            )
        arms.append(rows)
    paired = arms[0].merge(
        arms[1], on=list(_PAIR_KEYS), how="outer", suffixes=("_method", "_baseline")
    )
    records: list[dict[str, Any]] = []
    for record in paired.to_dict("records"):
        for metric in metrics:
            new = _finite(record[f"{metric}_method"])
            old = _finite(record[f"{metric}_baseline"])
            records.append(
                {
                    **{key: record[key] for key in _PAIR_KEYS},
                    "method": str(method),
                    "baseline": str(baseline),
                    "metric": metric,
                    "method_value": new,
                    "baseline_value": old,
                    "difference": None if new is None or old is None else new - old,
                    "change": _change(metric, new, old),
                }
            )
    differences = pd.DataFrame(records, columns=list(METHOD_DIFFERENCE_COLUMNS))
    differences = _as_float(differences, ("method_value", "baseline_value", "difference"))
    carried = [name for name in SITE_METADATA_COLUMNS[1:] if name in frame.columns]
    if carried:
        per_site = frame.drop_duplicates(subset="site").loc[:, ["site", *carried]]
        differences = differences.merge(per_site, on="site", how="left", validate="many_to_one")
        differences = differences.loc[:, ["site", *carried, *METHOD_DIFFERENCE_COLUMNS[1:]]]
    return _in_reporting_order(differences, ("site", "target", "gap_class", "subset", "metric"))


def _finite(value: object) -> float | None:
    """Return ``value`` as a float, or ``None`` where it is missing."""
    if _is_missing(value):
        return None
    number = float(cast("float", value))
    return number if math.isfinite(number) else None


def _change(metric: str, new: float | None, old: float | None) -> str | None:
    """Return whether ``new`` improves on ``old`` for ``metric``."""
    if new is None or old is None:
        return None
    score = _SCORES[metric]
    if score(new) > score(old):
        return "improved"
    if score(new) < score(old):
        return "worsened"
    return "unchanged"


def welch_comparison(
    report: pd.DataFrame,
    *,
    method: str,
    baseline: str,
    metrics: Sequence[str] = METRIC_COLUMNS,
    by: str | None = None,
    confidence: float = 0.95,
) -> pd.DataFrame:
    """Return Welch's test of one method against a baseline, across sites.

    The statistic of Supplementary Table S10: the per-site values of each method
    are two samples, and the mean difference ``method - baseline`` is tested with
    Welch's unequal-variance t-test, with a confidence interval from the
    Welch-Satterthwaite degrees of freedom. Reproduced as published - two
    independent samples - although the values are paired by site; the paired
    view is in the same row as counts of the sites that ``improved``,
    ``worsened`` and did not change (:func:`method_differences`).

    Statistical testing across sites is separate from gap filling and asserts
    nothing: ``significant`` is ``p < 1 - confidence``, reported, never
    required.

    A group needs two sites with a value on each side for a test; with fewer,
    or with no spread on either side, the means are reported and the test is
    missing.

    :param report: a :func:`site_report` frame, or any site-labelled tidy table.
    :param method: the method tested, e.g. ``"RFR3"``.
    :param baseline: the method it is tested against, e.g. ``"MDS"``.
    :param metrics: the metrics to test.
    :param by: a site-metadata column to repeat the test within, beside the
        pooled rows; ``None`` for the pooled rows only.
    :param confidence: the confidence level of the interval.
    :returns: a frame with :data:`WELCH_TABLE_COLUMNS`.
    :raises SiteReportError: as :func:`method_differences`, or if ``by`` is not
        a column of the report, or ``confidence`` is not strictly between 0 and 1.
    """
    if not 0.0 < float(confidence) < 1.0:
        raise SiteReportError(f"confidence must be strictly between 0 and 1, got {confidence}")
    if by is not None and (not isinstance(report, pd.DataFrame) or by not in report.columns):
        raise SiteReportError(
            f"the report has no {by!r} column to stratify by; attach site metadata with "
            "site_report(results, metadata)"
        )
    differences = method_differences(report, method=method, baseline=baseline, metrics=metrics)
    if by is not None and by not in differences.columns:
        per_site = report.assign(site=report["site"].astype(str)).drop_duplicates(subset="site")
        differences = differences.merge(
            per_site.loc[:, ["site", by]], on="site", how="left", validate="many_to_one"
        )
    strata = _strata(differences, by)
    keys = ["target", "gap_class", "subset", "stratifier", "stratum", "units", "metric"]
    rows: list[dict[str, Any]] = []
    for key, group in strata.groupby(keys, dropna=False, sort=False):
        new = group["method_value"].to_numpy(dtype=float)
        old = group["baseline_value"].to_numpy(dtype=float)
        changes = group["change"].tolist()
        rows.append(
            {
                **dict(zip(keys, cast("tuple[Any, ...]", key), strict=True)),
                "method": str(method),
                "baseline": str(baseline),
                **_welch(new[np.isfinite(new)], old[np.isfinite(old)], float(confidence)),
                "confidence": float(confidence),
                "n_paired": int(np.sum(np.isfinite(new) & np.isfinite(old))),
                "n_improved": changes.count("improved"),
                "n_worsened": changes.count("worsened"),
                "n_unchanged": changes.count("unchanged"),
            }
        )
    table = pd.DataFrame(rows, columns=list(WELCH_TABLE_COLUMNS))
    table = _as_float(
        table,
        (
            "mean_method",
            "mean_baseline",
            "mean_difference",
            "ci_lower",
            "ci_upper",
            "confidence",
            "df",
            "t_statistic",
            "p_value",
        ),
    )
    table = _as_count(
        table, ("n_method", "n_baseline", "n_paired", "n_improved", "n_worsened", "n_unchanged")
    )
    table["significant"] = pd.array(table["significant"].tolist(), dtype="boolean")
    return _in_reporting_order(
        table, ("target", "gap_class", "subset", "stratum", "units", "metric")
    )


def _welch(new: np.ndarray, old: np.ndarray, confidence: float) -> dict[str, Any]:
    """Return Welch's two-sample t-test of ``mean(new) - mean(old)``."""
    result: dict[str, Any] = {
        "n_method": int(new.size),
        "n_baseline": int(old.size),
        "mean_method": float(new.mean()) if new.size else None,
        "mean_baseline": float(old.mean()) if old.size else None,
        "mean_difference": None,
        "ci_lower": None,
        "ci_upper": None,
        "df": None,
        "t_statistic": None,
        "p_value": None,
        "significant": None,
    }
    if new.size and old.size:
        result["mean_difference"] = float(new.mean() - old.mean())
    if new.size < 2 or old.size < 2:
        return result
    if np.ptp(new) == 0.0 and np.ptp(old) == 0.0:
        # No spread on either side: the difference is exact, and a t statistic
        # would divide by zero - or, since a variance of identical floats is
        # rounding noise rather than zero, by something that only looks like it.
        return result
    share_new = float(new.var(ddof=1)) / new.size
    share_old = float(old.var(ddof=1)) / old.size
    variance = share_new + share_old
    df = variance**2 / (share_new**2 / (new.size - 1) + share_old**2 / (old.size - 1))
    error = math.sqrt(variance)
    difference = float(result["mean_difference"])
    statistic = difference / error
    half_width = float(stats.t.ppf(0.5 + confidence / 2.0, df)) * error
    p_value = float(2.0 * stats.t.sf(abs(statistic), df))
    result.update(
        {
            "ci_lower": difference - half_width,
            "ci_upper": difference + half_width,
            "df": df,
            "t_statistic": statistic,
            "p_value": p_value,
            "significant": p_value < 1.0 - confidence,
        }
    )
    return result


# ---------------------------------------------------------------------------
# Supplementary Table S10, as data
# ---------------------------------------------------------------------------

#: Where the Welch-test comparisons are tabulated.
TABLE_S10: Final = "Supplementary Table S10 (mmc9.docx)"


@dataclass(frozen=True)
class PublishedWelchComparison(FrozenRecord):
    """One published Welch comparison: a mean difference and its 95% interval.

    ``significant`` transcribes the table's asterisk, which marks ``p > 0.05``:
    it is ``False`` exactly where the supplement prints one. It is what the
    table says, kept even where the printed interval, rounded to one decimal,
    appears to disagree with it.
    """

    #: The flux: ``NEE``, ``H`` or ``LE``.
    target: str
    #: The method tested: ``RFR3`` or ``RFR10``.
    method: str
    #: The method it is tested against: ``MDS`` or ``RFR3``.
    baseline: str
    #: One of the four core metrics.
    metric: str
    #: Mean difference across sites, ``method - baseline``.
    mean_difference: float
    #: Lower end of the published 95% confidence interval.
    ci_lower: float
    #: Upper end of the published 95% confidence interval.
    ci_upper: float
    #: Whether the supplement reports ``p <= 0.05`` (no asterisk).
    significant: bool
    #: The supplementary table the values are transcribed from.
    source: str = TABLE_S10
    #: The site population the test was taken over.
    site_subset: str = SITE_SUBSET_94

    def __post_init__(self) -> None:
        if self.metric not in METRIC_COLUMNS:
            raise SiteReportError(
                f"unknown metric {self.metric!r}; expected one of {', '.join(METRIC_COLUMNS)}"
            )
        if self.ci_lower > self.ci_upper:
            raise SiteReportError(
                f"{self.target}/{self.metric}/{self.method}: published interval runs from "
                f"{self.ci_lower} down to {self.ci_upper}; a transcription has swapped its ends"
            )

    @property
    def units(self) -> str:
        """Units of the difference, as the Table S10 caption gives them."""
        if self.metric not in ("rmse", "bias"):
            return DIMENSIONLESS
        return NEE_BENCHMARK_UNITS if self.target == "NEE" else ENERGY_FLUX_UNITS

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable record of this published comparison."""
        return {
            "target": self.target,
            "method": self.method,
            "baseline": self.baseline,
            "metric": self.metric,
            "mean_difference": self.mean_difference,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
            "significant": self.significant,
            "units": self.units,
            "site_subset": self.site_subset,
            "source": self.source,
        }


def _s10(
    target: str,
    metric: str,
    method: str,
    baseline: str,
    mean_difference: float,
    ci_lower: float,
    ci_upper: float,
    significant: bool,
) -> PublishedWelchComparison:
    """Build one Table S10 cell."""
    return PublishedWelchComparison(
        target=target,
        method=method,
        baseline=baseline,
        metric=metric,
        mean_difference=mean_difference,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        significant=significant,
    )


#: Supplementary Table S10, as transcribed in section 5 of
#: ``docs/supplement_benchmarks.md``: RFR3 against MDS and RFR10 against RFR3,
#: every flux and metric, over the 94-site subset. Intervals printed in the
#: supplement as ``8.17x10-02`` are carried as ``0.0817``.
TABLE_S10_COMPARISONS: Final[tuple[PublishedWelchComparison, ...]] = (
    _s10("NEE", "r2", "RFR3", "MDS", 0.07, 0.0, 0.1, True),
    _s10("NEE", "r2", "RFR10", "RFR3", 0.05, 0.02, 0.09, True),
    _s10("NEE", "slope", "RFR3", "MDS", 0.01, -0.0274, 0.0484, False),
    _s10("NEE", "slope", "RFR10", "RFR3", 0.04, 0.00, 0.08, True),
    _s10("NEE", "bias", "RFR3", "MDS", 0.00, -0.1, 0.1, False),
    _s10("NEE", "bias", "RFR10", "RFR3", 0.03, -0.03, 0.08, False),
    _s10("NEE", "rmse", "RFR3", "MDS", -0.40, -0.9, 0.1, True),
    _s10("NEE", "rmse", "RFR10", "RFR3", -0.24, -0.67, 0.18, False),
    _s10("H", "r2", "RFR3", "MDS", 0.11, 0.0817, 0.146, True),
    _s10("H", "r2", "RFR10", "RFR3", 0.12, 0.10, 0.15, True),
    _s10("H", "slope", "RFR3", "MDS", 0.06, 0.0, 0.1, True),
    _s10("H", "slope", "RFR10", "RFR3", 0.11, 0.09, 0.13, True),
    _s10("H", "bias", "RFR3", "MDS", 1.97, 0.7, 3.2, True),
    _s10("H", "bias", "RFR10", "RFR3", -0.06, -0.81, 0.68, False),
    _s10("H", "rmse", "RFR3", "MDS", -10.55, -15.3, -5.8, True),
    _s10("H", "rmse", "RFR10", "RFR3", -13.05, -16.49, -9.62, True),
    _s10("LE", "r2", "RFR3", "MDS", 0.11, 0.0705, 0.141, True),
    _s10("LE", "r2", "RFR10", "RFR3", 0.10, 0.07, 0.13, True),
    _s10("LE", "slope", "RFR3", "MDS", 0.05, 0.0, 0.1, True),
    _s10("LE", "slope", "RFR10", "RFR3", 0.09, 0.06, 0.12, True),
    _s10("LE", "bias", "RFR3", "MDS", 2.99, 1.6, 4.4, True),
    _s10("LE", "bias", "RFR10", "RFR3", -0.85, -1.48, -0.22, True),
    _s10("LE", "rmse", "RFR3", "MDS", -7.89, -12.7, -3.0, True),
    _s10("LE", "rmse", "RFR10", "RFR3", -8.19, -11.60, -4.78, True),
)

#: Column order of :func:`published_welch_table`.
PUBLISHED_WELCH_TABLE_COLUMNS: Final[tuple[str, ...]] = (
    "target",
    "method",
    "baseline",
    "metric",
    "mean_difference",
    "ci_lower",
    "ci_upper",
    "significant",
    "units",
    "site_subset",
    "source",
)

#: Column order of :func:`compare_to_table_s10`.
S10_COMPARISON_COLUMNS: Final[tuple[str, ...]] = (
    "target",
    "method",
    "baseline",
    "metric",
    "run_mean_difference",
    "published_mean_difference",
    "difference",
    "run_ci_lower",
    "run_ci_upper",
    "published_ci_lower",
    "published_ci_upper",
    "run_significant",
    "published_significant",
    "significance_agrees",
    "run_units",
    "published_units",
    "comparable",
    "note",
    "site_subset",
    "source",
)


def published_welch_table() -> pd.DataFrame:
    """Return :data:`TABLE_S10_COMPARISONS` as a tidy frame."""
    rows = [record.to_dict() for record in TABLE_S10_COMPARISONS]
    table: pd.DataFrame = pd.DataFrame(rows, columns=list(PUBLISHED_WELCH_TABLE_COLUMNS))
    table["significant"] = table["significant"].astype("boolean")
    return table


def compare_to_table_s10(welch: pd.DataFrame) -> pd.DataFrame:
    """Return a reproduction's Welch comparisons beside Supplementary Table S10.

    Only the rows Table S10 describes are compared: every gap class and every
    observation pooled, every stratum pooled, RFR3 against MDS or RFR10 against
    RFR3. As in :func:`~rfrgapfill.benchmarks.compare_to_benchmarks`, a
    difference in units is reported rather than differenced (A10): NEE RMSE and
    bias are ``g C m-2 d-1`` in the table.

    Nothing is asserted. ``difference`` and ``significance_agrees`` say how
    close the reproduction came, for a reader to judge.

    :param welch: a :func:`welch_comparison` frame.
    :returns: a frame with :data:`S10_COMPARISON_COLUMNS`.
    :raises SiteReportError: if the frame is not a Welch table, repeats a cell,
        or has no row Table S10 describes.
    """
    frame = _require_frame(welch, WELCH_TABLE_COLUMNS, what="compare_to_table_s10")
    pooled = cast(
        "pd.DataFrame",
        frame.loc[
            (frame["gap_class"].astype(str) == POOLED_GAP_CLASS)
            & (frame["subset"].astype(str) == "all")
            & (frame["stratum"].astype(str) == POOLED_STRATUM)
        ],
    )
    keys = ["target", "method", "baseline", "metric"]
    repeated = pooled.duplicated(subset=keys, keep=False)
    if bool(repeated.any()):
        raise SiteReportError(
            "the Welch table has more than one pooled row for one target/method/metric - two "
            "unit labels for one flux - so there is no single value to compare"
        )
    merged = pooled.merge(
        published_welch_table(), on=keys, how="inner", suffixes=("_run", "_published")
    )
    if merged.empty:
        raise SiteReportError(
            "no row of this table is in Supplementary Table S10, which compares RFR3 with MDS "
            "and RFR10 with RFR3 over every gap class, every observation and every site"
        )
    rows = [_s10_row(record) for record in merged.to_dict("records")]
    comparison = pd.DataFrame(rows, columns=list(S10_COMPARISON_COLUMNS))
    comparison = _as_float(
        comparison,
        (
            "run_mean_difference",
            "published_mean_difference",
            "difference",
            "run_ci_lower",
            "run_ci_upper",
            "published_ci_lower",
            "published_ci_upper",
        ),
    )
    for column in ("run_significant", "published_significant", "significance_agrees"):
        comparison[column] = pd.array(comparison[column].tolist(), dtype="boolean")
    return _in_reporting_order(comparison, ("target", "method", "metric"))


def _s10_row(record: Mapping[str, Any]) -> dict[str, Any]:
    """Return one reproduction-against-Table-S10 row."""
    metric = str(record["metric"])
    dimensional = metric in ("rmse", "bias")
    run_units = _label(record["units_run"]) if dimensional else DIMENSIONLESS
    published_units = str(record["units_published"])
    comparable = run_units == published_units
    run = _finite(record["mean_difference_run"])
    published = float(record["mean_difference_published"])
    run_significant = (
        None if _is_missing(record["significant_run"]) else bool(record["significant_run"])
    )
    note: str | None = None
    if not comparable:
        note = (
            f"run reports {metric} in {run_units or 'unstated units'}; Table S10 reports "
            f"{published_units} (ambiguity A10)"
        )
    elif run is None:
        note = "the run has no mean difference for this cell"
    return {
        "target": record["target"],
        "method": record["method"],
        "baseline": record["baseline"],
        "metric": metric,
        "run_mean_difference": run,
        "published_mean_difference": published,
        "difference": run - published if comparable and run is not None else None,
        "run_ci_lower": _finite(record["ci_lower_run"]),
        "run_ci_upper": _finite(record["ci_upper_run"]),
        "published_ci_lower": float(record["ci_lower_published"]),
        "published_ci_upper": float(record["ci_upper_published"]),
        "run_significant": run_significant,
        "published_significant": bool(record["significant_published"]),
        "significance_agrees": (
            run_significant == bool(record["significant_published"])
            if comparable and run_significant is not None
            else None
        ),
        "run_units": run_units,
        "published_units": published_units,
        "comparable": comparable,
        "note": note,
        "site_subset": record["site_subset"],
        "source": record["source"],
    }


def _label(value: object) -> str | None:
    """Return a units label, or ``None`` where it is missing."""
    return None if _blank(value) else str(value).strip()


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _require_frame(table: object, columns: Iterable[str], *, what: str) -> pd.DataFrame:
    """Return ``table`` if it is a frame carrying every column ``what`` needs."""
    if not isinstance(table, pd.DataFrame):
        raise SiteReportError(f"{what} needs a DataFrame, got {type(table).__name__}")
    missing = [column for column in columns if column not in table.columns]
    if missing:
        raise SiteReportError(
            f"{what} needs column(s) {', '.join(missing)}; build the table with site_report()"
        )
    return table


def _require_metrics(metrics: Iterable[str]) -> None:
    """Raise unless every metric is one of the four core metrics."""
    unknown = [metric for metric in metrics if metric not in METRIC_COLUMNS]
    if unknown:
        raise SiteReportError(
            f"unknown metric(s) {', '.join(unknown)}; expected {', '.join(METRIC_COLUMNS)}"
        )


def _require_sites(table: pd.DataFrame, *, what: str) -> None:
    """Raise unless every row names its site."""
    unlabelled = int(sum(1 for value in table["site"] if _is_missing(value)))
    if unlabelled:
        raise SiteReportError(
            f"{unlabelled} row(s) carry no site label, so {what} is undefined; label the runs "
            "with a mapping of site label to result"
        )


def _require_unique_cells(table: pd.DataFrame) -> None:
    """Raise if one site contributes one cell twice."""
    repeated = table.duplicated(subset=list(_CELL_KEYS), keep=False)
    if bool(repeated.any()):
        offenders = table.loc[repeated, list(_CELL_KEYS)].drop_duplicates()
        shown = "; ".join(
            " ".join(str(value) for value in row) for row in offenders.head(3).to_numpy()
        )
        raise SiteReportError(
            f"{int(repeated.sum())} row(s) repeat a site/target/method/gap-class/subset cell, "
            f"which would count those sites twice: {shown}" + (" ..." if len(offenders) > 3 else "")
        )


def _stratum_label(value: object) -> str | None:
    """Return a stratum label for one site's value of a metadata field."""
    if _blank(value):
        return None
    if isinstance(value, bool | np.bool_):
        return str(bool(value))
    return str(value).strip()


def _strata(frame: pd.DataFrame, by: str | None) -> pd.DataFrame:
    """Return ``frame`` once pooled, and once per stratum of ``by``."""
    pooled: pd.DataFrame = frame.assign(stratifier=by, stratum=POOLED_STRATUM)
    if by is None:
        return pooled
    labels = [_stratum_label(value) for value in frame[by]]
    if POOLED_STRATUM in labels:
        raise SiteReportError(
            f"a site's {by} is {POOLED_STRATUM!r}, which is the label of the rows pooling every "
            "stratum"
        )
    stratified = frame.assign(stratifier=by, stratum=labels)
    kept = cast("pd.DataFrame", stratified.loc[stratified["stratum"].notna()])
    combined: pd.DataFrame = pd.concat([pooled, kept], ignore_index=True)
    return combined


def _as_float(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Return ``frame`` with ``columns`` float, undefined values missing."""
    typed: pd.DataFrame = frame.copy()
    for column in columns:
        typed[column] = pd.to_numeric(typed[column], errors="coerce").astype("float64")
    return typed


def _as_count(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Return ``frame`` with ``columns`` nullable integers."""
    typed: pd.DataFrame = frame.copy()
    for column in columns:
        typed[column] = pd.to_numeric(typed[column], errors="coerce").astype("Int64")
    return typed


def _in_reporting_order(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Return ``frame`` sorted by ``columns``, labelled columns in published order."""
    if frame.empty:
        return cast("pd.DataFrame", frame.reset_index(drop=True))
    keys = pd.DataFrame(index=frame.index)
    for column in columns:
        values = frame[column]
        if column in _ORDERS:
            keys[f"{column}_rank"] = _rank(values, _ORDERS[column])
        if column == "stratum":
            keys["stratum_rank"] = [0.0 if value == POOLED_STRATUM else 1.0 for value in values]
        keys[column] = ["" if _is_missing(value) else str(value) for value in values]
    order = keys.sort_values(list(keys.columns), kind="stable").index
    ordered: pd.DataFrame = frame.loc[order].reset_index(drop=True)
    return ordered
