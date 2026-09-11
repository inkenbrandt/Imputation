"""FLUXNET2015 adapter: optional helpers for the data product the paper used.

Zhu et al. (2022) ran on FLUXNET2015 FULLSET half-hourly files, so reproducing
their workflow means naming their columns. The core package must not know those
names - every driver is resolved through a
:class:`~rfrgapfill.schema.ColumnMap` (``docs/method_spec.md`` section 2) - so
the coupling lives here and nowhere else. Nothing in this module is required to
use the package; delete it and the science still runs.

What it does:

* :func:`prepare_fluxnet_frame` turns a raw FULLSET frame into one the rest of
  the package accepts: ``TIMESTAMP_START`` parsed into a ``DatetimeIndex``, and
  the ``-9999`` missing-value sentinel turned into ``NaN`` so it is never
  mistaken for a measurement;
* :func:`inspect_fluxnet` reports which canonical variables, fluxes and QC flags
  a frame actually carries, and which are missing, before anything is fitted;
* :meth:`FluxnetAvailability.column_map` builds the
  :class:`~rfrgapfill.schema.ColumnMap` from the columns that are really there;
* :func:`qc_summary` describes a flux QC column against the documented flag
  semantics below.

Nothing here downloads anything. FLUXNET2015 files are obtained by the user
under the data policy they agreed to, and no credential, site list or
site-specific assumption belongs in this package.

FLUXNET2015 QC flag semantics
-----------------------------

For the half-hourly and hourly FULLSET fluxes this package targets
(``NEE_VUT_REF_QC``, ``H_F_MDS_QC``, ``LE_F_MDS_QC``) the flag is an integer
class describing how the value on that row was produced:

===== ==========================================================
Flag  Meaning
===== ==========================================================
0     measured - a genuine observation
1     good-quality gap fill
2     medium-quality gap fill
3     poor-quality gap fill
===== ==========================================================

Only ``0`` is a measurement. The package's single structural use of the flag is
therefore ``flag == 0`` (:data:`OBSERVED_QC_VALUES`, the default of
``RFRConfig.observed_qc_values``); the 1/2/3 grading is reported by
:func:`qc_summary` but never acted on, because treating a "good" gap fill as
truth would train the model on another model's output and score predictions
against it.

Two caveats this module enforces or refuses to guess at:

* **Aggregated files do not use these codes.** In the daily, weekly, monthly and
  yearly FLUXNET2015 products the ``_QC`` column is the *fraction* of
  measured-or-good-quality records in the aggregation window, a float in
  ``[0, 1]``. :func:`qc_summary` raises rather than silently reading such a
  column as class codes. Use the half-hourly or hourly product.
* **Driver QC flags are not interpreted.** The paper used FLUXNET's pre-filled
  meteorological drivers as given (method_spec.md section 7), and the ``_F`` /
  ``_F_MDS`` driver flags encode a different provenance scheme (including
  ERA-downscaled values). This module reports them if asked and leaves the
  decision to the user.

``docs/fluxnet_adapter.md`` records these semantics and the four conventions
(``F1``-``F4``) this module adopts about the data product - timestamp labelling,
depth-indexed soil columns, the NEE variant, and the missing-value sentinel.
They are choices about the file format, not about the method, which is why they
carry their own identifiers and never appear in the paper's ``A1``-``A10``
ambiguity table.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

import numpy as np
import pandas as pd

from rfrgapfill.schema import (
    FLUXNET2015_COLUMNS,
    RFR10_DRIVERS,
    SOIL_TEMPERATURE,
    SOIL_WATER_CONTENT,
    ColumnMap,
    ColumnMapError,
    Mode,
)
from rfrgapfill.time import TIMESTAMP_INDEX_NAME

__all__ = [
    "DEPTH_INDEXED_VARIABLES",
    "FLUXNET2015_FLUXES",
    "FLUXNET2015_FLUX_QC",
    "FLUXNET2015_MISSING_VALUE",
    "FLUXNET2015_QC_FLAGS",
    "FLUXNET2015_TIMESTAMP_END",
    "FLUXNET2015_TIMESTAMP_FORMAT",
    "FLUXNET2015_TIMESTAMP_START",
    "OBSERVED_QC_VALUES",
    "QC_MEASURED",
    "FluxnetAvailability",
    "FluxnetError",
    "QCSummary",
    "fluxnet_column_map",
    "inspect_fluxnet",
    "prepare_fluxnet_frame",
    "qc_summary",
    "read_fluxnet_csv",
]


class FluxnetError(ValueError):
    """Raised when a frame does not look like the FLUXNET2015 product it claims to be."""


# ---------------------------------------------------------------------------
# Reference names
# ---------------------------------------------------------------------------

#: Interval-start timestamp column of a FULLSET file, ``YYYYMMDDHHMM``.
FLUXNET2015_TIMESTAMP_START: Final = "TIMESTAMP_START"
#: Interval-end timestamp column of a FULLSET file, ``YYYYMMDDHHMM``.
FLUXNET2015_TIMESTAMP_END: Final = "TIMESTAMP_END"
#: ``strftime`` pattern of both timestamp columns.
FLUXNET2015_TIMESTAMP_FORMAT: Final = "%Y%m%d%H%M"

#: The value FLUXNET2015 writes for missing data in every numeric column.
FLUXNET2015_MISSING_VALUE: Final = -9999.0

#: Target flux columns of the paper: short name -> FLUXNET2015 column.
#: ``NEE_VUT_REF`` is the variable-u*-threshold reference NEE, the series Zhu et
#: al. validated; the CUT variants are a documented alternative the user may
#: pass through ``overrides``.
FLUXNET2015_FLUXES: Final[Mapping[str, str]] = MappingProxyType(
    {"NEE": "NEE_VUT_REF", "H": "H_F_MDS", "LE": "LE_F_MDS"}
)

#: Quality flag of each target flux: short name -> FLUXNET2015 column.
FLUXNET2015_FLUX_QC: Final[Mapping[str, str]] = MappingProxyType(
    {"NEE": "NEE_VUT_REF_QC", "H": "H_F_MDS_QC", "LE": "LE_F_MDS_QC"}
)

#: Meaning of each half-hourly/hourly flux QC class (see the module docstring).
FLUXNET2015_QC_FLAGS: Final[Mapping[int, str]] = MappingProxyType(
    {
        0: "measured",
        1: "good-quality gap fill",
        2: "medium-quality gap fill",
        3: "poor-quality gap fill",
    }
)

#: The only flag that is a measurement. Matches ``RFRConfig.observed_qc_values``.
QC_MEASURED: Final = 0

#: Flags counted as genuine observations by default.
OBSERVED_QC_VALUES: Final[tuple[int, ...]] = (QC_MEASURED,)

#: Canonical variables whose FLUXNET column may carry a depth index (``_1`` is
#: the shallowest sensor). Resolved by :func:`inspect_fluxnet`; see
#: ``docs/fluxnet_adapter.md``.
DEPTH_INDEXED_VARIABLES: Final[tuple[str, ...]] = (SOIL_TEMPERATURE, SOIL_WATER_CONTENT)

#: Highest depth index scanned when resolving a depth-indexed variable.
_MAX_DEPTH_INDEX: Final = 10


def fluxnet_column_map(
    mode: Mode | str | None = None,
    *,
    overrides: Mapping[str, str] | None = None,
    timestamp: str | None = FLUXNET2015_TIMESTAMP_START,
) -> ColumnMap:
    """Return the reference FLUXNET2015 mapping, without looking at any data.

    A convenience wrapper over :meth:`~rfrgapfill.schema.ColumnMap.fluxnet2015`
    that additionally names the timestamp column, for files known to use the
    standard names. Prefer :func:`inspect_fluxnet` when the file is unseen: it
    resolves depth-indexed columns and reports what is missing instead of
    failing later inside a transformer.
    """
    return ColumnMap.fluxnet2015(mode, overrides=overrides, timestamp=timestamp)


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------


def prepare_fluxnet_frame(
    data: pd.DataFrame,
    *,
    timestamp: str | None = FLUXNET2015_TIMESTAMP_START,
    sentinel: float | None = FLUXNET2015_MISSING_VALUE,
    drop_timestamp_columns: bool = True,
) -> pd.DataFrame:
    """Return ``data`` with FLUXNET timestamps parsed and ``-9999`` turned into ``NaN``.

    Two things stand between a raw FULLSET frame and the package API, and both
    are silent failures if skipped: the timestamps are ``YYYYMMDDHHMM`` integers
    rather than datetimes, and missing data is the sentinel ``-9999``, which is
    a perfectly plausible-looking number that would otherwise be trained on.

    ``timestamp`` names the column to index by - ``TIMESTAMP_START`` by default,
    which labels each record by the start of its averaging interval; pass
    ``TIMESTAMP_END`` for the other convention or ``None`` to keep an index the
    frame already has. ``sentinel=None`` disables the replacement for files that
    already use ``NaN``.

    The input frame is never mutated.
    """
    if not isinstance(data, pd.DataFrame):
        raise FluxnetError(f"data must be a pandas DataFrame, got {type(data).__name__}")

    frame: pd.DataFrame = data.copy()

    if timestamp is not None:
        if timestamp not in frame.columns:
            raise FluxnetError(
                f"timestamp column {timestamp!r} is not in the data. A FLUXNET2015 FULLSET "
                f"file carries {FLUXNET2015_TIMESTAMP_START!r} and {FLUXNET2015_TIMESTAMP_END!r}; "
                "pass timestamp=None to use an index the frame already has."
            )
        index = _parse_fluxnet_timestamps(frame[timestamp], column=timestamp)
        frame = frame.drop(columns=[timestamp]) if drop_timestamp_columns else frame
        frame.index = index
        # The index carries the stamps, not the convention: which column they came
        # from is the caller's choice and is recorded in the run manifest, not in
        # an index name the rest of the package would have to special-case.
        frame.index.name = TIMESTAMP_INDEX_NAME
    if frame.index.name is None:
        frame.index.name = TIMESTAMP_INDEX_NAME

    if drop_timestamp_columns:
        leftovers = [
            column
            for column in (FLUXNET2015_TIMESTAMP_START, FLUXNET2015_TIMESTAMP_END)
            if column in frame.columns
        ]
        if leftovers and timestamp is not None:
            frame = frame.drop(columns=leftovers)

    if sentinel is not None:
        frame = _replace_sentinel(frame, sentinel)
    return frame


def read_fluxnet_csv(
    path: str | Path,
    *,
    timestamp: str | None = FLUXNET2015_TIMESTAMP_START,
    sentinel: float | None = FLUXNET2015_MISSING_VALUE,
    **read_csv_kwargs: Any,
) -> pd.DataFrame:
    """Read a local FLUXNET2015 CSV and return it ready for the package API.

    ``pd.read_csv`` followed by :func:`prepare_fluxnet_frame`. The file is a
    local path the user already holds: this package neither downloads FLUXNET
    data nor stores credentials for doing so.

    Numbers are parsed with ``float_precision="round_trip"`` unless the caller
    says otherwise: pandas' default parser can be one unit in the last place off,
    which is enough to change a fitted forest, and the command line reads files
    the same way.
    """
    read_csv_kwargs.setdefault("float_precision", "round_trip")
    frame = pd.read_csv(path, **read_csv_kwargs)
    return prepare_fluxnet_frame(frame, timestamp=timestamp, sentinel=sentinel)


def _parse_fluxnet_timestamps(values: pd.Series, *, column: str) -> pd.DatetimeIndex:
    """Return ``values`` as a :class:`~pandas.DatetimeIndex`, or raise :class:`FluxnetError`."""
    if pd.api.types.is_datetime64_any_dtype(values):
        existing: pd.DatetimeIndex = pd.DatetimeIndex(values)
        return existing
    text = values.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
    if text.isna().any():
        raise FluxnetError(f"timestamp column {column!r} contains missing values")
    try:
        parsed = pd.to_datetime(text, format=FLUXNET2015_TIMESTAMP_FORMAT, errors="raise")
    except (ValueError, TypeError) as error:
        raise FluxnetError(
            f"timestamp column {column!r} is not in the FLUXNET2015 "
            f"{FLUXNET2015_TIMESTAMP_FORMAT} form (e.g. 200401010030): {error}"
        ) from error
    index: pd.DatetimeIndex = pd.DatetimeIndex(parsed)
    return index


def _replace_sentinel(frame: pd.DataFrame, sentinel: float) -> pd.DataFrame:
    """Return ``frame`` with exact ``sentinel`` values in numeric columns set to ``NaN``.

    Only columns that actually contain the sentinel are touched, so a QC column
    of clean integer flags stays an integer column instead of being widened to
    float for a substitution it never needed.
    """
    numeric = frame.select_dtypes(include="number").columns
    affected = [column for column in numeric if bool((frame[column] == sentinel).any())]
    if affected:
        block = frame[affected].astype(float)
        frame[affected] = block.mask(block == sentinel)
    return frame


# ---------------------------------------------------------------------------
# QC flags
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QCSummary:
    """What one flux QC column says about how its flux was produced.

    Reported, not enforced: the package acts on ``flag == 0`` alone, and this is
    how a user sees the price of that rule before paying it - a site whose
    ``n_measured`` is a small share of ``n_rows`` has little to train on, which
    the aggregate row count would have hidden.
    """

    #: The QC column described.
    column: str
    #: The flux column it describes, when one was given.
    target: str | None
    #: Rows in the frame.
    n_rows: int
    #: Count of each flag value present, ordered by flag.
    counts: Mapping[int, int]
    #: Rows whose flag is absent or not an integer class.
    n_unflagged: int
    #: Flag values counted as measurements.
    observed_qc_values: tuple[int, ...]
    #: Rows counted as genuine measurements (flag accepted, and value finite when
    #: a target column was given).
    n_measured: int
    #: Rows carrying a value that arrived already gap-filled.
    n_prefilled: int
    #: Rows where the target itself is missing, when a target column was given.
    n_missing_value: int

    @property
    def measured_fraction(self) -> float:
        """Share of rows that are genuine measurements, in ``[0, 1]``."""
        return 0.0 if self.n_rows == 0 else self.n_measured / self.n_rows

    def describe(self) -> str:
        """Return a human-readable breakdown naming each flag's documented meaning."""
        lines = [f"{self.column}: {self.n_rows} rows, {self.measured_fraction:.1%} measured"]
        for flag, count in self.counts.items():
            meaning = FLUXNET2015_QC_FLAGS.get(flag, "undocumented flag value")
            share = 0.0 if self.n_rows == 0 else count / self.n_rows
            lines.append(f"  flag {flag}: {count} ({share:.1%}) - {meaning}")
        if self.n_unflagged:
            lines.append(f"  unflagged: {self.n_unflagged}")
        if self.target is not None:
            lines.append(f"  {self.target} missing: {self.n_missing_value}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation for the run manifest."""
        return {
            "column": self.column,
            "target": self.target,
            "n_rows": self.n_rows,
            "counts": {str(flag): count for flag, count in self.counts.items()},
            "flag_meanings": {
                str(flag): FLUXNET2015_QC_FLAGS.get(flag, "undocumented flag value")
                for flag in self.counts
            },
            "n_unflagged": self.n_unflagged,
            "observed_qc_values": list(self.observed_qc_values),
            "n_measured": self.n_measured,
            "n_prefilled": self.n_prefilled,
            "n_missing_value": self.n_missing_value,
            "measured_fraction": self.measured_fraction,
        }


def qc_summary(
    data: pd.DataFrame,
    qc_column: str,
    *,
    target: str | None = None,
    observed_qc_values: Sequence[int] = OBSERVED_QC_VALUES,
) -> QCSummary:
    """Describe ``qc_column`` against the documented FLUXNET2015 flag semantics.

    Raises :class:`FluxnetError` when the column does not hold integer class
    codes: in the daily and coarser FLUXNET2015 products ``_QC`` is the fraction
    of measured-or-good-quality records in the window, and reading that as a
    class code would count fractions as "measured" essentially at random.

    The mask itself comes from
    :func:`rfrgapfill.provenance.observed_mask`, which every fit and validation
    already routes through; this function only explains what that mask will do.
    """
    if qc_column not in data.columns:
        raise FluxnetError(f"QC column {qc_column!r} is not in the data")
    if target is not None and target not in data.columns:
        raise FluxnetError(f"flux column {target!r} is not in the data")

    flags = pd.to_numeric(data[qc_column], errors="coerce")
    values = flags.to_numpy(dtype=float)
    finite = np.isfinite(values)
    fractional = finite & (np.mod(values, 1.0) != 0.0)
    if fractional.any():
        example = float(values[fractional][0])
        raise FluxnetError(
            f"QC column {qc_column!r} holds non-integer values (e.g. {example}). In the daily "
            "and coarser FLUXNET2015 products the QC column is the fraction of "
            "measured-or-good-quality records, not a flag class; use the half-hourly or hourly "
            "product, or interpret the fraction yourself."
        )

    accepted = tuple(dict.fromkeys(int(flag) for flag in observed_qc_values))
    counts = {
        int(flag): int(count)
        for flag, count in sorted(pd.Series(values[finite]).value_counts().items())
    }
    is_accepted = finite & np.isin(values, np.asarray(accepted, dtype=float))

    if target is None:
        value_present = np.ones(len(data), dtype=bool)
        n_missing_value = 0
    else:
        value_present = np.isfinite(
            pd.to_numeric(data[target], errors="coerce").to_numpy(dtype=float)
        )
        n_missing_value = int((~value_present).sum())

    return QCSummary(
        column=qc_column,
        target=target,
        n_rows=len(data),
        counts=MappingProxyType(counts),
        n_unflagged=int((~finite).sum()),
        observed_qc_values=accepted,
        n_measured=int((is_accepted & value_present).sum()),
        n_prefilled=int((finite & ~is_accepted & value_present).sum()),
        n_missing_value=n_missing_value,
    )


# ---------------------------------------------------------------------------
# Availability reporting
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FluxnetAvailability:
    """Which FLUXNET2015 variables a frame carries, and which it does not.

    The answer to "can this site run RFR10?" - which the supplements say is not
    a given: the paper's full RFR3/RFR10 comparison used 94 of 194 sites,
    because complete QC and RFR10 driver availability were more restrictive
    (``docs/supplement_benchmarks.md``). Missing variables are reported here,
    before a fit, rather than discovered as a column error later.
    """

    #: Canonical driver name -> the column found for it.
    drivers: Mapping[str, str]
    #: Canonical driver name -> the reference column name that is absent.
    missing_drivers: Mapping[str, str]
    #: Flux short name (``NEE``/``H``/``LE``) -> the column found for it.
    fluxes: Mapping[str, str]
    #: Flux short name -> the reference column name that is absent.
    missing_fluxes: Mapping[str, str]
    #: Flux short name -> the QC column found for it.
    qc: Mapping[str, str]
    #: Flux short name -> the reference QC column name that is absent.
    missing_qc: Mapping[str, str]
    #: The FLUXNET timestamp column found, if the frame still carries one.
    timestamp: str | None = None

    def __post_init__(self) -> None:
        for name in ("drivers", "missing_drivers", "fluxes", "missing_fluxes", "qc", "missing_qc"):
            object.__setattr__(self, name, MappingProxyType(dict(getattr(self, name))))

    # -- readiness -----------------------------------------------------------

    def supports(self, mode: Mode | str) -> bool:
        """Whether every driver of ``mode`` was found."""
        return not self.missing(mode)

    def missing(self, mode: Mode | str) -> tuple[str, ...]:
        """Return the canonical drivers ``mode`` needs that were not found."""
        return tuple(name for name in Mode.coerce(mode).drivers if name not in self.drivers)

    @property
    def rfr3_ready(self) -> bool:
        """Whether all three RFR3 drivers were found."""
        return self.supports(Mode.RFR3)

    @property
    def rfr10_ready(self) -> bool:
        """Whether all ten RFR10 drivers were found."""
        return self.supports(Mode.RFR10)

    @property
    def best_mode(self) -> Mode | None:
        """The most complete mode this frame supports, or ``None`` if not even RFR3."""
        if self.rfr10_ready:
            return Mode.RFR10
        if self.rfr3_ready:
            return Mode.RFR3
        return None

    # -- handing over to the generic API -------------------------------------

    def column_map(
        self,
        mode: Mode | str | None = None,
        *,
        overrides: Mapping[str, str] | None = None,
        timestamp: str | None = None,
    ) -> ColumnMap:
        """Return the :class:`~rfrgapfill.schema.ColumnMap` for ``mode``, from real columns.

        Built from the columns actually found, so a depth-indexed
        ``SWC_F_MDS_1`` is mapped as ``soil_water_content`` without the caller
        editing anything. ``mode=None`` uses :attr:`best_mode`.

        Raises :class:`~rfrgapfill.schema.ColumnMapError` naming every canonical
        variable the frame does not carry.
        """
        if mode is None:
            resolved = self.best_mode
            if resolved is None:
                raise ColumnMapError(
                    "this frame does not carry the RFR3 drivers, so no mode can run: missing "
                    f"{', '.join(self.missing(Mode.RFR3))}. Expected FLUXNET2015 column(s): "
                    f"{', '.join(self.missing_drivers[name] for name in self.missing(Mode.RFR3))}"
                )
        else:
            resolved = Mode.coerce(mode)
        absent = self.missing(resolved)
        if absent:
            expected = ", ".join(self.missing_drivers[name] for name in absent)
            raise ColumnMapError(
                f"missing column mapping for mode={resolved.value}: {', '.join(absent)}. "
                f"Expected FLUXNET2015 column(s): {expected}. Map them with overrides, or use "
                f"mode={Mode.RFR3.value} if the extended drivers are unavailable at this site."
            )
        variables = {name: self.drivers[name] for name in resolved.drivers}
        if overrides:
            variables.update(overrides)
        return ColumnMap(
            variables=variables,
            timestamp=self.timestamp if timestamp is None else timestamp,
        )

    def target_columns(self, fluxes: Iterable[str] | None = None) -> tuple[str, ...]:
        """Return the flux column names found, for ``validate_rfr(targets=...)``."""
        return tuple(self.fluxes[name] for name in self._flux_names(fluxes))

    def qc_columns(self, fluxes: Iterable[str] | None = None) -> dict[str, str | None]:
        """Return flux column -> QC column, for ``validate_rfr(qc_columns=...)``.

        A flux whose QC column is absent maps to ``None``, which the package
        reads as "every finite value is a measurement" - acceptable only for a
        series the user knows is already cleaned.
        """
        return {self.fluxes[name]: self.qc.get(name) for name in self._flux_names(fluxes)}

    def heat_targets(self) -> tuple[str, str] | None:
        """Return the (H, LE) column pair for ``validate_rfr(energy_balance_targets=...)``.

        ``None`` when either flux is absent. The energy-balance ratio defaults to
        targets literally named ``H`` and ``LE`` (method_spec.md section 6.4);
        under FLUXNET naming they are ``H_F_MDS`` and ``LE_F_MDS``, so without
        this pair the ratio is quietly not computed.
        """
        if "H" in self.fluxes and "LE" in self.fluxes:
            return self.fluxes["H"], self.fluxes["LE"]
        return None

    def _flux_names(self, fluxes: Iterable[str] | None) -> tuple[str, ...]:
        if fluxes is None:
            return tuple(self.fluxes)
        requested = tuple(fluxes)
        unknown = [name for name in requested if name not in self.fluxes]
        if unknown:
            raise FluxnetError(
                f"flux column(s) not found in the data: {', '.join(sorted(unknown))}. "
                f"Found: {', '.join(self.fluxes) or 'none'}"
            )
        return requested

    def require(
        self,
        mode: Mode | str | None = None,
        *,
        fluxes: Iterable[str] | None = None,
        require_qc: bool = True,
    ) -> None:
        """Raise :class:`FluxnetError` listing everything a run of ``mode`` would lack.

        With ``require_qc`` left on, a flux without its QC column is a failure:
        on a raw FLUXNET file, running without it would treat MDS-filled values
        as measurements, hide them inside artificial gaps, and then score
        predictions against them (method_spec.md section 7).
        """
        problems: list[str] = []
        absent = self.missing(Mode.RFR3 if mode is None else mode)
        if absent:
            expected = ", ".join(self.missing_drivers[name] for name in absent)
            problems.append(f"missing driver(s): {', '.join(absent)} (expected {expected})")
        names = tuple(self.fluxes) if fluxes is None else tuple(fluxes)
        for name in names:
            if name not in self.fluxes:
                expected = self.missing_fluxes.get(name, FLUXNET2015_FLUXES.get(name, name))
                problems.append(f"missing flux {name} (expected {expected})")
            elif require_qc and name not in self.qc:
                expected = self.missing_qc.get(name, FLUXNET2015_FLUX_QC.get(name, ""))
                problems.append(f"missing QC flag for {name} (expected {expected})")
        if problems:
            raise FluxnetError("this frame cannot run as requested: " + "; ".join(problems))

    # -- reporting -----------------------------------------------------------

    def summary(self) -> str:
        """Return a short human-readable report of what was found and what was not."""
        mode = self.best_mode
        headline = "no mode supported" if mode is None else f"{mode.value} supported"
        lines = [f"FLUXNET2015 columns: {len(self.drivers)}/10 drivers, {headline}"]
        if self.drivers:
            lines.append(
                "  drivers: "
                + ", ".join(f"{name}={column}" for name, column in self.drivers.items())
            )
        if self.missing_drivers:
            lines.append(
                "  missing drivers: "
                + ", ".join(f"{name} ({column})" for name, column in self.missing_drivers.items())
            )
        if self.fluxes:
            lines.append(
                "  fluxes: "
                + ", ".join(
                    f"{column} (QC {self.qc.get(name, 'absent')})"
                    for name, column in self.fluxes.items()
                )
            )
        if self.missing_fluxes:
            lines.append("  missing fluxes: " + ", ".join(self.missing_fluxes.values()))
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation for the run manifest."""
        return {
            "drivers": dict(self.drivers),
            "missing_drivers": dict(self.missing_drivers),
            "fluxes": dict(self.fluxes),
            "missing_fluxes": dict(self.missing_fluxes),
            "qc": dict(self.qc),
            "missing_qc": dict(self.missing_qc),
            "timestamp": self.timestamp,
            "rfr3_ready": self.rfr3_ready,
            "rfr10_ready": self.rfr10_ready,
            "best_mode": None if self.best_mode is None else self.best_mode.value,
        }


def inspect_fluxnet(
    data: pd.DataFrame | Iterable[str],
    *,
    fluxes: Mapping[str, str] | None = None,
    overrides: Mapping[str, str] | None = None,
) -> FluxnetAvailability:
    """Report which FLUXNET2015 variables ``data`` carries.

    ``data`` is a DataFrame or any iterable of column names. Each canonical
    driver is looked for under its reference name and, for the depth-indexed
    soil variables, under ``NAME_1`` ... ``NAME_10``, the shallowest sensor
    present winning (``docs/fluxnet_adapter.md``).

    ``fluxes`` replaces the default ``NEE``/``H``/``LE`` target names - a CUT
    NEE variant, say - and ``overrides`` pins a canonical driver to a specific
    column, skipping the search.

    >>> info = inspect_fluxnet(df)                               # doctest: +SKIP
    >>> report = validate_rfr(                                   # doctest: +SKIP
    ...     df,
    ...     targets=list(info.target_columns()),
    ...     mode=info.best_mode,
    ...     column_map=info.column_map(),
    ...     qc_columns=info.qc_columns(),
    ...     latitude=51.5,
    ... )
    """
    if isinstance(data, pd.DataFrame):
        available = {str(column) for column in data.columns}
        index_is_time = pd.api.types.is_datetime64_any_dtype(data.index)
    else:
        available = {str(column) for column in data}
        index_is_time = False

    pinned = dict(overrides or {})
    found: dict[str, str] = {}
    absent: dict[str, str] = {}
    for name in RFR10_DRIVERS:
        reference = pinned.get(name, FLUXNET2015_COLUMNS[name])
        resolved = _resolve_column(name, reference, available)
        if resolved is None:
            absent[name] = reference
        else:
            found[name] = resolved

    flux_names = dict(fluxes) if fluxes is not None else dict(FLUXNET2015_FLUXES)
    flux_found: dict[str, str] = {}
    flux_absent: dict[str, str] = {}
    qc_found: dict[str, str] = {}
    qc_absent: dict[str, str] = {}
    for name, column in flux_names.items():
        if column in available:
            flux_found[name] = column
        else:
            flux_absent[name] = column
        qc_column = FLUXNET2015_FLUX_QC.get(name, f"{column}_QC")
        if qc_column in available:
            qc_found[name] = qc_column
        else:
            qc_absent[name] = qc_column

    timestamp: str | None = None
    if not index_is_time:
        for column in (FLUXNET2015_TIMESTAMP_START, FLUXNET2015_TIMESTAMP_END):
            if column in available:
                timestamp = column
                break

    return FluxnetAvailability(
        drivers=found,
        missing_drivers=absent,
        fluxes=flux_found,
        missing_fluxes=flux_absent,
        qc=qc_found,
        missing_qc=qc_absent,
        timestamp=timestamp,
    )


def _resolve_column(canonical: str, reference: str, available: set[str]) -> str | None:
    """Return the column holding ``canonical``, allowing a depth index, or ``None``."""
    if reference in available:
        return reference
    if canonical in DEPTH_INDEXED_VARIABLES:
        for depth in range(1, _MAX_DEPTH_INDEX + 1):
            candidate = f"{reference}_{depth}"
            if candidate in available:
                return candidate
    return None
