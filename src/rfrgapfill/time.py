"""Timestamp handling, cadence inference and elapsed-time utilities.

Every temporal quantity in this package is an elapsed-time quantity. Gap
durations, the ``time_distance_hours`` feature and interval selection are all
computed from timestamps, never from row positions: one day is 48 rows only at
30-minute cadence, and only when no rows are missing (``docs/method_spec.md``
sections 3.2, 4.1 and 7).

The module owns three responsibilities:

* **Validation** - resolve a timestamp column or ``DatetimeIndex`` into a
  checked :class:`pandas.DatetimeIndex`, sort it, and apply a documented
  duplicate policy (method_spec.md section 7 requires monotonic, unique
  timestamps unless a preprocessing option resolves duplicates).
* **Cadence** - infer or accept the time step and report the series against the
  regular grid it implies, separating *missing* grid points from *off-grid*
  timestamps instead of silently smoothing either away.
* **Arithmetic** - elapsed hours since a fixed origin, duration/row conversion,
  and half-open interval selection, which the gap generator and the receptive
  limiter both build on.

It sits between :mod:`rfrgapfill.schema` (vocabulary) and
:mod:`rfrgapfill.config` (validated configuration): configuration objects parse
their durations with :func:`to_timedelta` from here, so there is one duration
parser in the package.

Note: this module shadows the stdlib ``time`` only inside the package
namespace; absolute imports elsewhere are unaffected.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import timedelta
from enum import Enum
from typing import Any, Final, Literal

import numpy as np
import pandas as pd

from rfrgapfill.schema import ConfigError, coerce_enum

__all__ = [
    "TIME_DISTANCE_HOURS",
    "DuplicatePolicy",
    "IntervalClosed",
    "StepSource",
    "TimeAxis",
    "TimestampError",
    "as_datetime_index",
    "duplicate_timestamps",
    "duration_to_periods",
    "elapsed_hours",
    "infer_time_step",
    "interval_bounds",
    "interval_mask",
    "iso_duration",
    "periods_to_duration",
    "prepare_time_index",
    "resolve_duplicates",
    "resolve_time_step",
    "set_time_index",
    "sort_by_time",
    "to_timedelta",
]


class TimestampError(ValueError):
    """Raised when input timestamps cannot support a temporal operation.

    Distinct from :class:`~rfrgapfill.schema.ConfigError`, which reports an
    invalid *configuration*: a :class:`TimestampError` reports invalid or
    insufficient *data* - unparseable stamps, unsorted or duplicated stamps, a
    cadence that cannot be inferred, or a duration that does not fit the
    cadence. Both subclass :class:`ValueError`.
    """


#: Name of the elapsed-hours receptive-limiter feature (method_spec.md 3.2).
TIME_DISTANCE_HOURS: Final = "time_distance_hours"

#: Default name given to the index built from a timestamp column.
TIMESTAMP_INDEX_NAME: Final = "timestamp"

#: Fewest timestamps a cadence can be inferred from.
_MIN_TIMESTAMPS_FOR_INFERENCE: Final = 2

#: Guards against a declared cadence far finer than the data, which would
#: otherwise materialise an enormous expected grid before failing.
_MAX_EXPECTED_GRID: Final = 50_000_000

#: A bare ``d`` day unit, which pandas >= 3 deprecates in favour of ``D``.
_DAY_UNIT_RE: Final[re.Pattern[str]] = re.compile(r"(?<=[0-9 ])d(?![a-zA-Z])")

_ONE_HOUR: Final = pd.Timedelta(hours=1)
_ZERO: Final = pd.Timedelta(0)

#: How a time step was arrived at, recorded in the run manifest.
StepSource = Literal["declared", "inferred"]

#: Endpoint convention for interval selection, as in :meth:`pandas.Series.between`.
IntervalClosed = Literal["left", "right", "both", "neither"]


class DuplicatePolicy(str, Enum):
    """What to do with repeated timestamps (method_spec.md section 7).

    Duplicate timestamps are an error by default: half-hourly flux data has one
    row per stamp, and silently collapsing repeats would change which values a
    day's statistics are computed from. ``keep_first`` and ``keep_last`` are the
    documented preprocessing options; neither averages the duplicated rows,
    because combining flux observations is a scientific choice this package does
    not make on the user's behalf.
    """

    ERROR = "error"
    KEEP_FIRST = "keep_first"
    KEEP_LAST = "keep_last"

    @classmethod
    def coerce(cls, value: object) -> DuplicatePolicy:
        """Return ``value`` as a :class:`DuplicatePolicy`, accepting short spellings."""
        aliases = {"raise": cls.ERROR, "first": cls.KEEP_FIRST, "last": cls.KEEP_LAST}
        return coerce_enum(cls, value, field_name="on_duplicates", aliases=aliases)


# ---------------------------------------------------------------------------
# Duration parsing
# ---------------------------------------------------------------------------


def to_timedelta(value: object, *, field_name: str = "duration") -> timedelta:
    """Return ``value`` as a positive :class:`datetime.timedelta`.

    Accepts anything ``pandas.Timedelta`` understands (``"30min"``, ``"24h"``,
    ``"7d"``, ``"30d"``, ``timedelta(hours=24)``), so durations and cadences can
    be written the way users already write them for pandas. Bare numbers are
    rejected: a unitless ``24`` is ambiguous between rows, hours and days, and
    row counts are never a valid duration here.

    Raises :class:`~rfrgapfill.schema.ConfigError`, because a duration always
    arrives from configuration or from a caller's argument rather than from data.
    """
    if isinstance(value, (bool, int, float)):
        raise ConfigError(
            f"{field_name} must be a duration string or timedelta (e.g. '30min'), "
            f"not a bare number: {value!r}"
        )
    if isinstance(value, str):
        # pandas >= 3 deprecates the lowercase day unit, but '7d'/'30d' is how the
        # paper's gap classes are written; canonicalise rather than warn the user.
        candidate: str | timedelta = _DAY_UNIT_RE.sub("D", value)
    elif isinstance(value, timedelta):
        candidate = value
    else:
        raise ConfigError(
            f"{field_name} must be a duration string or timedelta (e.g. '30min'), "
            f"got {type(value).__name__}"
        )
    try:
        delta = pd.Timedelta(candidate)
    except (ValueError, TypeError) as exc:
        raise ConfigError(f"{field_name}={value!r} is not a valid duration: {exc}") from exc
    if delta != delta:  # NaT
        raise ConfigError(f"{field_name}={value!r} is not a valid duration")
    if delta <= _ZERO:
        raise ConfigError(f"{field_name} must be a positive duration, got {value!r}")
    result: timedelta = delta.to_pytimedelta()
    return result


def iso_duration(delta: timedelta) -> str:
    """Return an ISO-8601 duration string for manifests."""
    return str(pd.Timedelta(delta).isoformat())


# ---------------------------------------------------------------------------
# Timestamp validation
# ---------------------------------------------------------------------------


def as_datetime_index(
    values: object,
    *,
    field_name: str = "timestamps",
) -> pd.DatetimeIndex:
    """Return ``values`` as a validated :class:`pandas.DatetimeIndex`.

    Accepts a ``DatetimeIndex``, an index or series of parseable date strings, or
    any sequence of datetimes. Numeric values are refused rather than parsed:
    ``pandas`` would read them as epoch nanoseconds, which is exactly the
    row-position-as-time confusion this module exists to prevent.

    Missing (``NaT``) stamps and empty inputs raise, as does a mixture of UTC
    offsets, which ``pandas`` cannot represent as a single ``DatetimeIndex``.
    """
    if isinstance(values, pd.DatetimeIndex):
        index = values
    else:
        if isinstance(values, pd.Index):
            raw: pd.Index = values
        elif isinstance(values, pd.Series):
            raw = pd.Index(values.to_numpy())
        elif isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
            raw = pd.Index(list(values))
        else:
            raise TimestampError(
                f"{field_name} must be a DatetimeIndex, Series, Index or sequence of "
                f"timestamps, got {type(values).__name__}"
            )
        if raw.dtype.kind in "iufb":
            raise TimestampError(
                f"{field_name} has numeric dtype {raw.dtype}; numbers are ambiguous as "
                "timestamps (pandas would read them as epoch nanoseconds). Parse them "
                "first, e.g. with pandas.to_datetime(..., unit=...), or name a parsed "
                "timestamp column."
            )
        try:
            converted = pd.to_datetime(raw)
        except (ValueError, TypeError, OverflowError) as exc:
            raise TimestampError(f"{field_name} could not be parsed as timestamps: {exc}") from exc
        if not isinstance(converted, pd.DatetimeIndex):
            raise TimestampError(
                f"{field_name} did not parse to a single datetime dtype (got "
                f"{converted.dtype}); mixed UTC offsets must be converted to one "
                "timezone first."
            )
        index = converted

    if len(index) == 0:
        raise TimestampError(f"{field_name} is empty; there is no time axis to work with")
    if index.hasnans:
        n_missing = int(np.count_nonzero(index.isna()))
        raise TimestampError(
            f"{field_name} contains {n_missing} missing timestamp(s). Rows without a "
            "timestamp cannot be placed on the time axis; drop or repair them first."
        )
    return index


def set_time_index(
    data: pd.DataFrame,
    *,
    timestamp: str | None = None,
    drop: bool = True,
) -> pd.DataFrame:
    """Return a copy of ``data`` indexed by validated timestamps.

    ``timestamp`` names the column holding the stamps; when it is ``None`` the
    existing index is used and must already be datetime-like (or parseable
    strings). Passing ``ColumnMap.timestamp`` is the intended call for frames
    that carry a timestamp column.

    The timestamp column is dropped by default: its values live on in the index,
    and leaving a duplicate copy in the frame invites it being mistaken for a
    driver later.
    """
    if not isinstance(data, pd.DataFrame):
        raise TimestampError(f"data must be a pandas DataFrame, got {type(data).__name__}")
    if timestamp is None:
        index = as_datetime_index(data.index, field_name="the frame index")
        frame: pd.DataFrame = data.copy()
        frame.index = index
    else:
        if timestamp not in data.columns:
            available = ", ".join(str(column) for column in data.columns[:12])
            raise TimestampError(
                f"timestamp column {timestamp!r} is not in the data. Available columns: "
                f"{available}{', ...' if len(data.columns) > 12 else ''}"
            )
        index = as_datetime_index(data[timestamp], field_name=f"column {timestamp!r}")
        frame = data.drop(columns=[timestamp]) if drop else data.copy()
        frame.index = index
    if frame.index.name is None:
        # rename() rather than assigning to .name: a DatetimeIndex taken straight
        # from the caller's frame is the same object as theirs, and naming it in
        # place would reach back into a DataFrame this function promised to copy.
        frame.index = frame.index.rename(TIMESTAMP_INDEX_NAME)
    return frame


def sort_by_time(data: pd.DataFrame) -> pd.DataFrame:
    """Return ``data`` sorted by its timestamp index, stably.

    A stable sort keeps rows that share a timestamp in their original order, so
    :data:`DuplicatePolicy.KEEP_FIRST` means the first row as the user supplied
    it. Already-sorted frames are returned as a copy, unchanged.
    """
    index = as_datetime_index(data.index, field_name="the frame index")
    ordered: pd.DataFrame = (
        data.copy() if index.is_monotonic_increasing else data.sort_index(kind="stable")
    )
    return ordered


def duplicate_timestamps(values: object) -> pd.DatetimeIndex:
    """Return the distinct timestamps that occur more than once, in time order."""
    index = as_datetime_index(values)
    repeated = index[index.duplicated(keep="first")]
    distinct: pd.DatetimeIndex = pd.DatetimeIndex(repeated.unique()).sort_values()
    return distinct


def resolve_duplicates(
    data: pd.DataFrame,
    *,
    policy: DuplicatePolicy | str = DuplicatePolicy.ERROR,
) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    """Apply ``policy`` to repeated timestamps in ``data``.

    Returns the resolved frame and the distinct timestamps that were duplicated,
    so a caller can record how much data the preprocessing option removed. Under
    the default :data:`DuplicatePolicy.ERROR` a duplicate raises
    :class:`TimestampError` instead.
    """
    chosen = DuplicatePolicy.coerce(policy)
    index = as_datetime_index(data.index, field_name="the frame index")
    duplicates = duplicate_timestamps(index)
    if len(duplicates) == 0:
        return data.copy(), duplicates
    if chosen is DuplicatePolicy.ERROR:
        shown = ", ".join(str(stamp) for stamp in duplicates[:5])
        more = ", ..." if len(duplicates) > 5 else ""
        raise TimestampError(
            f"{len(duplicates)} timestamp(s) occur more than once: {shown}{more}. "
            "Unique timestamps are required (docs/method_spec.md section 7); pass "
            "on_duplicates='keep_first' or 'keep_last' to resolve them explicitly."
        )
    keep: Literal["first", "last"] = "first" if chosen is DuplicatePolicy.KEEP_FIRST else "last"
    resolved = data[~data.index.duplicated(keep=keep)].copy()
    return resolved, duplicates


# ---------------------------------------------------------------------------
# Cadence
# ---------------------------------------------------------------------------


def infer_time_step(values: object) -> timedelta:
    """Infer the time step of ``values`` from actual timestamp differences.

    The step is the most common positive difference between consecutive
    timestamps, with ties broken towards the shorter one. Missing rows do not
    defeat inference: their differences are whole multiples of the step, which is
    exactly the condition checked here. Differences that are *not* whole
    multiples mean the series has no single cadence, and that raises rather than
    being rounded away - declare ``frequency`` explicitly to work with such data.
    """
    index = as_datetime_index(values)
    if len(index) < _MIN_TIMESTAMPS_FOR_INFERENCE:
        raise TimestampError(
            f"a time step cannot be inferred from {len(index)} timestamp(s); supply "
            "frequency explicitly (e.g. frequency='30min')"
        )
    if not index.is_monotonic_increasing:
        raise TimestampError(
            "timestamps must be sorted before a time step can be inferred; call "
            "sort_by_time() first"
        )
    differences = pd.TimedeltaIndex(np.diff(index.to_numpy()))
    positive = differences[differences > _ZERO]
    if len(positive) == 0:
        raise TimestampError("timestamps do not advance; every difference is zero")

    counts = positive.value_counts()
    most_common = counts[counts == counts.max()]
    step = pd.Timedelta(min(most_common.index))

    remainders = positive % step
    off_step = positive[remainders != _ZERO]
    if len(off_step) > 0:
        shown = ", ".join(str(delta) for delta in pd.TimedeltaIndex(off_step).unique()[:5])
        raise TimestampError(
            f"no single cadence fits these timestamps: the most common step is {step}, "
            f"but {len(off_step)} difference(s) are not whole multiples of it ({shown}). "
            "Declare the intended cadence with frequency=... to treat the rest as "
            "off-grid timestamps."
        )
    result: timedelta = step.to_pytimedelta()
    return result


def resolve_time_step(
    values: object,
    *,
    frequency: timedelta | str | None = None,
) -> tuple[timedelta, StepSource]:
    """Return the time step to use and whether it was declared or inferred.

    A declared ``frequency`` always wins; it is the documented escape hatch for
    series whose cadence cannot be inferred. The source is reported so the run
    manifest can record which of the two produced the value.
    """
    if frequency is not None:
        return to_timedelta(frequency, field_name="frequency"), "declared"
    return infer_time_step(values), "inferred"


# ---------------------------------------------------------------------------
# Elapsed time and duration arithmetic
# ---------------------------------------------------------------------------


def elapsed_hours(
    values: object,
    *,
    origin: pd.Timestamp | str | None = None,
) -> pd.Series:
    """Return hours elapsed since ``origin`` for each timestamp.

    This is the ``time_distance_hours`` receptive-limiter feature of
    method_spec.md section 3.2: at 30-minute cadence it yields 0.0, 0.5, 1.0, ...
    computed from timestamp differences, never from row position, so rows around
    a missing period carry their true elapsed distance.

    ``origin`` defaults to the earliest timestamp in ``values``. Pass the origin
    of the full site series explicitly whenever a subset is transformed on its
    own - otherwise the same timestamp would receive different feature values in
    a subset than in the whole series.
    """
    index = as_datetime_index(values)
    if origin is None:
        start = index.min()
    else:
        start = pd.Timestamp(origin)
        if start is pd.NaT or start != start:
            raise TimestampError(f"origin is not a valid timestamp: {origin!r}")
        if (start.tz is None) != (index.tz is None):
            raise TimestampError(
                "origin and the timestamps must both be timezone-aware or both be "
                f"naive (origin tz={start.tz}, timestamps tz={index.tz})"
            )
    hours = np.asarray((index - start) / _ONE_HOUR, dtype=float)
    feature: pd.Series = pd.Series(hours, index=index, name=TIME_DISTANCE_HOURS)
    return feature


def duration_to_periods(
    duration: timedelta | str,
    time_step: timedelta | str,
    *,
    field_name: str = "duration",
) -> int:
    """Return how many rows ``duration`` spans at ``time_step`` on a regular grid.

    A convenience for reporting and for sizing arrays; the gap generator still
    selects rows by timestamp, because a real series may be missing some of the
    rows this count assumes. Durations that are not a whole multiple of the
    cadence raise, rather than being rounded into a silently wrong gap length.
    """
    delta = to_timedelta(duration, field_name=field_name)
    step = to_timedelta(time_step, field_name="time_step")
    if delta % step != timedelta(0):
        raise TimestampError(
            f"{field_name} {delta} is not a whole number of {step} steps "
            f"({delta / step:.6g} rows); choose a duration that fits the cadence"
        )
    return delta // step


def periods_to_duration(periods: int, time_step: timedelta | str) -> timedelta:
    """Return the elapsed duration of ``periods`` rows at ``time_step``."""
    if isinstance(periods, bool) or not isinstance(periods, (int, np.integer)):
        raise TimestampError(f"periods must be an integer, got {periods!r}")
    if periods < 0:
        raise TimestampError(f"periods must not be negative, got {periods}")
    return int(periods) * to_timedelta(time_step, field_name="time_step")


def interval_bounds(
    start: pd.Timestamp | str,
    duration: timedelta | str,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return the ``(start, end)`` timestamps of an interval of ``duration``.

    ``end = start + duration``. Intervals are half-open by convention here, so a
    24-hour interval starting at midnight ends at the following midnight and does
    **not** include it; see :func:`interval_mask`.
    """
    begin = pd.Timestamp(start)
    if begin is pd.NaT or begin != begin:
        raise TimestampError(f"interval start is not a valid timestamp: {start!r}")
    delta = to_timedelta(duration)
    return begin, begin + pd.Timedelta(delta)


def interval_mask(
    values: object,
    start: pd.Timestamp | str,
    duration: timedelta | str | None = None,
    *,
    end: pd.Timestamp | str | None = None,
    closed: IntervalClosed = "left",
) -> pd.Series:
    """Return a boolean Series marking the timestamps inside one interval.

    Give either ``duration`` or ``end``. The default ``closed="left"`` selects
    ``[start, end)``, so adjacent intervals of the same duration tile the axis
    without sharing a row - the property the artificial-gap generator relies on
    to keep gaps from overlapping.
    """
    index = as_datetime_index(values)
    if (duration is None) == (end is None):
        raise TimestampError("give exactly one of duration or end")
    if duration is not None:
        begin, stop = interval_bounds(start, duration)
    else:
        begin = pd.Timestamp(start)
        stop = pd.Timestamp(end)
        if begin is pd.NaT or stop is pd.NaT or begin != begin or stop != stop:
            raise TimestampError(f"interval bounds are not valid timestamps: {start!r}, {end!r}")
        if stop < begin:
            raise TimestampError(f"interval end {stop} precedes its start {begin}")
    if closed not in ("left", "right", "both", "neither"):
        raise TimestampError(f"closed must be 'left', 'right', 'both' or 'neither', got {closed!r}")
    lower = index >= begin if closed in ("left", "both") else index > begin
    upper = index <= stop if closed in ("right", "both") else index < stop
    mask: pd.Series = pd.Series(np.asarray(lower & upper, dtype=bool), index=index)
    return mask


# ---------------------------------------------------------------------------
# TimeAxis
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class TimeAxis:
    """A validated, sorted, unique time axis together with its cadence.

    Constructed by :meth:`from_index` or, for a whole frame, by
    :func:`prepare_time_index`. The axis compares the timestamps against the
    regular grid its time step implies and keeps the two failure modes apart:

    * :attr:`missing` - grid points with no row (ordinary data gaps, expected in
      eddy-covariance series and the reason gap lengths are elapsed-time
      quantities);
    * :attr:`off_grid` - rows that do not sit on the grid at all (a genuinely
      irregular series, or a mis-declared ``frequency``).

    Equality is identity-based (``eq=False``): the dataclass holds pandas objects
    whose ``==`` is element-wise.
    """

    #: Sorted, unique, NaT-free timestamps of the series.
    index: pd.DatetimeIndex
    #: Cadence of the series.
    time_step: timedelta
    #: Whether :attr:`time_step` was declared by the caller or inferred.
    step_source: StepSource
    #: Expected grid points that carry no row.
    missing: pd.DatetimeIndex
    #: Timestamps that do not lie on the expected grid.
    off_grid: pd.DatetimeIndex
    #: Distinct timestamps removed by a duplicate policy, if any.
    duplicates_removed: pd.DatetimeIndex

    # -- construction --------------------------------------------------------

    @classmethod
    def from_index(
        cls,
        values: object,
        *,
        frequency: timedelta | str | None = None,
        duplicates_removed: pd.DatetimeIndex | None = None,
    ) -> TimeAxis:
        """Build an axis from timestamps that are already sorted and unique.

        Sorting and duplicate resolution are deliberately not done here: they
        change which rows a frame contains, so they belong to
        :func:`prepare_time_index`, which can keep the frame and its axis in step.
        """
        index = as_datetime_index(values)
        if not index.is_monotonic_increasing:
            raise TimestampError(
                "timestamps must be sorted before building a TimeAxis; call "
                "sort_by_time() or prepare_time_index()"
            )
        repeated = duplicate_timestamps(index)
        if len(repeated) > 0:
            raise TimestampError(
                f"timestamps must be unique before building a TimeAxis; "
                f"{len(repeated)} value(s) repeat, starting at {repeated[0]}. Use "
                "prepare_time_index(on_duplicates=...) to resolve them explicitly."
            )
        step, source = resolve_time_step(index, frequency=frequency)
        grid = _expected_grid(index, step)
        return cls(
            index=index,
            time_step=step,
            step_source=source,
            missing=grid.difference(index),
            off_grid=index.difference(grid),
            duplicates_removed=(
                pd.DatetimeIndex([], dtype=index.dtype)
                if duplicates_removed is None
                else duplicates_removed
            ),
        )

    # -- description ---------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover - diagnostic only
        return (
            f"TimeAxis(start={self.start}, end={self.end}, n={len(self.index)}, "
            f"time_step={self.time_step} ({self.step_source}), regular={self.is_regular})"
        )

    def __len__(self) -> int:
        return len(self.index)

    @property
    def start(self) -> pd.Timestamp:
        """First timestamp; the origin of :meth:`elapsed_hours`."""
        first: pd.Timestamp = pd.Timestamp(self.index[0])
        return first

    @property
    def end(self) -> pd.Timestamp:
        """Last timestamp."""
        last: pd.Timestamp = pd.Timestamp(self.index[-1])
        return last

    @property
    def span(self) -> timedelta:
        """Elapsed time from the first to the last timestamp."""
        return (self.end - self.start).to_pytimedelta()

    @property
    def timezone(self) -> str | None:
        """Timezone name of the axis, or ``None`` when the timestamps are naive."""
        return None if self.index.tz is None else str(self.index.tz)

    @property
    def n_expected(self) -> int:
        """Rows a complete series at this cadence would have over the same span."""
        return len(self.index) + len(self.missing) - len(self.off_grid)

    @property
    def is_regular(self) -> bool:
        """Whether every grid point is present and every row sits on the grid."""
        return len(self.missing) == 0 and len(self.off_grid) == 0

    @property
    def coverage(self) -> float:
        """Fraction of the expected grid that carries a row, in [0, 1]."""
        expected = self.n_expected
        if expected <= 0:  # pragma: no cover - unreachable for a validated axis
            return 0.0
        return (len(self.index) - len(self.off_grid)) / expected

    def expected_index(self) -> pd.DatetimeIndex:
        """Return the complete regular grid implied by the cadence."""
        return _expected_grid(self.index, self.time_step)

    def require_regular(self) -> None:
        """Raise :class:`TimestampError` unless the series is a complete regular grid.

        Only for operations that genuinely cannot tolerate holes. Gap-length and
        feature calculations must not call it: real flux series have missing rows
        by construction, which is why durations are elapsed-time quantities.
        """
        if self.is_regular:
            return
        problems = []
        if len(self.missing) > 0:
            problems.append(f"{len(self.missing)} missing grid point(s) from {self.missing[0]}")
        if len(self.off_grid) > 0:
            problems.append(f"{len(self.off_grid)} off-grid timestamp(s) from {self.off_grid[0]}")
        raise TimestampError(
            f"the time axis is not a regular {self.time_step} grid: {'; '.join(problems)}"
        )

    # -- arithmetic ----------------------------------------------------------

    def elapsed_hours(self) -> pd.Series:
        """Return :func:`elapsed_hours` measured from :attr:`start`."""
        return elapsed_hours(self.index, origin=self.start)

    def periods(self, duration: timedelta | str, *, field_name: str = "duration") -> int:
        """Return how many rows ``duration`` spans at this cadence."""
        return duration_to_periods(duration, self.time_step, field_name=field_name)

    def duration(self, periods: int) -> timedelta:
        """Return the elapsed duration of ``periods`` rows at this cadence."""
        return periods_to_duration(periods, self.time_step)

    def bounds(
        self, start: pd.Timestamp | str, duration: timedelta | str
    ) -> tuple[pd.Timestamp, pd.Timestamp]:
        """Return the half-open ``(start, end)`` bounds of an interval."""
        return interval_bounds(start, duration)

    def mask(
        self,
        start: pd.Timestamp | str,
        duration: timedelta | str | None = None,
        *,
        end: pd.Timestamp | str | None = None,
        closed: IntervalClosed = "left",
    ) -> pd.Series:
        """Return a boolean Series over :attr:`index` marking one interval."""
        return interval_mask(self.index, start, duration, end=end, closed=closed)

    def with_duplicates_removed(self, removed: pd.DatetimeIndex) -> TimeAxis:
        """Return a copy recording which duplicate timestamps were resolved away."""
        return replace(self, duplicates_removed=removed)

    # -- serialisation -------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary for the run manifest."""
        return {
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "timezone": self.timezone,
            "n_timestamps": len(self.index),
            "time_step": iso_duration(self.time_step),
            "time_step_source": self.step_source,
            "is_regular": self.is_regular,
            "n_expected": self.n_expected,
            "n_missing": len(self.missing),
            "n_off_grid": len(self.off_grid),
            "n_duplicates_removed": len(self.duplicates_removed),
            "coverage": self.coverage,
        }


def _expected_grid(index: pd.DatetimeIndex, step: timedelta) -> pd.DatetimeIndex:
    """Return the regular grid from the first to the last timestamp at ``step``."""
    span = pd.Timestamp(index[-1]) - pd.Timestamp(index[0])
    expected = span // pd.Timedelta(step) + 1
    if expected > _MAX_EXPECTED_GRID:
        raise TimestampError(
            f"a {step} cadence over {span} implies {expected:,} rows, far more than the "
            "data holds; the declared frequency is almost certainly wrong"
        )
    return pd.date_range(start=index[0], end=index[-1], freq=pd.Timedelta(step))


# ---------------------------------------------------------------------------
# Frame preparation
# ---------------------------------------------------------------------------


def prepare_time_index(
    data: pd.DataFrame,
    *,
    timestamp: str | None = None,
    frequency: timedelta | str | None = None,
    on_duplicates: DuplicatePolicy | str = DuplicatePolicy.ERROR,
    require_regular: bool = False,
) -> tuple[pd.DataFrame, TimeAxis]:
    """Return ``data`` on a validated time axis, with that axis.

    The one call every downstream module should make before touching timestamps:
    it resolves the timestamp column or index, sorts, applies the duplicate
    policy, and resolves the cadence, returning a frame and a :class:`TimeAxis`
    that describe the same rows.

    ``require_regular=True`` additionally demands a complete grid. Leave it off
    for flux data: missing rows are normal, and every downstream duration is an
    elapsed-time quantity precisely so that they need not be filled in first.
    """
    frame = set_time_index(data, timestamp=timestamp)
    frame = sort_by_time(frame)
    frame, removed = resolve_duplicates(frame, policy=on_duplicates)
    axis = TimeAxis.from_index(frame.index, frequency=frequency, duplicates_removed=removed)
    if require_regular:
        axis.require_regular()
    return frame, axis
