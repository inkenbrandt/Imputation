"""Artificial-gap scenario generation (``docs/method_spec.md`` section 4).

The paper's validation experiment withholds roughly 25% of a site's genuinely
observed flux measurements as artificial gaps of three durations - 24 hours,
7 days and 30 days - mixed 20/30/50, and scores the model on exactly those
withheld values. This module builds that scenario and nothing else: it decides
*where* the gaps go, never what is done with them, so one manifest can be handed
to :mod:`rfrgapfill.leakage` for feature construction and to the metrics layer
for scoring by gap class - and, when NEE, H and LE are validated jointly, to all
three targets unchanged, which is the paper's shared-gap requirement.

Every quantity here is an elapsed-time quantity. A gap is a half-open interval
``[start, start + duration)`` matched against timestamps, so a 24-hour gap spans
24 hours whether or not the series happens to hold all 48 rows.

The 20/30/50 ambiguity (A3)
---------------------------

The article does not say whether those percentages count **gap events** or
**withheld half-hours**, and the two designs differ substantially: a mix that is
20/30/50 by event is nowhere near 20/30/50 by record, because one 30-day gap
withholds thirty times what a 24-hour gap does. :class:`AllocationBasis` exposes
the choice, :func:`allocate_gaps` implements both explicitly, and every manifest
reports the achieved mix *on both bases* next to the one that was requested.
Neither may be described as "paper exact".

Write ``S_c`` for the configured share of class ``c``, ``E_c = duration_c /
time_step`` for the records one event of that class covers on a complete grid,
``A`` for the number of available (genuinely observed) target values, and
``T = round(missing_fraction * A)`` for the records the scenario aims to
withhold in total. Then:

``allocation_basis="missing_records"`` (the default)
    The shares apply to withheld records. Each class independently receives the
    event count whose records come closest to its share of the total::

        events_c = round(S_c * T / E_c)

    The classes are allocated independently, so the counts sum to no particular
    total: what is apportioned is ``T``, not a number of gaps. This is the
    default because the paper frames the scenario as percentages of removed
    half-hours - an inference from the prose, not a statement in it.

``allocation_basis="gap_events"``
    The shares apply to the number of gap events. The mean event size under the
    mix is ``M = sum(S_c * E_c)``, so the design needs ``N = round(T / M)``
    events in total, apportioned across the classes by largest remainder::

        events_c = largest_remainder(N, S)

    Largest remainder is used so the counts sum to exactly ``N``; ties resolve in
    class order (short, long, very_long).

Both bases plan from ``E_c``, the records a *complete* grid would hold, because
the plan is made before any interval is placed. What the scenario actually
withholds is smaller wherever a chosen interval overlaps real missing data, and
that difference is reported rather than corrected away:
:attr:`GapManifest.achieved_fraction` and :attr:`GapManifest.record_shares` are
measured from the placed intervals, and the configured tolerances (A7) decide
whether the result is close enough to the request to pass without comment.

Placement
---------

Intervals are placed longest class first - a 30-day interval is far harder to
fit than a 24-hour one, so placing it while the series is still empty is what
makes the design achievable at all - and each proposal must satisfy two rules
before it is accepted:

* at least ``min_observed_fraction`` of the records it covers are genuinely
  observed target values, so an artificial gap is not quietly laid on top of a
  real one;
* it does not overlap an already-accepted interval, unless ``allow_overlap``.

Retries are bounded by ``max_attempts_per_gap``. A scenario that cannot be built
fails loudly with :class:`GapError` rather than returning a quietly thinner set
of gaps; ``on_shortfall="warn"`` is the deliberate opt-out that returns the
partial design for inspection.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import timedelta
from types import MappingProxyType
from typing import Any, Final, Literal

import numpy as np
import pandas as pd

from rfrgapfill.config import AllocationBasis, GapScenarioConfig, RFRConfig, ValidationConfig
from rfrgapfill.leakage import holdout_mask_from_intervals, observed_target_mask
from rfrgapfill.schema import ConfigError, FrozenRecord, GapClass
from rfrgapfill.time import (
    TimeAxis,
    as_datetime_index,
    duration_to_periods,
    interval_bounds,
    iso_duration,
    prepare_time_index,
    to_timedelta,
)

__all__ = [
    "GAP_MANIFEST_COLUMNS",
    "ArtificialGap",
    "GapAllocation",
    "GapError",
    "GapManifest",
    "GapScenarioGenerator",
    "GapScenarioWarning",
    "allocate_gaps",
]

#: What a gap manifest reports for every interval.
GAP_MANIFEST_COLUMNS: Final = (
    "gap_id",
    "gap_class",
    "start",
    "end",
    "duration",
    "n_expected",
    "n_rows",
    "n_observed_before_masking",
    "observed_fraction",
)

#: What to do when fewer intervals could be placed than the design asked for.
Shortfall = Literal["raise", "warn"]

#: Class order, for deterministic tie-breaking in the largest-remainder rule.
_CLASS_ORDER: Final = {gap_class: position for position, gap_class in enumerate(GapClass)}

#: Slack on the observed-fraction comparison, so a threshold that is exact in
#: decimal (0.5 of 48 records) is not missed by a binary representation.
_EPSILON: Final = 1e-9


class GapError(RuntimeError):
    """The requested artificial-gap scenario could not be constructed."""


class GapScenarioWarning(UserWarning):
    """The scenario was constructed, but not as it was requested.

    Emitted rather than raised when the dataset length prevents the requested
    design, when the achieved fraction or mix falls outside the configured
    tolerance (A7), or when ``on_shortfall="warn"`` accepts a partial set of
    intervals. Filter it to ``error`` to promote any of those to a failure.
    """


# ---------------------------------------------------------------------------
# Allocation: the 20/30/50 ambiguity, resolved explicitly (A3)
# ---------------------------------------------------------------------------


def _round_half_up(value: float) -> int:
    """Round to the nearest integer, halves upward.

    ``round`` is banker's rounding, which would make an allocation depend on the
    parity of a number nobody chose. Halves go up here, and the rule is stated in
    the module docstring rather than left to whoever reads the counts later.
    """
    return math.floor(value + 0.5)


def _largest_remainder(total: int, shares: Mapping[GapClass, float]) -> dict[GapClass, int]:
    """Apportion ``total`` indivisible items across ``shares``.

    Every class gets its floor and the leftover goes to the largest fractional
    remainders, so the counts sum to exactly ``total``. Ties resolve in
    :class:`~rfrgapfill.schema.GapClass` order, keeping the result a function of
    the configuration alone.
    """
    ideal = {gap_class: total * share for gap_class, share in shares.items()}
    counts = {gap_class: math.floor(value) for gap_class, value in ideal.items()}
    leftover = total - sum(counts.values())
    if leftover > 0:
        ranked = sorted(
            ideal,
            key=lambda gap_class: (
                -(ideal[gap_class] - counts[gap_class]),
                _CLASS_ORDER[gap_class],
            ),
        )
        for gap_class in ranked[:leftover]:
            counts[gap_class] += 1
    return counts


@dataclass(frozen=True)
class GapAllocation(FrozenRecord):
    """The gap design implied by the configured mix, before any interval is placed.

    A pure function of the configuration and the size of the series, so the
    interpretation of the paper's 20/30/50 can be inspected and tested without
    sampling anything. :attr:`notes` records every way the dataset prevents the
    requested design; :class:`GapScenarioGenerator` turns each into a
    :class:`GapScenarioWarning`.
    """

    #: Which interpretation of the mix produced this design (A3).
    basis: AllocationBasis
    #: Genuinely observed target values the withheld fraction is measured against.
    n_available: int
    #: Records the scenario aims to withhold in total.
    target_records: int
    #: Gap events planned per class.
    events: Mapping[GapClass, int]
    #: Records one event of each class covers on a complete grid.
    expected_records_per_event: Mapping[GapClass, int]
    #: Ways the dataset prevents the requested design, in human-readable form.
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "basis", AllocationBasis.coerce(self.basis))
        object.__setattr__(self, "events", MappingProxyType(dict(self.events)))
        object.__setattr__(
            self,
            "expected_records_per_event",
            MappingProxyType(dict(self.expected_records_per_event)),
        )
        object.__setattr__(self, "notes", tuple(self.notes))

    # -- accessors -----------------------------------------------------------

    @property
    def planned_records(self) -> Mapping[GapClass, int]:
        """Records each class is planned to withhold on a complete grid."""
        return MappingProxyType(
            {
                gap_class: count * self.expected_records_per_event[gap_class]
                for gap_class, count in self.events.items()
            }
        )

    @property
    def total_events(self) -> int:
        """Gap events planned across every class."""
        return sum(self.events.values())

    @property
    def total_planned_records(self) -> int:
        """Records the design is planned to withhold on a complete grid."""
        return sum(self.planned_records.values())

    @property
    def planned_fraction(self) -> float | None:
        """Planned withheld share of the available observations, or ``None`` if there are none."""
        if not self.n_available:
            return None
        return self.total_planned_records / self.n_available

    def summary(self) -> str:
        """Return a one-line account of the design and the basis behind it."""
        if not self.total_events:
            return (
                f"no gap could be allocated from {self.n_available} available observation(s) "
                f"under allocation_basis={self.basis.value!r}"
            )
        parts = [
            f"{gap_class.value} {count} x {self.expected_records_per_event[gap_class]} record(s)"
            for gap_class, count in self.events.items()
            if count
        ]
        return (
            f"allocation_basis={self.basis.value!r}: {self.total_events} gap(s) planned to "
            f"withhold {self.total_planned_records} of {self.n_available} available "
            "observation(s) - " + ", ".join(parts)
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation for the run manifest."""
        return {
            "allocation_basis": self.basis.value,
            "n_available": self.n_available,
            "target_records": self.target_records,
            "events": {gap_class.value: count for gap_class, count in self.events.items()},
            "expected_records_per_event": {
                gap_class.value: count
                for gap_class, count in self.expected_records_per_event.items()
            },
            "planned_records": {
                gap_class.value: count for gap_class, count in self.planned_records.items()
            },
            "total_events": self.total_events,
            "total_planned_records": self.total_planned_records,
            "planned_fraction": self.planned_fraction,
            "notes": list(self.notes),
        }


def allocate_gaps(
    config: GapScenarioConfig,
    *,
    n_available: int,
    time_step: timedelta | str,
    n_rows: int | None = None,
    span: timedelta | None = None,
) -> GapAllocation:
    """Turn the configured mix into a per-class event count.

    This function is the whole of ambiguity A3: both readings of the paper's
    20/30/50 live here, side by side, and the module docstring states each
    algorithm in full. Nothing is sampled - the same configuration and the same
    series size give the same design every time.

    :param config: the scenario configuration supplying the mix, the durations,
        the missing fraction and the allocation basis.
    :param n_available: genuinely observed target values in the series. The
        withheld fraction is measured against these rather than against every
        row: withholding a value that was never measured withholds nothing.
    :param time_step: the cadence, used to convert each duration into a record
        count. A duration that is not a whole number of steps is rejected by
        :func:`~rfrgapfill.time.duration_to_periods` rather than rounded.
    :param n_rows: rows on the time axis. Used only to notice that a design needs
        more non-overlapping rows than the series holds.
    :param span: elapsed time from the first to the last timestamp. Used only to
        notice that a gap class is longer than the series itself.
    """
    if isinstance(n_available, bool) or not isinstance(n_available, (int, np.integer)):
        raise ConfigError(f"n_available must be an integer, got {n_available!r}")
    if n_available < 0:
        raise ConfigError(f"n_available must not be negative, got {n_available}")
    step = to_timedelta(time_step, field_name="time_step")

    per_event = {
        gap_class: duration_to_periods(
            config.duration(gap_class),
            step,
            field_name=f"durations[{gap_class.value!r}]",
        )
        for gap_class in GapClass
    }
    for gap_class, records in per_event.items():
        if records <= 0:
            raise ConfigError(
                f"the {gap_class.value} duration {config.duration(gap_class)} covers no "
                f"records at a {step} cadence"
            )

    active = config.active_classes
    target_records = _round_half_up(config.missing_fraction * n_available)
    notes: list[str] = []

    fits: dict[GapClass, bool] = {}
    for gap_class in active:
        duration = config.duration(gap_class)
        fits[gap_class] = span is None or duration <= span + step
        if not fits[gap_class]:
            covered = span if span is not None else timedelta(0)
            notes.append(
                f"the {gap_class.value} class ({iso_duration(duration)}) is longer than the "
                f"series ({iso_duration(covered)}), so no interval of that class can be placed"
            )

    if config.basis is AllocationBasis.MISSING_RECORDS:
        events = {
            gap_class: _round_half_up(
                config.share(gap_class) * target_records / per_event[gap_class]
            )
            for gap_class in active
        }
    else:
        mean_event_size = math.fsum(
            config.share(gap_class) * per_event[gap_class] for gap_class in active
        )
        total = _round_half_up(target_records / mean_event_size) if mean_event_size else 0
        events = _largest_remainder(
            total, {gap_class: config.share(gap_class) for gap_class in active}
        )

    for gap_class in active:
        if not fits[gap_class]:
            events[gap_class] = 0
        elif events[gap_class] == 0:
            notes.append(
                f"the {gap_class.value} class has a {config.share(gap_class):.0%} share but was "
                f"allocated no gap: one {iso_duration(config.duration(gap_class))} interval "
                f"covers {per_event[gap_class]} record(s), more than the "
                f"{config.share(gap_class) * target_records:.0f} record(s) that share of "
                f"{target_records} allows"
            )

    allocation = GapAllocation(
        basis=config.basis,
        n_available=int(n_available),
        target_records=target_records,
        events={gap_class: events.get(gap_class, 0) for gap_class in GapClass},
        expected_records_per_event=per_event,
    )

    if allocation.total_planned_records > n_available:
        notes.append(
            f"the design plans to withhold {allocation.total_planned_records} record(s) but "
            f"only {n_available} observation(s) are available, so the achieved fraction will "
            "fall short of the request"
        )
    if (
        n_rows is not None
        and not config.allow_overlap
        and allocation.total_planned_records > n_rows
    ):
        notes.append(
            f"the design needs {allocation.total_planned_records} non-overlapping record(s) "
            f"but the series holds {n_rows}, so it cannot be constructed as requested"
        )
    if allocation.total_events == 0:
        notes.append(
            "no gap class received an event, so the scenario would withhold nothing; "
            "shorten the durations, lower missing_fraction, or use a longer series"
        )

    return replace(allocation, notes=tuple(notes))


# ---------------------------------------------------------------------------
# The manifest
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ArtificialGap(FrozenRecord):
    """One artificial gap: a half-open interval ``[start, end)`` and its accounting."""

    #: Stable identifier, ``"<class>-<n>"`` numbered chronologically within the class.
    gap_id: str
    #: Duration class this interval belongs to.
    gap_class: GapClass
    #: First timestamp inside the gap.
    start: pd.Timestamp
    #: First timestamp after the gap; not itself withheld.
    end: pd.Timestamp
    #: Records the interval covers on a complete grid at the series cadence.
    n_expected: int
    #: Rows the series actually holds inside the interval.
    n_rows: int
    #: Genuinely observed target values inside it, before the gap is applied.
    n_observed_before_masking: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "gap_class", GapClass.coerce(self.gap_class))
        object.__setattr__(self, "start", pd.Timestamp(self.start))
        object.__setattr__(self, "end", pd.Timestamp(self.end))

    @property
    def duration(self) -> timedelta:
        """Elapsed duration of the interval."""
        return (self.end - self.start).to_pytimedelta()

    @property
    def observed_fraction(self) -> float:
        """Observed values as a share of the records a complete grid would hold.

        The denominator is :attr:`n_expected`, not :attr:`n_rows`: the paper's 50%
        rule asks how much genuine measurement an interval destroys, and an
        interval laid over a real gap holds few rows *and* few observations,
        which would otherwise look like perfect coverage.
        """
        return self.n_observed_before_masking / self.n_expected if self.n_expected else 0.0

    def interval(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        """Return the ``(start, end)`` pair, as the holdout-mask helpers expect it."""
        return self.start, self.end

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation for the run manifest."""
        return {
            "gap_id": self.gap_id,
            "gap_class": self.gap_class.value,
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "duration": iso_duration(self.duration),
            "n_expected": self.n_expected,
            "n_rows": self.n_rows,
            "n_observed_before_masking": self.n_observed_before_masking,
            "observed_fraction": self.observed_fraction,
        }


@dataclass(frozen=True)
class GapManifest(FrozenRecord):
    """The placed scenario: every interval, what it withheld, and how it was designed.

    This is the object the specification's "returns a manifest" clause refers to,
    and it is where Step 10's exit criterion is met: :meth:`summary` and
    :meth:`to_dict` state exactly how the gap mixture was constructed - which
    reading of the 20/30/50 mix produced it, what the design asked for, and how
    far the result landed from that request on *both* bases.

    Per-class record counts are measured from the intervals themselves. With
    ``allow_overlap=True`` they may sum above :attr:`n_withheld_observed`, which
    counts each row once.
    """

    #: The placed intervals, in chronological order.
    gaps: tuple[ArtificialGap, ...]
    #: The design they were placed from.
    allocation: GapAllocation
    #: The scenario configuration in force.
    scenario: GapScenarioConfig
    #: Genuinely observed target values in the series.
    n_available: int
    #: Distinct rows falling inside any gap.
    n_withheld_rows: int
    #: Distinct genuinely observed values falling inside any gap - the test set.
    n_withheld_observed: int
    #: Cadence the durations were resolved against.
    time_step: timedelta
    #: Seed the placement used.
    random_state: int
    #: Targets the observed mask was derived from; empty when none was given.
    targets: tuple[str, ...] = ()
    #: Everything that did not go as requested, in human-readable form.
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "gaps", tuple(self.gaps))
        object.__setattr__(self, "time_step", to_timedelta(self.time_step, field_name="time_step"))
        object.__setattr__(self, "targets", tuple(self.targets))
        object.__setattr__(self, "warnings", tuple(self.warnings))

    # -- iteration -----------------------------------------------------------

    def __len__(self) -> int:
        return len(self.gaps)

    def __iter__(self) -> Iterator[ArtificialGap]:
        return iter(self.gaps)

    def by_class(self, gap_class: GapClass | str) -> tuple[ArtificialGap, ...]:
        """Return the placed intervals of one duration class."""
        wanted = GapClass.coerce(gap_class)
        return tuple(gap for gap in self.gaps if gap.gap_class is wanted)

    def intervals(self) -> tuple[tuple[pd.Timestamp, pd.Timestamp], ...]:
        """Return the ``(start, end)`` pair of every gap."""
        return tuple(gap.interval() for gap in self.gaps)

    def mask(self, values: object) -> pd.Series:
        """Return the holdout mask these gaps imply over ``values``.

        The bridge to :mod:`rfrgapfill.leakage`: this mask is step 1 of the
        validation procedure, built before any target-derived feature exists.
        """
        return holdout_mask_from_intervals(values, self.intervals())

    # -- what was achieved ---------------------------------------------------

    @property
    def basis(self) -> AllocationBasis:
        """The reading of the mix this scenario was built on (A3)."""
        return self.allocation.basis

    @property
    def achieved_fraction(self) -> float | None:
        """Withheld observations as a share of the available ones, or ``None`` if none are."""
        if not self.n_available:
            return None
        return self.n_withheld_observed / self.n_available

    @property
    def events_by_class(self) -> Mapping[GapClass, int]:
        """Intervals actually placed, per class."""
        return MappingProxyType(
            {gap_class: len(self.by_class(gap_class)) for gap_class in GapClass}
        )

    @property
    def records_by_class(self) -> Mapping[GapClass, int]:
        """Observed values actually withheld, per class."""
        return MappingProxyType(
            {
                gap_class: sum(gap.n_observed_before_masking for gap in self.by_class(gap_class))
                for gap_class in GapClass
            }
        )

    @property
    def event_shares(self) -> Mapping[GapClass, float]:
        """Achieved mix read as shares of the gap count."""
        return _shares(self.events_by_class)

    @property
    def record_shares(self) -> Mapping[GapClass, float]:
        """Achieved mix read as shares of the withheld records."""
        return _shares(self.records_by_class)

    @property
    def achieved_shares(self) -> Mapping[GapClass, float]:
        """Achieved mix on the basis that was requested."""
        if self.basis is AllocationBasis.GAP_EVENTS:
            return self.event_shares
        return self.record_shares

    @property
    def fraction_error(self) -> float | None:
        """Signed difference between the achieved and the requested withheld fraction."""
        achieved = self.achieved_fraction
        if achieved is None:
            return None
        return achieved - self.scenario.missing_fraction

    @property
    def mix_errors(self) -> Mapping[GapClass, float]:
        """Signed per-class difference between the achieved and the requested share."""
        achieved = self.achieved_shares
        return MappingProxyType(
            {
                gap_class: achieved[gap_class] - self.scenario.share(gap_class)
                for gap_class in GapClass
            }
        )

    @property
    def within_fraction_tolerance(self) -> bool:
        """Whether the achieved fraction is inside ``fraction_tolerance`` (A7)."""
        error = self.fraction_error
        return error is not None and abs(error) <= self.scenario.fraction_tolerance

    @property
    def within_mix_tolerance(self) -> bool:
        """Whether every class share is inside ``mix_tolerance`` (A7)."""
        return all(abs(error) <= self.scenario.mix_tolerance for error in self.mix_errors.values())

    @property
    def is_within_tolerance(self) -> bool:
        """Whether both the achieved fraction and the achieved mix are in tolerance."""
        return self.within_fraction_tolerance and self.within_mix_tolerance

    # -- reporting -----------------------------------------------------------

    def to_frame(self) -> pd.DataFrame:
        """Return the manifest as a table with :data:`GAP_MANIFEST_COLUMNS`."""
        rows: list[dict[str, Any]] = [
            {
                "gap_id": gap.gap_id,
                "gap_class": gap.gap_class.value,
                "start": gap.start,
                "end": gap.end,
                "duration": pd.Timedelta(gap.duration),
                "n_expected": gap.n_expected,
                "n_rows": gap.n_rows,
                "n_observed_before_masking": gap.n_observed_before_masking,
                "observed_fraction": gap.observed_fraction,
            }
            for gap in self.gaps
        ]
        frame: pd.DataFrame = pd.DataFrame(rows, columns=list(GAP_MANIFEST_COLUMNS))
        return frame

    def summary(self) -> str:
        """Return a one-line account of how the gap mixture was constructed."""
        requested = self.scenario.missing_fraction
        if not self.gaps:
            return (
                f"no artificial gap was placed; {requested:.1%} of {self.n_available} available "
                f"observation(s) was requested under allocation_basis={self.basis.value!r}"
            )
        achieved = self.achieved_fraction
        shares = self.achieved_shares
        unit = "gaps" if self.basis is AllocationBasis.GAP_EVENTS else "records"
        parts = [
            f"{gap_class.value} {count} x {iso_duration(self.scenario.duration(gap_class))} "
            f"({shares[gap_class]:.0%} of {unit}, requested {self.scenario.share(gap_class):.0%})"
            for gap_class, count in self.events_by_class.items()
            if count or self.scenario.share(gap_class)
        ]
        fraction = "n/a" if achieved is None else f"{achieved:.1%}"
        return (
            f"withheld {self.n_withheld_observed} of {self.n_available} available "
            f"observation(s) ({fraction}, requested {requested:.1%}) as {len(self.gaps)} gap(s) "
            f"under allocation_basis={self.basis.value!r}: " + "; ".join(parts)
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation for the run manifest."""
        return {
            "summary": self.summary(),
            "targets": list(self.targets),
            "random_state": self.random_state,
            "time_step": iso_duration(self.time_step),
            "scenario": self.scenario.to_dict(),
            "allocation": self.allocation.to_dict(),
            "n_available": self.n_available,
            "n_withheld_rows": self.n_withheld_rows,
            "n_withheld_observed": self.n_withheld_observed,
            "achieved_fraction": self.achieved_fraction,
            "fraction_error": self.fraction_error,
            "within_fraction_tolerance": self.within_fraction_tolerance,
            "events_by_class": _named(self.events_by_class),
            "records_by_class": _named(self.records_by_class),
            "event_shares": _named(self.event_shares),
            "record_shares": _named(self.record_shares),
            "achieved_shares": _named(self.achieved_shares),
            "mix_errors": _named(self.mix_errors),
            "within_mix_tolerance": self.within_mix_tolerance,
            "is_within_tolerance": self.is_within_tolerance,
            "gaps": [gap.to_dict() for gap in self.gaps],
            "warnings": list(self.warnings),
        }


def _shares(counts: Mapping[GapClass, int]) -> Mapping[GapClass, float]:
    """Return ``counts`` as shares of their total; all zero when the total is zero."""
    total = sum(counts.values())
    if not total:
        return MappingProxyType(dict.fromkeys(counts, 0.0))
    return MappingProxyType({gap_class: count / total for gap_class, count in counts.items()})


def _named(values: Mapping[GapClass, float]) -> dict[str, float]:
    """Return a per-class mapping keyed by class name, for JSON output."""
    return {gap_class.value: value for gap_class, value in values.items()}


# ---------------------------------------------------------------------------
# The generator
# ---------------------------------------------------------------------------


class GapScenarioGenerator:
    """Places the artificial gaps a configured scenario calls for.

    One generator is one scenario and one seed::

        manifest = GapScenarioGenerator(config).generate(df, target="LE", qc_column="LE_QC")
        holdout = manifest.mask(df.index)

    The manifest it returns is the input to the leakage-safe validation workflow
    and the label the metrics layer groups by, and it carries the full account of
    how the mixture was constructed - see :meth:`GapManifest.summary`.

    :param config: an :class:`~rfrgapfill.config.RFRConfig` (the scenario and the
        seed are read from it), a :class:`~rfrgapfill.config.ValidationConfig`, a
        :class:`~rfrgapfill.config.GapScenarioConfig`, or ``None`` for the
        paper's defaults.
    :param random_state: seed for the placement. Overrides the seed carried by an
        ``RFRConfig``; defaults to 42 when neither supplies one.
    """

    def __init__(
        self,
        config: RFRConfig | ValidationConfig | GapScenarioConfig | None = None,
        *,
        random_state: int | None = None,
    ) -> None:
        scenario, parent = _resolve_scenario(config)
        if random_state is None:
            seed = parent.random_state if parent is not None else 42
        else:
            seed = random_state
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ConfigError(f"random_state must be an integer, got {seed!r}")
        self._scenario = scenario
        self._config = parent
        self._random_state = int(seed)

    def __repr__(self) -> str:  # pragma: no cover - diagnostic only
        return (
            f"GapScenarioGenerator(basis={self._scenario.basis.value!r}, "
            f"missing_fraction={self._scenario.missing_fraction}, "
            f"random_state={self._random_state})"
        )

    @property
    def scenario(self) -> GapScenarioConfig:
        """The scenario configuration this generator places gaps for."""
        return self._scenario

    @property
    def config(self) -> RFRConfig | None:
        """The run configuration the scenario came from, when there was one."""
        return self._config

    @property
    def random_state(self) -> int:
        """The seed the placement uses. The same seed gives the same gaps."""
        return self._random_state

    def generate(
        self,
        data: object,
        *,
        target: str | Sequence[str] | None = None,
        qc_column: str | Sequence[str] | Mapping[str, str] | None = None,
        observed: object | None = None,
        frequency: timedelta | str | None = None,
        on_shortfall: Shortfall = "raise",
    ) -> GapManifest:
        """Place the scenario's gaps over ``data`` and return the manifest.

        :param data: the site frame, or any timestamps a
            :class:`~rfrgapfill.time.TimeAxis` can be built from. A frame is put
            on a validated time axis first, exactly as the rest of the package
            does.
        :param target: the target flux, or several of them for the paper's joint
            NEE/H/LE validation. With several, a row counts as available only
            where *every* target is observed, so the single set of gap locations
            the manifest describes is a fair test set for all of them.
        :param qc_column: the QC/provenance column, which decides what counts as
            a genuine measurement rather than a value that arrived already
            gap-filled. One name for one target, or a mapping from target to
            column when several are validated jointly.
        :param observed: an explicit availability mask, as an alternative to
            deriving one from ``target``. Mutually exclusive with it.
        :param frequency: the cadence, when it should not be inferred.
        :param on_shortfall: ``"raise"`` (the default) fails when fewer intervals
            could be placed than the design asked for; ``"warn"`` returns the
            partial scenario for inspection.
        """
        if on_shortfall not in ("raise", "warn"):
            raise ConfigError(f"on_shortfall must be 'raise' or 'warn', got {on_shortfall!r}")

        frame, axis = self._resolve_axis(data, frequency=frequency)
        targets = _resolve_targets(target)
        notes: list[str] = []
        available = self._availability(
            frame,
            axis,
            targets=targets,
            qc_column=qc_column,
            observed=observed,
            notes=notes,
        )

        allocation = allocate_gaps(
            self._scenario,
            n_available=int(np.count_nonzero(available)),
            time_step=axis.time_step,
            n_rows=len(axis),
            span=axis.span,
        )
        notes.extend(allocation.notes)

        gaps, withheld, shortfalls = self._place(axis, available, allocation)
        manifest = GapManifest(
            gaps=gaps,
            allocation=allocation,
            scenario=self._scenario,
            n_available=allocation.n_available,
            n_withheld_rows=int(np.count_nonzero(withheld)),
            n_withheld_observed=int(np.count_nonzero(withheld & available)),
            time_step=axis.time_step,
            random_state=self._random_state,
            targets=targets,
        )

        notes.extend(_tolerance_notes(manifest))
        if shortfalls:
            message = _shortfall_message(shortfalls, manifest, self._scenario)
            if on_shortfall == "raise":
                raise GapError(message)
            notes.append(message)
        elif not gaps:
            message = "the artificial-gap scenario withheld nothing: " + (
                "; ".join(allocation.notes) or "no gap class was allocated an event"
            )
            if on_shortfall == "raise":
                raise GapError(message)
            notes.append(message)

        manifest = replace(manifest, warnings=tuple(notes))
        for note in manifest.warnings:
            warnings.warn(note, GapScenarioWarning, stacklevel=2)
        return manifest

    # -- internals -----------------------------------------------------------

    def _resolve_axis(
        self, data: object, *, frequency: timedelta | str | None
    ) -> tuple[pd.DataFrame | None, TimeAxis]:
        """Return the frame (when there is one) on a validated axis, and that axis."""
        cadence = frequency
        if cadence is None and self._config is not None:
            cadence = self._config.frequency
        if isinstance(data, pd.DataFrame):
            frame, axis = prepare_time_index(data, frequency=cadence)
            return frame, axis
        index = as_datetime_index(data, field_name="timestamps")
        return None, TimeAxis.from_index(index, frequency=cadence)

    def _availability(
        self,
        frame: pd.DataFrame | None,
        axis: TimeAxis,
        *,
        targets: tuple[str, ...],
        qc_column: str | Sequence[str] | Mapping[str, str] | None,
        observed: object | None,
        notes: list[str],
    ) -> np.ndarray:
        """Return the rows whose target value the scenario is allowed to withhold."""
        if targets and observed is not None:
            raise ConfigError(
                "pass either target= or observed=, not both: two sources of the same mask "
                "cannot be reconciled in the gap manifest"
            )
        if observed is not None:
            if qc_column is not None:
                raise ConfigError(
                    "qc_column= describes how to derive the observed mask, so it cannot be "
                    "given alongside an explicit observed= mask"
                )
            return _mask_array(observed, axis.index, field_name="observed")
        if not targets:
            if qc_column is not None:
                raise ConfigError("qc_column= needs the target= it qualifies")
            notes.append(
                "no target was given, so every timestamp counts as available and the achieved "
                "fraction is measured against rows rather than against observations"
            )
            return np.ones(len(axis.index), dtype=bool)
        if frame is None:
            raise ConfigError(
                "target= names a column, so data must be a DataFrame; pass observed= to give "
                "the availability mask directly"
            )
        if len(targets) > 1 and not self._scenario.shared_gaps_across_targets:
            raise ConfigError(
                f"{len(targets)} targets were given but shared_gaps_across_targets is False; "
                "one manifest is one set of gap locations, so generate each target separately "
                "or enable the shared design the paper used"
            )
        available = np.ones(len(axis.index), dtype=bool)
        for name, qc_name in zip(targets, _resolve_qc_columns(targets, qc_column), strict=True):
            mask = observed_target_mask(frame, name, qc_column=qc_name, config=self._config)
            available &= mask.to_numpy(dtype=bool)
        return available

    def _place(
        self, axis: TimeAxis, available: np.ndarray, allocation: GapAllocation
    ) -> tuple[tuple[ArtificialGap, ...], np.ndarray, dict[GapClass, tuple[int, int]]]:
        """Sample intervals for the allocated design, longest class first."""
        index = axis.index
        step = pd.Timedelta(axis.time_step)
        rng = np.random.default_rng(self._random_state)
        withheld = np.zeros(len(index), dtype=bool)
        placed: list[ArtificialGap] = []
        shortfalls: dict[GapClass, tuple[int, int]] = {}

        wanted = sorted(
            (gap_class for gap_class in GapClass if allocation.events[gap_class] > 0),
            key=lambda gap_class: (-self._scenario.duration(gap_class), _CLASS_ORDER[gap_class]),
        )
        for gap_class in wanted:
            requested = allocation.events[gap_class]
            duration = self._scenario.duration(gap_class)
            n_expected = allocation.expected_records_per_event[gap_class]
            threshold = self._scenario.min_observed_fraction * n_expected
            # The last start whose half-open interval still ends inside the
            # series: the axis covers [start, end + one step).
            latest_start = index[-1] + step - pd.Timedelta(duration)
            eligible = np.flatnonzero(np.asarray(index <= latest_start))
            if eligible.size == 0:
                shortfalls[gap_class] = (requested, 0)
                continue

            count = 0
            for _ in range(requested):
                gap = self._sample_one(
                    index=index,
                    eligible=eligible,
                    rng=rng,
                    withheld=withheld,
                    available=available,
                    gap_class=gap_class,
                    ordinal=count + 1,
                    duration=duration,
                    n_expected=n_expected,
                    threshold=threshold,
                )
                if gap is None:
                    break
                placed.append(gap)
                count += 1
            if count < requested:
                shortfalls[gap_class] = (requested, count)

        # Placement runs longest class first, but a manifest is read in time
        # order, so number the gaps only once they are sorted.
        placed.sort(key=lambda gap: (gap.start, _CLASS_ORDER[gap.gap_class]))
        counters: dict[GapClass, int] = dict.fromkeys(GapClass, 0)
        numbered: list[ArtificialGap] = []
        for gap in placed:
            counters[gap.gap_class] += 1
            numbered.append(
                replace(gap, gap_id=f"{gap.gap_class.value}-{counters[gap.gap_class]:02d}")
            )
        return tuple(numbered), withheld, shortfalls

    def _sample_one(
        self,
        *,
        index: pd.DatetimeIndex,
        eligible: np.ndarray,
        rng: np.random.Generator,
        withheld: np.ndarray,
        available: np.ndarray,
        gap_class: GapClass,
        ordinal: int,
        duration: timedelta,
        n_expected: int,
        threshold: float,
    ) -> ArtificialGap | None:
        """Propose interval starts until one is acceptable, or the retries run out.

        Marks the accepted interval in ``withheld`` and returns it; returns
        ``None`` when ``max_attempts_per_gap`` proposals were all rejected.
        """
        for _ in range(self._scenario.max_attempts_per_gap):
            position = int(eligible[int(rng.integers(eligible.size))])
            start, end = interval_bounds(index[position], duration)
            lo = int(index.searchsorted(start, side="left"))
            hi = int(index.searchsorted(end, side="left"))
            if not self._scenario.allow_overlap and bool(withheld[lo:hi].any()):
                continue
            n_observed = int(np.count_nonzero(available[lo:hi]))
            if n_observed + _EPSILON < threshold:
                continue
            withheld[lo:hi] = True
            return ArtificialGap(
                gap_id=f"{gap_class.value}-{ordinal:02d}",  # renumbered chronologically
                gap_class=gap_class,
                start=start,
                end=end,
                n_expected=n_expected,
                n_rows=hi - lo,
                n_observed_before_masking=n_observed,
            )
        return None


def _resolve_scenario(
    config: RFRConfig | ValidationConfig | GapScenarioConfig | None,
) -> tuple[GapScenarioConfig, RFRConfig | None]:
    """Return the scenario configuration and the run configuration it came from."""
    if config is None:
        return GapScenarioConfig(), None
    if isinstance(config, GapScenarioConfig):
        return config, None
    if isinstance(config, ValidationConfig):
        return config.gaps, None
    if isinstance(config, RFRConfig):
        return config.validation.gaps, config
    raise ConfigError(
        "config must be an RFRConfig, ValidationConfig, GapScenarioConfig or None, got "
        f"{type(config).__name__}"
    )


def _resolve_targets(target: str | Sequence[str] | None) -> tuple[str, ...]:
    """Return the requested targets as a tuple, rejecting empty and duplicate entries."""
    if target is None:
        return ()
    if isinstance(target, str):
        return (target,)
    if not isinstance(target, Sequence):
        raise ConfigError(f"target must be a column name or a sequence of them, got {target!r}")
    targets = tuple(str(name) for name in target)
    if not targets:
        raise ConfigError("target must name at least one column")
    if len(set(targets)) != len(targets):
        raise ConfigError(f"target names a column more than once: {targets}")
    return targets


def _resolve_qc_columns(
    targets: tuple[str, ...],
    qc_column: str | Sequence[str] | Mapping[str, str] | None,
) -> tuple[str | None, ...]:
    """Pair each target with its QC column."""
    if qc_column is None:
        return (None,) * len(targets)
    if isinstance(qc_column, Mapping):
        return tuple(qc_column.get(name) for name in targets)
    if isinstance(qc_column, str):
        if len(targets) != 1:
            raise ConfigError(
                f"one QC column {qc_column!r} was given for {len(targets)} targets; pass a "
                "mapping from target to QC column when several are validated jointly"
            )
        return (qc_column,)
    if isinstance(qc_column, Sequence):
        columns = tuple(str(name) for name in qc_column)
        if len(columns) != len(targets):
            raise ConfigError(
                f"{len(columns)} QC column(s) were given for {len(targets)} target(s)"
            )
        return columns
    raise ConfigError(f"qc_column must be a name, a sequence or a mapping, got {qc_column!r}")


def _mask_array(mask: object, index: pd.DatetimeIndex, *, field_name: str) -> np.ndarray:
    """Return ``mask`` as a boolean array aligned to ``index``."""
    if isinstance(mask, pd.Series):
        if not mask.index.equals(index):
            raise ConfigError(f"{field_name} must be indexed by the same timestamps as the data")
        return np.asarray(mask.to_numpy(), dtype=bool)
    values = np.asarray(mask, dtype=bool)
    if values.shape != (len(index),):
        raise ConfigError(
            f"{field_name} must have one value per timestamp ({len(index)}), got {values.shape}"
        )
    return values


def _tolerance_notes(manifest: GapManifest) -> list[str]:
    """Return a note for every documented tolerance the achieved scenario misses (A7)."""
    scenario = manifest.scenario
    notes: list[str] = []
    if manifest.gaps and not manifest.within_fraction_tolerance:
        achieved = manifest.achieved_fraction
        shown = "n/a" if achieved is None else f"{achieved:.3f}"
        notes.append(
            f"the achieved withheld fraction {shown} differs from the requested "
            f"{scenario.missing_fraction:.3f} by more than the {scenario.fraction_tolerance:.3f} "
            "tolerance"
        )
    if manifest.gaps and not manifest.within_mix_tolerance:
        offenders = ", ".join(
            f"{gap_class.value} {manifest.achieved_shares[gap_class]:.3f} vs "
            f"{scenario.share(gap_class):.3f}"
            for gap_class, error in manifest.mix_errors.items()
            if abs(error) > scenario.mix_tolerance
        )
        notes.append(
            f"the achieved gap mix is outside the {scenario.mix_tolerance:.3f} tolerance on "
            f"allocation_basis={manifest.basis.value!r} ({offenders})"
        )
    return notes


def _shortfall_message(
    shortfalls: Mapping[GapClass, tuple[int, int]],
    manifest: GapManifest,
    scenario: GapScenarioConfig,
) -> str:
    """Explain which classes came up short, and what would let them fit."""
    detail = ", ".join(
        f"{gap_class.value} placed {placed} of {requested}"
        for gap_class, (requested, placed) in shortfalls.items()
    )
    return (
        f"the artificial-gap scenario could not be constructed as requested ({detail}) after up "
        f"to {scenario.max_attempts_per_gap} attempt(s) per gap; {len(manifest)} gap(s) were "
        f"placed, withholding {manifest.n_withheld_observed} of {manifest.n_available} available "
        "observation(s). Lower min_observed_fraction, lower missing_fraction, shorten the "
        "durations, allow overlap, or use a longer series."
    )
