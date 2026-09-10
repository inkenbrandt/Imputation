"""Artificial-gap scenario generation.

Builds the paper's validation scenario (``docs/method_spec.md`` section 4):
contiguous 24-hour, 7-day and 30-day intervals that collectively withhold about
25% of a site's genuinely observed target values, mixed 20/30/50, each interval
required to contain at least 50% real measurements before it is accepted.

Three properties are load-bearing and are what the module is organised around.

* **Durations are elapsed time.** An interval is ``[start, start + duration)``
  matched against timestamps. A 24-hour gap spans 24 hours whether or not the
  series happens to hold 48 rows there, so the ``n_expected`` a gap is scored
  against comes from the cadence, not from the rows that survived.
* **The withheld observations are the test set.** There is no row-wise random
  split anywhere in this package; that would destroy the contiguous temporal
  structure the paper exists to study (method_spec.md 4.5).
* **The scenario is reported, not asserted.** A real series with real gaps
  cannot always hit exactly 25% or exactly 20/30/50, so the generator records
  what it achieved against a documented tolerance and says so when it fell short
  (ambiguity A7) rather than raising or quietly pretending.

The generator knows nothing about features or models: it takes a time axis and a
mask of genuinely observed target values, and returns a mask and a manifest.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import timedelta
from types import MappingProxyType
from typing import Any, Final, cast

import numpy as np
import pandas as pd

from rfrgapfill.config import AllocationBasis, GapClass, GapScenarioConfig
from rfrgapfill.time import TimeAxis, duration_to_periods, iso_duration

__all__ = [
    "MANIFEST_COLUMNS",
    "ArtificialGaps",
    "GapError",
    "GapScenarioGenerator",
    "generate_artificial_gaps",
]


class GapError(ValueError):
    """Raised when the requested artificial-gap scenario cannot be constructed.

    Reserved for a request that is impossible rather than merely unsatisfied: a
    series shorter than the gaps asked of it, or no genuinely observed target
    value to withhold. A scenario that is placeable but falls short of its quota
    is *reported* on :class:`ArtificialGaps`, not raised - falling short is a
    property of the data and the run must still be able to describe it.
    """


#: Manifest columns, in order (implementation plan, Step 9).
MANIFEST_COLUMNS: Final[tuple[str, ...]] = (
    "gap_class",
    "start",
    "end",
    "duration",
    "n_expected",
    "n_rows",
    "n_observed_before_masking",
    "observed_fraction",
)


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ArtificialGaps:
    """A generated scenario: which rows are withheld, and how that came about.

    The three Series share the input time axis, so they can be used to select
    rows directly. :attr:`mask` marks every row inside an artificial interval;
    :attr:`gap_id` and :attr:`gap_class` say which interval and which duration
    class, and are the grouping keys the per-gap-class metrics of
    method_spec.md 6.3 are computed over.
    """

    #: One row per artificial gap, indexed by ``gap_id`` and ordered by start.
    manifest: pd.DataFrame
    #: True on every row inside an artificial interval, over the full time axis.
    mask: pd.Series
    #: Gap id per row, missing outside the artificial intervals.
    gap_id: pd.Series
    #: Gap class per row, missing outside the artificial intervals.
    gap_class: pd.Series
    #: Genuinely observed target values available before any were withheld.
    n_available: int
    #: Observed values actually withheld. This is the size of the test set.
    n_withheld: int
    #: Withheld observations per class.
    records_by_class: Mapping[GapClass, int]
    #: Gap events per class.
    events_by_class: Mapping[GapClass, int]
    #: The scenario that was asked for.
    config: GapScenarioConfig
    #: Seed the intervals were drawn with.
    random_state: int
    #: Why the scenario fell short, where it did. Empty when nothing did.
    warnings: tuple[str, ...]

    # -- achieved design -----------------------------------------------------

    @property
    def achieved_fraction(self) -> float:
        """Fraction of available observations actually withheld."""
        if self.n_available == 0:
            return 0.0
        return self.n_withheld / self.n_available

    @property
    def achieved_mix(self) -> Mapping[GapClass, float]:
        """Achieved class shares, on the configured allocation basis (A3).

        Shares of withheld *records* under ``missing_records`` and of *events*
        under ``gap_events``, so the reported mix is always measured the same way
        the scenario was requested.
        """
        counts = (
            self.records_by_class
            if self.config.basis is AllocationBasis.MISSING_RECORDS
            else self.events_by_class
        )
        total = sum(counts.values())
        if total == 0:
            return MappingProxyType({gap: 0.0 for gap in GapClass})
        return MappingProxyType({gap: counts[gap] / total for gap in GapClass})

    @property
    def n_gaps(self) -> int:
        """How many artificial intervals were placed."""
        return len(self.manifest)

    @property
    def fraction_within_tolerance(self) -> bool:
        """Whether the achieved fraction is within the configured tolerance (A7)."""
        return (
            abs(self.achieved_fraction - self.config.missing_fraction)
            <= self.config.fraction_tolerance
        )

    @property
    def mix_within_tolerance(self) -> bool:
        """Whether every class share is within the configured tolerance (A7)."""
        achieved = self.achieved_mix
        return all(
            abs(achieved[gap] - self.config.share(gap)) <= self.config.mix_tolerance
            for gap in GapClass
        )

    @property
    def satisfied(self) -> bool:
        """Whether the scenario met its requested design within tolerance.

        False does not invalidate a run: it means the series could not carry the
        requested design, and the metrics must be read against the achieved
        fraction and mix rather than the requested ones.
        """
        return not self.warnings and self.fraction_within_tolerance and self.mix_within_tolerance

    @property
    def shortfalls(self) -> tuple[str, ...]:
        """Every reason :attr:`satisfied` is false, in reader-facing prose.

        :attr:`warnings` alone is not enough. A scenario can place every gap it
        set out to and still miss the requested design, because whole gaps are a
        coarse quantum: two 30-day gaps either overshoot half of a 25% budget or
        undershoot it, with nothing in between. Those misses belong here rather
        than being visible only to a caller who thought to compare the achieved
        numbers against the tolerances themselves.
        """
        reasons = list(self.warnings)
        if not self.fraction_within_tolerance:
            reasons.append(
                f"withheld {self.achieved_fraction:.1%} of the available observations against "
                f"a requested {self.config.missing_fraction:.1%}, outside the "
                f"{self.config.fraction_tolerance:.1%} tolerance"
            )
        if not self.mix_within_tolerance:
            achieved = self.achieved_mix
            off = [
                f"{gap.value} {achieved[gap]:.1%} vs {self.config.share(gap):.1%}"
                for gap in GapClass
                if abs(achieved[gap] - self.config.share(gap)) > self.config.mix_tolerance
            ]
            reasons.append(
                f"the achieved class mix is outside the {self.config.mix_tolerance:.1%} "
                f"tolerance: {'; '.join(off)}"
            )
        return tuple(reasons)

    # -- selection -----------------------------------------------------------

    def class_mask(self, gap_class: GapClass | str) -> pd.Series:
        """Return the mask selecting rows inside gaps of one class."""
        chosen = GapClass.coerce(gap_class)
        selected: pd.Series = self.gap_class == chosen.value
        return selected.fillna(False).astype(bool)

    def gap_ids(self, gap_class: GapClass | str) -> tuple[int, ...]:
        """Return the ids of the gaps in one class, ordered by start."""
        chosen = GapClass.coerce(gap_class)
        matching = self.manifest.index[self.manifest["gap_class"] == chosen.value]
        return tuple(int(value) for value in matching)

    # -- serialisation -------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable manifest for the run report."""
        return {
            "random_state": self.random_state,
            "config": self.config.to_dict(),
            "n_available": self.n_available,
            "n_withheld": self.n_withheld,
            "n_gaps": self.n_gaps,
            "requested_fraction": self.config.missing_fraction,
            "achieved_fraction": self.achieved_fraction,
            "requested_mix": {gap.value: self.config.share(gap) for gap in GapClass},
            "achieved_mix": {gap.value: share for gap, share in self.achieved_mix.items()},
            "allocation_basis": self.config.basis.value,
            "records_by_class": {gap.value: self.records_by_class[gap] for gap in GapClass},
            "events_by_class": {gap.value: self.events_by_class[gap] for gap in GapClass},
            "fraction_within_tolerance": self.fraction_within_tolerance,
            "mix_within_tolerance": self.mix_within_tolerance,
            "satisfied": self.satisfied,
            "warnings": list(self.warnings),
            "shortfalls": list(self.shortfalls),
            "gaps": [
                {
                    "gap_id": int(cast("int", gap_id)),
                    "gap_class": str(row["gap_class"]),
                    "start": pd.Timestamp(row["start"]).isoformat(),
                    "end": pd.Timestamp(row["end"]).isoformat(),
                    "duration": iso_duration(pd.Timedelta(row["duration"]).to_pytimedelta()),
                    "n_expected": int(row["n_expected"]),
                    "n_rows": int(row["n_rows"]),
                    "n_observed_before_masking": int(row["n_observed_before_masking"]),
                    "observed_fraction": float(row["observed_fraction"]),
                }
                for gap_id, row in self.manifest.iterrows()
            ],
        }


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Placed:
    """One accepted interval, before the manifest is assembled."""

    gap_class: GapClass
    start: pd.Timestamp
    end: pd.Timestamp
    duration: timedelta
    n_expected: int
    n_rows: int
    n_observed: int

    @property
    def observed_fraction(self) -> float:
        return self.n_observed / self.n_expected


class GapScenarioGenerator:
    """Draws the artificial-gap scenario of method_spec.md section 4.

    Seeded and reproducible: the same axis, the same observation mask and the
    same seed give the same intervals, which is what makes a validation run
    repeatable at all.

    Intervals are placed longest class first. A 30-day interval needing 50%
    genuine measurements has far fewer viable positions than a 24-hour one, so
    placing it while the axis is still empty is what keeps the requested mix
    reachable; doing it the other way round strands the long gaps behind the
    short ones.
    """

    def __init__(
        self,
        config: GapScenarioConfig | None = None,
        *,
        random_state: int = 42,
    ) -> None:
        self.config = GapScenarioConfig() if config is None else config
        if isinstance(random_state, bool) or not isinstance(random_state, int):
            raise GapError(f"random_state must be an integer, got {random_state!r}")
        self.random_state = int(random_state)

    def generate(self, axis: TimeAxis, observed: object) -> ArtificialGaps:
        """Return a scenario over ``axis``, withholding from ``observed``.

        ``observed`` is a boolean mask over the axis marking genuinely measured,
        quality-controlled target values - never values that arrived already
        gap-filled, which are not truth to score against. For joint NEE/H/LE
        validation pass one mask covering all three targets so the gaps land in
        the same places for each (method_spec.md 4.4).
        """
        index = axis.index
        available = _as_mask(observed, index=index)
        n_available = int(available.sum())
        if n_available == 0:
            raise GapError(
                "no genuinely observed target values to withhold: the observation mask is "
                "empty. Check the QC column and the observed flag values."
            )

        config = self.config
        target_records = round(config.missing_fraction * n_available)
        if target_records == 0:
            raise GapError(
                f"missing_fraction={config.missing_fraction} withholds no observations from "
                f"{n_available} available value(s); the scenario would have no test set"
            )

        classes = self._placement_order(axis)
        quotas, warnings = self._quotas(axis, classes, target_records)

        rng = np.random.default_rng(self.random_state)
        cumulative = np.concatenate(([0], np.cumsum(available.to_numpy(dtype=np.int64))))
        placed: list[_Placed] = []

        for gap_class in classes:
            found, shortfall = self._place_class(
                gap_class,
                axis=axis,
                available=available,
                cumulative=cumulative,
                placed=placed,
                quota=quotas[gap_class],
                rng=rng,
            )
            placed.extend(found)
            if shortfall is not None:
                warnings.append(shortfall)

        return self._assemble(placed, axis=axis, available=available, warnings=tuple(warnings))

    # -- design --------------------------------------------------------------

    def _placement_order(self, axis: TimeAxis) -> tuple[GapClass, ...]:
        """Return the active classes, longest duration first."""
        active = self.config.active_classes
        if not active:  # pragma: no cover - gap_mix must sum to 1, so one share is positive
            raise GapError("the gap mix allocates nothing to any class")
        span = axis.span
        too_long = [gap for gap in active if self.config.duration(gap) > span]
        if too_long:
            raise GapError(
                "the series is shorter than the gap(s) requested of it: "
                + ", ".join(f"{gap.value} needs {self.config.duration(gap)}" for gap in too_long)
                + f", but the series spans {span}"
            )
        return tuple(sorted(active, key=self.config.duration, reverse=True))

    def _quotas(
        self,
        axis: TimeAxis,
        classes: Sequence[GapClass],
        target_records: int,
    ) -> tuple[dict[GapClass, float], list[str]]:
        """Return each class's quota on the configured basis, with any warnings (A3).

        Under ``missing_records`` a quota is a number of withheld observations and
        the 20/30/50 shares apply directly. Under ``gap_events`` the shares apply
        to a count of intervals, which the paper's prose does not give: it is
        derived from the same 25% budget by dividing it by the records one average
        event withholds, ``sum(share * rows_per_event)``. That derivation is this
        package's, and it is recorded in the manifest alongside the basis.
        """
        warnings: list[str] = []
        config = self.config
        if config.basis is AllocationBasis.MISSING_RECORDS:
            return {gap: config.share(gap) * target_records for gap in classes}, warnings

        per_event = {
            gap: duration_to_periods(
                config.duration(gap), axis.time_step, field_name=f"durations[{gap.value!r}]"
            )
            for gap in classes
        }
        mean_records = sum(config.share(gap) * per_event[gap] for gap in classes)
        n_events = round(target_records / mean_records) if mean_records > 0 else 0
        if n_events < len(classes):
            warnings.append(
                f"allocation_basis='gap_events' implies only {n_events} event(s) for "
                f"{len(classes)} class(es) at this series length; each class is given at "
                "least one gap, so the achieved fraction will exceed the requested one"
            )
        counts = _largest_remainder(
            {gap: config.share(gap) for gap in classes}, total=max(n_events, len(classes))
        )
        return {gap: float(max(counts[gap], 1)) for gap in classes}, warnings

    # -- placement -----------------------------------------------------------

    def _place_class(
        self,
        gap_class: GapClass,
        *,
        axis: TimeAxis,
        available: pd.Series,
        cumulative: np.ndarray,
        placed: Sequence[_Placed],
        quota: float,
        rng: np.random.Generator,
    ) -> tuple[list[_Placed], str | None]:
        """Place gaps of one class until its quota is met or attempts run out."""
        config = self.config
        duration = config.duration(gap_class)
        n_expected = duration_to_periods(
            duration, axis.time_step, field_name=f"durations[{gap_class.value!r}]"
        )
        index = axis.index
        latest_start = axis.end + pd.Timedelta(axis.time_step) - pd.Timedelta(duration)

        # Candidate starts are observed rows: an interval that begins on a real
        # measurement is far likelier to clear the observed-fraction test than one
        # beginning inside an existing data gap, and the draw stays reproducible.
        eligible = np.flatnonzero(available.to_numpy() & (index <= latest_start))
        if eligible.size == 0:
            return [], (
                f"no position on this series can hold a {gap_class.value} gap of {duration}: "
                f"it needs to start on or before {latest_start}"
            )

        accepted: list[_Placed] = []
        taken = 0.0
        attempts = 0
        exhausted = False
        while taken < quota:
            if attempts >= config.max_attempts_per_gap:
                exhausted = True
                break
            attempts += 1
            position = int(eligible[rng.integers(0, eligible.size)])
            start = pd.Timestamp(index[position])
            end = start + pd.Timedelta(duration)

            if not config.allow_overlap and _overlaps(start, end, placed, accepted):
                continue
            stop = int(index.searchsorted(end, side="left"))
            n_observed = int(cumulative[stop] - cumulative[position])
            if n_observed / n_expected < config.min_observed_fraction:
                continue

            # A class's quota rarely divides evenly into whole gaps, so the last
            # one either overshoots or is left out. Take whichever lands closer to
            # the quota - but always take at least one gap, or a class whose quota
            # is smaller than a single gap would never appear at all.
            contribution = n_observed if config.basis is AllocationBasis.MISSING_RECORDS else 1.0
            if accepted and abs(taken + contribution - quota) > abs(taken - quota):
                break

            accepted.append(
                _Placed(
                    gap_class=gap_class,
                    start=start,
                    end=end,
                    duration=duration,
                    n_expected=n_expected,
                    n_rows=stop - position,
                    n_observed=n_observed,
                )
            )
            taken += contribution
            attempts = 0

        # Stopping one gap short of the quota because the next one would overshoot
        # it further is the design working, not failing: only exhausted attempts
        # mean the series could not carry what was asked of it.
        if not exhausted:
            return accepted, None
        unit = "observation" if config.basis is AllocationBasis.MISSING_RECORDS else "gap"
        return accepted, (
            f"{gap_class.value} gaps fell short: {taken:.0f} of {quota:.0f} {unit}(s) placed "
            f"after {config.max_attempts_per_gap} consecutive rejected intervals. The series "
            f"may not hold enough stretches with {config.min_observed_fraction:.0%} genuine "
            f"measurements over {duration}."
        )

    # -- assembly ------------------------------------------------------------

    def _assemble(
        self,
        placed: Sequence[_Placed],
        *,
        axis: TimeAxis,
        available: pd.Series,
        warnings: tuple[str, ...],
    ) -> ArtificialGaps:
        """Turn accepted intervals into a manifest, masks and achieved counts."""
        index = axis.index
        ordered = sorted(placed, key=lambda gap: (gap.start, gap.end))
        ids = list(range(1, len(ordered) + 1))

        manifest = pd.DataFrame(
            {
                "gap_class": [gap.gap_class.value for gap in ordered],
                "start": [gap.start for gap in ordered],
                "end": [gap.end for gap in ordered],
                "duration": [pd.Timedelta(gap.duration) for gap in ordered],
                "n_expected": [gap.n_expected for gap in ordered],
                "n_rows": [gap.n_rows for gap in ordered],
                "n_observed_before_masking": [gap.n_observed for gap in ordered],
                "observed_fraction": [gap.observed_fraction for gap in ordered],
            },
            index=pd.Index(ids, name="gap_id"),
        ).loc[:, list(MANIFEST_COLUMNS)]

        # Built as arrays and wrapped once: assigning into a nullable Int64 or an
        # object Series row-slice by row-slice is both slower and easier to get
        # subtly wrong than filling plain arrays and constructing at the end.
        withheld = np.zeros(len(index), dtype=bool)
        identifiers = np.zeros(len(index), dtype=np.int64)
        classes: list[str | None] = [None] * len(index)
        records = {gap: 0 for gap in GapClass}
        events = {gap: 0 for gap in GapClass}

        observed = available.to_numpy()
        for identifier, gap in zip(ids, ordered, strict=True):
            lo = int(index.searchsorted(gap.start, side="left"))
            hi = int(index.searchsorted(gap.end, side="left"))
            withheld[lo:hi] = True
            identifiers[lo:hi] = identifier
            classes[lo:hi] = [gap.gap_class.value] * (hi - lo)
            records[gap.gap_class] += int(np.sum(observed[lo:hi]))
            events[gap.gap_class] += 1

        mask = pd.Series(withheld, index=index, name="artificial_gap")
        gap_id = pd.Series(pd.array(identifiers, dtype="Int64"), index=index, name="gap_id").where(
            mask
        )
        gap_class = pd.Series(classes, index=index, dtype=object, name="gap_class")

        return ArtificialGaps(
            manifest=manifest,
            mask=mask,
            gap_id=gap_id,
            gap_class=gap_class,
            n_available=int(observed.sum()),
            n_withheld=int(np.sum(mask.to_numpy() & observed)),
            records_by_class=MappingProxyType(records),
            events_by_class=MappingProxyType(events),
            config=self.config,
            random_state=self.random_state,
            warnings=warnings,
        )


def generate_artificial_gaps(
    axis: TimeAxis,
    observed: object,
    *,
    config: GapScenarioConfig | None = None,
    random_state: int = 42,
) -> ArtificialGaps:
    """Generate one artificial-gap scenario. A shorthand for :class:`GapScenarioGenerator`."""
    return GapScenarioGenerator(config, random_state=random_state).generate(axis, observed)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _as_mask(values: object, *, index: pd.DatetimeIndex) -> pd.Series:
    """Return ``values`` as a boolean Series aligned to ``index``."""
    if isinstance(values, pd.Series):
        if not values.index.equals(index):
            raise GapError(
                "the observation mask must be indexed by the same timestamps as the time "
                f"axis ({len(values)} vs {len(index)} rows)"
            )
        array = values.to_numpy()
    else:
        array = np.asarray(values)
        if array.ndim != 1 or array.size != len(index):
            raise GapError(
                f"the observation mask must be 1-D of length {len(index)}, got shape {array.shape}"
            )
    if array.dtype != bool:
        array = pd.Series(array).fillna(False).astype(bool).to_numpy()
    mask: pd.Series = pd.Series(array, index=index, name="observed")
    return mask


def _overlaps(
    start: pd.Timestamp,
    end: pd.Timestamp,
    *groups: Sequence[_Placed],
) -> bool:
    """Return whether ``[start, end)`` intersects any already-placed interval.

    Compared as time intervals, not as row sets: two gaps separated by nothing but
    a stretch of missing rows share no rows yet still overlap in time, and letting
    that through would make the manifest's durations misleading.
    """
    return any(gap.start < end and start < gap.end for group in groups for gap in group)


def _largest_remainder(shares: Mapping[GapClass, float], *, total: int) -> dict[GapClass, int]:
    """Apportion ``total`` across ``shares`` so the parts sum to it exactly.

    Plain rounding of each share can miss the total by a unit or two; the largest
    remainder method gives the leftovers to the classes rounded down hardest, so
    the achieved event counts are the closest integers to the requested mix.
    """
    exact = {gap: share * total for gap, share in shares.items()}
    counts = {gap: int(value) for gap, value in exact.items()}
    remaining = total - sum(counts.values())
    order = sorted(exact, key=lambda gap: (exact[gap] - counts[gap], shares[gap]), reverse=True)
    for gap in order[:remaining]:
        counts[gap] += 1
    return counts
