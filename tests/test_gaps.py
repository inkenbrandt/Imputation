"""Artificial-gap generator tests. Covers acceptance tests 15-21.

The allocation tests are the substance of ambiguity A3: both readings of the
paper's 20/30/50 are checked against hand-calculated event counts, so a change to
either algorithm has to be a deliberate one. Placement tests use short custom
durations on small series wherever the paper's own 24 h / 7 d / 30 d would need
years of half-hourly rows; the duration tests use the real ones.

See ``docs/method_spec.md`` section 4 and ambiguities A3 and A7.
"""

from __future__ import annotations

import warnings
from datetime import timedelta
from itertools import pairwise

import numpy as np
import pandas as pd
import pytest

from rfrgapfill.config import AllocationBasis, GapScenarioConfig, RFRConfig, ValidationConfig
from rfrgapfill.gaps import (
    GAP_MANIFEST_COLUMNS,
    ArtificialGap,
    GapError,
    GapManifest,
    GapScenarioGenerator,
    GapScenarioWarning,
    allocate_gaps,
)
from rfrgapfill.schema import ConfigError, GapClass

HALF_HOURLY = "30min"

#: One day, one week and one month of half-hourly records.
DAY_RECORDS = 48
WEEK_RECORDS = 7 * 48
MONTH_RECORDS = 30 * 48

#: Durations short enough to place many of them in a small test series, keeping
#: the same 1:7:30 ratio the paper's classes have.
SMALL_DURATIONS = {"short": "2h", "long": "14h", "very_long": "60h"}


def site_frame(
    days: int = 400,
    *,
    start: str = "2018-01-01",
    freq: str = HALF_HOURLY,
    missing: float = 0.0,
    seed: int = 0,
) -> pd.DataFrame:
    """A half-hourly frame with a QC column and, optionally, real gaps in the target."""
    periods = int(days * pd.Timedelta("1D") / pd.Timedelta(freq))
    index = pd.date_range(start, periods=periods, freq=freq)
    values = pd.Series(np.sin(np.arange(len(index)) / 48.0), index=index)
    if missing:
        rng = np.random.default_rng(seed)
        values[rng.random(len(index)) < missing] = np.nan
    return pd.DataFrame({"LE": values, "H": values * 2.0, "LE_QC": 0.0, "H_QC": 0.0}, index=index)


def generate(
    frame: pd.DataFrame,
    scenario: GapScenarioConfig | None = None,
    **kwargs: object,
) -> GapManifest:
    """Generate a manifest, ignoring the tolerance warnings a test is not about."""
    generator = GapScenarioGenerator(scenario, random_state=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", GapScenarioWarning)
        return generator.generate(frame, target="LE", qc_column="LE_QC", **kwargs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Allocation: the 20/30/50 ambiguity resolved explicitly (A3, Step 10)
# ---------------------------------------------------------------------------


def test_missing_records_basis_apportions_withheld_records_not_events() -> None:
    """20/30/50 of 25% of 40 000 observations, hand-calculated per class.

    ``T = round(0.25 * 40000) = 10000``. Short gets ``0.2 * 10000 = 2000``
    records at 48 per event (41.7 -> 42); long ``3000`` at 336 (8.9 -> 9);
    very_long ``5000`` at 1440 (3.5 -> 3).
    """
    allocation = allocate_gaps(
        GapScenarioConfig(allocation_basis="missing_records"),
        n_available=40_000,
        time_step=HALF_HOURLY,
    )

    assert allocation.basis is AllocationBasis.MISSING_RECORDS
    assert allocation.target_records == 10_000
    assert dict(allocation.events) == {
        GapClass.SHORT: 42,
        GapClass.LONG: 9,
        GapClass.VERY_LONG: 3,
    }
    assert dict(allocation.expected_records_per_event) == {
        GapClass.SHORT: DAY_RECORDS,
        GapClass.LONG: WEEK_RECORDS,
        GapClass.VERY_LONG: MONTH_RECORDS,
    }
    assert allocation.total_planned_records == 42 * 48 + 9 * 336 + 3 * 1440
    assert allocation.planned_fraction == pytest.approx(allocation.total_planned_records / 40_000)


def test_gap_events_basis_apportions_the_number_of_gaps() -> None:
    """The same request read as event shares gives a completely different design.

    Mean event size ``M = 0.2*48 + 0.3*336 + 0.5*1440 = 830.4``, so
    ``N = round(10000 / 830.4) = 12`` events, apportioned 2.4 / 3.6 / 6.0 ->
    floors 2 / 3 / 6 with one left over, which the largest remainder (0.6, long)
    takes.
    """
    allocation = allocate_gaps(
        GapScenarioConfig(allocation_basis="gap_events"),
        n_available=40_000,
        time_step=HALF_HOURLY,
    )

    assert allocation.basis is AllocationBasis.GAP_EVENTS
    assert allocation.total_events == 12
    assert dict(allocation.events) == {
        GapClass.SHORT: 2,
        GapClass.LONG: 4,
        GapClass.VERY_LONG: 6,
    }


def test_the_two_bases_disagree_by_far_more_than_a_rounding_step() -> None:
    """The ambiguity is worth exposing: the designs are not near each other."""
    records = allocate_gaps(
        GapScenarioConfig(allocation_basis="missing_records"),
        n_available=40_000,
        time_step=HALF_HOURLY,
    )
    events = allocate_gaps(
        GapScenarioConfig(allocation_basis="gap_events"),
        n_available=40_000,
        time_step=HALF_HOURLY,
    )

    assert records.events[GapClass.SHORT] == 42
    assert events.events[GapClass.SHORT] == 2
    # Under gap_events, half the gaps are 30-day ones, so the withheld records
    # are dominated by the very_long class rather than split 20/30/50.
    planned = events.planned_records
    very_long_share = planned[GapClass.VERY_LONG] / events.total_planned_records
    assert very_long_share > 0.80


def test_largest_remainder_makes_the_event_counts_sum_exactly() -> None:
    """No event is lost or invented by rounding on the gap_events basis."""
    for n_available in range(1_000, 60_000, 3_137):
        allocation = allocate_gaps(
            GapScenarioConfig(allocation_basis="gap_events"),
            n_available=n_available,
            time_step=HALF_HOURLY,
        )
        assert sum(allocation.events.values()) == allocation.total_events


def test_allocation_is_a_pure_function_of_configuration_and_size() -> None:
    first = allocate_gaps(GapScenarioConfig(), n_available=12_345, time_step=HALF_HOURLY)
    second = allocate_gaps(GapScenarioConfig(), n_available=12_345, time_step=HALF_HOURLY)
    assert dict(first.events) == dict(second.events)
    assert first.to_dict() == second.to_dict()


def test_allocation_records_the_basis_it_used() -> None:
    allocation = allocate_gaps(
        GapScenarioConfig(allocation_basis="gap_events"),
        n_available=40_000,
        time_step=HALF_HOURLY,
    )
    assert allocation.to_dict()["allocation_basis"] == "gap_events"
    assert "gap_events" in allocation.summary()


def test_a_class_longer_than_the_series_is_allocated_nothing_and_says_so() -> None:
    """Step 10: warn when dataset length prevents the requested design."""
    allocation = allocate_gaps(
        GapScenarioConfig(),
        n_available=2_000,
        time_step=HALF_HOURLY,
        span=timedelta(days=10),
    )

    assert allocation.events[GapClass.VERY_LONG] == 0
    assert any(
        "very_long" in note and "longer than the series" in note for note in allocation.notes
    )


def test_a_class_too_small_a_share_to_earn_a_gap_says_so() -> None:
    """A short series cannot afford a 30-day gap even though it would fit."""
    allocation = allocate_gaps(
        GapScenarioConfig(),
        n_available=2_000,
        time_step=HALF_HOURLY,
        span=timedelta(days=60),
    )

    assert allocation.events[GapClass.VERY_LONG] == 0
    assert any("very_long" in note and "allocated no gap" in note for note in allocation.notes)


def test_a_design_that_needs_more_rows_than_the_series_holds_says_so() -> None:
    allocation = allocate_gaps(
        GapScenarioConfig(missing_fraction=0.9),
        n_available=40_000,
        time_step=HALF_HOURLY,
        n_rows=1_000,
        span=timedelta(days=400),
    )
    assert any("non-overlapping" in note for note in allocation.notes)


def test_a_duration_that_is_not_a_whole_number_of_steps_is_rejected() -> None:
    from rfrgapfill.time import TimestampError

    with pytest.raises(TimestampError, match="durations"):
        allocate_gaps(
            GapScenarioConfig(durations={"short": "45min"}),
            n_available=1_000,
            time_step=HALF_HOURLY,
        )


@pytest.mark.parametrize("n_available", [-1, 2.5, True, "many"])
def test_a_nonsensical_available_count_is_rejected(n_available: object) -> None:
    with pytest.raises(ConfigError, match="n_available"):
        allocate_gaps(GapScenarioConfig(), n_available=n_available, time_step=HALF_HOURLY)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Durations are elapsed time (acceptance tests 15-17)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("gap_class", "expected"),
    [
        (GapClass.SHORT, timedelta(hours=24)),
        (GapClass.LONG, timedelta(days=7)),
        (GapClass.VERY_LONG, timedelta(days=30)),
    ],
)
def test_each_gap_class_spans_its_configured_elapsed_duration(
    gap_class: GapClass, expected: timedelta
) -> None:
    manifest = generate(site_frame(days=400))
    placed = manifest.by_class(gap_class)

    assert placed, f"no {gap_class.value} gap was placed"
    for gap in placed:
        assert gap.end - gap.start == pd.Timedelta(expected)
        assert gap.duration == expected


def test_durations_are_elapsed_time_and_not_row_counts() -> None:
    """A 24-hour gap is 24 hours at hourly cadence too - with half the rows."""
    manifest = generate(
        site_frame(days=400, freq="1h"),
        GapScenarioConfig(gap_mix={"short": 1.0}),
    )

    assert manifest.gaps
    for gap in manifest.gaps:
        assert gap.duration == timedelta(hours=24)
        assert gap.n_expected == 24


def test_a_gap_ends_before_the_row_that_closes_it() -> None:
    """Intervals are half-open, so consecutive gaps could tile without sharing a row."""
    manifest = generate(site_frame(days=400))
    frame = site_frame(days=400)

    for gap in manifest.gaps:
        held = manifest.mask(frame.index)
        assert held.loc[gap.start]
        if gap.end in frame.index:
            assert not held.loc[gap.end] or any(
                other.start <= gap.end < other.end for other in manifest.gaps if other is not gap
            )


# ---------------------------------------------------------------------------
# Reproducibility (acceptance test 18)
# ---------------------------------------------------------------------------


def test_the_same_seed_gives_the_same_gaps() -> None:
    frame = site_frame(days=400)
    first = generate(frame)
    second = generate(frame)

    assert first.intervals() == second.intervals()
    assert first.to_dict() == second.to_dict()


def test_a_different_seed_gives_different_gaps() -> None:
    frame = site_frame(days=400)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", GapScenarioWarning)
        first = GapScenarioGenerator(random_state=1).generate(frame, target="LE")
        second = GapScenarioGenerator(random_state=2).generate(frame, target="LE")

    assert first.intervals() != second.intervals()


def test_the_seed_is_taken_from_the_run_configuration() -> None:
    config = RFRConfig(mode="RFR3", hemisphere="north", random_state=7)
    assert GapScenarioGenerator(config).random_state == 7
    assert GapScenarioGenerator(config, random_state=9).random_state == 9
    assert GapScenarioGenerator().random_state == 42


def test_the_scenario_is_read_from_whichever_configuration_object_carries_it() -> None:
    scenario = GapScenarioConfig(allocation_basis="gap_events")
    assert GapScenarioGenerator(scenario).scenario is scenario
    assert GapScenarioGenerator(ValidationConfig(gaps=scenario)).scenario is scenario
    config = RFRConfig(mode="RFR3", hemisphere="north", validation=ValidationConfig(gaps=scenario))
    assert GapScenarioGenerator(config).scenario is scenario
    assert GapScenarioGenerator(config).config is config


def test_an_unusable_configuration_object_is_rejected() -> None:
    with pytest.raises(ConfigError, match="config must be"):
        GapScenarioGenerator("RFR3")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# The observed-fraction rule (acceptance test 19)
# ---------------------------------------------------------------------------


def test_an_interval_over_a_real_gap_is_rejected() -> None:
    """Half the series is genuinely missing; no accepted gap may sit on it."""
    frame = site_frame(days=200, freq="1h")
    holed = frame.copy()
    holed.loc["2018-03-01":"2018-05-01", "LE"] = np.nan

    manifest = generate(
        holed,
        GapScenarioConfig(durations=SMALL_DURATIONS, min_observed_fraction=0.5),
    )

    assert manifest.gaps
    for gap in manifest.gaps:
        assert gap.observed_fraction >= 0.5
        assert not (pd.Timestamp("2018-03-02") <= gap.start <= pd.Timestamp("2018-04-28"))


def test_the_observed_fraction_is_measured_against_a_complete_grid() -> None:
    """An interval half-covered by real gaps must not look fully observed."""
    frame = site_frame(days=200, freq="1h", missing=0.4, seed=3)
    manifest = generate(frame, GapScenarioConfig(durations=SMALL_DURATIONS))

    for gap in manifest.gaps:
        assert gap.observed_fraction == gap.n_observed_before_masking / gap.n_expected
        assert gap.n_observed_before_masking <= gap.n_rows <= gap.n_expected


def test_a_qc_flagged_prefilled_value_does_not_count_as_observed() -> None:
    """A value that arrived already gap-filled is not a genuine measurement."""
    frame = site_frame(days=60, freq="1h")
    frame.loc["2018-01-10":"2018-02-10", "LE_QC"] = 1.0

    unflagged = generate(frame.assign(LE_QC=0.0), GapScenarioConfig(durations=SMALL_DURATIONS))
    flagged = generate(frame, GapScenarioConfig(durations=SMALL_DURATIONS))

    assert flagged.n_available < unflagged.n_available
    for gap in flagged.gaps:
        assert not (pd.Timestamp("2018-01-11") <= gap.start <= pd.Timestamp("2018-02-08"))


def comb_frame(days: int = 60) -> pd.DataFrame:
    """Every second target value is missing, so no interval is ever fully observed."""
    frame = site_frame(days=days, freq="1h")
    frame.iloc[1::2, frame.columns.get_loc("LE")] = np.nan
    return frame


def test_an_impossible_observed_fraction_fails_loudly_rather_than_quietly() -> None:
    with pytest.raises(GapError, match="could not be constructed"):
        generate(
            comb_frame(),
            GapScenarioConfig(durations=SMALL_DURATIONS, min_observed_fraction=1.0),
        )


def test_a_shortfall_can_be_downgraded_to_a_warning_for_inspection() -> None:
    frame = comb_frame()
    scenario = GapScenarioConfig(durations=SMALL_DURATIONS, min_observed_fraction=1.0)

    with pytest.warns(GapScenarioWarning, match="could not be constructed"):
        manifest = GapScenarioGenerator(scenario, random_state=42).generate(
            frame, target="LE", on_shortfall="warn"
        )

    assert any("could not be constructed" in note for note in manifest.warnings)


def test_on_shortfall_only_accepts_the_two_documented_behaviours() -> None:
    with pytest.raises(ConfigError, match="on_shortfall"):
        GapScenarioGenerator().generate(site_frame(days=60), target="LE", on_shortfall="ignore")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Overlap and boundaries
# ---------------------------------------------------------------------------


def test_gaps_do_not_overlap_by_default() -> None:
    frame = site_frame(days=200, freq="1h")
    manifest = generate(frame, GapScenarioConfig(durations=SMALL_DURATIONS))

    ordered = sorted(manifest.gaps, key=lambda gap: gap.start)
    assert len(ordered) > 1
    for earlier, later in pairwise(ordered):
        assert earlier.end <= later.start


def test_every_gap_lies_inside_the_series() -> None:
    frame = site_frame(days=200, freq="1h")
    manifest = generate(frame, GapScenarioConfig(durations=SMALL_DURATIONS))
    step = pd.Timedelta("1h")

    for gap in manifest.gaps:
        assert gap.start >= frame.index[0]
        assert gap.end <= frame.index[-1] + step


def test_the_withheld_rows_are_exactly_the_rows_the_mask_marks() -> None:
    frame = site_frame(days=200, freq="1h", missing=0.1, seed=5)
    manifest = generate(frame, GapScenarioConfig(durations=SMALL_DURATIONS))
    held = manifest.mask(frame.index)

    assert int(held.sum()) == manifest.n_withheld_rows
    observed = frame["LE"].notna() & frame["LE_QC"].eq(0.0)
    assert int((held & observed).sum()) == manifest.n_withheld_observed


# ---------------------------------------------------------------------------
# Shared gaps across targets (acceptance test 20)
# ---------------------------------------------------------------------------


def test_joint_validation_uses_one_set_of_gap_locations_for_every_target() -> None:
    frame = site_frame(days=200, freq="1h")
    scenario = GapScenarioConfig(durations=SMALL_DURATIONS)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", GapScenarioWarning)
        manifest = GapScenarioGenerator(scenario, random_state=42).generate(
            frame, target=["LE", "H"], qc_column={"LE": "LE_QC", "H": "H_QC"}
        )

    assert manifest.targets == ("LE", "H")
    mask = manifest.mask(frame.index)
    # The paper's requirement: the identical mask scores every target.
    for target in ("LE", "H"):
        assert mask.equals(manifest.mask(frame[target].index))


def test_a_row_is_available_only_where_every_joint_target_is_observed() -> None:
    frame = site_frame(days=200, freq="1h")
    frame.loc["2018-02-01":"2018-03-01", "H"] = np.nan

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", GapScenarioWarning)
        joint = GapScenarioGenerator(
            GapScenarioConfig(durations=SMALL_DURATIONS), random_state=42
        ).generate(frame, target=["LE", "H"])
        single = GapScenarioGenerator(
            GapScenarioConfig(durations=SMALL_DURATIONS), random_state=42
        ).generate(frame, target="LE")

    assert joint.n_available < single.n_available


def test_several_targets_need_the_shared_design_the_paper_used() -> None:
    scenario = GapScenarioConfig(shared_gaps_across_targets=False)
    with pytest.raises(ConfigError, match="shared_gaps_across_targets"):
        GapScenarioGenerator(scenario).generate(site_frame(days=60), target=["LE", "H"])


def test_one_qc_column_cannot_stand_for_several_targets() -> None:
    with pytest.raises(ConfigError, match="mapping from target to QC column"):
        GapScenarioGenerator().generate(site_frame(days=60), target=["LE", "H"], qc_column="LE_QC")


def test_a_target_named_twice_is_rejected() -> None:
    with pytest.raises(ConfigError, match="more than once"):
        GapScenarioGenerator().generate(site_frame(days=60), target=["LE", "LE"])


# ---------------------------------------------------------------------------
# Achieved fraction and mix within tolerance (acceptance test 21)
# ---------------------------------------------------------------------------


def test_the_achieved_fraction_and_mix_are_within_tolerance_on_a_complete_series() -> None:
    """The paper's own scenario on a four-year half-hourly series with no real gaps."""
    frame = site_frame(days=4 * 365)
    manifest = generate(frame)

    assert manifest.achieved_fraction == pytest.approx(0.25, abs=0.05)
    assert manifest.within_fraction_tolerance
    for gap_class in GapClass:
        assert manifest.record_shares[gap_class] == pytest.approx(
            manifest.scenario.share(gap_class), abs=0.05
        )
    assert manifest.within_mix_tolerance
    assert manifest.is_within_tolerance


def test_the_achieved_mix_is_measured_on_the_requested_basis() -> None:
    frame = site_frame(days=4 * 365)
    records = generate(frame, GapScenarioConfig(allocation_basis="missing_records"))
    events = generate(frame, GapScenarioConfig(allocation_basis="gap_events"))

    assert records.achieved_shares == records.record_shares
    assert events.achieved_shares == events.event_shares
    # Both readings are always reported, whichever was requested.
    assert set(records.to_dict()) >= {"record_shares", "event_shares", "achieved_shares"}


def test_a_scenario_outside_tolerance_warns_rather_than_passing_quietly() -> None:
    """Real gaps eat into every interval, so the achieved fraction falls short."""
    frame = site_frame(days=2 * 365, missing=0.5, seed=11)
    scenario = GapScenarioConfig(fraction_tolerance=0.01, mix_tolerance=0.01)

    with pytest.warns(GapScenarioWarning):
        manifest = GapScenarioGenerator(scenario, random_state=42).generate(
            frame, target="LE", qc_column="LE_QC"
        )

    assert not manifest.is_within_tolerance
    assert manifest.warnings


def test_the_fraction_is_measured_against_observations_not_rows() -> None:
    """Withholding a value that was never measured withholds nothing."""
    frame = site_frame(days=2 * 365, missing=0.3, seed=13)
    manifest = generate(frame)

    observed = int((frame["LE"].notna() & frame["LE_QC"].eq(0.0)).sum())
    assert manifest.n_available == observed
    assert manifest.achieved_fraction == manifest.n_withheld_observed / observed
    assert manifest.n_withheld_observed < manifest.n_withheld_rows


# ---------------------------------------------------------------------------
# The manifest as a report (Step 10 exit criterion)
# ---------------------------------------------------------------------------


def test_the_manifest_states_how_the_mixture_was_constructed() -> None:
    manifest = generate(site_frame(days=4 * 365))
    summary = manifest.summary()

    assert "allocation_basis='missing_records'" in summary
    assert "requested 25.0%" in summary
    for gap_class in GapClass:
        assert gap_class.value in summary


def test_the_manifest_dictionary_is_json_serialisable_and_complete() -> None:
    import json

    payload = generate(site_frame(days=4 * 365)).to_dict()
    json.dumps(payload)

    assert payload["allocation"]["allocation_basis"] == "missing_records"
    assert payload["scenario"]["allocation_basis"] == "missing_records"
    assert payload["n_withheld_observed"] > 0
    assert len(payload["gaps"]) == payload["allocation"]["total_events"]
    assert set(payload["gaps"][0]) == set(GAP_MANIFEST_COLUMNS)


def test_the_manifest_table_carries_the_documented_columns() -> None:
    frame_out = generate(site_frame(days=4 * 365)).to_frame()

    assert list(frame_out.columns) == list(GAP_MANIFEST_COLUMNS)
    assert frame_out["start"].is_monotonic_increasing
    assert (frame_out["observed_fraction"] >= 0.5).all()


def test_gap_ids_are_unique_and_numbered_in_time_order() -> None:
    manifest = generate(site_frame(days=4 * 365))
    ids = [gap.gap_id for gap in manifest.gaps]

    assert len(set(ids)) == len(ids)
    for gap_class in GapClass:
        placed = manifest.by_class(gap_class)
        expected = [f"{gap_class.value}-{n + 1:02d}" for n in range(len(placed))]
        assert [gap.gap_id for gap in placed] == expected


def test_the_manifest_is_iterable_and_groups_by_class() -> None:
    manifest = generate(site_frame(days=4 * 365))

    assert len(manifest) == len(manifest.gaps)
    assert list(manifest) == list(manifest.gaps)
    regrouped = [gap for cls in GapClass for gap in manifest.by_class(cls)]
    assert sorted(gap.gap_id for gap in regrouped) == sorted(gap.gap_id for gap in manifest)


def test_the_manifest_survives_a_pickle_round_trip() -> None:
    import pickle

    manifest = generate(site_frame(days=4 * 365))
    restored = pickle.loads(pickle.dumps(manifest))

    assert restored.to_dict() == manifest.to_dict()
    assert restored.scenario.basis is manifest.scenario.basis


# ---------------------------------------------------------------------------
# Availability inputs
# ---------------------------------------------------------------------------


def test_an_explicit_availability_mask_may_stand_in_for_a_target() -> None:
    frame = site_frame(days=200, freq="1h")
    observed = pd.Series(True, index=frame.index)
    observed.loc["2018-02-01":"2018-04-01"] = False

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", GapScenarioWarning)
        manifest = GapScenarioGenerator(
            GapScenarioConfig(durations=SMALL_DURATIONS), random_state=42
        ).generate(frame, observed=observed)

    assert manifest.n_available == int(observed.sum())
    assert manifest.targets == ()


def test_a_target_and_an_explicit_mask_cannot_both_be_given() -> None:
    frame = site_frame(days=60)
    with pytest.raises(ConfigError, match="not both"):
        GapScenarioGenerator().generate(
            frame, target="LE", observed=pd.Series(True, index=frame.index)
        )


def test_a_mask_that_does_not_match_the_timestamps_is_rejected() -> None:
    frame = site_frame(days=60)
    with pytest.raises(ConfigError, match="same timestamps"):
        GapScenarioGenerator().generate(frame, observed=pd.Series(True, index=range(len(frame))))


def test_without_a_target_the_fraction_is_measured_against_rows_and_says_so() -> None:
    frame = site_frame(days=200, freq="1h")
    scenario = GapScenarioConfig(durations=SMALL_DURATIONS)

    with pytest.warns(GapScenarioWarning, match="every timestamp counts as available"):
        manifest = GapScenarioGenerator(scenario, random_state=42).generate(frame)

    assert manifest.n_available == len(frame)


def test_gaps_can_be_placed_over_a_bare_timestamp_index() -> None:
    index = pd.date_range("2018-01-01", periods=200 * 24, freq="1h")
    scenario = GapScenarioConfig(durations=SMALL_DURATIONS)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", GapScenarioWarning)
        manifest = GapScenarioGenerator(scenario, random_state=42).generate(index)

    assert manifest.gaps
    assert manifest.n_available == len(index)


def test_a_target_needs_a_frame_to_read_it_from() -> None:
    index = pd.date_range("2018-01-01", periods=100, freq="1h")
    with pytest.raises(ConfigError, match="must be a DataFrame"):
        GapScenarioGenerator().generate(index, target="LE")


# ---------------------------------------------------------------------------
# ArtificialGap
# ---------------------------------------------------------------------------


def test_an_artificial_gap_reports_its_own_arithmetic() -> None:
    gap = ArtificialGap(
        gap_id="short-01",
        gap_class="short",
        start=pd.Timestamp("2020-01-01"),
        end=pd.Timestamp("2020-01-02"),
        n_expected=48,
        n_rows=40,
        n_observed_before_masking=36,
    )

    assert gap.gap_class is GapClass.SHORT
    assert gap.duration == timedelta(days=1)
    assert gap.observed_fraction == pytest.approx(36 / 48)
    assert gap.interval() == (pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02"))
    assert gap.to_dict()["duration"] == "P1DT0H0M0S"
