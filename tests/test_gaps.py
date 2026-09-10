"""Artificial-gap generator tests. Covers acceptance tests 15-21.

The durations are checked as elapsed time rather than row counts, because that
is the whole point of the design: a 24-hour gap spans 24 hours whether or not
the series holds 48 rows there (``docs/method_spec.md`` 4.1).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rfrgapfill.config import AllocationBasis, GapClass, GapScenarioConfig
from rfrgapfill.gaps import MANIFEST_COLUMNS, GapError, GapScenarioGenerator
from rfrgapfill.time import prepare_time_index


def axis_over(days: int = 400, *, freq: str = "30min") -> object:
    index = pd.date_range("2019-01-01", periods=days * 48, freq=freq, name="timestamp")
    frame = pd.DataFrame({"NEE": np.zeros(len(index))}, index=index)
    _, axis = prepare_time_index(frame, frequency=freq)
    return axis


def observed_everywhere(axis: object) -> pd.Series:
    return pd.Series(True, index=axis.index)  # type: ignore[attr-defined]


def observed_with_holes(axis: object, *, fraction: float = 0.12, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    index = axis.index  # type: ignore[attr-defined]
    return pd.Series(rng.random(len(index)) >= fraction, index=index)


# ---------------------------------------------------------------------------
# Durations are elapsed time (acceptance tests 15-17)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("gap_class", "expected"),
    [
        (GapClass.SHORT, pd.Timedelta(hours=24)),
        (GapClass.LONG, pd.Timedelta(days=7)),
        (GapClass.VERY_LONG, pd.Timedelta(days=30)),
    ],
)
def test_each_gap_class_spans_its_nominal_duration(
    gap_class: GapClass, expected: pd.Timedelta
) -> None:
    axis = axis_over()
    config = GapScenarioConfig(gap_mix={gap_class.value: 1.0})
    gaps = GapScenarioGenerator(config, random_state=1).generate(axis, observed_everywhere(axis))

    assert len(gaps.manifest) > 0
    spans = gaps.manifest["end"] - gaps.manifest["start"]
    assert (spans == expected).all()


def test_a_gap_spans_its_duration_even_where_rows_are_missing() -> None:
    # Drop every row of one afternoon, then require a 24-hour gap to start there.
    index = pd.date_range("2019-01-01", periods=400 * 48, freq="30min", name="timestamp")
    frame = pd.DataFrame({"NEE": np.zeros(len(index))}, index=index)
    hole = (frame.index >= "2019-03-01 12:00") & (frame.index < "2019-03-02 00:00")
    frame = frame.loc[~hole]
    _, axis = prepare_time_index(frame, frequency="30min")

    config = GapScenarioConfig(gap_mix={"short": 1.0})
    gaps = GapScenarioGenerator(config, random_state=3).generate(axis, observed_everywhere(axis))

    assert (gaps.manifest["end"] - gaps.manifest["start"] == pd.Timedelta(hours=24)).all()
    # n_expected comes from the cadence; n_rows from what the series actually has.
    assert (gaps.manifest["n_expected"] == 48).all()
    assert (gaps.manifest["n_rows"] <= gaps.manifest["n_expected"]).all()


def test_hourly_data_gets_the_same_durations_with_half_the_rows() -> None:
    axis = axis_over(days=200, freq="1h")
    config = GapScenarioConfig(gap_mix={"short": 1.0})
    gaps = GapScenarioGenerator(config, random_state=1).generate(axis, observed_everywhere(axis))

    assert (gaps.manifest["end"] - gaps.manifest["start"] == pd.Timedelta(hours=24)).all()
    assert (gaps.manifest["n_expected"] == 24).all()


# ---------------------------------------------------------------------------
# Reproducibility (acceptance test 18)
# ---------------------------------------------------------------------------


def test_the_same_seed_reproduces_the_same_gaps() -> None:
    axis = axis_over()
    observed = observed_with_holes(axis)
    first = GapScenarioGenerator(random_state=42).generate(axis, observed)
    second = GapScenarioGenerator(random_state=42).generate(axis, observed)

    pd.testing.assert_frame_equal(first.manifest, second.manifest)
    pd.testing.assert_series_equal(first.mask, second.mask)


def test_a_different_seed_gives_different_gaps() -> None:
    axis = axis_over()
    observed = observed_with_holes(axis)
    first = GapScenarioGenerator(random_state=42).generate(axis, observed)
    second = GapScenarioGenerator(random_state=7).generate(axis, observed)

    assert not first.manifest.equals(second.manifest)


# ---------------------------------------------------------------------------
# Rejection rules (acceptance test 19)
# ---------------------------------------------------------------------------


def test_every_accepted_gap_clears_the_observed_fraction_threshold() -> None:
    axis = axis_over()
    observed = observed_with_holes(axis, fraction=0.3)
    config = GapScenarioConfig(min_observed_fraction=0.60)
    gaps = GapScenarioGenerator(config, random_state=5).generate(axis, observed)

    assert (gaps.manifest["observed_fraction"] >= 0.60).all()


def test_a_series_too_sparse_for_the_threshold_reports_instead_of_pretending() -> None:
    axis = axis_over(days=120)
    # Only a fifth of the rows are observed, so no interval can be 90% measured.
    observed = observed_with_holes(axis, fraction=0.80, seed=1)
    config = GapScenarioConfig(min_observed_fraction=0.90, max_attempts_per_gap=20)
    gaps = GapScenarioGenerator(config, random_state=5).generate(axis, observed)

    assert not gaps.satisfied
    assert gaps.warnings
    assert len(gaps.manifest) == 0


def test_gaps_do_not_overlap_by_default() -> None:
    axis = axis_over()
    gaps = GapScenarioGenerator(random_state=11).generate(axis, observed_with_holes(axis))

    ordered = gaps.manifest.sort_values("start")
    assert (ordered["start"].to_numpy()[1:] >= ordered["end"].to_numpy()[:-1]).all()


def test_a_series_shorter_than_the_requested_gaps_is_rejected_outright() -> None:
    axis = axis_over(days=10)
    with pytest.raises(GapError, match="shorter than the gap"):
        GapScenarioGenerator(random_state=1).generate(axis, observed_everywhere(axis))


def test_an_empty_observation_mask_is_rejected_outright() -> None:
    axis = axis_over()
    with pytest.raises(GapError, match="no genuinely observed"):
        GapScenarioGenerator(random_state=1).generate(
            axis,
            pd.Series(False, index=axis.index),  # type: ignore[attr-defined]
        )


def test_a_misaligned_observation_mask_is_rejected() -> None:
    axis = axis_over()
    with pytest.raises(GapError, match="same timestamps"):
        GapScenarioGenerator(random_state=1).generate(axis, pd.Series([True, False]))


# ---------------------------------------------------------------------------
# Achieved design and reporting (acceptance tests 20-21)
# ---------------------------------------------------------------------------


def test_the_achieved_fraction_lands_near_the_requested_one() -> None:
    axis = axis_over(days=800)
    observed = observed_with_holes(axis)
    gaps = GapScenarioGenerator(random_state=42).generate(axis, observed)

    assert gaps.achieved_fraction == pytest.approx(0.25, abs=0.05)
    assert gaps.fraction_within_tolerance


def test_the_achieved_mix_lands_near_twenty_thirty_fifty() -> None:
    axis = axis_over(days=1200)
    observed = observed_with_holes(axis)
    gaps = GapScenarioGenerator(random_state=42).generate(axis, observed)

    achieved = gaps.achieved_mix
    assert achieved[GapClass.SHORT] == pytest.approx(0.20, abs=0.06)
    assert achieved[GapClass.LONG] == pytest.approx(0.30, abs=0.06)
    assert achieved[GapClass.VERY_LONG] == pytest.approx(0.50, abs=0.06)


def test_the_withheld_count_only_counts_genuine_observations() -> None:
    axis = axis_over()
    observed = observed_with_holes(axis, fraction=0.4)
    gaps = GapScenarioGenerator(random_state=9).generate(axis, observed)

    withheld = gaps.mask & observed
    assert gaps.n_withheld == int(withheld.sum())
    # Rows inside a gap that were never observed are masked but not withheld.
    assert int(gaps.mask.sum()) > gaps.n_withheld


def test_the_manifest_carries_every_documented_field() -> None:
    axis = axis_over()
    gaps = GapScenarioGenerator(random_state=42).generate(axis, observed_with_holes(axis))

    assert tuple(gaps.manifest.columns) == MANIFEST_COLUMNS
    assert gaps.manifest.index.name == "gap_id"
    assert gaps.manifest["start"].is_monotonic_increasing


def test_the_class_masks_partition_the_gap_mask() -> None:
    axis = axis_over()
    gaps = GapScenarioGenerator(random_state=42).generate(axis, observed_with_holes(axis))

    union = np.zeros(len(gaps.mask), dtype=bool)
    for gap_class in GapClass:
        union |= gaps.class_mask(gap_class).to_numpy()
    assert (union == gaps.mask.to_numpy()).all()


def test_a_scenario_that_misses_its_design_says_why() -> None:
    # Two 30-day gaps is a coarse quantum on a one-year series: the class either
    # overshoots half of a 25% budget or undershoots it. Whichever it does, the
    # reason has to be readable rather than left to the caller to work out.
    axis = axis_over(days=400)
    config = GapScenarioConfig(mix_tolerance=0.01, fraction_tolerance=0.01)
    gaps = GapScenarioGenerator(config, random_state=42).generate(axis, observed_with_holes(axis))

    assert not gaps.satisfied
    assert gaps.shortfalls
    assert any("tolerance" in reason for reason in gaps.shortfalls)
    assert gaps.to_dict()["shortfalls"] == list(gaps.shortfalls)


def test_a_satisfied_scenario_reports_no_shortfall() -> None:
    axis = axis_over(days=800)
    gaps = GapScenarioGenerator(random_state=42).generate(axis, observed_with_holes(axis))

    assert gaps.satisfied
    assert gaps.shortfalls == ()


def test_the_report_states_which_allocation_basis_was_used() -> None:
    axis = axis_over()
    gaps = GapScenarioGenerator(random_state=42).generate(axis, observed_with_holes(axis))
    manifest = gaps.to_dict()

    assert manifest["allocation_basis"] == "missing_records"
    assert manifest["requested_fraction"] == 0.25
    assert set(manifest["achieved_mix"]) == {"short", "long", "very_long"}


def test_the_gap_events_basis_allocates_events_rather_than_records() -> None:
    axis = axis_over(days=800)
    config = GapScenarioConfig(allocation_basis=AllocationBasis.GAP_EVENTS)
    gaps = GapScenarioGenerator(config, random_state=42).generate(axis, observed_with_holes(axis))

    events = gaps.events_by_class
    total = sum(events.values())
    assert total > 0
    # Twenty percent of events are short under this basis, against roughly two
    # percent of the withheld records - which is exactly why ambiguity A3 matters.
    assert events[GapClass.SHORT] / total == pytest.approx(0.20, abs=0.10)
    assert gaps.to_dict()["allocation_basis"] == "gap_events"


def test_the_whole_scenario_serialises_for_the_run_manifest() -> None:
    import json

    axis = axis_over()
    gaps = GapScenarioGenerator(random_state=42).generate(axis, observed_with_holes(axis))
    manifest = gaps.to_dict()

    assert json.loads(json.dumps(manifest)) == manifest
    assert len(manifest["gaps"]) == gaps.n_gaps
