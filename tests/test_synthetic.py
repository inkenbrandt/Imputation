"""Synthetic scientific fixtures. Covers Step 16 and acceptance tests 34-35.

The generator's value is that things can be asserted about it, so this module is
organised by claim rather than by function: the record is a year of half-hourly
data; the radiation has a diurnal cycle and the temperature a seasonal one; VPD
follows temperature and humidity exactly; soil temperature lags the air; the
fluxes are functions of those drivers with a known energy-balance closure; the
whole thing is reproducible; and the known gaps span 24 hours, 7 days and 30 days
of elapsed time.

Nothing here downloads anything, which is Step 16's exit criterion, and the
integration section at the end shows the fixture reaching the feature, leakage,
gap and model layers unchanged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rfrgapfill.config import FeatureConfig, GapScenarioConfig
from rfrgapfill.features import build_feature_matrix
from rfrgapfill.gaps import GapError, GapScenarioGenerator
from rfrgapfill.leakage import build_validation_features, require_no_target_leakage
from rfrgapfill.metrics import compare_energy_balance, core_metrics
from rfrgapfill.model import RFRModel
from rfrgapfill.schema import ConfigError, GapClass, Hemisphere
from rfrgapfill.synthetic import (
    DEFAULT_KNOWN_GAPS,
    MISSING_QC,
    OBSERVED_QC,
    PRE_FILLED_QC,
    TARGETS,
    KnownGap,
    known_gap_manifest,
    synthetic_site,
)
from rfrgapfill.time import prepare_time_index

HALF_HOURLY = "30min"
STEPS_PER_DAY = 48
DAYTIME_THRESHOLD = 20.0

DRIVER_COLUMNS = (
    "SW_IN_F",
    "VPD_F_MDS",
    "TA_F_MDS",
    "NETRAD",
    "WS",
    "WD",
    "G_F_MDS",
    "TS_F_MDS",
    "RH",
    "SWC_F_MDS",
)


@pytest.fixture(scope="module")
def site():
    """The reference site: one year at 45 degrees north, default seed."""
    return synthetic_site()


@pytest.fixture(scope="module")
def clean():
    """The same site with no measurement noise, no outages and no QC flags."""
    return synthetic_site(noise_scale=0.0, real_gap_fraction=0.0, pre_filled_fraction=0.0)


def hour_of_day(index: pd.DatetimeIndex) -> pd.Series:
    """Fractional hour of each timestamp, for grouping a diurnal cycle."""
    return pd.Series(index.hour + index.minute / 60.0, index=index)


def diurnal_amplitude(values: pd.Series) -> float:
    """Peak-to-trough range of the mean diurnal cycle of ``values``."""
    cycle = values.groupby(hour_of_day(values.index)).mean()
    return float(cycle.max() - cycle.min())


# ---------------------------------------------------------------------------
# The record itself
# ---------------------------------------------------------------------------


def test_default_site_is_one_year_of_half_hourly_data(site):
    index = site.frame.index
    assert len(site.frame) == 365 * STEPS_PER_DAY == 17520
    assert index[0] == pd.Timestamp("2018-01-01 00:00")
    assert index[-1] == pd.Timestamp("2018-12-31 23:30")
    assert site.time_step == pd.Timedelta(HALF_HOURLY).to_pytimedelta()
    assert site.index.equals(index)


def test_time_axis_is_valid_for_the_package(site):
    """The frame passes the same axis validation every other input does."""
    frame, axis = prepare_time_index(site.frame, frequency=HALF_HOURLY, require_regular=True)
    assert axis.time_step == pd.Timedelta(HALF_HOURLY).to_pytimedelta()
    assert frame.index.is_monotonic_increasing
    assert frame.index.is_unique


def test_frame_carries_every_driver_target_and_flag(site):
    for column in DRIVER_COLUMNS:
        assert column in site.frame.columns
    for target in TARGETS:
        assert target in site.frame.columns
        assert f"{target}_QC" in site.frame.columns
    assert site.targets == ("NEE", "H", "LE")
    assert dict(site.qc_columns()) == {"NEE": "NEE_QC", "H": "H_QC", "LE": "LE_QC"}
    assert set(site.column_map) == {
        "shortwave",
        "vpd",
        "air_temperature",
        "net_radiation",
        "wind_speed",
        "wind_direction",
        "soil_heat_flux",
        "soil_temperature",
        "relative_humidity",
        "soil_water_content",
    }


def test_drivers_are_never_missing(site):
    """The paper's meteorological drivers arrive pre-filled (method_spec.md 7)."""
    assert int(site.frame[list(DRIVER_COLUMNS)].isna().sum().sum()) == 0


def test_truth_is_complete_and_noise_free(site):
    assert int(site.truth.isna().sum().sum()) == 0
    assert list(site.truth.columns) == ["NEE", "H", "LE", "GPP", "RECO"]
    assert site.truth.index.equals(site.frame.index)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_same_arguments_give_identical_data(site):
    again = synthetic_site()
    assert again.frame.equals(site.frame)
    assert again.truth.equals(site.truth)


def test_a_different_seed_gives_different_data(site):
    other = synthetic_site(seed=site.seed + 1)
    assert other.frame.index.equals(site.frame.index)
    assert not other.frame["TA_F_MDS"].equals(site.frame["TA_F_MDS"])
    assert not other.frame["NEE"].equals(site.frame["NEE"])


def test_streams_are_independent_of_one_another():
    """Changing the outage rate must not reshuffle the weather that preceded it."""
    a = synthetic_site(real_gap_fraction=0.05)
    b = synthetic_site(real_gap_fraction=0.20)
    assert a.frame["TA_F_MDS"].equals(b.frame["TA_F_MDS"])
    assert a.truth["NEE"].equals(b.truth["NEE"])
    assert b.frame["NEE"].isna().mean() > a.frame["NEE"].isna().mean()


# ---------------------------------------------------------------------------
# Radiation: the diurnal cycle
# ---------------------------------------------------------------------------


def test_shortwave_is_zero_at_night_and_never_negative(clean):
    frame = clean.frame
    index = frame.index
    deep_night = (index.hour >= 0) & (index.hour < 3)
    assert float(frame["SW_IN_F"][deep_night].max()) == 0.0
    assert float(frame["SW_IN_F"].min()) >= 0.0


def test_shortwave_peaks_at_solar_noon(clean):
    cycle = clean.frame["SW_IN_F"].groupby(hour_of_day(clean.frame.index)).mean()
    assert cycle.idxmax() == pytest.approx(12.0, abs=0.5)


def test_summer_noon_radiation_exceeds_winter_noon(clean):
    frame = clean.frame
    index = frame.index
    noon = index.hour == 12
    summer = index.month.isin([6, 7, 8]) & noon
    winter = index.month.isin([12, 1, 2]) & noon
    assert float(frame["SW_IN_F"][summer].mean()) > 2.0 * float(frame["SW_IN_F"][winter].mean())


# ---------------------------------------------------------------------------
# Air temperature: seasonality, diurnal cycle, hemisphere
# ---------------------------------------------------------------------------


def test_air_temperature_is_seasonal(site):
    index = site.frame.index
    summer = float(site.frame["TA_F_MDS"][index.month.isin([6, 7, 8])].mean())
    winter = float(site.frame["TA_F_MDS"][index.month.isin([12, 1, 2])].mean())
    assert summer - winter > 15.0


def test_air_temperature_peaks_after_solar_noon(site):
    """A thermal lag, not a copy of the radiation curve."""
    summer = site.frame[site.frame.index.month.isin([6, 7, 8])]
    peak = summer["TA_F_MDS"].groupby(hour_of_day(summer.index)).mean().idxmax()
    assert 13.0 <= float(peak) <= 17.0


def test_southern_latitude_flips_the_seasons():
    south = synthetic_site(latitude=-35.0, days=365)
    index = south.frame.index
    june = float(south.frame["TA_F_MDS"][index.month.isin([6, 7, 8])].mean())
    december = float(south.frame["TA_F_MDS"][index.month.isin([12, 1, 2])].mean())
    assert december - june > 10.0
    assert south.hemisphere is Hemisphere.SOUTH


def test_southern_latitude_flips_the_growing_season():
    south = synthetic_site(latitude=-35.0, days=365)
    index = south.truth.index
    june = float(south.truth["GPP"][index.month.isin([6, 7, 8])].mean())
    december = float(south.truth["GPP"][index.month.isin([12, 1, 2])].mean())
    assert december > june


def test_hemisphere_follows_the_documented_latitude_rule(site):
    assert site.hemisphere is Hemisphere.NORTH
    assert synthetic_site(latitude=0.0, days=40).hemisphere is Hemisphere.NORTH


# ---------------------------------------------------------------------------
# VPD and its relationship to temperature and humidity
# ---------------------------------------------------------------------------


def test_vpd_is_consistent_with_humidity_and_temperature(site):
    """VPD is ``es(TA) * (1 - RH/100)`` exactly, not an independent series."""
    temperature = site.frame["TA_F_MDS"].to_numpy()
    saturation = 6.1078 * np.exp(17.27 * temperature / (temperature + 237.3))
    expected = saturation * (1.0 - site.frame["RH"].to_numpy() / 100.0)
    assert np.allclose(site.frame["VPD_F_MDS"].to_numpy(), expected)


def test_vpd_is_never_negative_and_humidity_is_a_percentage(site):
    assert float(site.frame["VPD_F_MDS"].min()) >= 0.0
    assert float(site.frame["RH"].min()) > 0.0
    assert float(site.frame["RH"].max()) <= 100.0


def test_vpd_tracks_temperature_and_radiation(site):
    assert site.frame["VPD_F_MDS"].corr(site.frame["TA_F_MDS"]) > 0.5
    assert site.frame["VPD_F_MDS"].corr(site.frame["RH"]) < -0.5


def test_vpd_peaks_in_the_afternoon(site):
    summer = site.frame[site.frame.index.month.isin([6, 7, 8])]
    peak = summer["VPD_F_MDS"].groupby(hour_of_day(summer.index)).mean().idxmax()
    assert 12.0 <= float(peak) <= 18.0


# ---------------------------------------------------------------------------
# Soil temperature: damped and lagged
# ---------------------------------------------------------------------------


def test_soil_temperature_damps_the_diurnal_cycle(site):
    summer = site.frame[site.frame.index.month.isin([6, 7, 8])]
    air = diurnal_amplitude(summer["TA_F_MDS"])
    soil = diurnal_amplitude(summer["TS_F_MDS"])
    assert 0.0 < soil < 0.4 * air


def test_soil_temperature_lags_the_air(site):
    summer = site.frame[site.frame.index.month.isin([6, 7, 8])]
    air_peak = float(summer["TA_F_MDS"].groupby(hour_of_day(summer.index)).mean().idxmax())
    soil_peak = float(summer["TS_F_MDS"].groupby(hour_of_day(summer.index)).mean().idxmax())
    assert soil_peak > air_peak


def test_soil_temperature_still_follows_the_season(site):
    index = site.frame.index
    summer = float(site.frame["TS_F_MDS"][index.month.isin([6, 7, 8])].mean())
    winter = float(site.frame["TS_F_MDS"][index.month.isin([12, 1, 2])].mean())
    assert summer - winter > 10.0


# ---------------------------------------------------------------------------
# The fluxes
# ---------------------------------------------------------------------------


def test_daytime_growing_season_nee_is_uptake(clean):
    index = clean.truth.index
    daytime = clean.frame["SW_IN_F"] > DAYTIME_THRESHOLD
    growing = index.month.isin([6, 7, 8])
    assert float(clean.truth["NEE"][daytime & growing].mean()) < -3.0


def test_nighttime_nee_is_respiration(clean):
    """In the dark there is no photosynthesis, so NEE is exactly RECO."""
    dark = clean.frame["SW_IN_F"] == 0.0
    assert bool(dark.any())
    assert float(clean.truth["GPP"][dark].max()) == 0.0
    assert np.allclose(clean.truth["NEE"][dark], clean.truth["RECO"][dark])
    below_threshold = clean.frame["SW_IN_F"] <= DAYTIME_THRESHOLD
    assert float(clean.truth["NEE"][below_threshold].mean()) > 0.0


def test_respiration_rises_with_soil_temperature(clean):
    """A Q10 response: doubling per 10 degrees, at a constant canopy."""
    summer = clean.frame.index.month.isin([7])
    warm = clean.frame["TS_F_MDS"][summer] > clean.frame["TS_F_MDS"][summer].median()
    respiration = clean.truth["RECO"][summer]
    assert float(respiration[warm].mean()) > float(respiration[~warm].mean())


def test_latent_heat_is_never_negative_and_needs_a_canopy_or_energy(clean):
    assert float(clean.truth["LE"].min()) >= 0.0
    night = clean.frame["SW_IN_F"] <= DAYTIME_THRESHOLD
    assert float(clean.truth["LE"][night].mean()) < 5.0


def test_latent_heat_responds_to_soil_water(clean):
    """Dry soil suppresses evaporation, which is what makes SWC a real driver."""
    summer = clean.frame.index.month.isin([7, 8])
    daytime = clean.frame["SW_IN_F"] > 300.0
    subset = summer & daytime
    water = clean.frame["SWC_F_MDS"][subset]
    wet = water > water.median()
    assert float(clean.truth["LE"][subset][wet].mean()) > float(
        clean.truth["LE"][subset][~wet].mean()
    )


def test_soil_water_stays_inside_the_bucket(site):
    assert float(site.frame["SWC_F_MDS"].min()) > 0.0
    assert float(site.frame["SWC_F_MDS"].max()) <= 34.0
    assert diurnal_amplitude(site.frame["SWC_F_MDS"]) < 1.0


def test_wind_direction_is_a_compass_bearing(site):
    assert float(site.frame["WD"].min()) >= 0.0
    assert float(site.frame["WD"].max()) < 360.0
    assert float(site.frame["WS"].min()) > 0.0


# ---------------------------------------------------------------------------
# Energy balance: the known closure ratio
# ---------------------------------------------------------------------------


def test_noise_free_fluxes_close_the_energy_balance_at_the_stated_ratio(clean):
    available = clean.frame["NETRAD"] - clean.frame["G_F_MDS"]
    ratio = float((clean.truth["H"] + clean.truth["LE"]).sum() / available.sum())
    assert ratio == pytest.approx(clean.energy_balance_closure, abs=1e-12)


@pytest.mark.parametrize("closure", [1.0, 0.75])
def test_closure_is_configurable(closure):
    other = synthetic_site(days=90, energy_balance_closure=closure)
    available = other.frame["NETRAD"] - other.frame["G_F_MDS"]
    ratio = float((other.truth["H"] + other.truth["LE"]).sum() / available.sum())
    assert ratio == pytest.approx(closure, abs=1e-12)


def test_measured_energy_balance_matches_the_metric_module(clean):
    """The same ratio, computed by the module the paper's EBR lives in."""
    comparison = compare_energy_balance(
        measured_sensible_heat=clean.frame["H"],
        measured_latent_heat=clean.frame["LE"],
        filled_sensible_heat=clean.truth["H"],
        filled_latent_heat=clean.truth["LE"],
        net_radiation=clean.frame["NETRAD"],
        soil_heat_flux=clean.frame["G_F_MDS"],
    )
    assert comparison.measured == pytest.approx(clean.energy_balance_closure, abs=1e-12)
    assert comparison.difference == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# Noise, outages and QC provenance
# ---------------------------------------------------------------------------


def test_noise_scale_zero_reproduces_the_truth_exactly(clean):
    for target in TARGETS:
        assert np.allclose(clean.frame[target].to_numpy(), clean.truth[target].to_numpy())


def test_noise_is_present_by_default_and_unbiased(site):
    residual = site.frame["LE"] - site.truth["LE"]
    assert float(residual.abs().mean()) > 1.0
    assert float(residual.mean()) == pytest.approx(0.0, abs=1.0)


def test_outages_are_shared_by_every_target(site):
    """One instrument system: a row missing from NEE is missing from H and LE."""
    missing = site.frame["NEE"].isna()
    assert missing.equals(site.frame["H"].isna())
    assert missing.equals(site.frame["LE"].isna())
    assert 0.03 < float(missing.mean()) < 0.06


def test_outages_are_bursts_rather_than_scattered_rows(site):
    missing = site.frame["NEE"].isna().to_numpy()
    starts = int(np.count_nonzero(missing[1:] & ~missing[:-1])) + int(missing[0])
    assert starts > 0
    assert int(missing.sum()) / starts > 4.0


def test_qc_flags_partition_the_rows(site):
    for target in TARGETS:
        flags = site.frame[f"{target}_QC"]
        assert set(flags.unique()) <= {OBSERVED_QC, PRE_FILLED_QC, MISSING_QC}
        assert flags.eq(MISSING_QC).equals(site.frame[target].isna())
        assert 0.05 < float(flags.eq(PRE_FILLED_QC).mean()) < 0.12


def test_pre_filled_nee_is_concentrated_at_night(site):
    """As friction-velocity filtering removes calm nights."""
    night = site.frame["SW_IN_F"] <= DAYTIME_THRESHOLD
    flagged = site.frame["NEE_QC"].eq(PRE_FILLED_QC)
    assert float((flagged & night).sum()) / float(flagged.sum()) > 0.7


def test_observed_mask_excludes_missing_and_pre_filled_rows(site):
    observed = site.observed("LE")
    assert observed.equals(site.frame["LE_QC"].eq(OBSERVED_QC) & site.frame["LE"].notna())
    assert 0.80 < float(observed.mean()) < 0.95


def test_gaps_and_flags_can_be_turned_off(clean):
    assert int(clean.frame[list(TARGETS)].isna().sum().sum()) == 0
    for target in TARGETS:
        assert clean.frame[f"{target}_QC"].eq(OBSERVED_QC).all()


# ---------------------------------------------------------------------------
# The growth trend behind time_distance_hours
# ---------------------------------------------------------------------------


def test_growth_trend_raises_productivity_over_the_record():
    flat = synthetic_site(growth_per_year=0.0)
    growing = synthetic_site(growth_per_year=0.5)
    first_day = slice(0, STEPS_PER_DAY)
    last_month = slice(-30 * STEPS_PER_DAY, None)
    assert float(growing.truth["GPP"].iloc[first_day].sum()) == pytest.approx(
        float(flat.truth["GPP"].iloc[first_day].sum()), rel=1e-3
    )
    assert float(growing.truth["GPP"].iloc[last_month].sum()) > float(
        flat.truth["GPP"].iloc[last_month].sum()
    )


# ---------------------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "changes",
    [
        {"days": 0},
        {"days": 1.5},
        {"real_gap_fraction": 1.0},
        {"real_gap_fraction": -0.1},
        {"pre_filled_fraction": 2.0},
        {"noise_scale": -1.0},
        {"latitude": 120.0},
        {"frequency": "7min"},
    ],
)
def test_impossible_arguments_are_rejected(changes):
    settings = {"days": 10} | changes
    with pytest.raises((ConfigError, ValueError)):
        synthetic_site(**settings)


# ---------------------------------------------------------------------------
# Known gaps
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def manifest(site):
    return site.known_gaps()


def test_known_gaps_span_the_paper_durations(manifest):
    expected = {
        GapClass.SHORT: pd.Timedelta(hours=24),
        GapClass.LONG: pd.Timedelta(days=7),
        GapClass.VERY_LONG: pd.Timedelta(days=30),
    }
    for gap in manifest:
        assert pd.Timedelta(gap.duration) == expected[gap.gap_class]
        assert gap.end - gap.start == expected[gap.gap_class]


def test_known_gaps_cover_the_records_a_complete_grid_holds(manifest):
    expected = {GapClass.SHORT: 48, GapClass.LONG: 336, GapClass.VERY_LONG: 1440}
    for gap in manifest:
        assert gap.n_expected == expected[gap.gap_class]
        assert gap.n_rows == gap.n_expected  # the synthetic grid is complete


def test_known_gap_plan_places_every_class(manifest):
    counts = manifest.events_by_class
    assert counts[GapClass.SHORT] == 3
    assert counts[GapClass.LONG] == 2
    assert counts[GapClass.VERY_LONG] == 1
    assert len(manifest) == len(DEFAULT_KNOWN_GAPS)


def test_known_gaps_are_chronological_and_identified_per_class(manifest):
    starts = [gap.start for gap in manifest]
    assert starts == sorted(starts)
    assert [gap.gap_id for gap in manifest.by_class(GapClass.SHORT)] == [
        "short-1",
        "short-2",
        "short-3",
    ]


def test_known_gaps_do_not_overlap(manifest, site):
    mask = manifest.mask(site.frame.index)
    assert int(mask.sum()) == sum(gap.n_rows for gap in manifest)
    assert int(mask.sum()) == manifest.n_withheld_rows


def test_known_gaps_are_reproducible(site):
    assert site.known_gaps().to_dict() == site.known_gaps().to_dict()


def test_known_gaps_clear_the_observed_fraction_rule(manifest):
    """Every default interval withholds real measurement on the reference site."""
    for gap in manifest:
        assert gap.observed_fraction >= manifest.scenario.min_observed_fraction


def test_manifest_reports_what_the_intervals_actually_withhold(manifest):
    assert manifest.n_withheld_observed < manifest.n_withheld_rows
    assert manifest.achieved_fraction == pytest.approx(manifest.scenario.missing_fraction)
    assert manifest.fraction_error == pytest.approx(0.0)
    assert manifest.is_within_tolerance


def test_manifest_is_serialisable(manifest):
    document = manifest.to_dict()
    assert document["targets"] == list(TARGETS)
    assert len(document["gaps"]) == len(DEFAULT_KNOWN_GAPS)
    assert document["time_step"] == "P0DT0H30M0S"


def test_known_gaps_are_shared_across_targets(site, manifest):
    """One set of locations for NEE, H and LE, as the paper requires."""
    joint = manifest.mask(site.frame.index)
    for target in TARGETS:
        single = site.known_gaps(targets=target)
        assert single.mask(site.frame.index).equals(joint)


def test_a_single_target_manifest_has_more_available_rows(site, manifest):
    """Availability is the intersection: fewer targets, fewer rows excluded."""
    assert site.known_gaps(targets="LE").n_available > manifest.n_available


def test_gap_mix_can_be_read_on_the_event_basis(site):
    events = site.known_gaps(
        scenario=GapScenarioConfig(allocation_basis="gap_events"),
    )
    assert events.basis.value == "gap_events"
    assert events.achieved_shares[GapClass.SHORT] == pytest.approx(3 / 6)
    assert events.fraction_error == pytest.approx(0.0)


def test_a_plan_that_runs_past_the_record_is_rejected(site):
    with pytest.raises(GapError, match="past the end"):
        site.known_gaps(plan=[KnownGap(GapClass.VERY_LONG, 350.0)])


def test_overlapping_plans_are_rejected(site):
    plan = [KnownGap(GapClass.LONG, 100.0), KnownGap(GapClass.SHORT, 103.0)]
    with pytest.raises(GapError, match="overlap"):
        site.known_gaps(plan=plan)


def test_overlap_can_be_allowed_deliberately(site):
    plan = [KnownGap(GapClass.LONG, 100.0), KnownGap(GapClass.SHORT, 103.0)]
    allowed = site.known_gaps(plan=plan, scenario=GapScenarioConfig(allow_overlap=True))
    assert len(allowed) == 2
    assert allowed.n_withheld_rows < sum(gap.n_rows for gap in allowed)


def test_an_empty_plan_is_rejected(site):
    with pytest.raises(GapError, match="empty"):
        site.known_gaps(plan=[])


def test_a_frame_without_observations_is_rejected(site):
    blank = site.frame.copy()
    blank["LE"] = np.nan
    with pytest.raises(GapError, match="withhold nothing"):
        known_gap_manifest(
            blank,
            time_step=site.time_step,
            targets=("LE",),
            qc_columns={"LE": "LE_QC"},
        )


def test_an_unknown_target_is_rejected(site):
    with pytest.raises(ConfigError, match="unknown target"):
        site.qc_column("FCH4")


@pytest.mark.parametrize("start_day", [-1.0, float("nan")])
def test_a_known_gap_needs_a_real_offset(start_day):
    with pytest.raises(ConfigError, match="start_day"):
        KnownGap(GapClass.SHORT, start_day)


def test_a_shorter_record_takes_its_own_plan():
    short = synthetic_site(days=60)
    manifest = short.known_gaps(
        plan=[KnownGap(GapClass.VERY_LONG, 5.0), KnownGap(GapClass.SHORT, 40.0)]
    )
    assert manifest.events_by_class[GapClass.VERY_LONG] == 1
    assert manifest.events_by_class[GapClass.LONG] == 0


# ---------------------------------------------------------------------------
# Integration: the fixture reaching the rest of the package
# ---------------------------------------------------------------------------


def test_features_build_completely_from_the_fixture(site):
    config = site.config("RFR10")
    features = build_feature_matrix(site.frame, config=config, target="LE")
    assert len(features) == len(site.frame)
    assert int(features.notna().all(axis=1).sum()) == len(features)


def test_config_carries_the_sites_own_metadata(site):
    config = site.config("RFR10", random_state=7)
    assert config.mode.value == "RFR10"
    assert config.latitude == site.latitude
    assert config.site_id == site.site_id
    assert config.random_state == 7
    assert config.frequency == site.time_step


def test_long_gaps_have_no_complete_features_under_the_default_strategy(site, manifest):
    """Ambiguity A4, visible: a gap covering a whole day has no daily statistics."""
    features = build_validation_features(
        site.frame,
        config=site.config("RFR3"),
        target="LE",
        holdout=manifest.mask(site.frame.index),
        qc_column="LE_QC",
    )
    assert int((features.holdout_mask & features.complete_mask).sum()) == 0


def test_a_reaching_strategy_restores_them(site, manifest):
    features = build_validation_features(
        site.frame,
        config=site.config(
            "RFR3", features=FeatureConfig(daily_statistic_strategy="rolling_available")
        ),
        target="LE",
        holdout=manifest.mask(site.frame.index),
        qc_column="LE_QC",
    )
    assert int((features.holdout_mask & features.complete_mask).sum()) > 1000


def test_the_fixture_is_leakage_safe(site, manifest):
    """Acceptance tests 11-14 on a full year rather than a toy frame."""
    require_no_target_leakage(
        site.frame,
        config=site.config(
            "RFR3", features=FeatureConfig(daily_statistic_strategy="rolling_available")
        ),
        target="LE",
        holdout=manifest.mask(site.frame.index),
        qc_column="LE_QC",
    )


def test_the_generator_can_place_its_own_scenario(site):
    """The paper's 25% scenario runs on this frame; A7 reports the shortfall."""
    config = site.config("RFR3")
    with pytest.warns(UserWarning, match="withheld fraction"):
        generated = GapScenarioGenerator(config).generate(
            site.frame,
            target=list(TARGETS),
            qc_column={target: f"{target}_QC" for target in TARGETS},
        )
    assert len(generated) > len(DEFAULT_KNOWN_GAPS)
    assert generated.events_by_class[GapClass.VERY_LONG] >= 1
    assert generated.achieved_fraction < generated.scenario.missing_fraction


@pytest.mark.slow
def test_end_to_end_fit_and_score_on_known_gaps(site, manifest):
    """Acceptance tests 34-35: train, withhold 24 h / 7 d / 30 d, fill, score."""
    config = site.config(
        "RFR10",
        hyperparameter_grid={"n_estimators": (25,)},
        cv_folds=3,
        features=FeatureConfig(daily_statistic_strategy="rolling_available"),
    )
    features = build_validation_features(
        site.frame,
        config=config,
        target="LE",
        holdout=manifest.mask(site.frame.index),
        qc_column="LE_QC",
    )
    model = RFRModel(config, target="LE").fit(
        features.features[features.training_mask], features.truth[features.training_mask]
    )
    scored = features.scoring_mask & features.complete_mask
    predictions = model.predict(features.features[scored])
    metrics = core_metrics(features.truth[scored], predictions)
    assert metrics.n > 1000
    assert metrics.r2 is not None and metrics.r2 > 0.8
    assert metrics.slope is not None and 0.7 < metrics.slope < 1.3
    for gap_class in GapClass:
        rows = manifest.mask(site.frame.index) & scored
        for gap in manifest.by_class(gap_class):
            inside = rows & (site.frame.index >= gap.start) & (site.frame.index < gap.end)
            assert int(inside.sum()) > 0
