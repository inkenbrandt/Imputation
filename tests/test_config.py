"""Validated-configuration tests.

Covers the contract of ``docs/method_spec.md`` section 15 and the ambiguity table:
unknown modes, missing RFR3/RFR10 driver mappings, invalid thresholds, invalid gap
proportions, and the latitude-to-hemisphere rule. The governing requirement is that
every scientific choice is reachable through configuration, so these tests also
pin the documented defaults (A1-A7, A9) and their manifest serialisation.

See ``docs/method_spec.md`` for the contract and
``docs/supplement_benchmarks.md`` for benchmark provenance.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import pickle
from datetime import timedelta
from typing import Any

import pytest

from rfrgapfill.config import (
    DEFAULT_FALLBACK_WINDOW_DAYS,
    DEFAULT_HYPERPARAMETER_GRID,
    AllocationBasis,
    BoundaryConvention,
    ColumnMap,
    ColumnMapError,
    ConfigError,
    CVStrategy,
    DailyStatisticStrategy,
    FeatureConfig,
    FeatureMode,
    GapClass,
    GapScenarioConfig,
    Hemisphere,
    MetricSubset,
    Mode,
    RFRConfig,
    ValidationConfig,
    orf_pairing_differences,
    require_orf_pairing,
)
from rfrgapfill.schema import RFR3_DRIVERS, RFR10_DRIVERS

RFR3_MAPPING = {"shortwave": "SW_IN", "vpd": "VPD", "air_temperature": "TA"}
RFR10_MAPPING = {
    **RFR3_MAPPING,
    "net_radiation": "NETRAD",
    "wind_speed": "WS",
    "wind_direction": "WD",
    "soil_heat_flux": "G",
    "soil_temperature": "TS",
    "relative_humidity": "RH",
    "soil_water_content": "SWC",
}


def config(**changes: Any) -> RFRConfig:
    """Return a valid RFR3 configuration with ``changes`` applied."""
    kwargs: dict[str, Any] = {"mode": "RFR3", "hemisphere": "north"}
    kwargs.update(changes)
    return RFRConfig(**kwargs)


# ---------------------------------------------------------------------------
# Mode
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["RFR5", "RFR", "ORF", "", None, 10])
def test_unknown_modes_are_rejected(mode: object) -> None:
    with pytest.raises(ConfigError, match="not a recognised Mode"):
        config(mode=mode)


def test_mode_is_normalised_and_exposes_its_drivers() -> None:
    assert config(mode="rfr10").rfr_mode is Mode.RFR10
    assert config(mode="RFR3").drivers == RFR3_DRIVERS
    assert config(mode="RFR10").drivers == RFR10_DRIVERS


def test_mode_has_no_default() -> None:
    with pytest.raises(TypeError):
        RFRConfig()  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# Required column mappings
# ---------------------------------------------------------------------------


def test_rfr3_requires_exactly_the_three_canonical_drivers() -> None:
    assert config(column_map=RFR3_MAPPING).required_variables() == RFR3_DRIVERS


def test_rfr10_requires_the_ten_canonical_drivers() -> None:
    assert config(mode="RFR10", column_map=RFR10_MAPPING).required_variables() == RFR10_DRIVERS


def test_missing_rfr3_mapping_is_rejected_at_construction() -> None:
    incomplete = {"shortwave": "SW_IN", "vpd": "VPD"}

    with pytest.raises(ColumnMapError, match=r"missing column mapping.*air_temperature"):
        config(column_map=incomplete)


def test_missing_rfr10_mapping_names_every_absent_driver() -> None:
    with pytest.raises(ColumnMapError) as excinfo:
        config(mode="RFR10", column_map=RFR3_MAPPING)

    message = str(excinfo.value)
    assert "mode=RFR10" in message
    for name in RFR10_DRIVERS[3:]:
        assert name in message


def test_column_map_may_be_supplied_later_instead_of_on_the_config() -> None:
    deferred = config()

    assert deferred.columns == ColumnMap()
    with pytest.raises(ColumnMapError, match="missing column mapping"):
        deferred.require_column_map({"shortwave": "SW_IN"})
    assert deferred.require_column_map(RFR3_MAPPING).column("vpd") == "VPD"


def test_energy_balance_adds_net_radiation_and_soil_heat_flux_to_rfr3() -> None:
    rfr3 = config(column_map=RFR3_MAPPING)

    assert rfr3.required_variables(include_ebr=True) == (
        "shortwave",
        "vpd",
        "air_temperature",
        "net_radiation",
        "soil_heat_flux",
    )
    with pytest.raises(ColumnMapError, match="net_radiation, soil_heat_flux"):
        rfr3.require_column_map(include_ebr=True)


# ---------------------------------------------------------------------------
# Thresholds (A2) and the daytime split
# ---------------------------------------------------------------------------


def test_radiation_thresholds_default_to_the_paper_values() -> None:
    assert FeatureConfig().radiation_thresholds == (10.0, 100.0)
    assert FeatureConfig().convention is BoundaryConvention.MEDIUM_INCLUSIVE


@pytest.mark.parametrize(
    "thresholds",
    [
        (100.0, 10.0),  # decreasing
        (10.0, 10.0),  # not strictly increasing
        (10.0,),  # wrong length
        (10.0, 50.0, 100.0),  # wrong length
        (float("nan"), 100.0),
        (10.0, float("inf")),
        ("10", "100"),
        10.0,
    ],
)
def test_invalid_radiation_thresholds_are_rejected(thresholds: object) -> None:
    with pytest.raises(ConfigError, match="radiation_thresholds"):
        FeatureConfig(radiation_thresholds=thresholds)  # type: ignore[arg-type]


def test_radiation_thresholds_are_normalised_to_floats() -> None:
    assert FeatureConfig(radiation_thresholds=(5, 200)).radiation_thresholds == (5.0, 200.0)


def test_boundary_convention_is_configurable_and_validated() -> None:
    assert (
        FeatureConfig(boundary_convention="medium_exclusive").convention
        is BoundaryConvention.MEDIUM_EXCLUSIVE
    )
    with pytest.raises(ConfigError, match="not a recognised BoundaryConvention"):
        FeatureConfig(boundary_convention="round_half_up")


def test_daytime_threshold_defaults_to_20_and_rejects_non_finite_values() -> None:
    assert ValidationConfig().daytime_threshold == 20.0
    assert ValidationConfig(daytime_threshold=5.0).daytime_threshold == 5.0
    with pytest.raises(ConfigError, match="daytime_threshold"):
        ValidationConfig(daytime_threshold=float("nan"))
    with pytest.raises(ConfigError, match="daytime_threshold"):
        ValidationConfig(daytime_threshold="20")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Feature mode and the ORF switch
# ---------------------------------------------------------------------------


def test_paper_safe_is_the_default_feature_mode() -> None:
    assert FeatureConfig().mode is FeatureMode.PAPER_SAFE
    assert FeatureConfig().use_receptive_limiter is True


def test_legacy_fluxlib_mode_is_reserved_and_refused_until_evidence_exists() -> None:
    with pytest.raises(ConfigError, match="reserved and not implemented"):
        FeatureConfig(feature_mode="legacy_fluxlib")


def test_unknown_feature_modes_are_rejected() -> None:
    with pytest.raises(ConfigError, match="not a recognised FeatureMode"):
        FeatureConfig(feature_mode="leaky")


def test_min_daily_observations_controls_the_empty_day_policy() -> None:
    assert FeatureConfig().min_daily_observations == 1
    assert FeatureConfig(min_daily_observations=4).min_daily_observations == 4
    with pytest.raises(ConfigError, match="min_daily_observations"):
        FeatureConfig(min_daily_observations=0)


def test_the_thinly_observed_day_strategy_defaults_to_leaving_it_missing() -> None:
    # A4: the paper does not say what it did with a day it could not summarise,
    # so the default imputes nothing and borrows nothing.
    settings = FeatureConfig()
    assert settings.statistic_strategy is DailyStatisticStrategy.MISSING
    assert settings.fallback_window is None
    assert settings.to_dict()["daily_statistic_strategy"] == "missing"


@pytest.mark.parametrize(
    ("strategy", "reaches"),
    [
        ("missing", False),
        ("within_day_available", False),
        ("neighbor_day_fallback", True),
        ("rolling_available", True),
    ],
)
def test_only_the_reaching_strategies_take_a_fallback_window(strategy: str, reaches: bool) -> None:
    window = 5 if reaches else None
    settings = FeatureConfig(daily_statistic_strategy=strategy, fallback_window_days=window)
    assert settings.statistic_strategy.uses_other_days is reaches
    assert settings.fallback_window == window


def test_a_reaching_strategy_fills_in_the_documented_default_window() -> None:
    settings = FeatureConfig(daily_statistic_strategy="rolling_available")
    assert settings.fallback_window == DEFAULT_FALLBACK_WINDOW_DAYS


def test_a_window_that_could_not_apply_is_rejected_rather_than_ignored() -> None:
    with pytest.raises(ConfigError, match="meaningful only"):
        FeatureConfig(daily_statistic_strategy="within_day_available", fallback_window_days=3)


def test_an_unknown_strategy_is_rejected() -> None:
    with pytest.raises(ConfigError, match="daily_statistic_strategy"):
        FeatureConfig(daily_statistic_strategy="ask_a_neighbour")


def test_a_non_positive_window_is_rejected() -> None:
    with pytest.raises(ConfigError, match="fallback_window_days"):
        FeatureConfig(daily_statistic_strategy="rolling_available", fallback_window_days=0)


def test_the_strategy_reaches_the_orf_pairing_check() -> None:
    # Every feature setting is carried over to the benchmark arm unchanged, so a
    # comparison that also changed the daily-statistic policy is refused.
    rfr = RFRConfig(mode="RFR3", hemisphere="north")
    diverged = rfr.as_orf().replace(
        features=rfr.as_orf().features.replace(daily_statistic_strategy="within_day_available")
    )
    assert "features.daily_statistic_strategy" in orf_pairing_differences(rfr, diverged)


def test_orf_benchmark_switch_drops_the_hemisphere_requirement() -> None:
    orf = RFRConfig(mode="RFR3", features=FeatureConfig(use_receptive_limiter=False))

    assert orf.features.use_receptive_limiter is False
    assert orf.is_paper_faithful is False
    assert orf.hemisphere_source is None


# ---------------------------------------------------------------------------
# Hemisphere and latitude (A9)
# ---------------------------------------------------------------------------


def test_latitude_is_enough_to_resolve_the_hemisphere() -> None:
    north = config(hemisphere=None, latitude=51.5)
    south = config(hemisphere=None, latitude=-33.9)

    assert north.resolve_hemisphere() is Hemisphere.NORTH
    assert north.hemisphere_source == "latitude"
    assert south.resolve_hemisphere() is Hemisphere.SOUTH


def test_equator_resolves_north_by_the_documented_tie_break() -> None:
    assert config(hemisphere=None, latitude=0.0).resolve_hemisphere() is Hemisphere.NORTH


def test_explicit_hemisphere_overrides_latitude() -> None:
    conflicting = config(hemisphere="south", latitude=51.5)

    assert conflicting.resolve_hemisphere() is Hemisphere.SOUTH
    assert conflicting.hemisphere_source == "explicit"


def test_the_season_feature_refuses_to_guess_a_hemisphere() -> None:
    with pytest.raises(ConfigError, match="season feature needs a hemisphere"):
        RFRConfig(mode="RFR3")


def test_invalid_latitude_is_rejected() -> None:
    with pytest.raises(ConfigError, match=r"\[-90, 90\]"):
        config(hemisphere=None, latitude=120.0)


# ---------------------------------------------------------------------------
# Gap scenario (A3, A7)
# ---------------------------------------------------------------------------


def test_gap_scenario_defaults_reproduce_the_papers_scenario() -> None:
    scenario = GapScenarioConfig()

    assert scenario.missing_fraction == 0.25
    assert scenario.share(GapClass.SHORT) == 0.20
    assert scenario.share(GapClass.LONG) == 0.30
    assert scenario.share(GapClass.VERY_LONG) == 0.50
    assert scenario.duration(GapClass.SHORT) == timedelta(hours=24)
    assert scenario.duration(GapClass.LONG) == timedelta(days=7)
    assert scenario.duration(GapClass.VERY_LONG) == timedelta(days=30)
    assert scenario.min_observed_fraction == 0.50
    assert scenario.allow_overlap is False
    assert scenario.shared_gaps_across_targets is True
    assert scenario.basis is AllocationBasis.MISSING_RECORDS


def test_gap_mix_may_be_given_with_duration_aliases_as_in_the_public_api() -> None:
    scenario = GapScenarioConfig(gap_mix={"24h": 0.20, "7d": 0.30, "30d": 0.50})

    assert scenario.share("short") == 0.20
    assert scenario.share("very_long") == 0.50


@pytest.mark.parametrize(
    "gap_mix",
    [
        {"short": 0.2, "long": 0.3, "very_long": 0.4},  # sums to 0.9
        {"short": 0.5, "long": 0.5, "very_long": 0.5},  # sums to 1.5
        {"short": 0.0, "long": 0.0, "very_long": 0.0},  # withholds nothing
    ],
)
def test_gap_proportions_that_do_not_sum_to_one_are_rejected(gap_mix: dict[str, float]) -> None:
    with pytest.raises(ConfigError, match=r"sum to 1\.0"):
        GapScenarioConfig(gap_mix=gap_mix)


def test_a_single_class_scenario_is_allowed_when_the_shares_still_sum_to_one() -> None:
    scenario = GapScenarioConfig(gap_mix={"very_long": 1.0})

    assert scenario.active_classes == (GapClass.VERY_LONG,)
    assert scenario.share("short") == 0.0


@pytest.mark.parametrize(
    "gap_mix",
    [
        {"short": -0.2, "long": 0.7, "very_long": 0.5},
        {"short": 1.5, "long": 0.0, "very_long": -0.5},
        {"short": "0.2", "long": 0.3, "very_long": 0.5},
    ],
)
def test_gap_proportions_outside_zero_to_one_are_rejected(gap_mix: dict[str, object]) -> None:
    with pytest.raises(ConfigError, match="gap_mix"):
        GapScenarioConfig(gap_mix=gap_mix)  # type: ignore[arg-type]


def test_unknown_gap_classes_are_rejected() -> None:
    with pytest.raises(ConfigError, match="not a recognised GapClass"):
        GapScenarioConfig(gap_mix={"medium": 1.0})


def test_duplicate_gap_class_spellings_are_rejected() -> None:
    with pytest.raises(ConfigError, match="more than once"):
        GapScenarioConfig(gap_mix={"very_long": 0.5, "30d": 0.5})


@pytest.mark.parametrize("fraction", [0.0, 1.0, -0.1, 1.5, float("nan"), "0.25"])
def test_invalid_missing_fractions_are_rejected(fraction: object) -> None:
    with pytest.raises(ConfigError, match="missing_fraction"):
        GapScenarioConfig(missing_fraction=fraction)  # type: ignore[arg-type]


@pytest.mark.parametrize("fraction", [-0.1, 1.5, float("inf")])
def test_invalid_observed_fraction_thresholds_are_rejected(fraction: float) -> None:
    with pytest.raises(ConfigError, match="min_observed_fraction"):
        GapScenarioConfig(min_observed_fraction=fraction)


def test_gap_durations_are_configurable_and_must_be_positive_durations() -> None:
    scenario = GapScenarioConfig(durations={"short": "12h", "long": timedelta(days=3)})

    assert scenario.duration("short") == timedelta(hours=12)
    assert scenario.duration("long") == timedelta(days=3)
    assert scenario.duration("very_long") == timedelta(days=30)
    with pytest.raises(ConfigError, match="positive duration"):
        GapScenarioConfig(durations={"short": "-1d"})
    with pytest.raises(ConfigError, match="not a bare number"):
        GapScenarioConfig(durations={"short": 24})


def test_allocation_basis_is_configurable_and_validated() -> None:
    assert GapScenarioConfig(allocation_basis="gap_events").basis is AllocationBasis.GAP_EVENTS
    with pytest.raises(ConfigError, match="not a recognised AllocationBasis"):
        GapScenarioConfig(allocation_basis="gap_hours")


def test_retry_bound_is_configurable_and_must_be_positive() -> None:
    assert GapScenarioConfig(max_attempts_per_gap=10).max_attempts_per_gap == 10
    with pytest.raises(ConfigError, match="max_attempts_per_gap"):
        GapScenarioConfig(max_attempts_per_gap=0)


# ---------------------------------------------------------------------------
# Validation configuration
# ---------------------------------------------------------------------------


def test_validation_reports_all_daytime_and_nighttime_by_default() -> None:
    assert ValidationConfig().metric_subsets == (
        MetricSubset.ALL,
        MetricSubset.DAYTIME,
        MetricSubset.NIGHTTIME,
    )
    assert ValidationConfig().report_by_gap_class is True
    assert ValidationConfig().bias_iqr_by_gap_class is True


def test_validation_subsets_are_deduplicated_and_validated() -> None:
    assert ValidationConfig(subsets=["all", "all", "daytime"]).metric_subsets == (
        MetricSubset.ALL,
        MetricSubset.DAYTIME,
    )
    with pytest.raises(ConfigError, match="not a recognised MetricSubset"):
        ValidationConfig(subsets=["dusk"])
    with pytest.raises(ConfigError, match="at least one metric subset"):
        ValidationConfig(subsets=[])


def test_validation_requires_a_gap_scenario_config() -> None:
    with pytest.raises(ConfigError, match="GapScenarioConfig"):
        ValidationConfig(gaps={"missing_fraction": 0.25})  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Estimator, reproducibility and cross-validation (A1, A5)
# ---------------------------------------------------------------------------


def test_frequency_accepts_pandas_style_durations_or_is_inferred() -> None:
    assert config(frequency="30min").time_step == timedelta(minutes=30)
    assert config(frequency=timedelta(hours=1)).time_step == timedelta(hours=1)
    assert config().time_step is None


@pytest.mark.parametrize("frequency", [30, 0.5, "-30min", "sometimes"])
def test_invalid_frequencies_are_rejected(frequency: object) -> None:
    with pytest.raises(ConfigError, match="frequency"):
        config(frequency=frequency)


def test_random_state_is_required_to_be_an_integer_seed() -> None:
    assert config().random_state == 42
    with pytest.raises(ConfigError, match="random_state"):
        config(random_state=None)
    with pytest.raises(ConfigError, match="random_state"):
        config(random_state=-1)


def test_n_jobs_is_configurable_but_may_not_be_zero() -> None:
    assert config(n_jobs=-1).n_jobs == -1
    assert config().n_jobs is None
    with pytest.raises(ConfigError, match="n_jobs"):
        config(n_jobs=0)


def test_default_grid_is_ours_not_the_papers_and_is_overridable() -> None:
    assert set(DEFAULT_HYPERPARAMETER_GRID) <= {
        "max_features",
        "min_samples_leaf",
        "n_estimators",
    }
    custom = config(hyperparameter_grid={"n_estimators": [50]})
    assert custom.hyperparameter_grid == {"n_estimators": (50,)}


@pytest.mark.parametrize(
    "grid",
    [
        {},
        {"n_estimators": []},
        {"n_estimators": 100},
        {"n_estimators": "100"},
        {"n_trees": [100]},
        {"random_state": [0, 1]},
        {"n_jobs": [1]},
        [("n_estimators", [100])],
    ],
)
def test_invalid_hyperparameter_grids_are_rejected(grid: object) -> None:
    with pytest.raises(ConfigError, match="hyperparameter_grid"):
        config(hyperparameter_grid=grid)  # type: ignore[arg-type]


def test_cross_validation_defaults_are_conventional_folds() -> None:
    assert config().cv is CVStrategy.KFOLD
    assert config().cv.is_paper_default is True
    assert config().cv_folds == 5
    assert config().cv_shuffle is False


def test_time_aware_cross_validation_is_available_but_marked_as_an_enhancement() -> None:
    enhanced = config(cv_strategy="time_series_split")

    assert enhanced.cv is CVStrategy.TIME_SERIES_SPLIT
    assert enhanced.cv.is_paper_default is False
    assert enhanced.is_paper_faithful is False


def test_incompatible_cross_validation_options_are_rejected_early() -> None:
    with pytest.raises(ConfigError, match="incompatible"):
        config(cv_strategy="time_series_split", cv_shuffle=True)
    with pytest.raises(ConfigError, match="cv_folds"):
        config(cv_folds=1)
    with pytest.raises(ConfigError, match="not a recognised CVStrategy"):
        config(cv_strategy="bootstrap")


def test_observed_qc_values_are_configurable_and_never_empty() -> None:
    assert config().observed_qc_values == (0,)
    assert config(observed_qc_values=[0, 1]).observed_qc_values == (0, 1)
    with pytest.raises(ConfigError, match="observed_qc_values"):
        config(observed_qc_values=[])
    with pytest.raises(ConfigError, match="observed_qc_values"):
        config(observed_qc_values="0")


def test_a_default_configuration_is_paper_faithful() -> None:
    assert config(column_map=RFR3_MAPPING).is_paper_faithful is True


# ---------------------------------------------------------------------------
# Immutability, copying and manifest serialisation
# ---------------------------------------------------------------------------


def test_configurations_are_frozen() -> None:
    frozen = config()

    with pytest.raises(dataclasses.FrozenInstanceError):
        frozen.random_state = 1  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        frozen.features.radiation_thresholds = (1.0, 2.0)  # type: ignore[misc]


def test_replace_revalidates_instead_of_bypassing_validation() -> None:
    base = config(column_map=RFR3_MAPPING)

    assert base.replace(random_state=7).random_state == 7
    with pytest.raises(ConfigError, match="not a recognised Mode"):
        base.replace(mode="RFR5")
    with pytest.raises(ConfigError, match=r"sum to 1\.0"):
        GapScenarioConfig().replace(gap_mix={"short": 0.1, "long": 0.1, "very_long": 0.1})


def test_configuration_serialises_to_json_for_the_run_manifest() -> None:
    manifest = config(
        mode="RFR10",
        frequency="30min",
        latitude=51.5,
        site_id="GB-Ham",
        column_map=RFR10_MAPPING,
    ).to_dict()

    assert json.loads(json.dumps(manifest)) == manifest
    assert manifest["mode"] == "RFR10"
    assert manifest["site_id"] == "GB-Ham"
    assert manifest["hemisphere_source"] == "explicit"
    assert manifest["resolved_hemisphere"] == "north"
    assert manifest["features"]["feature_mode"] == "paper_safe"
    assert manifest["validation"]["gaps"]["allocation_basis"] == "missing_records"
    assert manifest["validation"]["gaps"]["durations"]["very_long"] == "P30DT0H0M0S"
    assert manifest["column_map"]["variables"]["soil_water_content"] == "SWC"


# ---------------------------------------------------------------------------
# ORF benchmark pairing (method_spec.md 3.6, Supplementary Figure S1)
# ---------------------------------------------------------------------------


def test_as_orf_changes_the_limiter_and_nothing_else() -> None:
    # Acceptance test 7: the benchmark keeps the estimator family, the driver
    # set, the seed and the search policy; only the feature stage differs.
    rfr = config(
        mode="RFR10",
        column_map=RFR10_MAPPING,
        random_state=1234,
        n_jobs=3,
        cv_folds=7,
        site_id="AA-Bbb",
    )
    orf = rfr.as_orf()

    assert orf.is_orf is True
    assert rfr.is_orf is False
    assert orf.drivers == rfr.drivers == RFR10_DRIVERS
    assert orf.random_state == rfr.random_state
    assert orf.n_jobs == rfr.n_jobs
    assert orf.cv_folds == rfr.cv_folds
    assert orf.cv == rfr.cv
    assert orf.hyperparameter_grid == rfr.hyperparameter_grid
    assert orf.columns == rfr.columns
    assert orf.observed_qc_values == rfr.observed_qc_values
    assert orf.validation == rfr.validation
    assert orf.site_id == rfr.site_id


def test_as_orf_preserves_every_other_feature_setting() -> None:
    # A benchmark that also reset the thresholds would measure that too.
    rfr = config(
        features=FeatureConfig(
            radiation_thresholds=(20.0, 200.0),
            boundary_convention="medium_exclusive",
            min_daily_observations=3,
            daily_std_ddof=0,
        )
    )
    orf = rfr.as_orf()

    assert orf.features.use_receptive_limiter is False
    assert orf.features.radiation_thresholds == (20.0, 200.0)
    assert orf.features.convention is BoundaryConvention.MEDIUM_EXCLUSIVE
    assert orf.features.min_daily_observations == 3
    assert orf.features.daily_std_ddof == 0


def test_as_orf_is_idempotent() -> None:
    orf = config().as_orf()
    assert orf.as_orf() == orf


def test_the_orf_benchmark_is_not_labelled_paper_faithful() -> None:
    # It is a supplementary comparison, not the published method.
    assert config().is_paper_faithful is True
    assert config().as_orf().is_paper_faithful is False


def test_a_derived_pair_is_accepted() -> None:
    rfr = config()
    require_orf_pairing(rfr, rfr.as_orf())
    assert orf_pairing_differences(rfr, rfr.as_orf()) == ()


@pytest.mark.parametrize(
    ("changes", "reported"),
    [
        ({"random_state": 99}, "random_state"),
        ({"cv_folds": 3}, "cv_folds"),
        ({"hyperparameter_grid": {"n_estimators": (50,)}}, "hyperparameter_grid.n_estimators"),
        ({"observed_qc_values": (0, 1)}, "observed_qc_values"),
    ],
)
def test_a_benchmark_that_changed_anything_else_is_rejected(
    changes: dict[str, Any], reported: str
) -> None:
    # "Do not redefine ORF to mean a different estimator, parameter grid, or
    # training dataset" is a checked precondition, not a comment.
    rfr = config()
    tampered = rfr.as_orf().replace(**changes)

    assert reported in orf_pairing_differences(rfr, tampered)
    with pytest.raises(ConfigError, match=reported.replace(".", r"\.")):
        require_orf_pairing(rfr, tampered)


def test_a_benchmark_with_a_different_driver_set_is_rejected() -> None:
    rfr = config(mode="RFR10", column_map=RFR10_MAPPING)
    rfr3_arm = config(mode="RFR3").as_orf()
    with pytest.raises(ConfigError, match="mode"):
        require_orf_pairing(rfr, rfr3_arm)


def test_both_arms_must_actually_be_the_two_arms() -> None:
    rfr = config()
    with pytest.raises(ConfigError, match="use_receptive_limiter=False"):
        require_orf_pairing(rfr, rfr)
    with pytest.raises(ConfigError, match="compares ORF with itself"):
        require_orf_pairing(rfr.as_orf(), rfr.as_orf())


def test_the_pairing_difference_report_names_nested_settings() -> None:
    rfr = config()
    tampered = rfr.as_orf().replace(
        features=rfr.features.replace(use_receptive_limiter=False, daily_std_ddof=0)
    )
    assert orf_pairing_differences(rfr, tampered) == ("features.daily_std_ddof",)


# ---------------------------------------------------------------------------
# Serialisation (method_spec.md section 5: the configuration travels with the model)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "configuration",
    [
        RFRConfig(mode="RFR3", hemisphere="north", column_map=RFR3_MAPPING),
        RFRConfig(
            mode="RFR10",
            latitude=-33.5,
            column_map=RFR10_MAPPING,
            hyperparameter_grid={"n_estimators": [10, 20], "max_depth": [None, 8]},
            cv_strategy=CVStrategy.TIME_SERIES_SPLIT,
            features=FeatureConfig(daily_statistic_strategy="rolling_available"),
            validation=ValidationConfig(gaps=GapScenarioConfig(missing_fraction=0.1)),
        ),
        GapScenarioConfig(),
        FeatureConfig(),
        ValidationConfig(),
        ColumnMap(RFR10_MAPPING, timestamp="TIMESTAMP_START"),
    ],
)
def test_a_configuration_survives_a_pickle_round_trip(configuration: Any) -> None:
    """A joblib-saved model carries its configuration, which read-only mappings block."""
    restored = pickle.loads(pickle.dumps(configuration))

    assert restored == configuration
    assert restored.to_dict() == configuration.to_dict()
    assert copy.deepcopy(configuration) == configuration


def test_a_restored_configuration_is_revalidated_rather_than_trusted() -> None:
    """A tampered model file must fail on load, not predict under rejected settings."""
    state = {**RFRConfig(mode="RFR3", hemisphere="north").__getstate__(), "random_state": -1}

    with pytest.raises(ConfigError, match="random_state"):
        object.__new__(RFRConfig).__setstate__(state)


def test_a_restored_mapping_field_is_read_only_again() -> None:
    restored = pickle.loads(pickle.dumps(RFRConfig(mode="RFR3", hemisphere="north")))

    with pytest.raises(TypeError):
        restored.hyperparameter_grid["n_estimators"] = (1,)  # type: ignore[index]
