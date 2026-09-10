"""End-to-end artificial-gap validation tests. Covers acceptance tests 34-35 and 40.

The exit criterion for the orchestration step is that one call runs the whole
Zhu-style experiment transparently, so most of these tests assert on the shape
and the provenance of what comes back rather than on any particular metric
value: the synthetic site is not the paper's, and its numbers are not the
paper's either.

The exceptions are the two that must hold for the run to mean anything at all -
that the gaps are contiguous 24-hour, 7-day and 30-day intervals rather than a
random row-wise holdout, and that no held-out truth reached the features built
to predict it.

See ``docs/method_spec.md`` for the contract and
``docs/supplement_benchmarks.md`` for benchmark provenance.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from fixtures import synthetic_site
from rfrgapfill.config import (
    ColumnMap,
    FeatureConfig,
    GapClass,
    GapScenarioConfig,
    RFRConfig,
    ValidationConfig,
)
from rfrgapfill.schema import ConfigError
from rfrgapfill.validation import (
    SCENARIOS,
    ValidationError,
    ValidationReport,
    compare_receptive_limiter,
    validate_rfr,
)

pytestmark = pytest.mark.slow

#: Small enough to fit repeatedly, large enough to be a real search.
FAST_GRID = {"n_estimators": (30,), "min_samples_leaf": (1, 5)}

QC_COLUMNS = {"NEE": "NEE_QC", "H": "H_QC", "LE": "LE_QC"}


def config(mode: str = "RFR10", **changes: object) -> RFRConfig:
    settings: dict[str, object] = {
        "mode": mode,
        "frequency": "30min",
        "latitude": 51.5,
        "site_id": "SYN-01",
        "column_map": ColumnMap.fluxnet2015("RFR10"),
        "hyperparameter_grid": FAST_GRID,
        "cv_folds": 3,
    }
    settings.update(changes)
    return RFRConfig(**settings)  # type: ignore[arg-type]


@pytest.fixture(scope="module")
def site() -> pd.DataFrame:
    # Long enough to carry two 30-day gaps and the shorter classes around them.
    return synthetic_site(days=400, seed=0)


@pytest.fixture(scope="module")
def report(site: pd.DataFrame) -> ValidationReport:
    return validate_rfr(site, targets=["NEE", "H", "LE"], config=config(), qc_columns=QC_COLUMNS)


# ---------------------------------------------------------------------------
# One call runs the whole experiment (acceptance tests 34-35)
# ---------------------------------------------------------------------------


def test_one_call_returns_predictions_metrics_manifest_and_configuration(
    report: ValidationReport,
) -> None:
    assert report.targets == ("NEE", "H", "LE")
    assert not report.predictions().empty
    assert not report.metrics_frame().empty
    assert not report.gap_manifest.empty
    assert report.config.rfr_mode.value == "RFR10"


def test_all_three_gap_classes_are_exercised_and_reported(report: ValidationReport) -> None:
    classes = set(report.gap_manifest["gap_class"])
    assert classes == {"short", "long", "very_long"}

    for target in report.targets:
        assert set(report[target].by_gap_class) == {
            GapClass.SHORT,
            GapClass.LONG,
            GapClass.VERY_LONG,
        }


def test_the_gaps_are_contiguous_intervals_not_a_random_row_holdout(
    report: ValidationReport,
) -> None:
    manifest = report.gap_manifest
    durations = {
        "short": pd.Timedelta(hours=24),
        "long": pd.Timedelta(days=7),
        "very_long": pd.Timedelta(days=30),
    }
    for gap_class, expected in durations.items():
        rows = manifest[manifest["gap_class"] == gap_class]
        assert len(rows) > 0
        assert ((rows["end"] - rows["start"]) == expected).all()


def test_the_withheld_rows_are_the_test_set_and_the_rest_is_training(
    report: ValidationReport, site: pd.DataFrame
) -> None:
    result = report["NEE"]
    withheld = result.n_withheld
    trained = result.training.n_trained

    # Roughly a quarter withheld, the remainder trained on: Figure 2's design.
    assert result.gaps.achieved_fraction == pytest.approx(0.25, abs=0.06)
    assert trained > withheld * 2
    # Nothing withheld was trained on.
    assert (
        not (result.gaps.mask & site["NEE"].notna())[report.time_axis.index]
        .reindex(result.predictions.index)
        .eq(False)
        .any()
    )


def test_every_subset_is_reported_never_the_aggregate_alone(report: ValidationReport) -> None:
    frame = report.metrics_frame()
    assert set(frame["subset"]) == {"all", "daytime", "nighttime"}

    for target in report.targets:
        metrics = report[target].metrics
        assert metrics.all.n == metrics.daytime.n + metrics.nighttime.n
        assert np.isfinite(metrics.nighttime.rmse)


def test_the_model_learns_something_on_a_learnable_site(report: ValidationReport) -> None:
    # Not a paper benchmark: the fixture is synthetic. It only asserts the
    # orchestration wired the pieces together rather than predicting noise.
    for target in report.targets:
        assert report[target].metrics.daytime.r2 > 0.5


# ---------------------------------------------------------------------------
# Shared gaps across targets (acceptance test 20)
# ---------------------------------------------------------------------------


def test_the_three_targets_are_scored_on_identical_gap_locations(
    report: ValidationReport,
) -> None:
    assert report.shared_gaps
    masks = [report[target].gaps.mask for target in report.targets]
    for mask in masks[1:]:
        pd.testing.assert_series_equal(mask, masks[0])


def test_separate_scenarios_are_available_and_are_not_identical(site: pd.DataFrame) -> None:
    settings = config(
        validation=ValidationConfig(gaps=GapScenarioConfig(shared_gaps_across_targets=False))
    )
    separate = validate_rfr(site, targets=["NEE", "H"], config=settings, qc_columns=QC_COLUMNS)

    assert not separate.shared_gaps
    assert not separate["NEE"].gaps.mask.equals(separate["H"].gaps.mask)
    assert "target" in separate.gap_manifest.columns


# ---------------------------------------------------------------------------
# Leakage, end to end
# ---------------------------------------------------------------------------


def test_altering_hidden_truth_does_not_change_the_features_that_predict_it(
    site: pd.DataFrame,
) -> None:
    settings = config("RFR3")
    base = validate_rfr(site, targets="NEE", config=settings, qc_columns=QC_COLUMNS)

    # Corrupt only the values the artificial gaps hide, then rerun.
    corrupted = site.copy()
    hidden = base["NEE"].gaps.mask.reindex(corrupted.index, fill_value=False)
    corrupted.loc[hidden, "NEE"] = 1e6

    altered = validate_rfr(corrupted, targets="NEE", config=settings, qc_columns=QC_COLUMNS)

    pd.testing.assert_frame_equal(base["NEE"].features.frame, altered["NEE"].features.frame)
    pd.testing.assert_series_equal(
        base["NEE"].predictions["predicted"], altered["NEE"].predictions["predicted"]
    )


def test_the_same_seed_reproduces_the_whole_run(site: pd.DataFrame) -> None:
    settings = config("RFR3")
    first = validate_rfr(site, targets="LE", config=settings, qc_columns=QC_COLUMNS)
    second = validate_rfr(site, targets="LE", config=settings, qc_columns=QC_COLUMNS)

    pd.testing.assert_frame_equal(first.gap_manifest, second.gap_manifest)
    pd.testing.assert_frame_equal(first["LE"].predictions, second["LE"].predictions)


# ---------------------------------------------------------------------------
# Energy-balance ratio (acceptance test 33, applied end to end)
# ---------------------------------------------------------------------------


def test_the_energy_balance_ratio_is_reported_for_h_and_le(report: ValidationReport) -> None:
    balance = report.energy_balance
    assert balance is not None
    assert balance.n_rows > 0
    assert np.isfinite(balance.measured)
    assert np.isfinite(balance.filled)
    assert balance.difference == balance.filled - balance.measured


def test_no_energy_balance_is_reported_when_only_one_heat_flux_was_validated(
    site: pd.DataFrame,
) -> None:
    single = validate_rfr(site, targets="H", config=config("RFR3"), qc_columns=QC_COLUMNS)
    assert single.energy_balance is None


def test_naming_an_unvalidated_target_for_the_ebr_is_an_error(site: pd.DataFrame) -> None:
    with pytest.raises(ValidationError, match="did not validate"):
        validate_rfr(
            site,
            targets=["H", "LE"],
            config=config(),
            qc_columns=QC_COLUMNS,
            energy_balance_targets=("H", "NEE"),
        )


# ---------------------------------------------------------------------------
# Bias IQR by gap class (acceptance test 40)
# ---------------------------------------------------------------------------


def test_bias_iqr_is_reported_per_gap_class(report: ValidationReport) -> None:
    for gap_class, result in report["NEE"].by_gap_class.items():
        assert result.n_gaps >= 1
        assert len(result.bias_by_gap) == result.n_gaps
        if result.n_gaps >= 2:
            assert np.isfinite(result.bias_iqr)
        assert gap_class in GapClass


def test_bias_iqr_is_missing_rather_than_zero_for_a_single_gap() -> None:
    from rfrgapfill.metrics import CoreMetrics, SubsetMetrics
    from rfrgapfill.validation import GapClassMetrics

    single = GapClassMetrics(
        gap_class=GapClass.VERY_LONG,
        metrics=SubsetMetrics(metrics={}, daytime_threshold=20.0, n_missing_shortwave=0),
        n_gaps=1,
        bias_by_gap=(0.5,),
    )
    assert np.isnan(single.bias_iqr)
    assert CoreMetrics  # imported for the fixture's shape, not asserted on


# ---------------------------------------------------------------------------
# The ORF benchmark (acceptance test 10)
# ---------------------------------------------------------------------------


def test_rfr_and_orf_are_compared_on_identical_gaps(site: pd.DataFrame) -> None:
    rfr, orf, comparison = compare_receptive_limiter(
        site, targets="LE", config=config("RFR3"), qc_columns=QC_COLUMNS
    )

    pd.testing.assert_frame_equal(rfr.gap_manifest, orf.gap_manifest)
    assert rfr["LE"].features.use_receptive_limiter
    assert not orf["LE"].features.use_receptive_limiter
    # Same drivers, same estimator family, same grid: only the features differ.
    assert orf["LE"].training.feature_names == rfr["LE"].training.feature_names[:3]
    assert rfr.config.hyperparameter_grid == orf.config.hyperparameter_grid

    # Recorded, not asserted: the supplement does not claim RFR wins every metric.
    assert {"r2_rfr", "r2_orf", "r2_difference"} <= set(comparison.columns)
    assert not comparison.empty


# ---------------------------------------------------------------------------
# Provenance and reporting
# ---------------------------------------------------------------------------


def test_the_run_manifest_records_everything_the_specification_asks_for(
    report: ValidationReport,
) -> None:
    manifest = report.manifest()
    assert json.loads(json.dumps(manifest)) == manifest

    assert manifest["config"]["mode"] == "RFR10"
    assert manifest["config"]["site_id"] == "SYN-01"
    assert manifest["config"]["random_state"] == 42
    assert manifest["config"]["features"]["feature_mode"] == "paper_safe"
    assert manifest["config"]["resolved_hemisphere"] == "north"
    assert manifest["config"]["validation"]["gaps"]["allocation_basis"] == "missing_records"
    assert manifest["time_axis"]["time_step"] == "P0DT0H30M0S"
    assert manifest["gaps"]["shared"] is True
    assert manifest["training"]["NEE"]["best_params"]
    assert manifest["environment"]["scikit_learn"]
    assert manifest["paper_doi"] == "10.1016/j.agrformet.2021.108777"


def test_the_whole_report_serialises_to_json(report: ValidationReport) -> None:
    payload = report.to_dict()
    assert json.loads(json.dumps(payload)) == payload
    assert set(payload["results"]) == {"NEE", "H", "LE"}


def test_the_report_states_the_achieved_fraction_and_mix_against_the_request(
    report: ValidationReport,
) -> None:
    manifest = report.manifest()["gaps"]["by_target"]["NEE"]

    assert manifest["requested_fraction"] == 0.25
    assert manifest["achieved_fraction"] == pytest.approx(0.25, abs=0.06)
    assert set(manifest["achieved_mix"]) == {"short", "long", "very_long"}
    assert isinstance(manifest["satisfied"], bool)
    # A report never says the design was missed without saying why.
    assert bool(report.warnings) is not report.satisfied


def test_the_daily_statistic_fallback_is_counted_not_hidden(report: ValidationReport) -> None:
    daily = report["NEE"].features.daily_statistics
    assert daily is not None
    # A 30-day gap has no visible target observation of its own, so its days must
    # have taken a neighbour's statistics - and the count must say so.
    assert daily.n_days_from_neighbour > 0
    assert daily.to_dict()["strategy"] == "nearest_visible_day"


def test_every_withheld_row_carries_its_gap_id_and_class(report: ValidationReport) -> None:
    predictions = report["NEE"].predictions
    assert predictions["gap_id"].notna().all()
    assert predictions["gap_class"].notna().all()
    assert set(predictions["gap_class"]) <= {"short", "long", "very_long"}


# ---------------------------------------------------------------------------
# Argument handling
# ---------------------------------------------------------------------------


def test_the_named_scenario_is_the_published_design() -> None:
    scenario = SCENARIOS["zhu2022"]
    assert scenario.missing_fraction == 0.25
    assert scenario.share(GapClass.SHORT) == 0.20
    assert scenario.share(GapClass.LONG) == 0.30
    assert scenario.share(GapClass.VERY_LONG) == 0.50
    assert scenario.min_observed_fraction == 0.50


def test_an_unknown_scenario_name_lists_the_registered_ones(site: pd.DataFrame) -> None:
    with pytest.raises(ValidationError, match="unknown scenario"):
        validate_rfr(site, targets="NEE", config=config(), scenario="made_up")


def test_a_missing_driver_set_is_refused_rather_than_defaulted(site: pd.DataFrame) -> None:
    with pytest.raises(ConfigError, match="no default driver set"):
        validate_rfr(site, targets="NEE", column_map=ColumnMap.fluxnet2015("RFR3"))


def test_mixing_a_config_with_loose_settings_is_refused(site: pd.DataFrame) -> None:
    with pytest.raises(ConfigError, match="not both"):
        validate_rfr(site, targets="NEE", config=config(), mode="RFR3")


def test_a_target_the_frame_does_not_carry_is_named(site: pd.DataFrame) -> None:
    with pytest.raises(ValidationError, match="FCH4"):
        validate_rfr(site, targets=["NEE", "FCH4"], config=config())


def test_the_loose_settings_build_the_same_configuration(site: pd.DataFrame) -> None:
    loose = validate_rfr(
        site.head(48 * 200),
        targets="NEE",
        mode="RFR3",
        latitude=51.5,
        frequency="30min",
        column_map=ColumnMap.fluxnet2015("RFR3"),
        qc_columns=QC_COLUMNS,
        scenario=GapScenarioConfig(gap_mix={"short": 1.0}),
    )
    assert loose.config.rfr_mode.value == "RFR3"
    assert loose.config.resolve_hemisphere().value == "north"
    assert set(loose.gap_manifest["gap_class"]) == {"short"}


def test_the_orf_switch_is_reachable_from_the_loose_settings(site: pd.DataFrame) -> None:
    orf = validate_rfr(
        site.head(48 * 200),
        targets="NEE",
        mode="RFR3",
        latitude=51.5,
        frequency="30min",
        use_receptive_limiter=False,
        column_map=ColumnMap.fluxnet2015("RFR3"),
        qc_columns=QC_COLUMNS,
        scenario=GapScenarioConfig(gap_mix={"short": 0.5, "long": 0.5}),
    )
    assert not orf.config.features.use_receptive_limiter
    assert not orf.config.is_paper_faithful
    assert orf["NEE"].features.daily_statistics is None


def test_a_within_day_strategy_leaves_long_gaps_all_but_unpredictable(
    site: pd.DataFrame,
) -> None:
    # The literal reading of ambiguity A4, and what it costs. A 30-day gap holds
    # no visible target observation, so no whole day inside it has statistics of
    # its own and almost nothing inside it can be predicted - the survivors are
    # the partial days at each end, where the gap starts and stops mid-day.
    settings = config(
        "RFR3",
        features=FeatureConfig(daily_statistics_strategy="within_day"),
        validation=ValidationConfig(gaps=GapScenarioConfig(gap_mix={"very_long": 1.0})),
    )
    strict = validate_rfr(site, targets="NEE", config=settings, qc_columns=QC_COLUMNS)
    daily = strict["NEE"].features.daily_statistics
    assert daily is not None

    assert strict["NEE"].coverage < 0.10
    assert daily.n_days_missing > 0
    assert daily.n_days_from_neighbour == 0

    # The default recovers what the strict reading forfeits, without ever reading
    # a held-out value: see the leakage test above, which uses the same default.
    lenient = validate_rfr(
        site,
        targets="NEE",
        config=config("RFR3", validation=settings.validation),
        qc_columns=QC_COLUMNS,
    )
    assert lenient["NEE"].coverage == 1.0
