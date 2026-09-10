"""End-to-end artificial-gap validation. Covers Steps 13 and 17, acceptance 34-35.

Step 17 asks for one thing that no unit test can give: both published
configurations, run whole, on data with known structure, reporting R2, slope,
RMSE, bias, day/night, metrics by gap class and the gap manifest - and leaving
the observed values exactly as they were. The ``arms`` fixture is that run, done
once for RFR3 and once for RFR10 over the same fixed intervals, and most of this
module reads its output rather than producing more of it.

The forests are deliberately tiny (15 trees, 3 folds, a one-point grid): nothing
here tests predictive quality, and Step 17 explicitly refuses to require RFR10 to
beat RFR3 on any individual metric. What is asserted is that the workflow runs,
that it is deterministic, that the numbers it reports are the ones its own
predictions imply, and that nothing it did touched a measurement.

The intervals come from :meth:`SyntheticSite.known_gaps` rather than from the
sampler, so a change in placement cannot move a metric here; the sampled
generator gets its own test.
"""

from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
import pytest

from rfrgapfill.config import FeatureConfig, MetricSubset, RFRConfig, ValidationConfig
from rfrgapfill.gaps import GapManifest
from rfrgapfill.metrics import compare_energy_balance, core_metrics
from rfrgapfill.provenance import RunManifest
from rfrgapfill.schema import NET_RADIATION, SOIL_HEAT_FLUX, ColumnMap, ConfigError, GapClass
from rfrgapfill.synthetic import PRE_FILLED_QC, TARGETS, synthetic_site
from rfrgapfill.validation import (
    ENERGY_BALANCE_TABLE_COLUMNS,
    METRIC_TABLE_COLUMNS,
    BiasSpread,
    EnergyBalanceCheck,
    ValidationError,
    ValidationReport,
    ValidationWarning,
    daytime_mask,
    gap_class_labels,
    subset_masks,
    validate_rfr,
)

DAYTIME_THRESHOLD = 20.0

#: One grid point and three folds. This module scores nothing for quality.
TEST_GRID = {"n_estimators": (15,)}

#: The reaching strategy a long-gap run has to choose deliberately (A4): under
#: the default, a 7-day gap has no daily statistics and nothing in it is scored.
REACHING = FeatureConfig(daily_statistic_strategy="rolling_available")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def site():
    """The reference synthetic site: one year of half-hourly data, seeded."""
    return synthetic_site()


@pytest.fixture(scope="module")
def gaps(site) -> GapManifest:
    """The fixed 30-day / 7-day / 24-hour intervals, shared by every arm."""
    return site.known_gaps()


def _config(site, mode: str) -> RFRConfig:
    """Return a fast validation configuration for one arm of the Step 17 run."""
    return site.config(
        mode,
        hyperparameter_grid=TEST_GRID,
        cv_folds=3,
        features=REACHING,
    )


def _run(site, gaps: GapManifest, mode: str) -> ValidationReport:
    """Validate all three fluxes on the fixed intervals under one driver set."""
    return validate_rfr(
        site.frame,
        config=_config(site, mode),
        targets=list(TARGETS),
        qc_columns=dict(site.qc_columns()),
        gaps=gaps,
    )


@pytest.fixture(scope="module")
def arms(site, gaps) -> dict[str, ValidationReport]:
    """Step 17 itself: RFR3 and RFR10, run end to end over identical gaps."""
    return {"RFR3": _run(site, gaps, "RFR3"), "RFR10": _run(site, gaps, "RFR10")}


@pytest.fixture(scope="module")
def report(arms) -> ValidationReport:
    """The RFR3 arm, for assertions that do not care which arm produced them."""
    return arms["RFR3"]


# ---------------------------------------------------------------------------
# Row selections
# ---------------------------------------------------------------------------


class TestSubsets:
    """The day/night split of Step 12 and method_spec.md section 6.2."""

    def test_boundary_value_is_night(self):
        radiation = pd.Series(
            [19.9, 20.0, 20.1], index=pd.date_range("2020-06-01", periods=3, freq="30min")
        )
        masks = subset_masks(radiation, threshold=DAYTIME_THRESHOLD)
        assert list(masks[MetricSubset.DAYTIME]) == [False, False, True]
        assert list(masks[MetricSubset.NIGHTTIME]) == [True, True, False]

    def test_threshold_is_configurable(self):
        radiation = pd.Series([5.0, 15.0], index=pd.date_range("2020-06-01", periods=2, freq="h"))
        assert list(daytime_mask(radiation, threshold=10.0)) == [False, True]
        assert list(daytime_mask(radiation, threshold=DAYTIME_THRESHOLD)) == [False, False]

    def test_missing_radiation_is_neither_day_nor_night(self):
        radiation = pd.Series(
            [np.nan, 100.0], index=pd.date_range("2020-06-01", periods=2, freq="h")
        )
        masks = subset_masks(radiation)
        assert list(masks[MetricSubset.ALL]) == [True, True]
        assert list(masks[MetricSubset.DAYTIME]) == [False, True]
        assert list(masks[MetricSubset.NIGHTTIME]) == [False, False]

    def test_day_and_night_partition_the_known_rows(self, site):
        masks = subset_masks(site.frame["SW_IN_F"], threshold=DAYTIME_THRESHOLD)
        day = masks[MetricSubset.DAYTIME].to_numpy()
        night = masks[MetricSubset.NIGHTTIME].to_numpy()
        assert not (day & night).any()
        assert (day | night).all()  # the synthetic radiation is never missing


class TestGapClassLabels:
    """Labelling each row with the duration class that withheld it."""

    def test_labels_follow_the_intervals(self, site, gaps):
        labels = gap_class_labels(gaps, site.index)
        for gap in gaps:
            inside = labels.loc[gap.start : gap.end - site.time_step]
            assert set(inside) == {gap.gap_class.value}
        assert labels.notna().sum() == gaps.n_withheld_rows

    def test_the_interval_end_is_not_labelled(self, site, gaps):
        """The intervals are half-open, so the row at ``end`` was never withheld."""
        labels = gap_class_labels(gaps, site.index)
        for gap in gaps:
            assert labels.get(gap.end) is None

    def test_rejects_something_that_is_not_a_manifest(self, site):
        with pytest.raises(ValidationError, match="GapManifest"):
            gap_class_labels("every 30 days", site.index)


# ---------------------------------------------------------------------------
# Step 17: both published configurations, end to end
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestBothArmsRun:
    """Step 17's exit criterion: RFR3 and RFR10 both run, and report what is asked."""

    def test_each_arm_validates_every_target(self, arms):
        for method, run in arms.items():
            assert run.method == method
            assert run.targets == TARGETS
            assert len(run) == 3

    def test_each_arm_uses_its_own_driver_set(self, arms):
        assert len(arms["RFR3"].config.drivers) == 3
        assert len(arms["RFR10"].config.drivers) == 10
        for method, run in arms.items():
            for result in run:
                assert set(arms[method].config.drivers) <= set(result.model.get_feature_names())

    @pytest.mark.parametrize("metric", ["r2", "slope", "rmse", "bias"])
    def test_the_four_core_metrics_are_reported(self, arms, metric):
        for run in arms.values():
            for result in run:
                assert getattr(result.overall, metric) is not None

    def test_day_and_night_are_reported_separately(self, arms):
        for run in arms.values():
            for result in run:
                day = result.metric(subset="daytime")
                night = result.metric(subset="nighttime")
                assert day.n > 0 and night.n > 0
                assert day.n + night.n == result.overall.n

    def test_metrics_are_reported_by_gap_class(self, arms):
        for run in arms.values():
            for result in run:
                classes = set(result.metrics_by_gap_class)
                assert classes == set(GapClass)
                scored = sum(result.metric(gap_class=gap_class).n for gap_class in GapClass)
                assert scored == result.overall.n

    def test_the_gap_manifest_travels_with_the_result(self, arms, gaps):
        for run in arms.values():
            assert run.gaps is gaps
            assert {gap.gap_class for gap in run.gaps} == set(GapClass)
            assert len(run.gaps.to_frame()) == len(gaps)

    def test_all_three_durations_are_present_and_elapsed(self, gaps):
        expected = {
            GapClass.SHORT: pd.Timedelta(hours=24),
            GapClass.LONG: pd.Timedelta(days=7),
            GapClass.VERY_LONG: pd.Timedelta(days=30),
        }
        for gap in gaps:
            assert gap.duration == expected[gap.gap_class]

    def test_the_run_summarises_itself(self, report):
        text = report.summary()
        assert "RFR3 validation of NEE, H, LE" in text
        assert "very_long" in text and "nighttime" in text


@pytest.mark.slow
class TestObservedValuesAreUnchanged:
    """Step 17 task 4: nothing the run did touched a measurement."""

    def test_the_input_frame_is_not_modified(self, site, gaps):
        before = site.frame.copy(deep=True)
        _run(site, gaps, "RFR3")
        pd.testing.assert_frame_equal(site.frame, before)

    def test_the_scored_truth_is_the_input_column(self, site, report):
        for target in TARGETS:
            truth = report[target].features.truth
            original = site.frame[target].astype(float)
            pd.testing.assert_series_equal(truth, original, check_names=False)

    def test_no_prediction_lands_outside_an_artificial_gap(self, report):
        for result in report:
            outside = ~result.features.holdout_mask.to_numpy()
            assert result.predictions.to_numpy()[outside].size
            assert np.isnan(result.predictions.to_numpy()[outside]).all()

    def test_observed_rows_keep_their_observed_flag(self, site, report):
        for target in TARGETS:
            observed = report[target].features.observed_mask
            expected = site.observed(target)
            pd.testing.assert_series_equal(observed, expected, check_names=False)


@pytest.mark.slow
class TestDeterminism:
    """Step 17's exit criterion: both workflows run *deterministically*."""

    def test_the_same_seed_reproduces_the_metrics(self, site, gaps, report):
        again = _run(site, gaps, "RFR3")
        pd.testing.assert_frame_equal(report.to_frame(), again.to_frame())

    def test_the_same_seed_reproduces_the_predictions(self, site, gaps, report):
        again = _run(site, gaps, "RFR3")
        pd.testing.assert_frame_equal(report.predictions(), again.predictions())


@pytest.mark.slow
class TestTheNumbersAreTheirOwnPredictions:
    """The reported metrics describe the predictions this run actually made."""

    def test_bias_is_the_paper_formula_over_the_scored_rows(self, report):
        for result in report:
            rows = result.features.scoring_mask.to_numpy()
            measured = result.features.truth.to_numpy()[rows]
            predicted = result.predictions.to_numpy()[rows]
            complete = np.isfinite(measured) & np.isfinite(predicted)
            expected = (predicted[complete].sum() - measured[complete].sum()) / complete.sum()
            assert result.overall.bias == pytest.approx(expected)
            assert result.overall.n == int(complete.sum())

    def test_a_subset_scores_exactly_its_own_rows(self, site, report):
        result = report["LE"]
        night = subset_masks(site.frame["SW_IN_F"], threshold=DAYTIME_THRESHOLD)[
            MetricSubset.NIGHTTIME
        ]
        rows = (result.features.scoring_mask & night).to_numpy()
        expected = core_metrics(
            result.features.truth.loc[rows],
            result.predictions.loc[rows],
            definition=result.config.validation.r2,
        )
        assert result.metric(subset="nighttime").to_dict() == expected.to_dict()

    def test_a_row_the_model_could_not_predict_is_not_scored(self, report):
        for result in report:
            withheld = result.features.holdout_mask
            complete = int((withheld & result.features.complete_mask).sum())
            assert result.overall.n <= complete
            assert result.overall.n_offered == int(result.features.scoring_mask.sum())


@pytest.mark.slow
class TestSharedGapsAndComparability:
    """One mask, every target and every arm (method_spec.md 4.4 and 3.6)."""

    def test_the_targets_share_one_holdout_mask(self, report):
        masks = [result.features.holdout_mask for result in report]
        for mask in masks[1:]:
            pd.testing.assert_series_equal(masks[0], mask)

    def test_the_arms_are_scored_on_the_same_rows(self, arms):
        for target in TARGETS:
            three = arms["RFR3"][target].features.scoring_mask
            ten = arms["RFR10"][target].features.scoring_mask
            pd.testing.assert_series_equal(three, ten)
            assert arms["RFR3"][target].overall.n == arms["RFR10"][target].overall.n

    def test_the_arms_differ_only_in_their_features(self, arms):
        three = set(arms["RFR3"]["LE"].model.get_feature_names())
        ten = set(arms["RFR10"]["LE"].model.get_feature_names())
        assert three < ten
        assert ten - three == set(arms["RFR10"].config.drivers) - set(arms["RFR3"].config.drivers)


@pytest.mark.slow
class TestReportingSurfaces:
    """The tables, the manifest and the JSON a validation run has to produce."""

    def test_the_tidy_table_has_one_row_per_target_class_and_subset(self, report):
        table = report.to_frame()
        assert tuple(table.columns) == METRIC_TABLE_COLUMNS
        assert len(table) == 3 * (1 + len(GapClass)) * 3
        assert set(table["gap_class"]) == {"all", "short", "long", "very_long"}
        assert set(table["subset"]) == {"all", "daytime", "nighttime"}

    def test_the_table_agrees_with_the_records(self, report):
        table = report.to_frame()
        row = table.query("target == 'H' and gap_class == 'long' and subset == 'daytime'")
        scores = report["H"].metric(subset="daytime", gap_class="long")
        assert row["r2"].item() == pytest.approx(scores.r2)
        assert row["n"].item() == scores.n

    def test_bias_spread_is_reported_per_gap_class(self, report):
        for result in report:
            assert set(result.bias_spread) == set(GapClass)
            for gap_class, spread in result.bias_spread.items():
                assert isinstance(spread, BiasSpread)
                assert spread.n_gaps_offered == len(result.gaps.by_class(gap_class))
                if spread.n_gaps > 1:
                    assert spread.iqr == pytest.approx(spread.q3 - spread.q1)
                    assert spread.q1 <= spread.median <= spread.q3

    def test_bias_spread_of_a_single_gap_has_no_iqr(self, report):
        very_long = report["LE"].bias_spread[GapClass.VERY_LONG]
        assert very_long.n_gaps == 1
        assert very_long.iqr is None
        assert very_long.median is not None

    def test_the_manifest_is_complete_and_serialisable(self, report):
        manifest = report.manifest("LE")
        assert isinstance(manifest, RunManifest)
        manifest.require_complete()
        document = json.loads(json.dumps(manifest.to_dict()))
        assert document["kind"] == "validation"
        assert document["gaps"]["manifest"]["gaps"]
        assert document["metrics"]["metrics"]["nighttime"]["rmse"] is not None

    def test_the_whole_run_is_serialisable(self, report):
        document = json.loads(json.dumps(report.to_dict()))
        assert document["method"] == "RFR3"
        assert set(document["results"]) == set(TARGETS)
        assert document["energy_balance"]["measured_ebr"] is not None

    def test_predictions_come_back_on_the_run_axis(self, site, report):
        frame = report.predictions()
        assert list(frame.columns) == [f"{target}_predicted" for target in TARGETS]
        assert frame.index.equals(site.index)


def _drivers(site, report: ValidationReport) -> pd.DataFrame:
    """NETRAD and G as the run read them; the site frame is on the run's time axis."""
    columns = report.column_map
    return site.frame[[columns.column(NET_RADIATION), columns.column(SOIL_HEAT_FLUX)]]


def _complete_ebr_rows(site, report: ValidationReport) -> pd.Series:
    """Rows carrying every component of the energy-balance check."""
    heat, latent = report["H"], report["LE"]
    complete: pd.Series = (
        heat.features.scoring_mask
        & latent.features.scoring_mask
        & heat.predictions.notna()
        & latent.predictions.notna()
        & _drivers(site, report).notna().all(axis=1)
    )
    return complete


@pytest.mark.slow
class TestEnergyBalance:
    """Section 6.4's independent check, over the artificial-gap rows (Step 19)."""

    def test_measured_and_filled_ratios_share_one_row_set(self, report):
        balance = report.energy_balance
        assert balance is not None
        assert balance.n > 0
        assert balance.difference == pytest.approx(balance.filled - balance.measured)

    def test_the_measured_ratio_is_the_arms_common_ground(self, arms):
        assert arms["RFR3"].energy_balance.measured == pytest.approx(
            arms["RFR10"].energy_balance.measured
        )

    def test_every_withheld_row_is_offered(self, report):
        check = report.energy_balance_check
        assert isinstance(check, EnergyBalanceCheck)
        assert check.targets == ("H", "LE")
        assert check.overall.n_offered == report["H"].withheld_rows

    def test_the_ratios_are_the_formula_over_this_runs_own_rows(self, site, report):
        heat, latent = report["H"], report["LE"]
        rows = _complete_ebr_rows(site, report).to_numpy()
        netrad = site.frame[report.column_map.column(NET_RADIATION)].to_numpy()[rows]
        soil = site.frame[report.column_map.column(SOIL_HEAT_FLUX)].to_numpy()[rows]
        available = (netrad - soil).sum()
        measured = (heat.features.truth.to_numpy() + latent.features.truth.to_numpy())[rows]
        filled = (heat.predictions.to_numpy() + latent.predictions.to_numpy())[rows]

        overall = report.energy_balance_check.overall
        assert overall.n == int(rows.sum())
        assert overall.available_energy == pytest.approx(available)
        assert overall.measured == pytest.approx(measured.sum() / available)
        assert overall.filled == pytest.approx(filled.sum() / available)

    def test_the_accounting_explains_every_dropped_row(self, site, report):
        heat, latent = report["H"], report["LE"]
        withheld = heat.features.holdout_mask
        genuine = heat.features.scoring_mask & latent.features.scoring_mask
        unpredicted = heat.predictions.isna() | latent.predictions.isna()
        drivers = _drivers(site, report)
        no_energy = drivers.isna().any(axis=1)

        overall = report.energy_balance_check.overall
        assert overall.n_missing_measured == int((withheld & ~genuine).sum())
        assert overall.n_missing_filled == int((withheld & unpredicted).sum())
        assert overall.n_missing_available_energy == int((withheld & no_energy).sum())
        assert overall.dropped_incomplete == int(
            (withheld & ~_complete_ebr_rows(site, report)).sum()
        )

    def test_each_gap_class_is_checked_and_the_classes_partition_the_run(self, report):
        check = report.energy_balance_check
        assert set(check.by_gap_class) == set(GapClass)
        assert sum(item.n for item in check.by_gap_class.values()) == check.overall.n
        assert (
            sum(item.n_offered for item in check.by_gap_class.values()) == check.overall.n_offered
        )
        labels = gap_class_labels(report.gaps, report.time_axis.index)
        for gap_class, item in check.by_gap_class.items():
            assert item.n_offered == int((labels == gap_class.value).sum())
            assert item.n > 0
            assert item.measured is not None and item.filled is not None
            assert item.difference == pytest.approx(item.filled - item.measured)

    def test_an_unchecked_gap_class_is_named(self, report):
        with pytest.raises(ConfigError):
            report.energy_balance_check.comparison("fortnight")

    def test_the_h_and_le_manifests_carry_the_check_and_nee_does_not(self, report):
        for target in ("H", "LE"):
            document = json.loads(json.dumps(report.manifest(target).to_dict()))
            balance = document["metrics"]["energy_balance"]
            assert (balance["sensible_heat"], balance["latent_heat"]) == ("H", "LE")
            assert balance["measured_ebr"] == pytest.approx(report.energy_balance.measured)
            assert balance["filled_ebr"] == pytest.approx(report.energy_balance.filled)
            assert set(balance["by_gap_class"]) == {gap_class.value for gap_class in GapClass}
        assert report["NEE"].to_dict()["energy_balance"] is None
        assert report["NEE"].energy_balance_check is None

    def test_the_energy_balance_table_has_the_run_and_each_class(self, report):
        table = report.energy_balance_frame()
        assert tuple(table.columns) == ENERGY_BALANCE_TABLE_COLUMNS
        assert list(table["gap_class"]) == ["all"] + [gap_class.value for gap_class in GapClass]
        assert set(table["method"]) == {"RFR3"}
        long = report.energy_balance_check.comparison("long")
        row = table.set_index("gap_class").loc["long"]
        assert row["n"] == long.n
        assert row["filled_ebr"] == pytest.approx(long.filled)
        assert table["measured_ebr"].dtype == np.float64

    def test_the_summary_reports_the_run_and_each_class(self, report):
        lines = [line for line in report.summary().splitlines() if "EBR" in line]
        assert len(lines) == 1 + len(GapClass)
        assert all("measured=" in line and "filled=" in line for line in lines)

    def test_a_run_without_both_fluxes_reports_no_ratio(self, site, gaps):
        run = validate_rfr(
            site.frame,
            config=_config(site, "RFR3"),
            targets="NEE",
            qc_columns=site.qc_column("NEE"),
            gaps=gaps,
        )
        assert run.energy_balance is None
        assert run.energy_balance_check is None
        assert run.energy_balance_frame().empty
        assert tuple(run.energy_balance_frame().columns) == ENERGY_BALANCE_TABLE_COLUMNS
        assert run.targets == ("NEE",)

    def test_pre_filled_fluxes_and_missing_radiation_are_counted_not_used(self, site, gaps, report):
        """Step 19's missing-data handling, on rows that were complete before.

        Six withheld H values are re-flagged as pre-filled and four withheld NETRAD
        values removed. The RFR3 driver set does not read NETRAD, and withheld
        rows are never trained on, so the forests are unchanged and exactly those
        ten rows leave both ratios - each counted under what it lost.
        """
        short = next(gap for gap in gaps if gap.gap_class is GapClass.SHORT)
        index = site.frame.index
        inside = pd.Series((index >= short.start) & (index < short.end), index=index)
        candidates = index[(inside & _complete_ebr_rows(site, report)).to_numpy()]
        pre_filled, no_radiation = candidates[:6], candidates[6:10]

        frame = site.frame.copy()
        frame.loc[pre_filled, site.qc_column("H")] = PRE_FILLED_QC
        frame.loc[no_radiation, report.column_map.column(NET_RADIATION)] = np.nan
        run = validate_rfr(
            frame,
            config=_config(site, "RFR3"),
            targets=["H", "LE"],
            qc_columns={"H": "H_QC", "LE": "LE_QC"},
            gaps=gaps,
        )

        before = report.energy_balance_check
        after = run.energy_balance_check
        for scope in (None, GapClass.SHORT):
            old, new = before.comparison(scope), after.comparison(scope)
            assert new.n_offered == old.n_offered
            assert new.n == old.n - 10
            assert new.n_missing_measured == old.n_missing_measured + 6
            assert new.n_missing_available_energy == old.n_missing_available_energy + 4
            assert new.n_missing_filled == old.n_missing_filled
        # The other classes' rows were not touched, and nor were the forests.
        for gap_class in (GapClass.LONG, GapClass.VERY_LONG):
            assert after.comparison(gap_class) == before.comparison(gap_class)


class TestEnergyBalanceCheckRecord:
    """The record itself, without a run behind it."""

    @staticmethod
    def _check(by_gap_class=None) -> EnergyBalanceCheck:
        comparison = compare_energy_balance(
            measured_sensible_heat=[30.0, 50.0],
            measured_latent_heat=[20.0, 30.0],
            filled_sensible_heat=[35.0, 45.0],
            filled_latent_heat=[20.0, 30.0],
            net_radiation=[100.0, 150.0],
            soil_heat_flux=[10.0, 20.0],
        )
        return EnergyBalanceCheck(
            sensible_heat="H",
            latent_heat="LE",
            overall=comparison,
            by_gap_class={GapClass.SHORT: comparison} if by_gap_class is None else by_gap_class,
        )

    def test_its_mapping_is_read_only(self):
        with pytest.raises(TypeError):
            self._check().by_gap_class[GapClass.LONG] = None  # type: ignore[index]

    def test_it_survives_a_pickle_round_trip(self):
        import pickle

        check = self._check()
        assert pickle.loads(pickle.dumps(check)) == check

    def test_a_run_without_gap_classes_says_so(self):
        with pytest.raises(ValidationError, match="report_by_gap_class=False"):
            self._check(by_gap_class={}).comparison("short")

    def test_its_document_keeps_the_overall_ratios_at_the_top(self):
        document = json.loads(json.dumps(self._check().to_dict()))
        assert document["measured_ebr"] == pytest.approx(130.0 / 220.0)
        assert document["available_energy"] == pytest.approx(220.0)
        assert set(document["by_gap_class"]) == {"short"}


# ---------------------------------------------------------------------------
# The sampled generator, and ambiguity A4
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_a_sampled_scenario_runs_end_to_end(site):
    """The default path: gaps sampled from the configured scenario, not handed in."""
    with warnings.catch_warnings():
        # The achieved fraction can miss the requested 25% (A7); the manifest
        # reports it, and this test is about the workflow completing.
        warnings.simplefilter("ignore")
        run = validate_rfr(
            site.frame,
            config=_config(site, "RFR3"),
            targets=["H", "LE"],
            qc_columns={"H": "H_QC", "LE": "LE_QC"},
        )
    assert run.gaps.random_state == site.seed
    assert len(run.gaps) > 0
    assert run["H"].overall.n > 0
    assert run.gaps.achieved_fraction is not None


@pytest.mark.slow
def test_the_default_daily_statistic_strategy_scores_nothing_and_says_so(site, gaps):
    """Ambiguity A4: under ``missing``, a whole-day gap has no feature rows at all."""
    config = site.config("RFR3", hyperparameter_grid=TEST_GRID, cv_folds=3)
    assert config.features.statistic_strategy.value == "missing"
    with pytest.warns(ValidationWarning, match="ambiguity A4"):
        run = validate_rfr(
            site.frame,
            config=config,
            targets="LE",
            qc_columns="LE_QC",
            gaps=gaps,
        )
    result = run["LE"]
    assert result.withheld_rows > 0
    assert result.scored_rows == 0
    assert result.overall.r2 is None
    # The row accounting says why, rather than leaving an empty table unexplained.
    assert result.features.to_dict()["holdout_rows_with_complete_features"] == 0


# ---------------------------------------------------------------------------
# What the request layer refuses
# ---------------------------------------------------------------------------


class TestRejectedRequests:
    """Bad requests fail before anything is fitted, naming what is wrong."""

    def test_data_must_be_a_frame(self, site):
        with pytest.raises(ValidationError, match="DataFrame"):
            validate_rfr("the whole site", config=_config(site, "RFR3"), targets="LE")

    def test_config_must_be_an_rfr_config(self, site):
        with pytest.raises(ConfigError, match="RFRConfig"):
            validate_rfr(site.frame, config={"mode": "RFR3"}, targets="LE")

    def test_an_unknown_target_column_is_named(self, site):
        with pytest.raises(ValidationError, match="target column"):
            validate_rfr(site.frame, config=_config(site, "RFR3"), targets="CH4")

    def test_a_missing_qc_column_is_named(self, site):
        with pytest.raises(ValidationError, match="QC column"):
            validate_rfr(
                site.frame, config=_config(site, "RFR3"), targets="LE", qc_columns="LE_FLAG"
            )

    def test_a_repeated_target_is_rejected(self, site):
        with pytest.raises(ValidationError, match="more than once"):
            validate_rfr(site.frame, config=_config(site, "RFR3"), targets=["LE", "LE"])

    def test_no_target_is_rejected(self, site):
        with pytest.raises(ValidationError, match="at least one target"):
            validate_rfr(site.frame, config=_config(site, "RFR3"), targets=[])

    def test_one_qc_column_for_several_targets_is_ambiguous(self, site):
        with pytest.raises(ValidationError, match="ambiguous"):
            validate_rfr(
                site.frame, config=_config(site, "RFR3"), targets=["H", "LE"], qc_columns="LE_QC"
            )

    def test_qc_columns_for_an_unvalidated_target_are_rejected(self, site):
        with pytest.raises(ValidationError, match="does not validate"):
            validate_rfr(
                site.frame,
                config=_config(site, "RFR3"),
                targets=["LE"],
                qc_columns={"LE": "LE_QC", "H": "H_QC"},
            )

    def test_handed_gaps_must_be_a_manifest(self, site):
        with pytest.raises(ValidationError, match="GapManifest"):
            validate_rfr(site.frame, config=_config(site, "RFR3"), targets="LE", gaps=[("a", "b")])

    def test_the_energy_balance_needs_its_own_variables_mapped(self, site):
        """Section 6.4 requires NETRAD and G whatever the driver mode is."""
        config = site.config(
            "RFR3",
            column_map=ColumnMap.fluxnet2015("RFR3"),
            hyperparameter_grid=TEST_GRID,
        )
        with pytest.raises(ValidationError, match="compute_energy_balance_ratio=False"):
            validate_rfr(
                site.frame,
                config=config,
                targets=["H", "LE"],
                qc_columns={"H": "H_QC", "LE": "LE_QC"},
            )

    @pytest.mark.slow
    def test_switching_the_energy_balance_off_lets_that_run_proceed(self, site, gaps):
        config = site.config(
            "RFR3",
            column_map=ColumnMap.fluxnet2015("RFR3"),
            hyperparameter_grid=TEST_GRID,
            cv_folds=3,
            features=REACHING,
            validation=ValidationConfig(compute_energy_balance_ratio=False),
        )
        run = validate_rfr(
            site.frame,
            config=config,
            targets=["H", "LE"],
            qc_columns={"H": "H_QC", "LE": "LE_QC"},
            gaps=gaps,
        )
        assert run.energy_balance is None

    @pytest.mark.slow
    def test_an_unknown_gap_class_is_named(self, report):
        with pytest.raises(ConfigError):
            report["LE"].metric(gap_class="fortnight")

    @pytest.mark.slow
    def test_an_unvalidated_target_is_named(self, report):
        with pytest.raises(ValidationError, match="no result for target"):
            report["CH4"]


class TestMissingDriverColumns:
    """A mapped column the data does not carry fails as a name, not a KeyError."""

    def test_an_absent_driver_column_is_named(self, site):
        config = site.config(
            "RFR3",
            column_map=ColumnMap(
                {"shortwave": "SW", "vpd": "VPD_F_MDS", "air_temperature": "TA_F_MDS"}
            ),
            hyperparameter_grid=TEST_GRID,
        )
        with pytest.raises(ValidationError, match="mapped driver column"):
            validate_rfr(site.frame, config=config, targets="LE", qc_columns="LE_QC")


def test_a_manifest_from_another_record_is_rejected(site):
    """A handed-in manifest that withholds nothing would score an empty test set."""
    elsewhere = synthetic_site(days=365, start="2001-01-01", seed=7)
    with pytest.raises(ValidationError, match="withholds no row"):
        validate_rfr(
            site.frame,
            config=_config(site, "RFR3"),
            targets="LE",
            qc_columns="LE_QC",
            gaps=elsewhere.known_gaps(),
        )
