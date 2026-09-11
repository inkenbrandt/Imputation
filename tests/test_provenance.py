"""Run-manifest tests. Covers Step 15.

The exit criterion of Step 15 is that a gap-filled result can be traced back to
the exact settings that produced it, so the central test here is not that some
JSON came out: it is that **every item on Step 15's list** is present in the
exported document, checked one at a time against
:data:`~rfrgapfill.provenance.REQUIRED_FIELDS`. The rest of the module covers the
settings digest, the resolved-ambiguity block the specification requires in run
output, the pairings a manifest must refuse, and the JSON contract.

The forests here are deliberately tiny (10 trees, 3 folds): nothing in this
module tests predictive quality. See ``docs/method_spec.md`` section 7 for the
contract.
"""

from __future__ import annotations

import copy
import json
import math
import pickle
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from rfrgapfill.config import FeatureConfig, GapScenarioConfig, RFRConfig, ValidationConfig
from rfrgapfill.features import feature_names
from rfrgapfill.fill import FillResult, RFRGapFiller
from rfrgapfill.gaps import GapManifest, GapScenarioGenerator
from rfrgapfill.leakage import ValidationFeatureSet, build_validation_features
from rfrgapfill.metrics import CoreMetrics
from rfrgapfill.model import RFRModel
from rfrgapfill.provenance import (
    MANIFEST_FORMAT,
    MANIFEST_VERSION,
    REQUIRED_FIELDS,
    VALIDATION_REQUIRED_FIELDS,
    ProvenanceError,
    RowCounts,
    RunKind,
    RunManifest,
    ambiguity_choices,
    environment_versions,
    load_manifest,
)
from rfrgapfill.schema import ColumnMap, ConfigError

#: Every ambiguity ``docs/method_spec.md`` documents; all must reach run output.
AMBIGUITY_IDS = tuple(f"A{number}" for number in range(1, 13))

HALF_HOURLY = "30min"

COLUMN_MAP = ColumnMap({"shortwave": "SW", "vpd": "VPD", "air_temperature": "TA"})

#: One candidate per grid point is enough; this module never scores a prediction.
TEST_GRID = {"n_estimators": (10,)}

#: A gap short enough to leave its calendar day plenty of visible observations.
GAP_START = "2020-06-20 06:00"
GAP_END = "2020-06-20 12:00"


def site_frame(days: int = 60, *, seed: int = 0) -> pd.DataFrame:
    """A half-hourly site frame whose target is a nonlinear function of its drivers.

    The same construction as ``tests/test_fill.py`` and ``tests/test_model.py``.
    """
    index = pd.date_range("2020-06-01", periods=days * 48, freq=HALF_HOURLY)
    hour = index.hour + index.minute / 60.0
    shortwave = np.clip(600.0 * np.sin(np.pi * (hour - 6.0) / 12.0), 0.0, None)
    air_temperature = 12.0 + 8.0 * np.sin(2.0 * np.pi * index.dayofyear / 365.0) + 0.4 * hour
    vpd = np.clip(0.02 * shortwave + 0.3 * air_temperature, 0.0, None)
    noise = np.random.default_rng(seed).normal(0.0, 2.0, len(index))
    latent_heat = 0.03 * shortwave * air_temperature + 20.0 * np.sqrt(vpd) + noise
    return pd.DataFrame(
        {
            "SW": shortwave,
            "VPD": vpd,
            "TA": air_temperature,
            "LE": latent_heat,
            "LE_QC": 0.0,
        },
        index=index,
    )


def with_gap(data: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``data`` with the target missing over the fixture gap."""
    gapped = data.copy()
    index = pd.DatetimeIndex(gapped.index)
    inside = (index >= pd.Timestamp(GAP_START)) & (index < pd.Timestamp(GAP_END))
    gapped.loc[inside, "LE"] = np.nan
    gapped.loc[inside, "LE_QC"] = np.nan
    return gapped


def rfr_config(**changes: object) -> RFRConfig:
    """An RFR3 configuration with a one-candidate grid and few folds."""
    settings: dict[str, object] = {
        "mode": "RFR3",
        "frequency": HALF_HOURLY,
        "hemisphere": "north",
        "random_state": 42,
        "hyperparameter_grid": TEST_GRID,
        "cv_folds": 3,
        "column_map": COLUMN_MAP,
    }
    settings.update(changes)
    return RFRConfig(**settings)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Fixtures: one fitted fill and one fitted validation run, shared
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def data() -> pd.DataFrame:
    """A half-hourly site frame with one six-hour gap in the target."""
    return with_gap(site_frame())


@pytest.fixture(scope="module")
def filler(data: pd.DataFrame) -> RFRGapFiller:
    """One fitted filler shared by the read-only tests."""
    return RFRGapFiller(rfr_config()).fit(data, target="LE", qc_column="LE_QC")


@pytest.fixture(scope="module")
def result(filler: RFRGapFiller, data: pd.DataFrame) -> FillResult:
    """The fill that filler produced."""
    return filler.fill(data)


@pytest.fixture(scope="module")
def fill_manifest(filler: RFRGapFiller, result: FillResult) -> RunManifest:
    """The manifest of an operational fill."""
    return RunManifest.from_fill(filler, result)


@pytest.fixture(scope="module")
def validation_config() -> RFRConfig:
    """A configuration whose scenario is 24-hour gaps only, so the fixture stays small."""
    scenario = GapScenarioConfig(
        missing_fraction=0.20,
        gap_mix={"short": 1.0, "long": 0.0, "very_long": 0.0},
    )
    return rfr_config(validation=ValidationConfig(gaps=scenario))


@pytest.fixture(scope="module")
def gaps(validation_config: RFRConfig) -> GapManifest:
    """The placed artificial-gap scenario."""
    generator = GapScenarioGenerator(validation_config)
    return generator.generate(site_frame(), target="LE", qc_column="LE_QC")


@pytest.fixture(scope="module")
def feature_set(validation_config: RFRConfig, gaps: GapManifest) -> ValidationFeatureSet:
    """Leakage-safe validation features for the placed scenario."""
    frame = site_frame()
    return build_validation_features(
        frame,
        config=validation_config,
        target="LE",
        holdout=gaps.mask(frame.index),
        qc_column="LE_QC",
    )


@pytest.fixture(scope="module")
def fitted_model(validation_config: RFRConfig, feature_set: ValidationFeatureSet) -> RFRModel:
    """A model fitted on the validation run's training rows."""
    model = RFRModel(validation_config, target="LE", feature_names=feature_set.feature_names)
    model.fit(feature_set.training_features(), feature_set.training_target())
    return model


@pytest.fixture(scope="module")
def validation_manifest(
    validation_config: RFRConfig,
    fitted_model: RFRModel,
    gaps: GapManifest,
    feature_set: ValidationFeatureSet,
) -> RunManifest:
    """The manifest of an artificial-gap validation run."""
    return RunManifest.from_validation(
        config=validation_config,
        model=fitted_model,
        gaps=gaps,
        features=feature_set,
        qc_column="LE_QC",
    )


def lookup(document: dict[str, Any], path: str) -> Any:
    """Return the value at a dotted path, failing the test when it is absent."""
    current: Any = document
    for part in path.split("."):
        assert isinstance(current, dict) and part in current, f"{path} is not in the manifest"
        current = current[part]
    return current


# ---------------------------------------------------------------------------
# The exit criterion: every Step 15 item is recorded
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize(("path", "item"), sorted(REQUIRED_FIELDS.items()))
def test_fill_manifest_records_every_required_item(
    fill_manifest: RunManifest, path: str, item: str
) -> None:
    document = fill_manifest.to_dict()
    value = lookup(document, path)
    assert value is not None, f"Step 15 requires {item} ({path})"
    if isinstance(value, (str, dict, list)):
        assert len(value) > 0, f"Step 15 requires {item} ({path}), which is empty"


@pytest.mark.slow
def test_fill_manifest_is_complete(fill_manifest: RunManifest) -> None:
    assert fill_manifest.missing_fields() == {}
    assert fill_manifest.is_complete
    assert fill_manifest.require_complete() is fill_manifest


@pytest.mark.slow
@pytest.mark.parametrize(("path", "item"), sorted(VALIDATION_REQUIRED_FIELDS.items()))
def test_validation_manifest_records_the_gap_manifest(
    validation_manifest: RunManifest, path: str, item: str
) -> None:
    assert lookup(validation_manifest.to_dict(), path), f"a validation run must record {item}"


@pytest.mark.slow
def test_validation_manifest_is_complete(validation_manifest: RunManifest) -> None:
    assert validation_manifest.missing_fields() == {}
    assert validation_manifest.run_kind is RunKind.VALIDATION


def test_an_operational_fill_needs_no_gap_manifest() -> None:
    """A fill has a gap *configuration* but places no artificial intervals."""
    manifest = RunManifest(
        kind="fill",
        target="LE",
        config=rfr_config(),
        rows=RowCounts(1, 1, 1, 1),
    )
    assert "gaps.manifest" not in manifest.required_fields()
    assert (
        "gaps.manifest"
        in RunManifest(
            kind="validation",
            target="LE",
            config=rfr_config(),
            rows=RowCounts(1, 1, 1, 1),
        ).required_fields()
    )


def test_require_complete_names_the_missing_audit_item() -> None:
    manifest = RunManifest(
        kind="fill", target="LE", config=rfr_config(), rows=RowCounts(1, 1, 1, 1)
    )
    assert set(manifest.missing_fields()) == {
        "model.hyperparameter_grid",
        "model.best_params",
    }
    with pytest.raises(ProvenanceError, match="hyperparameter grid"):
        manifest.require_complete()


def test_the_hemisphere_is_required_only_when_the_season_feature_is_built() -> None:
    """The ORF benchmark builds no season feature, and its config accepts no hemisphere."""
    rfr = RunManifest(kind="fill", target="LE", config=rfr_config(), rows=RowCounts(1, 1, 1, 1))
    assert "run.hemisphere" in rfr.required_fields()

    orf = RunManifest(
        kind="fill",
        target="LE",
        config=rfr_config(features=FeatureConfig(use_receptive_limiter=False), hemisphere=None),
        rows=RowCounts(1, 1, 1, 1),
    )
    assert "run.hemisphere" not in orf.required_fields()
    assert orf.to_dict()["run"]["hemisphere"] is None
    assert orf.to_dict()["run"]["is_orf"] is True


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("package", ["rfr_gapfill", "python", "scikit_learn"])
def test_environment_records_the_three_specified_versions(package: str) -> None:
    versions = environment_versions()
    assert isinstance(versions[package], str)
    assert versions[package]


def test_the_model_and_the_manifest_agree_about_the_environment(
    fill_manifest: RunManifest,
) -> None:
    """One definition of "the environment", so a saved model and its manifest cannot drift."""
    fitted = (fill_manifest.section("model") or {})["versions"]
    assert fitted == dict(fill_manifest.versions or {})
    assert fill_manifest.environment_drift == {}


def test_environment_drift_reports_a_model_fitted_elsewhere() -> None:
    """A model loaded from another environment shows up next to the numbers it produced."""
    manifest = RunManifest(
        kind="fill",
        target="LE",
        config=rfr_config(),
        rows=RowCounts(1, 1, 1, 1),
        versions={"rfr_gapfill": "9.9.9", "python": "3.12.0", "scikit_learn": "1.5.0"},
        model={"versions": {"rfr_gapfill": "0.1.0", "scikit_learn": "1.5.0"}},
    )
    assert manifest.environment_drift == {"rfr_gapfill": ["0.1.0", "9.9.9"]}


# ---------------------------------------------------------------------------
# Row counts
# ---------------------------------------------------------------------------


def test_row_counts_report_what_was_dropped() -> None:
    counts = RowCounts(
        training_rows=90,
        training_rows_offered=100,
        prediction_rows=5,
        prediction_rows_offered=48,
    )
    assert counts.training_rows_dropped == 10
    assert counts.prediction_rows_dropped == 43
    assert counts.to_dict()["prediction_rows_dropped"] == 43


@pytest.mark.parametrize(
    "counts",
    [
        (-1, 10, 0, 0),
        (10, 5, 0, 0),
        (0, 0, 5, 1),
    ],
)
def test_row_counts_reject_an_impossible_account(counts: tuple[int, int, int, int]) -> None:
    with pytest.raises(ProvenanceError):
        RowCounts(*counts)


def test_row_counts_reject_a_non_integer() -> None:
    with pytest.raises(ProvenanceError, match="must be an integer"):
        RowCounts(1.5, 2, 0, 0)  # type: ignore[arg-type]


@pytest.mark.slow
def test_fill_row_counts_come_from_the_two_reports(
    fill_manifest: RunManifest, filler: RFRGapFiller, result: FillResult
) -> None:
    rows = fill_manifest.rows
    assert rows.training_rows == filler.model.fit_report.fitted_rows
    assert rows.training_rows_offered == filler.model.fit_report.rows
    # Candidates, not the whole frame: an observed row was never a prediction
    # the model declined to make.
    assert rows.prediction_rows == result.report.filled_rows
    assert rows.prediction_rows_offered == result.report.candidate_rows


@pytest.mark.slow
def test_validation_prediction_rows_are_the_rows_features_reach(
    validation_manifest: RunManifest, feature_set: ValidationFeatureSet
) -> None:
    """Ambiguity A4 shows up here: a blocked run predicts far fewer rows than it withheld."""
    rows = validation_manifest.rows
    assert rows.prediction_rows_offered == int(feature_set.holdout_mask.sum())
    assert rows.prediction_rows == int((feature_set.holdout_mask & feature_set.complete_mask).sum())


def test_validation_row_counts_need_a_source(
    validation_config: RFRConfig, fitted_model: RFRModel, gaps: GapManifest
) -> None:
    with pytest.raises(ProvenanceError, match="row counts are required"):
        RunManifest.from_validation(
            config=validation_config, model=fitted_model, gaps=gaps, target="LE"
        )


# ---------------------------------------------------------------------------
# Settings digest
# ---------------------------------------------------------------------------


def manifest_with(**changes: Any) -> RunManifest:
    """A minimal manifest whose configuration carries ``changes``."""
    settings: dict[str, Any] = {
        "kind": "fill",
        "target": "LE",
        "config": rfr_config(**changes.pop("config_changes", {})),
        "rows": RowCounts(1, 1, 1, 1),
        "versions": {"rfr_gapfill": "1.0", "python": "3.12.0", "scikit_learn": "1.5.0"},
    }
    settings.update(changes)
    return RunManifest(**settings)


def test_the_digest_ignores_what_varies_between_two_runs_of_one_design() -> None:
    first = manifest_with(created_at="2020-01-01T00:00:00+00:00")
    second = manifest_with(
        created_at="2024-09-10T12:00:00+00:00",
        rows=RowCounts(500, 900, 7, 48),
        metrics={"r2": 0.81},
    )
    assert first.settings_digest == second.settings_digest


@pytest.mark.parametrize(
    "change",
    [
        {"random_state": 7},
        {"mode": "RFR10", "column_map": ColumnMap.fluxnet2015()},
        {"hyperparameter_grid": {"n_estimators": (10, 20)}},
        {"features": FeatureConfig(daily_std_ddof=0)},
        {"validation": ValidationConfig(r2_definition="squared_correlation")},
    ],
)
def test_the_digest_changes_with_the_settings(change: dict[str, Any]) -> None:
    baseline = manifest_with()
    assert manifest_with(config_changes=change).settings_digest != baseline.settings_digest


def test_the_digest_changes_with_the_environment() -> None:
    baseline = manifest_with()
    drifted = manifest_with(
        versions={"rfr_gapfill": "1.0", "python": "3.12.0", "scikit_learn": "1.6.0"}
    )
    assert drifted.settings_digest != baseline.settings_digest


def test_the_digest_is_in_the_exported_document() -> None:
    manifest = manifest_with()
    assert manifest.to_dict()["settings_digest"] == manifest.settings_digest


# ---------------------------------------------------------------------------
# Resolved ambiguities travel with the result
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ambiguity", AMBIGUITY_IDS)
def test_every_documented_ambiguity_reaches_run_output(ambiguity: str) -> None:
    choices = ambiguity_choices(rfr_config())["choices"]
    entry = choices[ambiguity]
    assert entry["topic"]
    assert entry["settings"]


def test_the_ambiguity_block_never_claims_the_paper() -> None:
    block = ambiguity_choices(rfr_config())
    assert block["paper_exact"] is False
    assert "none of them reproduces the paper exactly" in block["note"]


def test_the_ambiguity_block_reports_the_choice_this_run_made() -> None:
    config = rfr_config(
        features=FeatureConfig(
            daily_statistic_strategy="rolling_available", fallback_window_days=3
        ),
        validation=ValidationConfig(
            gaps=GapScenarioConfig(allocation_basis="gap_events"),
            r2_definition="squared_correlation",
        ),
    )
    choices = ambiguity_choices(config)["choices"]
    assert choices["A3"]["settings"]["allocation_basis"] == "gap_events"
    assert choices["A4"]["settings"]["daily_statistic_strategy"] == "rolling_available"
    assert choices["A4"]["settings"]["fallback_window_days"] == 3
    assert choices["A12"]["settings"]["r2_definition"] == "squared_correlation"


def test_a_supplied_grid_is_not_reported_as_the_package_default() -> None:
    default = ambiguity_choices(RFRConfig(mode="RFR3", hemisphere="north"))
    assert default["choices"]["A1"]["settings"]["is_package_default"] is True
    # rfr_config ships a one-candidate grid so the tests stay fast.
    assert (
        ambiguity_choices(rfr_config())["choices"]["A1"]["settings"]["is_package_default"] is False
    )


@pytest.mark.slow
def test_the_manifest_carries_the_ambiguity_block(fill_manifest: RunManifest) -> None:
    block = fill_manifest.to_dict()["ambiguities"]
    assert set(block["choices"]) == set(AMBIGUITY_IDS)


def test_ambiguity_choices_rejects_a_non_configuration() -> None:
    """``ConfigError``, as everywhere else a configuration object is checked."""
    with pytest.raises(ConfigError, match="must be an RFRConfig"):
        ambiguity_choices(object())  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# JSON export
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_the_document_survives_a_strict_json_round_trip(fill_manifest: RunManifest) -> None:
    text = fill_manifest.to_json()
    document = json.loads(text)
    assert document["manifest_format"] == MANIFEST_FORMAT
    assert document["manifest_version"] == MANIFEST_VERSION
    assert document == json.loads(json.dumps(document, allow_nan=False))


def test_an_undefined_value_is_exported_as_null_not_nan() -> None:
    """The package's own convention: undefined is ``None``, never ``NaN`` (metrics)."""
    manifest = manifest_with(
        metrics=CoreMetrics(n=0, n_offered=0, r2=None, slope=None, rmse=None, bias=None),
        extra={"a_nan": math.nan, "an_inf": math.inf},
    )
    document = json.loads(manifest.to_json())
    assert document["metrics"]["r2"] is None
    assert document["extra"]["a_nan"] is None
    assert document["extra"]["an_inf"] is None


def test_sections_accept_the_objects_that_own_them(result: FillResult, gaps: GapManifest) -> None:
    """A caller hands over the record, not a rendering of it."""
    manifest = manifest_with(kind="validation", time_axis=result.time_axis, gap_manifest=gaps)
    document = manifest.to_dict()
    assert document["time_axis"] == result.time_axis.to_dict()
    assert document["gaps"]["manifest"] == gaps.to_dict()


def test_an_exotic_value_is_rendered_rather_than_breaking_the_export() -> None:
    manifest = manifest_with(
        extra={
            "when": pd.Timestamp("2020-06-01T00:00:00"),
            "how_long": timedelta(minutes=30),
            "which": np.array([1, 2, 3]),
            "kind": RunKind.FILL,
            "unknown": object(),
        }
    )
    extra = json.loads(manifest.to_json())["extra"]
    assert extra["when"].startswith("2020-06-01")
    assert extra["how_long"] == "P0DT0H30M0S"
    assert extra["which"] == [1, 2, 3]
    assert extra["kind"] == "fill"
    assert isinstance(extra["unknown"], str)


def test_a_section_that_is_not_a_mapping_is_refused() -> None:
    with pytest.raises(ProvenanceError, match="must be a mapping"):
        manifest_with(metrics=[1, 2, 3])


def test_section_names_are_checked() -> None:
    with pytest.raises(ProvenanceError, match="not a manifest section"):
        manifest_with().section("config")


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_save_and_load_round_trip(fill_manifest: RunManifest, tmp_path: Path) -> None:
    written = fill_manifest.save(tmp_path / "LE_run.json")
    assert written.is_file()
    assert load_manifest(written) == fill_manifest.to_dict()


def test_save_refuses_an_incomplete_manifest(tmp_path: Path) -> None:
    """The file outlives the session that could have explained it."""
    manifest = manifest_with()
    destination = tmp_path / "partial.json"
    with pytest.raises(ProvenanceError, match="missing required provenance"):
        manifest.save(destination)
    assert not destination.exists()
    assert manifest.save(destination, check=False).is_file()


def test_to_json_does_not_check_completeness() -> None:
    """An unfinished manifest is worth printing to see what it is short of."""
    assert json.loads(manifest_with().to_json())["kind"] == "fill"


def test_load_manifest_refuses_a_foreign_file(tmp_path: Path) -> None:
    other = tmp_path / "other.json"
    other.write_text(json.dumps({"hello": "world"}), encoding="utf-8")
    with pytest.raises(ProvenanceError, match="not an rfr-gapfill run manifest"):
        load_manifest(other)

    missing = tmp_path / "absent.json"
    with pytest.raises(ProvenanceError, match="cannot read a run manifest"):
        load_manifest(missing)


# ---------------------------------------------------------------------------
# Pairings a manifest must refuse
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_from_fill_refuses_an_unfitted_filler(result: FillResult) -> None:
    with pytest.raises(ProvenanceError, match="unfitted filler"):
        RunManifest.from_fill(RFRGapFiller(rfr_config()), result)


@pytest.mark.slow
def test_from_fill_refuses_another_runs_result(
    filler: RFRGapFiller, data: pd.DataFrame, result: FillResult
) -> None:
    """A manifest that paired a result with a different run would describe neither."""
    other = RFRGapFiller(rfr_config()).fit(
        data.rename(columns={"LE": "H", "LE_QC": "H_QC"}), target="H", qc_column="H_QC"
    )
    with pytest.raises(ProvenanceError, match="may not pair a result with another run"):
        RunManifest.from_fill(other, result)


@pytest.mark.slow
def test_from_validation_refuses_a_target_disagreement(
    validation_config: RFRConfig,
    fitted_model: RFRModel,
    gaps: GapManifest,
    feature_set: ValidationFeatureSet,
) -> None:
    with pytest.raises(ProvenanceError, match="target is ambiguous"):
        RunManifest.from_validation(
            config=validation_config,
            model=fitted_model,
            gaps=gaps,
            features=feature_set,
            target="H",
        )


def test_from_validation_refuses_an_unfitted_model(
    validation_config: RFRConfig, gaps: GapManifest
) -> None:
    model = RFRModel(validation_config, target="LE")
    with pytest.raises(ProvenanceError, match="unfitted model"):
        RunManifest.from_validation(config=validation_config, model=model, gaps=gaps)


def test_a_manifest_needs_a_target_and_a_configuration() -> None:
    with pytest.raises(ProvenanceError, match="target must be a non-empty string"):
        RunManifest(kind="fill", target="  ", config=rfr_config(), rows=RowCounts(1, 1, 1, 1))
    with pytest.raises(ConfigError, match="must be an RFRConfig"):
        RunManifest(
            kind="fill",
            target="LE",
            config=object(),  # type: ignore[arg-type]
            rows=RowCounts(1, 1, 1, 1),
        )


# ---------------------------------------------------------------------------
# Defaults filled in from the objects that know them
# ---------------------------------------------------------------------------


def test_the_feature_description_defaults_to_the_configuration() -> None:
    config = rfr_config()
    manifest = RunManifest(kind="fill", target="LE", config=config, rows=RowCounts(1, 1, 1, 1))
    described = manifest.to_dict()["features"]
    assert described["feature_names"] == list(feature_names(config, target="LE"))
    assert described["target"] == "LE"


def test_the_column_map_defaults_to_the_configured_one() -> None:
    manifest = RunManifest(
        kind="fill", target="LE", config=rfr_config(), rows=RowCounts(1, 1, 1, 1)
    )
    assert manifest.columns == rfr_config().columns


@pytest.mark.slow
def test_the_column_map_and_qc_column_come_from_the_fit(
    fill_manifest: RunManifest, filler: RFRGapFiller
) -> None:
    document = fill_manifest.to_dict()
    assert document["column_map"] == filler.column_map.to_dict()
    assert document["qc"]["qc_column"] == "LE_QC"
    assert document["qc"]["observed_qc_values"] == [0]


def test_the_time_resolution_falls_back_to_the_configuration() -> None:
    manifest = RunManifest(
        kind="fill", target="LE", config=rfr_config(), rows=RowCounts(1, 1, 1, 1)
    )
    assert manifest.time_resolution == "P0DT0H30M0S"


@pytest.mark.slow
def test_the_time_resolution_is_the_cadence_the_run_used(
    fill_manifest: RunManifest, result: FillResult
) -> None:
    assert fill_manifest.time_resolution == result.time_axis.to_dict()["time_step"]


def test_created_at_defaults_to_now_and_is_recorded() -> None:
    manifest = RunManifest(
        kind="fill", target="LE", config=rfr_config(), rows=RowCounts(1, 1, 1, 1)
    )
    assert manifest.created_at is not None
    assert pd.Timestamp(manifest.created_at).tzinfo is not None


# ---------------------------------------------------------------------------
# The manifest is an ordinary frozen record
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_a_manifest_survives_pickling(fill_manifest: RunManifest) -> None:
    restored = pickle.loads(pickle.dumps(fill_manifest))
    assert restored.to_dict() == fill_manifest.to_dict()


def test_a_manifest_survives_deep_copying() -> None:
    manifest = manifest_with()
    assert copy.deepcopy(manifest).settings_digest == manifest.settings_digest
