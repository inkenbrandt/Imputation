"""Configuration-file tests (Step 22).

``RFRConfig.from_dict`` and ``load_config`` read back the form ``to_dict`` writes,
so the ``config`` section of a run manifest is a configuration file as it stands
and a command-line run rebuilds exactly the configuration a script would. The
round trips below pin that across the settings every ambiguity is reachable
through; the rest pins the refusals - unknown keys, a missing mode, derived keys
that disagree with the settings, and files that are not JSON or YAML.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from rfrgapfill.config import (
    ColumnMap,
    ConfigError,
    FeatureConfig,
    GapScenarioConfig,
    Mode,
    RFRConfig,
    ValidationConfig,
    load_config,
)

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

CONFIGS: dict[str, Callable[[], RFRConfig]] = {
    "rfr3-defaults": lambda: RFRConfig(mode="RFR3", latitude=45.0, column_map=RFR3_MAPPING),
    "rfr10-everything-changed": lambda: RFRConfig(
        mode="RFR10",
        frequency="1h",
        hemisphere="south",
        latitude=-33.0,
        site_id="AU-Xxx",
        random_state=7,
        n_jobs=2,
        hyperparameter_preset="legacy_fluxlib_fixed",
        cv_strategy="time_series_split",
        cv_folds=4,
        observed_qc_values=(0, 1),
        features=FeatureConfig(
            radiation_thresholds=(5.0, 150.0),
            boundary_convention="medium_exclusive",
            min_daily_observations=4,
            daily_statistic_strategy="neighbor_day_fallback",
            fallback_window_days=3,
            daily_std_ddof=0,
        ),
        validation=ValidationConfig(
            gaps=GapScenarioConfig(
                missing_fraction=0.2,
                gap_mix={"short": 0.5, "long": 0.5, "very_long": 0.0},
                durations={"short": "12h", "long": "3d", "very_long": "20d"},
                allocation_basis="gap_events",
                min_observed_fraction=0.6,
                allow_overlap=True,
                max_attempts_per_gap=50,
                shared_gaps_across_targets=False,
                fraction_tolerance=0.1,
                mix_tolerance=0.1,
            ),
            daytime_threshold=10.0,
            subsets=("all", "daytime"),
            r2_definition="squared_correlation",
            report_by_gap_class=False,
            bias_iqr_by_gap_class=False,
            compute_energy_balance_ratio=False,
        ),
        column_map=ColumnMap(RFR10_MAPPING, timestamp="TIMESTAMP"),
    ),
    "legacy-fluxlib": lambda: RFRConfig(
        mode="RFR3",
        latitude=45.0,
        features=FeatureConfig(feature_mode="legacy_fluxlib"),
        hyperparameter_preset="legacy_fluxlib",
    ),
    "orf": lambda: RFRConfig(
        mode="RFR3", features=FeatureConfig(use_receptive_limiter=False), column_map=RFR3_MAPPING
    ),
    "custom-grid": lambda: RFRConfig(
        mode="RFR3",
        latitude=10.0,
        hyperparameter_grid={"n_estimators": (10, 20), "max_features": ("sqrt", 1.0)},
    ),
}


def through_json(config: RFRConfig) -> dict[str, Any]:
    """``config.to_dict()`` as it comes back out of a JSON file."""
    document: dict[str, Any] = json.loads(json.dumps(config.to_dict()))
    return document


def base(**changes: Any) -> dict[str, Any]:
    return {"mode": "RFR3", "latitude": 45.0, "column_map": RFR3_MAPPING, **changes}


class TestRoundTrips:
    @pytest.mark.parametrize("name", CONFIGS)
    def test_from_dict_inverts_to_dict(self, name):
        config = CONFIGS[name]()
        assert RFRConfig.from_dict(through_json(config)).to_dict() == config.to_dict()

    @pytest.mark.parametrize("name", CONFIGS)
    def test_a_json_file_of_to_dict_loads_the_same_configuration(self, name, tmp_path):
        config = CONFIGS[name]()
        path = tmp_path / "run.json"
        path.write_text(json.dumps(config.to_dict()), encoding="utf-8")
        assert load_config(path).to_dict() == config.to_dict()

    def test_the_flat_column_map_may_name_the_timestamp(self):
        config = RFRConfig.from_dict(base(column_map={**RFR3_MAPPING, "timestamp": "TIMESTAMP"}))
        assert config.columns.to_dict() == {"variables": RFR3_MAPPING, "timestamp": "TIMESTAMP"}

    def test_nested_sections_read_their_own_keys(self):
        config = RFRConfig.from_dict(
            base(
                features={"daily_statistic_strategy": "rolling_available"},
                validation={"gaps": {"missing_fraction": 0.1}, "r2_definition": "residual"},
            )
        )
        assert config.features.statistic_strategy.value == "rolling_available"
        assert config.features.fallback_window == 7
        assert config.validation.gaps.missing_fraction == 0.1


class TestRefusals:
    def test_a_misspelt_setting_is_named(self):
        with pytest.raises(ConfigError, match="randm_state"):
            RFRConfig.from_dict(base(randm_state=1))

    @pytest.mark.parametrize(
        ("section", "document", "key"),
        [
            ("features", {"features": {"daily_stat": "missing"}}, "daily_stat"),
            ("validation", {"validation": {"r2": "residual"}}, "r2"),
            ("validation.gaps", {"validation": {"gaps": {"missing": 0.1}}}, "missing"),
        ],
    )
    def test_a_misspelt_nested_setting_names_its_section(self, section, document, key):
        with pytest.raises(ConfigError, match=rf"unknown {section} setting\(s\): {key}\b"):
            RFRConfig.from_dict(base(**document))

    def test_the_mode_is_required(self):
        document = base()
        del document["mode"]
        with pytest.raises(ConfigError, match="must name a mode"):
            RFRConfig.from_dict(document)

    def test_the_document_must_be_a_mapping(self):
        with pytest.raises(ConfigError, match="mapping"):
            RFRConfig.from_dict(["RFR3"])  # type: ignore[arg-type]

    def test_a_structured_column_map_takes_only_its_own_keys(self):
        with pytest.raises(ConfigError, match="structured form"):
            RFRConfig.from_dict(base(column_map={"variables": RFR3_MAPPING, "site": "x"}))

    def test_a_derived_key_that_disagrees_is_refused(self):
        document = through_json(CONFIGS["orf"]())
        assert document["is_paper_faithful"] is False
        document["is_paper_faithful"] = True
        with pytest.raises(ConfigError, match="is_paper_faithful"):
            RFRConfig.from_dict(document)

    def test_null_is_not_a_quiet_default_outside_legacy_mode(self):
        with pytest.raises(ConfigError):
            FeatureConfig.from_dict({"daily_statistic_strategy": None})


class TestLoadConfig:
    def write(self, tmp_path: Path, name: str, text: str) -> Path:
        path = tmp_path / name
        path.write_text(text, encoding="utf-8")
        return path

    def test_overrides_apply_before_validation(self, tmp_path):
        path = self.write(tmp_path, "run.json", json.dumps(base(column_map=RFR10_MAPPING)))
        assert load_config(path).rfr_mode is Mode.RFR3
        assert load_config(path, mode="RFR10").rfr_mode is Mode.RFR10

    def test_an_override_is_checked_like_any_setting(self, tmp_path):
        path = self.write(tmp_path, "run.json", json.dumps(base()))
        with pytest.raises(ConfigError, match="missing column mapping"):
            load_config(path, mode="RFR10")
        with pytest.raises(ConfigError, match="modes"):
            load_config(path, modes="RFR10")

    def test_a_missing_file(self, tmp_path):
        with pytest.raises(ConfigError, match="does not exist"):
            load_config(tmp_path / "absent.json")

    def test_an_unknown_suffix(self, tmp_path):
        path = self.write(tmp_path, "run.toml", 'mode = "RFR3"\n')
        with pytest.raises(ConfigError, match=r"\*\.json, \*\.yaml or \*\.yml"):
            load_config(path)

    def test_invalid_json(self, tmp_path):
        path = self.write(tmp_path, "run.json", '{"mode": "RFR3",}')
        with pytest.raises(ConfigError, match="not valid JSON"):
            load_config(path)

    def test_the_top_level_must_be_a_mapping(self, tmp_path):
        path = self.write(tmp_path, "run.json", '["RFR3"]')
        with pytest.raises(ConfigError, match="mapping of settings at the top level"):
            load_config(path)

    def test_yaml(self, tmp_path):
        pytest.importorskip("yaml")
        path = self.write(
            tmp_path,
            "run.yaml",
            "mode: RFR10\n"
            "latitude: 45.0\n"
            "features:\n"
            "  daily_statistic_strategy: rolling_available\n"
            "column_map:\n" + "".join(f"  {k}: {v}\n" for k, v in RFR10_MAPPING.items()),
        )
        config = load_config(path)
        assert config.rfr_mode is Mode.RFR10
        assert config.columns.to_dict()["variables"] == RFR10_MAPPING

    def test_yaml_without_pyyaml_says_what_to_install(self, tmp_path, monkeypatch):
        monkeypatch.setitem(sys.modules, "yaml", None)
        path = self.write(tmp_path, "run.yml", "mode: RFR3\n")
        with pytest.raises(ConfigError, match=r"PyYAML.*rfr-gapfill\[yaml\]"):
            load_config(path)
