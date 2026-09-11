"""Command-line tests (Step 22).

Step 22's exit criterion is that the command line invokes the same tested Python
API rather than duplicating its logic. The end-to-end tests below therefore run
each subcommand and the API call it stands for on the same input, and require the
same numbers back. The rest covers what the command line owns itself: reading the
CSV, the timestamp column, output that is never silently overwritten, and failures
reported as one-line messages rather than tracebacks.

The forests are deliberately tiny (10 trees, 3 folds): nothing here tests
predictive quality.
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

import rfrgapfill.cli as cli
from rfrgapfill import __version__
from rfrgapfill.cli import CLIError, main, read_table
from rfrgapfill.config import FeatureConfig, GapScenarioConfig, RFRConfig, ValidationConfig
from rfrgapfill.fill import FillError, RFRGapFiller
from rfrgapfill.provenance import load_manifest
from rfrgapfill.schema import Mode
from rfrgapfill.synthetic import SyntheticSite, synthetic_site
from rfrgapfill.validation import ValidationError, validate_rfr

TEST_GRID = {"n_estimators": (10,)}

RFR10_COLUMNS = {
    "shortwave": "SW",
    "vpd": "VPD",
    "air_temperature": "TA",
    "net_radiation": "NETRAD",
    "wind_speed": "WS",
    "wind_direction": "WD",
    "soil_heat_flux": "G",
    "soil_temperature": "TS",
    "relative_humidity": "RH",
    "soil_water_content": "SWC",
}

#: A small configuration file in the hand-written form: the flat column map,
#: with the timestamp column named inside it.
MINIMAL_CONFIG: dict[str, Any] = {
    "mode": "RFR3",
    "latitude": 45.0,
    "column_map": {
        "shortwave": "SW",
        "vpd": "VPD",
        "air_temperature": "TA",
        "timestamp": "TIMESTAMP",
    },
}

TINY_CSV = (
    "TIMESTAMP,SW,VPD,TA,LE,LE_QC\n"
    "2020-06-01 00:00,0,1.5,10,5.0,0\n"
    "2020-06-01 00:30,-9999,1.5,10,-9999.0,0\n"
)


@pytest.fixture(scope="module")
def site() -> SyntheticSite:
    return synthetic_site(days=60)


def run_config(site: SyntheticSite, **changes: Any) -> RFRConfig:
    """A fast configuration whose artificial-gap scenario fits in 60 days."""
    settings: dict[str, Any] = {
        "hyperparameter_grid": TEST_GRID,
        "cv_folds": 3,
        "features": FeatureConfig(daily_statistic_strategy="rolling_available"),
        "validation": ValidationConfig(
            gaps=GapScenarioConfig(
                missing_fraction=0.1, gap_mix={"short": 1.0, "long": 0.0, "very_long": 0.0}
            )
        ),
    }
    settings.update(changes)
    return site.config("RFR3", **settings)


def write_json(document: dict[str, Any], path: Path) -> Path:
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def minimal_config(tmp_path: Path, **changes: Any) -> Path:
    return write_json({**MINIMAL_CONFIG, **changes}, tmp_path / "run.json")


def tiny_csv(tmp_path: Path) -> Path:
    path = tmp_path / "tiny.csv"
    path.write_text(TINY_CSV, encoding="utf-8")
    return path


def as_table(frame: pd.DataFrame) -> pd.DataFrame:
    """``frame`` after the same CSV round trip the command line's output takes."""
    return read_exact(io.StringIO(frame.to_csv(index=False)))


def read_exact(source: Any, **kwargs: Any) -> pd.DataFrame:
    """Read a CSV back without the default parser's last-digit rounding."""
    read = pd.read_csv
    frame: pd.DataFrame = read(source, float_precision="round_trip", **kwargs)
    return frame


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# The exit criterion: same API, same numbers
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestTheCommandLineReproducesTheAPI:
    def test_validate_writes_what_validate_rfr_returns(self, site, tmp_path, capsys):
        # The timestamp column is named in the configuration, not on the command line.
        config = run_config(site, column_map=site.column_map.with_overrides(timestamp="TIMESTAMP"))
        frame = site.frame.rename_axis("TIMESTAMP").reset_index()
        data = tmp_path / "site.csv"
        frame.to_csv(data, index=False)
        config_file = write_json(config.to_dict(), tmp_path / "run.json")
        qc = site.qc_columns()
        output = tmp_path / "validation"

        code = main(
            [
                "validate", str(data),
                "--target", "H", "--target", "LE",
                "--config", str(config_file),
                "--qc-column", f"H={qc['H']}", "--qc-column", f"LE={qc['LE']}",
                "--output", str(output),
            ]
        )  # fmt: skip
        assert code == 0

        expected = validate_rfr(
            frame, config=config, targets=["H", "LE"], qc_columns={"H": qc["H"], "LE": qc["LE"]}
        )
        for name, table in {
            "metrics.csv": expected.to_frame(),
            "bias_spread.csv": expected.bias_spread_frame(),
            "energy_balance.csv": expected.energy_balance_frame(),
            "gaps.csv": expected.gaps.to_frame(),
        }.items():
            pd.testing.assert_frame_equal(read_exact(output / name), as_table(table), obj=name)
        assert len(read_exact(output / "energy_balance.csv")) > 0

        predictions = read_exact(output / "predictions.csv", index_col="TIMESTAMP")
        assert list(predictions.columns) == ["H_predicted", "LE_predicted"]
        np.testing.assert_array_equal(predictions.to_numpy(), expected.predictions().to_numpy())

        for target in ("H", "LE"):
            manifest = load_manifest(output / f"RFR3_{target}_manifest.json")
            assert manifest["kind"] == "validation"
            # The configuration file rebuilt exactly the configuration it was written from.
            assert manifest["settings_digest"] == expected.manifest(target).settings_digest
            invocation = manifest["extra"]["command_line"]
            assert invocation["command"] == "validate"
            assert invocation["data_sha256"] == sha256(data)
            assert invocation["config_sha256"] == sha256(config_file)

        printed = capsys.readouterr().out
        assert "RFR3 validation of H, LE" in printed
        assert f"wrote {output / 'metrics.csv'}" in printed

    def test_fill_of_a_fluxnet_style_file_matches_rfr_gap_filler(self, site, tmp_path):
        # FLUXNET's conventions: an integer YYYYMMDDHHMM timestamp and -9999 for missing.
        config = run_config(site)
        raw = site.frame.fillna(-9999.0)
        stamps = site.frame.index.strftime("%Y%m%d%H%M")
        raw.insert(0, "TIMESTAMP_START", stamps.astype("int64"))
        data = tmp_path / "FLX_site.csv"
        raw.to_csv(data, index=False)
        config_file = write_json(config.to_dict(), tmp_path / "run.json")
        qc = site.qc_column("LE")
        output = tmp_path / "out" / "filled.csv"

        code = main(
            [
                "fill", str(data),
                "--target", "LE",
                "--mode", "RFR10",
                "--config", str(config_file),
                "--qc-column", qc,
                "--timestamp", "TIMESTAMP_START",
                "--timestamp-format", "%Y%m%d%H%M",
                "--na-value", "-9999",
                "--refill-pre-filled",
                "--output", str(output),
            ]
        )  # fmt: skip
        assert code == 0

        filler = RFRGapFiller(config.replace(mode="RFR10")).fit(
            site.frame, target="LE", qc_column=qc
        )
        expected = filler.fill(site.frame, refill_pre_filled=True).frame
        written = read_exact(output, index_col="TIMESTAMP_START")
        assert list(pd.to_datetime(written.index).strftime("%Y%m%d%H%M")) == list(stamps)
        for column in ("LE_original", "LE_filled"):
            np.testing.assert_array_equal(
                written[column].to_numpy(dtype=float), expected[column].to_numpy(dtype=float)
            )
        for column in ("LE_is_observed", "LE_is_filled", "LE_fill_method"):
            np.testing.assert_array_equal(written[column].to_numpy(), expected[column].to_numpy())
        assert written["LE_is_filled"].any()
        assert (written["LE_fill_method"] == "RFR10").any()

        manifest = load_manifest(output.with_name("filled.manifest.json"))
        assert manifest["kind"] == "fill"
        assert manifest["run"]["mode"] == "RFR10"
        assert manifest["fill"]["filled_rows"] == int(expected["LE_is_filled"].sum())
        assert manifest["extra"]["command_line"]["data_sha256"] == sha256(data)


# ---------------------------------------------------------------------------
# The command line delegates rather than reimplements
# ---------------------------------------------------------------------------


class TestTheCommandLineDelegates:
    def test_validate_hands_its_arguments_to_validate_rfr(self, tmp_path, monkeypatch, capsys):
        seen: dict[str, Any] = {}

        def fake_validate_rfr(data: pd.DataFrame, **kwargs: Any) -> None:
            seen["data"] = data
            seen.update(kwargs)
            raise ValidationError("stopped by the test")

        monkeypatch.setattr(cli, "validate_rfr", fake_validate_rfr)
        config_file = minimal_config(
            tmp_path, column_map={**RFR10_COLUMNS, "timestamp": "TIMESTAMP"}
        )
        output = tmp_path / "out"

        code = main(
            [
                "validate", str(tiny_csv(tmp_path)),
                "--target", "LE",
                "--config", str(config_file),
                "--mode", "RFR10",
                "--qc-column", "LE_QC",
                "--na-value", "-9999",
                "--on-shortfall", "warn",
                "--check-leakage",
                "--output", str(output),
            ]
        )  # fmt: skip

        assert code == 1
        assert "stopped by the test" in capsys.readouterr().err
        assert isinstance(seen["config"], RFRConfig)
        assert seen["config"].rfr_mode is Mode.RFR10
        assert seen["targets"] == ["LE"]
        assert seen["qc_columns"] == "LE_QC"
        assert seen["timestamp"] == "TIMESTAMP"
        assert seen["on_shortfall"] == "warn"
        assert seen["check_leakage"] is True
        assert seen["energy_balance_targets"] is None
        assert seen["data"]["TIMESTAMP"].iloc[0] == "2020-06-01 00:00"
        assert seen["data"]["SW"].isna().tolist() == [False, True]
        assert not output.exists(), "a failed run must not leave an output directory"

    def test_fill_goes_through_rfr_gap_filler(self, tmp_path, monkeypatch):
        seen: dict[str, Any] = {}

        class FakeFiller:
            def __init__(self, config: RFRConfig) -> None:
                seen["config"] = config

            def fit(self, data: pd.DataFrame, **kwargs: Any) -> FakeFiller:
                seen.update(kwargs)
                raise FillError("stopped by the test")

        monkeypatch.setattr(cli, "RFRGapFiller", FakeFiller)
        output = tmp_path / "filled.csv"
        code = main(
            [
                "fill", str(tiny_csv(tmp_path)),
                "--target", "LE",
                "--config", str(minimal_config(tmp_path)),
                "--qc-column", "LE_QC",
                "--min-training-rows", "5",
                "--output", str(output),
            ]
        )  # fmt: skip

        assert code == 1
        assert seen["target"] == "LE"
        assert seen["qc_column"] == "LE_QC"
        assert seen["timestamp"] == "TIMESTAMP"
        assert seen["min_training_rows"] == 5
        assert not output.exists()


# ---------------------------------------------------------------------------
# The parser
# ---------------------------------------------------------------------------


class TestTheParser:
    def test_no_command_prints_help_naming_both_subcommands(self, capsys):
        assert main([]) == 0
        printed = capsys.readouterr().out
        assert "validate" in printed
        assert "fill" in printed

    def test_version(self, capsys):
        with pytest.raises(SystemExit) as exit_info:
            main(["--version"])
        assert exit_info.value.code == 0
        assert __version__ in capsys.readouterr().out

    def test_a_missing_required_option_is_a_usage_error(self, capsys):
        with pytest.raises(SystemExit) as exit_info:
            main(["fill", "data.csv"])
        assert exit_info.value.code == 2
        assert "--config" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Reading the CSV
# ---------------------------------------------------------------------------


class TestReadTable:
    def test_the_timestamp_column_is_read_as_text(self, tmp_path):
        path = tmp_path / "fluxnet.csv"
        path.write_text("TIMESTAMP_START,LE\n200506010000,1.0\n200506010030,2.0\n")
        frame = read_table(path, timestamp="TIMESTAMP_START")
        assert frame["TIMESTAMP_START"].tolist() == ["200506010000", "200506010030"]

    def test_a_timestamp_format_is_applied(self, tmp_path):
        path = tmp_path / "fluxnet.csv"
        path.write_text("TIMESTAMP_START,LE\n200506010000,1.0\n200506010030,2.0\n")
        frame = read_table(path, timestamp="TIMESTAMP_START", timestamp_format="%Y%m%d%H%M")
        assert frame["TIMESTAMP_START"].tolist() == [
            pd.Timestamp("2005-06-01 00:00"),
            pd.Timestamp("2005-06-01 00:30"),
        ]

    def test_a_timestamp_that_does_not_match_the_format_is_reported(self, tmp_path):
        with pytest.raises(CLIError, match="--timestamp-format"):
            read_table(tiny_csv(tmp_path), timestamp="TIMESTAMP", timestamp_format="%Y%m%d%H%M")

    def test_a_numeric_na_value_catches_its_float_spelling(self, tmp_path):
        frame = read_table(tiny_csv(tmp_path), timestamp="TIMESTAMP", na_values=["-9999"])
        assert frame["SW"].isna().tolist() == [False, True]
        assert frame["LE"].isna().tolist() == [False, True]

    def test_without_na_values_minus_9999_is_a_number(self, tmp_path):
        frame = read_table(tiny_csv(tmp_path), timestamp="TIMESTAMP")
        assert frame["LE"].tolist() == [5.0, -9999.0]


# ---------------------------------------------------------------------------
# Failures are messages
# ---------------------------------------------------------------------------


class TestFailuresAreMessages:
    def fill(self, tmp_path: Path, *extra: str, config: Path | None = None) -> list[str]:
        return [
            "fill", str(tiny_csv(tmp_path)),
            "--target", "LE",
            "--config", str(config if config is not None else minimal_config(tmp_path)),
            "--output", str(tmp_path / "filled.csv"),
            *extra,
        ]  # fmt: skip

    def test_a_missing_configuration_file(self, tmp_path, capsys):
        code = main(self.fill(tmp_path, config=tmp_path / "absent.json"))
        assert code == 1
        err = capsys.readouterr().err
        assert err.startswith("rfr-gapfill fill: error:")
        assert "absent.json" in err

    def test_a_misspelt_configuration_setting(self, tmp_path, capsys):
        config = minimal_config(tmp_path, randm_state=1)
        assert main(self.fill(tmp_path, config=config)) == 1
        assert "randm_state" in capsys.readouterr().err

    def test_an_unknown_mode(self, tmp_path, capsys):
        assert main(self.fill(tmp_path, "--mode", "RFR7")) == 1
        assert "RFR7" in capsys.readouterr().err

    def test_the_timestamp_column_must_be_named(self, tmp_path, capsys):
        config = minimal_config(
            tmp_path, column_map={"shortwave": "SW", "vpd": "VPD", "air_temperature": "TA"}
        )
        assert main(self.fill(tmp_path, config=config)) == 1
        assert "--timestamp" in capsys.readouterr().err

    def test_a_timestamp_column_absent_from_the_file(self, tmp_path, capsys):
        assert main(self.fill(tmp_path, "--timestamp", "WHEN")) == 1
        assert "WHEN" in capsys.readouterr().err

    def test_fill_never_overwrites_silently(self, tmp_path, capsys):
        existing = tmp_path / "filled.csv"
        existing.write_text("keep me\n")
        assert main(self.fill(tmp_path)) == 1
        assert "--overwrite" in capsys.readouterr().err
        assert existing.read_text() == "keep me\n"

    def test_fill_refuses_to_write_the_manifest_over_the_output(self, tmp_path, capsys):
        code = main(self.fill(tmp_path, "--manifest", str(tmp_path / "filled.csv")))
        assert code == 1
        assert "same file" in capsys.readouterr().err

    def test_validate_refuses_a_non_empty_directory(self, tmp_path, capsys):
        output = tmp_path / "validation"
        output.mkdir()
        (output / "metrics.csv").write_text("from an earlier run\n")
        code = main(
            [
                "validate", str(tiny_csv(tmp_path)),
                "--target", "LE",
                "--config", str(minimal_config(tmp_path)),
                "--output", str(output),
            ]
        )  # fmt: skip
        assert code == 1
        assert "--overwrite" in capsys.readouterr().err
        assert (output / "metrics.csv").read_text() == "from an earlier run\n"

    @pytest.mark.parametrize(
        ("values", "message"),
        [
            (["LE_QC", "H=H_QC"], "TARGET=COLUMN"),
            (["LE_QC", "H_QC"], "TARGET=COLUMN"),
            (["=LE_QC"], "TARGET=COLUMN"),
            (["LE=LE_QC", "LE=OTHER"], "more than once"),
        ],
    )
    def test_qc_columns_must_be_one_form(self, tmp_path, capsys, values, message):
        arguments = [
            "validate", str(tiny_csv(tmp_path)),
            "--target", "LE",
            "--config", str(minimal_config(tmp_path)),
            "--output", str(tmp_path / "out"),
        ]  # fmt: skip
        for value in values:
            arguments += ["--qc-column", value]
        assert main(arguments) == 1
        assert message in capsys.readouterr().err
