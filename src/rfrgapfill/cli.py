"""Command-line entry point (Step 22).

Two subcommands, each a thin shell around the tested Python API::

    rfr-gapfill validate site.csv --target LE --config run.json --output validation/
    rfr-gapfill fill site.csv --target LE --config run.json --output filled.csv

``validate`` runs :func:`~rfrgapfill.validation.validate_rfr` and writes its
tables and one run manifest per target into a directory. ``fill`` fits
:class:`~rfrgapfill.fill.RFRGapFiller`, fills, and writes the filled frame beside
its run manifest.

Nothing scientific is decided here. The configuration file is read by
:func:`~rfrgapfill.config.load_config` into the same validated
:class:`~rfrgapfill.config.RFRConfig` a script would build, and the command line
owns only what the library deliberately leaves to its caller: reading the CSV,
handing its timestamp column to the time layer as text, and writing results. A
batch run and a notebook run of one configuration therefore go through the same
code and produce the same numbers.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from rfrgapfill import __version__
from rfrgapfill.config import RFRConfig, load_config
from rfrgapfill.features import FeatureError
from rfrgapfill.fill import FillError, RFRGapFiller
from rfrgapfill.gaps import GapError
from rfrgapfill.leakage import LeakageError
from rfrgapfill.metrics import MetricError
from rfrgapfill.model import IncompletePolicy, ModelError
from rfrgapfill.provenance import ProvenanceError, RunManifest
from rfrgapfill.schema import ConfigError
from rfrgapfill.time import DuplicatePolicy, TimestampError
from rfrgapfill.validation import ValidationError, ValidationReport, validate_rfr

__all__ = ["CLIError", "build_parser", "main", "read_table", "write_validation"]

#: The console-script name (``[project.scripts]`` in ``pyproject.toml``).
PROG = "rfr-gapfill"


class CLIError(RuntimeError):
    """Raised for a command-line request that cannot be carried out as given."""


#: Failures reported as one line on stderr with exit status 1. Anything else is a
#: defect and keeps its traceback.
_RUN_ERRORS: tuple[type[Exception], ...] = (
    CLIError,
    ConfigError,
    TimestampError,
    FeatureError,
    MetricError,
    ModelError,
    FillError,
    GapError,
    LeakageError,
    ValidationError,
    ProvenanceError,
    OSError,
    pd.errors.ParserError,
    pd.errors.EmptyDataError,
)


# ---------------------------------------------------------------------------
# The parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Return the top-level argument parser."""
    parser = argparse.ArgumentParser(
        prog=PROG,
        description=(
            "Leakage-safe Random Forest Robust gap filling for eddy-covariance data "
            "(Zhu et al., 2022)."
        ),
    )
    parser.add_argument("--version", action="version", version=f"{PROG} {__version__}")
    commands = parser.add_subparsers(dest="command", metavar="COMMAND")

    validate = commands.add_parser(
        "validate",
        help="score the method on artificial gaps placed over observed data",
        description=(
            "Place the configured artificial gaps, fit on the remaining observations, "
            "predict the withheld intervals and score them (validate_rfr). Writes "
            "metrics.csv, bias_spread.csv, energy_balance.csv, gaps.csv, predictions.csv, "
            "summary.txt and one run manifest per target into --output."
        ),
    )
    _add_run_arguments(validate)
    validate.add_argument(
        "--target",
        dest="targets",
        action="append",
        required=True,
        metavar="COLUMN",
        help="target flux column; repeat it for a joint run on one shared set of gaps",
    )
    validate.add_argument(
        "--qc-column",
        action="append",
        metavar="[TARGET=]COLUMN",
        help=(
            "QC column separating measured from pre-filled values: COLUMN for a single "
            "target, or TARGET=COLUMN once per target"
        ),
    )
    validate.add_argument(
        "--output",
        type=Path,
        required=True,
        metavar="DIRECTORY",
        help="directory to write tables and manifests into; must be empty or new",
    )
    validate.add_argument(
        "--energy-balance-targets",
        nargs=2,
        metavar=("H", "LE"),
        help="the H and LE target columns, when they are not called H and LE",
    )
    validate.add_argument(
        "--on-shortfall",
        choices=("raise", "warn"),
        default="raise",
        help="when the gap scenario cannot be placed as configured (default: raise)",
    )
    validate.add_argument(
        "--check-leakage",
        action="store_true",
        help="prove the features cannot see withheld values before fitting (slower)",
    )
    validate.set_defaults(handler=run_validate)

    fill = commands.add_parser(
        "fill",
        help="fill the real gaps of one target",
        description=(
            "Fit on the target's observed rows and fill its gaps (RFRGapFiller). Writes "
            "the input columns plus the six provenance columns to --output, and the run "
            "manifest beside it."
        ),
    )
    _add_run_arguments(fill)
    fill.add_argument("--target", required=True, metavar="COLUMN", help="target flux column")
    fill.add_argument(
        "--qc-column",
        metavar="COLUMN",
        help="QC column separating measured from pre-filled values (strongly recommended)",
    )
    fill.add_argument(
        "--output", type=Path, required=True, metavar="FILE", help="CSV file to write"
    )
    fill.add_argument(
        "--manifest",
        type=Path,
        metavar="FILE",
        help="where to write the run manifest (default: <output stem>.manifest.json)",
    )
    fill.add_argument(
        "--refill-pre-filled",
        action="store_true",
        help="also replace values the QC flag marks as gap-filled before ingestion",
    )
    fill.set_defaults(handler=run_fill)
    return parser


def _add_run_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the input and run options both subcommands share."""
    parser.add_argument("data", type=Path, help="input CSV file, one row per timestamp")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        metavar="FILE",
        help="run configuration, JSON or YAML, in the shape of RFRConfig.to_dict()",
    )
    parser.add_argument(
        "--mode", metavar="{RFR3,RFR10}", help="driver set; overrides the configuration's mode"
    )
    parser.add_argument(
        "--timestamp",
        metavar="COLUMN",
        help="timestamp column (default: column_map.timestamp in the configuration)",
    )
    parser.add_argument(
        "--timestamp-format",
        metavar="FORMAT",
        help=(
            "strftime format of the timestamp column, e.g. %%Y%%m%%d%%H%%M for FLUXNET's "
            "TIMESTAMP_START; without it the column must hold date-time text"
        ),
    )
    parser.add_argument(
        "--na-value",
        action="append",
        default=[],
        metavar="VALUE",
        help="a value marking missing data, e.g. -9999 for FLUXNET; repeatable",
    )
    parser.add_argument(
        "--on-duplicates",
        choices=[policy.value for policy in DuplicatePolicy],
        default=DuplicatePolicy.ERROR.value,
        help="what to do with repeated timestamps (default: %(default)s)",
    )
    parser.add_argument(
        "--on-incomplete",
        choices=[policy.value for policy in IncompletePolicy],
        default=IncompletePolicy.MISSING.value,
        help="a row lacking a predictor is left unpredicted, or fails the run",
    )
    parser.add_argument(
        "--min-training-rows", type=int, metavar="N", help="floor on usable training rows"
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="replace existing output instead of refusing"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line interface and return a process exit code.

    ``0`` on success, ``1`` when the run cannot be carried out - reported as one
    line on stderr, not a traceback - and ``2`` for a malformed command line.
    """
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(arguments)
    handler: Callable[[argparse.Namespace], int] | None = getattr(args, "handler", None)
    if handler is None:
        parser.print_help()
        return 0
    args.argv = arguments
    try:
        return handler(args)
    except _RUN_ERRORS as exc:
        print(f"{PROG} {args.command}: error: {exc}", file=sys.stderr)
        return 1


def run_validate(args: argparse.Namespace) -> int:
    """``rfr-gapfill validate``: :func:`validate_rfr`, then its tables and manifests."""
    config = _config(args)
    timestamp = _timestamp_column(args.timestamp, config)
    _require_empty_directory(args.output, overwrite=args.overwrite)
    qc_columns = _validation_qc_columns(args.qc_column)
    pair = args.energy_balance_targets
    data = read_table(
        args.data,
        timestamp=timestamp,
        timestamp_format=args.timestamp_format,
        na_values=args.na_value,
    )
    report = validate_rfr(
        data,
        config=config,
        targets=args.targets,
        qc_columns=qc_columns,
        timestamp=timestamp,
        on_duplicates=args.on_duplicates,
        on_shortfall=args.on_shortfall,
        on_incomplete=args.on_incomplete,
        min_training_rows=args.min_training_rows,
        check_leakage=args.check_leakage,
        energy_balance_targets=None if pair is None else (str(pair[0]), str(pair[1])),
    )
    written = write_validation(report, args.output, timestamp=timestamp, extra=_invocation(args))
    print(report.summary())
    _report_written(written)
    return 0


def run_fill(args: argparse.Namespace) -> int:
    """``rfr-gapfill fill``: fit :class:`RFRGapFiller`, fill, write frame and manifest."""
    config = _config(args)
    timestamp = _timestamp_column(args.timestamp, config)
    output: Path = args.output
    manifest: Path = (
        args.manifest
        if args.manifest is not None
        else output.with_name(f"{output.stem}.manifest.json")
    )
    if manifest.resolve() == output.resolve():
        raise CLIError("--manifest and --output name the same file")
    for path in (output, manifest):
        _require_new_file(path, overwrite=args.overwrite)
    data = read_table(
        args.data,
        timestamp=timestamp,
        timestamp_format=args.timestamp_format,
        na_values=args.na_value,
    )
    filler = RFRGapFiller(config).fit(
        data,
        target=args.target,
        qc_column=args.qc_column,
        timestamp=timestamp,
        on_duplicates=args.on_duplicates,
        min_training_rows=args.min_training_rows,
    )
    result = filler.fill(
        data,
        on_incomplete=args.on_incomplete,
        refill_pre_filled=args.refill_pre_filled,
        timestamp=timestamp,
        on_duplicates=args.on_duplicates,
    )
    # Built, and checked complete, before anything is written: a filled file must
    # never exist without the manifest that explains it.
    record = RunManifest.from_fill(filler, result, extra=_invocation(args)).require_complete()
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    result.frame.to_csv(output, index_label=timestamp)
    record.save(manifest)
    print(result.report.summary())
    _report_written([output, manifest])
    return 0


# ---------------------------------------------------------------------------
# Input and output
# ---------------------------------------------------------------------------


def read_table(
    path: str | Path,
    *,
    timestamp: str,
    timestamp_format: str | None = None,
    na_values: Sequence[str] = (),
) -> pd.DataFrame:
    """Return the CSV at ``path`` with its timestamp column ready for the time layer.

    The timestamp column is read as text, never as a number: FLUXNET's
    ``TIMESTAMP_START`` (``200501010030``) would otherwise arrive as an integer,
    which :mod:`rfrgapfill.time` refuses because pandas reads numbers as epoch
    nanoseconds. With ``timestamp_format`` the text is parsed here with exactly
    that format; without it the time layer parses and validates it.

    ``na_values`` are extra missing-value markers. A numeric marker also matches
    its float spelling, so ``"-9999"`` catches ``-9999.0``.

    Numbers are parsed with ``float_precision="round_trip"``: pandas' default
    parser can be one unit in the last place off, and a model fitted on values
    that differ from the file's would not be the model the API fits on them.
    """
    frame: pd.DataFrame = pd.read_csv(
        path,
        dtype={timestamp: str},
        na_values=list(na_values) or None,
        float_precision="round_trip",
    )
    if timestamp_format is not None and timestamp in frame.columns:
        try:
            frame[timestamp] = pd.to_datetime(frame[timestamp], format=timestamp_format)
        except (TypeError, ValueError) as exc:
            raise CLIError(
                f"timestamp column {timestamp!r} does not match --timestamp-format "
                f"{timestamp_format!r}: {exc}"
            ) from exc
    return frame


def write_validation(
    report: ValidationReport,
    directory: Path,
    *,
    timestamp: str,
    extra: Mapping[str, Any] | None = None,
) -> list[Path]:
    """Write a validation report into ``directory`` and return the paths written.

    ``metrics.csv``, ``bias_spread.csv``, ``energy_balance.csv`` (a header only
    when the run had no H/LE pair) and ``gaps.csv`` are the report's own tables,
    always all four, so a rerun never leaves a stale table beside fresh ones.
    ``predictions.csv`` holds every target's predictions on the run's time axis,
    ``<method>_<target>_manifest.json`` one complete run manifest per target, and
    ``summary.txt`` :meth:`ValidationReport.summary`.
    """
    directory.mkdir(parents=True, exist_ok=True)
    tables = {
        "metrics.csv": report.to_frame(),
        "bias_spread.csv": report.bias_spread_frame(),
        "energy_balance.csv": report.energy_balance_frame(),
        "gaps.csv": report.gaps.to_frame(),
    }
    written: list[Path] = []
    for name, table in tables.items():
        path = directory / name
        table.to_csv(path, index=False)
        written.append(path)
    predictions = directory / "predictions.csv"
    report.predictions().to_csv(predictions, index_label=timestamp)
    written.append(predictions)
    for target in report.targets:
        path = directory / f"{report.method}_{target}_manifest.json"
        written.append(report.manifest(target, extra=extra).save(path))
    summary = directory / "summary.txt"
    summary.write_text(report.summary() + "\n", encoding="utf-8")
    written.append(summary)
    return written


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _config(args: argparse.Namespace) -> RFRConfig:
    """Return the run configuration, with ``--mode`` applied before validation."""
    overrides: dict[str, Any] = {}
    if args.mode is not None:
        overrides["mode"] = args.mode
    return load_config(args.config, **overrides)


def _timestamp_column(requested: str | None, config: RFRConfig) -> str:
    """Return the timestamp column: ``--timestamp``, else the configured one."""
    column = requested if requested is not None else config.columns.timestamp
    if column is None:
        raise CLIError(
            "name the timestamp column: pass --timestamp COLUMN, or set "
            "column_map.timestamp in the configuration"
        )
    return column


def _validation_qc_columns(values: Sequence[str] | None) -> str | dict[str, str] | None:
    """Return ``--qc-column`` values in the form :func:`validate_rfr` takes."""
    if not values:
        return None
    bare = [value for value in values if "=" not in value]
    if bare:
        if len(values) > 1:
            raise CLIError(
                "give --qc-column once as COLUMN for a single target, or once per target "
                "as TARGET=COLUMN"
            )
        return bare[0]
    mapping: dict[str, str] = {}
    for value in values:
        target, _, column = value.partition("=")
        if not target or not column:
            raise CLIError(f"--qc-column {value!r} must look like TARGET=COLUMN")
        if target in mapping:
            raise CLIError(f"--qc-column names target {target!r} more than once")
        mapping[target] = column
    return mapping


def _require_empty_directory(path: Path, *, overwrite: bool) -> None:
    """Refuse to write into a non-empty directory unless ``--overwrite`` was given."""
    if path.exists() and not path.is_dir():
        raise CLIError(f"--output {path} exists and is not a directory")
    if path.is_dir() and any(path.iterdir()) and not overwrite:
        raise CLIError(f"--output {path} is not empty; pass --overwrite to write into it")


def _require_new_file(path: Path, *, overwrite: bool) -> None:
    """Refuse to replace an existing file unless ``--overwrite`` was given."""
    if path.is_dir():
        raise CLIError(f"{path} is a directory; name a file to write")
    if path.exists() and not overwrite:
        raise CLIError(f"{path} already exists; pass --overwrite to replace it")


def _invocation(args: argparse.Namespace) -> dict[str, Any]:
    """Return the manifest's record of how this command was invoked.

    The input and configuration files are identified by content as well as by
    path, so a manifest still says which data produced it after files move.
    """
    return {
        "command_line": {
            "command": args.command,
            "arguments": list(args.argv),
            "data": str(args.data),
            "data_sha256": _sha256(args.data),
            "config": str(args.config),
            "config_sha256": _sha256(args.config),
        }
    }


def _sha256(path: Path) -> str:
    """Return the SHA-256 hex digest of the file at ``path``."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _report_written(paths: Sequence[Path]) -> None:
    """Print one line per file written."""
    for path in paths:
        print(f"wrote {path}")


if __name__ == "__main__":  # pragma: no cover - module executed as a script
    raise SystemExit(main())
