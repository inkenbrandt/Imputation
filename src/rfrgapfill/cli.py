"""Command-line entry point.

Subcommands (``fill``, ``validate``, ``benchmark``) are added alongside the
modules that implement them. For now the CLI reports the installed version so the
console-script wiring is exercised by the scaffold tests.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from rfrgapfill import __version__

__all__ = ["build_parser", "main"]


def build_parser() -> argparse.ArgumentParser:
    """Return the top-level argument parser."""
    parser = argparse.ArgumentParser(
        prog="rfr-gapfill",
        description=(
            "Leakage-safe Random Forest Robust gap filling for eddy-covariance data "
            "(Zhu et al., 2022)."
        ),
    )
    parser.add_argument("--version", action="version", version=f"rfr-gapfill {__version__}")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line interface and return a process exit code."""
    parser = build_parser()
    parser.parse_args(argv)
    parser.print_help()
    return 0


if __name__ == "__main__":  # pragma: no cover - module executed as a script
    raise SystemExit(main())
