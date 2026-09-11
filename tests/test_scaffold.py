"""Scaffold smoke tests.

These guard the packaging contract only: the package imports, every module named in
the planned architecture exists and is importable, the typing marker ships, and the
console-script entry point runs. Scientific behaviour is tested in the per-module
test files as each implementation step lands.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

MODULES = [
    "config",
    "schema",
    "time",
    "features",
    "gaps",
    "model",
    "fill",
    "metrics",
    "validation",
    "fluxnet",
    "provenance",
    "cli",
]


def test_package_imports_and_reports_a_version() -> None:
    import rfrgapfill

    assert isinstance(rfrgapfill.__version__, str)
    assert rfrgapfill.__version__
    assert rfrgapfill.PAPER_DOI == "10.1016/j.agrformet.2021.108777"


@pytest.mark.parametrize("name", MODULES)
def test_planned_module_is_importable(name: str) -> None:
    module = importlib.import_module(f"rfrgapfill.{name}")
    assert module.__doc__, f"rfrgapfill.{name} must document its contract"


def test_package_ships_typing_marker() -> None:
    import rfrgapfill

    package_dir = Path(rfrgapfill.__file__).parent
    assert (package_dir / "py.typed").is_file()


def test_runtime_dependencies_are_importable() -> None:
    for name in ("numpy", "pandas", "scipy", "sklearn", "joblib"):
        importlib.import_module(name)


def test_cli_entry_point_runs(capsys: pytest.CaptureFixture[str]) -> None:
    from rfrgapfill.cli import main

    assert main([]) == 0
    assert "rfr-gapfill" in capsys.readouterr().out


def test_specification_documents_are_present() -> None:
    docs = Path(__file__).resolve().parents[1] / "docs"
    for name in (
        "method_spec.md",
        "method_spec.yaml",
        "supplement_benchmarks.md",
        "fluxnet_adapter.md",
    ):
        assert (docs / name).is_file(), f"missing specification document: docs/{name}"
