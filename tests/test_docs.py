"""Documentation tests (Step 21).

The exit criterion of Step 21 is that a new user can run the synthetic example
from the README, so the quick start is executed here exactly as it is written.
The rest keeps the user documentation honest as the code moves: the pages exist
and are linked, every relative link resolves, and the ambiguities the user guide
explains are exactly the ones the specification defines.
"""

from __future__ import annotations

import json
import re
import sys
import warnings
from pathlib import Path

import pytest

import rfrgapfill

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
DOCS = ROOT / "docs"
EXAMPLES = ROOT / "examples"
NOTEBOOK_DIR = EXAMPLES / "notebooks"
NOTEBOOKS = sorted(NOTEBOOK_DIR.glob("*.ipynb"))

#: The user documentation Step 21 asks for, and the index of the example notebooks.
USER_DOCS = (
    "method.md",
    "validation.md",
    "assumptions.md",
    "fluxnet.md",
    "supplement_benchmarks.md",
    "examples.md",
)

#: What a notebook may import besides the standard library: the package and the
#: `notebooks` extra's plotting stack, nothing a reader would have to go and find.
NOTEBOOK_IMPORTS = {"rfrgapfill", "pandas", "numpy", "matplotlib"}

#: A saved output carrying one of these was run on somebody's machine and says so.
MACHINE_PATHS = re.compile(r"[A-Za-z]:\\\\?Users|/home/|/Users/|AppData|site-packages")

QUICKSTART = re.compile(
    r"<!-- quickstart:begin -->\s*```python\n(?P<code>.*?)```\s*<!-- quickstart:end -->",
    re.DOTALL,
)
LINK = re.compile(r"\]\((?P<target>[^)\s]+)\)")
TABLE_AMBIGUITY = re.compile(r"^\| (A\d+) \|", re.MULTILINE)
HEADED_AMBIGUITY = re.compile(r"^### (A\d+)\b", re.MULTILINE)


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def quickstart_source() -> str:
    matches = list(QUICKSTART.finditer(read(README)))
    assert len(matches) == 1, "README.md must carry exactly one marked quick-start block"
    return matches[0].group("code")


class TestTheDocumentsExist:
    @pytest.mark.parametrize("name", USER_DOCS)
    def test_step_21_page_is_present(self, name):
        assert (DOCS / name).is_file(), f"missing user documentation: docs/{name}"

    @pytest.mark.parametrize("name", USER_DOCS)
    def test_the_readme_links_it(self, name):
        assert f"](docs/{name}" in read(README)


class TestLinksResolve:
    @pytest.mark.parametrize(
        "path", [README, *(DOCS / name for name in USER_DOCS)], ids=lambda path: path.name
    )
    def test_every_relative_link_points_at_a_file_that_exists(self, path):
        broken = []
        for match in LINK.finditer(read(path)):
            target = match.group("target")
            if target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            if not (path.parent / target.split("#", 1)[0]).exists():
                broken.append(target)
        assert not broken, f"{path.name} links to missing files: {broken}"


class TestAmbiguitiesAreExplained:
    def specification(self) -> set[str]:
        ids = set(TABLE_AMBIGUITY.findall(read(DOCS / "method_spec.md")))
        assert ids, "method_spec.md defines no ambiguities; the pattern has drifted"
        return ids

    def test_the_user_guide_has_a_section_for_every_ambiguity(self):
        assert set(HEADED_AMBIGUITY.findall(read(DOCS / "assumptions.md"))) == (
            self.specification()
        )

    def test_the_readme_summarises_every_ambiguity(self):
        assert set(TABLE_AMBIGUITY.findall(read(README))) == self.specification()


class TestTheQuickStart:
    def test_it_is_valid_python(self):
        compile(quickstart_source(), "README.md quick start", "exec")

    def test_it_needs_nothing_but_the_package(self):
        imports = re.findall(r"^(?:from|import) (\S+)", quickstart_source(), re.MULTILINE)
        assert imports
        assert set(imports) == {"rfrgapfill"}

    @pytest.mark.slow
    def test_it_runs_as_written(self, capsys):
        namespace: dict[str, object] = {"__name__": "readme_quickstart"}
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            exec(compile(quickstart_source(), "README.md quick start", "exec"), namespace)
        printed = capsys.readouterr().out

        # A first run should not greet a new user with the package's own warnings.
        ours = [
            str(warning.message)
            for warning in caught
            if warning.category.__module__.startswith("rfrgapfill")
        ]
        assert not ours

        report = namespace["report"]
        assert isinstance(report, rfrgapfill.ValidationReport)
        assert {result.target for result in report} == {"NEE", "H", "LE"}

        site = namespace["site"]
        result = namespace["result"]
        assert isinstance(site, rfrgapfill.SyntheticSite)
        assert isinstance(result, rfrgapfill.FillResult)
        frame = result.frame
        observed = frame["LE_is_observed"].to_numpy(dtype=bool)
        assert observed.any()
        filled = frame["LE_filled"].to_numpy()[observed]
        assert (filled == site.frame["LE"].to_numpy()[observed]).all()
        assert frame["LE_is_filled"].any()
        assert "LE: filled" in printed


def notebook_cells(path: Path, kind: str) -> list[dict]:
    return [cell for cell in json.loads(read(path))["cells"] if cell["cell_type"] == kind]


def joined(value: str | list[str]) -> str:
    return value if isinstance(value, str) else "".join(value)


def printed_text(output: dict) -> str:
    """Return what an output shows as text; images are left out."""
    if output["output_type"] == "stream":
        return joined(output["text"])
    data = output.get("data", {})
    return "".join(joined(data[kind]) for kind in ("text/plain", "text/html") if kind in data)


class TestTheNotebooks:
    """The example notebooks are documentation, so they are held to the same standard."""

    def test_there_are_notebooks_to_check(self):
        assert len(NOTEBOOKS) >= 4

    @pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda path: path.name)
    def test_the_index_and_the_examples_readme_link_it(self, path):
        assert f"](../examples/notebooks/{path.name})" in read(DOCS / "examples.md")
        assert f"](notebooks/{path.name})" in read(EXAMPLES / "README.md")

    @pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda path: path.name)
    def test_it_is_a_python_notebook_in_format_4(self, path):
        document = json.loads(read(path))
        assert document["nbformat"] == 4
        assert document["metadata"]["kernelspec"]["language"] == "python"

    @pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda path: path.name)
    def test_its_code_is_valid_python(self, path):
        for number, cell in enumerate(notebook_cells(path, "code")):
            compile(joined(cell["source"]), f"{path.name}, code cell {number}", "exec")

    @pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda path: path.name)
    def test_it_needs_only_the_package_and_the_notebooks_extra(self, path):
        source = "\n".join(joined(cell["source"]) for cell in notebook_cells(path, "code"))
        modules = re.findall(r"^(?:from|import) (\S+)", source, re.MULTILINE)
        imported = {module.split(".")[0] for module in modules}
        assert "rfrgapfill" in imported
        assert not imported - NOTEBOOK_IMPORTS - set(sys.stdlib_module_names)

    @pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda path: path.name)
    def test_its_saved_outputs_come_from_one_clean_run(self, path):
        cells = notebook_cells(path, "code")
        counts = [cell["execution_count"] for cell in cells]
        assert counts == list(range(1, len(cells) + 1)), (
            f"{path.name} was not run top to bottom in one kernel; rerun it before committing"
        )
        errors = [
            output
            for cell in cells
            for output in cell["outputs"]
            if output["output_type"] == "error"
        ]
        assert not errors

    @pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda path: path.name)
    def test_its_outputs_name_no_local_paths(self, path):
        shown = [
            printed_text(output)
            for cell in notebook_cells(path, "code")
            for output in cell["outputs"]
        ]
        leaks = [text for text in shown if MACHINE_PATHS.search(text)]
        assert not leaks, f"{path.name} outputs show a local path: {leaks[0][:200]}"

    @pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda path: path.name)
    def test_every_relative_link_points_at_a_file_that_exists(self, path):
        broken = []
        for cell in notebook_cells(path, "markdown"):
            for match in LINK.finditer(joined(cell["source"])):
                target = match.group("target")
                if target.startswith(("http://", "https://", "mailto:", "#")):
                    continue
                if not (path.parent / target.split("#", 1)[0]).exists():
                    broken.append(target)
        assert not broken, f"{path.name} links to missing files: {broken}"

    @pytest.mark.slow
    @pytest.mark.notebooks
    @pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda path: path.name)
    def test_it_runs_from_top_to_bottom(self, path):
        nbformat = pytest.importorskip("nbformat")
        nbclient = pytest.importorskip("nbclient")
        pytest.importorskip("ipykernel")
        document = nbformat.read(path, as_version=4)
        nbclient.NotebookClient(
            document,
            timeout=1800,
            kernel_name="python3",
            resources={"metadata": {"path": str(NOTEBOOK_DIR)}},
        ).execute()
