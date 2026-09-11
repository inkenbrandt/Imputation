"""Documentation tests (Step 21).

The exit criterion of Step 21 is that a new user can run the synthetic example
from the README, so the quick start is executed here exactly as it is written.
The rest keeps the user documentation honest as the code moves: the pages exist
and are linked, every relative link resolves, and the ambiguities the user guide
explains are exactly the ones the specification defines.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import pytest

import rfrgapfill

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
DOCS = ROOT / "docs"

#: The user documentation Step 21 asks for.
USER_DOCS = (
    "method.md",
    "validation.md",
    "assumptions.md",
    "fluxnet.md",
    "supplement_benchmarks.md",
)

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
