"""Sphinx configuration for the rfr-gapfill documentation (Read the Docs).

The pages are the Markdown files in this directory, read with MyST, so the same
files render on GitHub and here. Some of their links point outside ``docs/``,
where Sphinx cannot follow: the README (this site's landing page), the example
notebooks in ``examples/notebooks/`` (copied in at build time) and repository
files such as ``CITATION.cff`` (sent to GitHub). ``retarget`` rewrites those
links as each page is read, so the sources keep the links that work on GitHub.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import rfrgapfill

DOCS = Path(__file__).resolve().parent
ROOT = DOCS.parent
README = ROOT / "README.md"
NOTEBOOK_SOURCE = ROOT / "examples" / "notebooks"
#: Generated and ignored by git: Sphinx can only render files under ``docs/``.
NOTEBOOK_COPY = DOCS / "notebooks"

REPOSITORY = "https://github.com/inkenbrandt/Imputation"
#: On Read the Docs, repository links point at the ref being built, so a tagged
#: release's documentation links that release's files.
GIT_REF = os.environ.get("READTHEDOCS_GIT_IDENTIFIER", "main")

# -- Project -----------------------------------------------------------------

project = "rfr-gapfill"
author = "Paul Inkenbrandt"
copyright = f"2026, {author}"
release = rfrgapfill.__version__
version = release

# -- General -----------------------------------------------------------------

extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "sphinxarg.ext",
]
templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    # Planning documents written for the implementation, not for users.
    "rfr_gapfill_ai_steps_v2.md",
    "rfr_gapfill_context_v2.md",
]

# Headings get GitHub-style anchors, so `assumptions.md#a4-...` links work here too.
myst_heading_anchors = 4
myst_enable_extensions = ["colon_fence"]

# The notebooks are committed with their outputs and CI checks those come from
# one clean run, so the documentation renders them rather than re-running them.
nb_execution_mode = "off"

autosummary_generate = True
# Several modules re-export names in __all__ (config re-exports schema's
# Hemisphere, for one). Ignoring __all__ documents each object once, in the
# module that defines it.
autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
    "ignore-module-all": True,
}
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_class_signature = "separated"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
}

# -- HTML --------------------------------------------------------------------

html_theme = "furo"
html_title = f"rfr-gapfill {release}"
html_theme_options = {
    "top_of_page_buttons": [],
}
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True

# -- Links that leave docs/ --------------------------------------------------

LINK = re.compile(r"\]\((?P<target>[^)\s]+)\)")


def retarget(text: str, source_dir: Path, output_dir: Path) -> str:
    """Rewrite the Markdown links in ``text`` for a page built into ``output_dir``.

    ``source_dir`` is where the links were written relative to. Links to pages
    Sphinx builds stay relative; anything else in the repository goes to GitHub.
    """

    def replace(match: re.Match[str]) -> str:
        target = match.group("target")
        if target.startswith(("http://", "https://", "mailto:", "#")):
            return match.group(0)
        path, hash_, anchor = target.partition("#")
        resolved = (source_dir / path).resolve()
        if resolved == README:
            page = DOCS / "index.md"
        elif resolved.parent == NOTEBOOK_SOURCE and resolved.suffix == ".ipynb":
            page = NOTEBOOK_COPY / resolved.name
        elif resolved.suffix in {".md", ".ipynb"} and resolved.is_relative_to(DOCS):
            page = resolved
        else:
            kind = "tree" if resolved.is_dir() else "blob"
            location = resolved.relative_to(ROOT).as_posix()
            return f"]({REPOSITORY}/{kind}/{GIT_REF}/{location}{hash_}{anchor})"
        relative = Path(os.path.relpath(page, output_dir)).as_posix()
        return f"]({relative}{hash_}{anchor})"

    return LINK.sub(replace, text)


def copy_notebooks() -> None:
    """Copy the example notebooks under ``docs/``, retargeting their links."""
    NOTEBOOK_COPY.mkdir(exist_ok=True)
    wanted = {path.name for path in NOTEBOOK_SOURCE.glob("*.ipynb")}
    for stale in NOTEBOOK_COPY.glob("*.ipynb"):
        if stale.name not in wanted:
            stale.unlink()
    for source in sorted(NOTEBOOK_SOURCE.glob("*.ipynb")):
        document = json.loads(source.read_text(encoding="utf-8"))
        for cell in document["cells"]:
            if cell["cell_type"] == "markdown":
                text = retarget("".join(cell["source"]), NOTEBOOK_SOURCE, NOTEBOOK_COPY)
                cell["source"] = text.splitlines(keepends=True)
        copied = json.dumps(document, indent=1, ensure_ascii=False) + "\n"
        target = NOTEBOOK_COPY / source.name
        # Rewriting an unchanged copy would make Sphinx reread it on every build.
        if not target.exists() or target.read_text(encoding="utf-8") != copied:
            target.write_text(copied, encoding="utf-8")


def retarget_page(app: Any, docname: str, source: list[str]) -> None:
    path = Path(app.env.doc2path(docname))
    if path.suffix == ".md":
        source[0] = retarget(source[0], path.parent, path.parent)


def retarget_include(
    app: Any, relative_path: Path, parent_docname: str, content: list[str]
) -> None:
    included = (DOCS / relative_path).resolve()
    parent = Path(app.env.doc2path(parent_docname)).parent
    content[0] = retarget(content[0], included.parent, parent)


def setup(app: Any) -> None:
    copy_notebooks()
    app.connect("source-read", retarget_page)
    app.connect("include-read", retarget_include)
