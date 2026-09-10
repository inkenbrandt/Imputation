"""Observation provenance and run manifests.

Answers two questions the rest of the package keeps asking.

*Which target values are genuinely measured?* Flux archives ship already
gap-filled values alongside real ones, distinguished by a QC flag
(``docs/method_spec.md`` section 7). Training and validation must prefer real
measurements, and artificial gaps must withhold only real measurements - hiding
a value that was itself gap-filled and then scoring against it measures nothing.
:func:`observed_mask` is the single place that decision is made.

*What produced this number?* Every run records the package, Python and
scikit-learn versions, the column mapping, the mode, the feature mode, the grid,
the chosen parameters, the seed, the gap manifest, the cadence, the hemisphere
and the QC rule (method_spec.md section 7). :func:`run_manifest` assembles that,
and :func:`environment` captures the version half of it.
"""

from __future__ import annotations

import platform
import sys
from collections.abc import Mapping, Sequence
from typing import Any, Final

import numpy as np
import pandas as pd

from rfrgapfill.config import RFRConfig

__all__ = [
    "FILL_METHOD_RFR",
    "FILL_SUFFIXES",
    "environment",
    "fill_column_names",
    "observed_mask",
    "run_manifest",
]

#: Value written to ``<target>_fill_method`` for a Random Forest prediction.
FILL_METHOD_RFR: Final = "RFR"

#: Suffixes of the provenance columns filling adds (method_spec.md section 7).
FILL_SUFFIXES: Final[tuple[str, ...]] = (
    "original",
    "filled",
    "is_observed",
    "is_filled",
    "fill_method",
    "model_version",
)


def fill_column_names(target: str) -> tuple[str, ...]:
    """Return the provenance column names filling adds for ``target``, in order."""
    return tuple(f"{target}_{suffix}" for suffix in FILL_SUFFIXES)


def observed_mask(
    data: pd.DataFrame,
    target: str,
    *,
    qc_column: str | None = None,
    observed_qc_values: Sequence[int] = (0,),
) -> pd.Series:
    """Return the rows where ``target`` is a genuine, quality-controlled measurement.

    A row qualifies when the target value is finite **and**, if ``qc_column`` is
    given, its flag is one of ``observed_qc_values`` - 0 in FLUXNET2015, meaning
    measured rather than gap-filled.

    Without a QC column every finite value counts as observed, and the caller has
    accepted that any pre-filled values in the series will be treated as truth.
    That is the right default for data a user has already cleaned and the wrong
    one for a raw FLUXNET file, which is why the QC column is a parameter and not
    an inference.
    """
    if target not in data.columns:
        raise KeyError(f"target column {target!r} is not in the data")
    values = pd.to_numeric(data[target], errors="coerce")
    mask = np.isfinite(values.to_numpy(dtype=float))

    if qc_column is not None:
        if qc_column not in data.columns:
            raise KeyError(
                f"QC column {qc_column!r} is not in the data; pass qc_col=None to treat "
                "every finite value as observed"
            )
        flags = pd.to_numeric(data[qc_column], errors="coerce")
        accepted = flags.isin(list(observed_qc_values)).to_numpy()
        mask = mask & accepted

    result: pd.Series = pd.Series(mask, index=data.index, name=f"{target}_is_observed")
    return result


def environment() -> dict[str, str]:
    """Return the package, Python and library versions a run depends on.

    Recorded in every manifest. The same seed and the same grid can still produce
    different trees across scikit-learn releases, so a reported metric is only
    reproducible against the versions that produced it.
    """
    import sklearn

    from rfrgapfill import __version__

    return {
        "rfr_gapfill": __version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
    }


def run_manifest(
    config: RFRConfig,
    *,
    targets: Sequence[str],
    qc_columns: Mapping[str, str | None] | None = None,
    time_axis: Mapping[str, Any] | None = None,
    gaps: Mapping[str, Any] | None = None,
    training: Mapping[str, Any] | None = None,
    features: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the JSON-serialisable manifest for one run.

    Everything method_spec.md section 7 asks a run to record, in one dictionary:
    the environment, the whole configuration (which already carries the column
    map, mode, feature mode, grid, seed and QC values), the resolved time axis,
    the artificial-gap manifest, and the per-target training reports.

    ``is_paper_faithful`` from the configuration travels with it, so a report
    that used a labelled enhancement - time-aware folds, the ORF benchmark -
    cannot be mistaken later for a reproduction of the published setup.
    """
    manifest: dict[str, Any] = {
        "paper_doi": _paper_doi(),
        "environment": environment(),
        "targets": list(targets),
        "qc_columns": {} if qc_columns is None else dict(qc_columns),
        "config": config.to_dict(),
    }
    if time_axis is not None:
        manifest["time_axis"] = dict(time_axis)
    if gaps is not None:
        manifest["gaps"] = dict(gaps)
    if features is not None:
        manifest["features"] = dict(features)
    if training is not None:
        manifest["training"] = dict(training)
    if extra:
        manifest.update(extra)
    return manifest


def _paper_doi() -> str:
    """Return the DOI of the method's source publication."""
    from rfrgapfill import PAPER_DOI

    return PAPER_DOI
