"""Run manifests: what produced a result, in one auditable record.

Every other module already describes its own stage - :meth:`RFRConfig.to_dict`
the settings, :meth:`RFRModel.to_dict` the grid and the parameters it chose,
:meth:`GapManifest.to_dict` the placed scenario, :meth:`FillReport.to_dict` the
row accounting. This module is the one place they are assembled, checked against
the list ``docs/method_spec.md`` section 7 requires of a run, and written out as
JSON.

The point is the exit criterion of Step 15: a gap-filled series that arrives
months later, in a file, next to nobody who remembers the run, can still be
traced back to the exact settings that produced it::

    filler = RFRGapFiller(config).fit(data, target="LE", qc_column="LE_QC")
    result = filler.fill(data)
    RunManifest.from_fill(filler, result).save("LE_run.json")

Two things make that more than a dictionary dump.

**The required list is checked, not hoped for.** :data:`REQUIRED_FIELDS` is Step
15's list of audit items mapped onto the exported document, and
:meth:`RunManifest.require_complete` names any that a manifest cannot supply.
:meth:`RunManifest.save` runs it before writing, because the file is the artifact
someone will have to trust later.

**The resolved ambiguities travel with the result.** ``docs/method_spec.md``
requires every ambiguity A1-A12 to be reported in run output, not merely exposed
in configuration, so :func:`ambiguity_choices` writes the choice this run made
for each one into the manifest. A manifest therefore records not just what was
run but which of the paper's open questions were answered which way - none of
them, ever, as "paper exact".

Export
------

:meth:`RunManifest.to_dict` returns plain Python containers only: every value is
normalised at construction time, so a manifest that cannot be serialised fails
when it is built rather than when it is written. Non-finite floats become
``null``, matching :mod:`rfrgapfill.metrics`, where an undefined quantity is
``None`` and never ``NaN``. Objects the normaliser does not recognise are
rendered as text, the same fallback the model layer applies to estimator
parameters.

YAML needs no support here: :meth:`~RunManifest.to_dict` is a plain mapping, so
``yaml.safe_dump(manifest.to_dict())`` works with any YAML library the caller
already has, and the package takes no dependency on one.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, Protocol

import numpy as np
import pandas as pd

from rfrgapfill.config import (
    DEFAULT_HYPERPARAMETER_GRID,
    FeatureMode,
    RFRConfig,
)
from rfrgapfill.features import describe_features
from rfrgapfill.fill import FillReport, FillResult, RFRGapFiller, method_label
from rfrgapfill.gaps import GapManifest
from rfrgapfill.leakage import ValidationFeatureSet
from rfrgapfill.legacy import LEGACY_DAILY_STATISTIC_RULE, LEGACY_RADIATION_RULE
from rfrgapfill.model import FitReport, RFRModel
from rfrgapfill.schema import ColumnMap, ConfigError, FrozenRecord, coerce_enum
from rfrgapfill.time import TimeAxis, iso_duration

__all__ = [
    "MANIFEST_FORMAT",
    "MANIFEST_VERSION",
    "REQUIRED_FIELDS",
    "VALIDATION_REQUIRED_FIELDS",
    "ProvenanceError",
    "RowCounts",
    "RunKind",
    "RunManifest",
    "ambiguity_choices",
    "environment_versions",
    "load_manifest",
]

#: Identifies an exported document as one of this package's run manifests.
MANIFEST_FORMAT: Final = "rfr-gapfill/run-manifest"

#: Layout version of the exported document. Bumped when a field moves or changes
#: meaning, so a reader can tell an old manifest from a new one.
MANIFEST_VERSION: Final = 1


class ProvenanceError(RuntimeError):
    """Raised when a run manifest cannot be assembled, completed, or read back."""


class SupportsToDict(Protocol):
    """Anything that already describes itself for a manifest.

    Every record in this package does - a configuration, a report, a gap manifest,
    a time axis, a metric - so a manifest section accepts the object itself rather
    than making the caller render it first.
    """

    def to_dict(self) -> Mapping[str, Any]:
        """Return a JSON-serialisable representation."""


#: What a manifest section accepts: a mapping, one of this package's records, or
#: nothing. Whatever is given is rendered to plain containers at construction.
Section = Mapping[str, Any] | SupportsToDict | None


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


def environment_versions() -> dict[str, str]:
    """Return the versions a reproduction of this run would have to match.

    The three ``docs/method_spec.md`` section 7 names - the package, Python and
    scikit-learn - plus numpy and pandas, which decide the quantile and standard
    deviation conventions of ambiguity A11.
    """
    import platform

    import sklearn

    from rfrgapfill import __version__

    return {
        "rfr_gapfill": __version__,
        "python": platform.python_version(),
        "scikit_learn": sklearn.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }


# ---------------------------------------------------------------------------
# JSON normalisation
# ---------------------------------------------------------------------------


def _number(value: int | float) -> int | float | None:
    """Return ``value`` as a plain Python number, with non-finite floats as ``None``.

    ``NaN`` and infinities are not JSON, and this package's own convention is that
    an undefined quantity is ``None`` (see :mod:`rfrgapfill.metrics`). Writing
    ``null`` keeps a manifest readable by a strict parser and keeps an undefined
    value from propagating silently through a later mean.
    """
    if isinstance(value, int):
        return int(value)
    number = float(value)
    return number if math.isfinite(number) else None


def _jsonable(value: Any) -> Any:
    """Return ``value`` as plain JSON-serialisable Python containers.

    Recognises this package's own records (anything with ``to_dict``), enums,
    timestamps, durations, numpy scalars and arrays. Anything else is rendered as
    text rather than raising, which is the fallback
    :func:`rfrgapfill.model._serialisable_params` already applies to estimator
    parameters: a manifest that loses fidelity on an exotic value is better than
    a run whose provenance cannot be written at all.
    """
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, Enum):
        return _jsonable(value.value)
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, (int, float)):
        return _number(value)
    if isinstance(value, datetime):  # pd.Timestamp is a datetime subclass
        return value.isoformat()
    if isinstance(value, timedelta):  # pd.Timedelta is a timedelta subclass
        return iso_duration(value)
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _jsonable(to_dict())
    if isinstance(value, (set, frozenset)):
        return sorted(_jsonable(item) for item in value)
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, Sequence):
        return [_jsonable(item) for item in value]
    return str(value)


def _section(value: object, *, field_name: str) -> Mapping[str, Any] | None:
    """Return ``value`` as a frozen, JSON-ready manifest section, or ``None``."""
    if value is None:
        return None
    rendered = _jsonable(value)
    if not isinstance(rendered, Mapping):
        raise ProvenanceError(
            f"{field_name} must be a mapping or an object with to_dict(), got "
            f"{type(value).__name__}"
        )
    return MappingProxyType(dict(rendered))


# ---------------------------------------------------------------------------
# Row accounting
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RowCounts(FrozenRecord):
    """The two counts Step 15 requires, each beside what it was selected from.

    "Training-row count" and "prediction-row count" are only readable next to the
    number of rows that were *offered*: under ambiguity A4's default a gap
    covering a whole calendar day has no daily statistics, so its rows are offered
    to the model and receive nothing. A manifest that recorded only
    ``prediction_rows`` would show a small number with no way to tell a small gap
    from a blocked one.
    """

    #: Rows the model was actually fitted on.
    training_rows: int
    #: Rows offered to the fit, before incomplete ones were dropped.
    training_rows_offered: int
    #: Rows that received a prediction.
    prediction_rows: int
    #: Rows prediction was attempted on.
    prediction_rows_offered: int

    def __post_init__(self) -> None:
        for item in fields(self):
            value = getattr(self, item.name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ProvenanceError(f"{item.name} must be an integer, got {value!r}")
            if value < 0:
                raise ProvenanceError(f"{item.name} cannot be negative, got {value}")
        if self.training_rows > self.training_rows_offered:
            raise ProvenanceError(
                f"fitted {self.training_rows} row(s) out of {self.training_rows_offered} "
                "offered; dropping incomplete rows cannot increase the count"
            )
        if self.prediction_rows > self.prediction_rows_offered:
            raise ProvenanceError(
                f"predicted {self.prediction_rows} row(s) out of "
                f"{self.prediction_rows_offered} offered; a row that was not offered "
                "cannot have received a prediction"
            )

    @property
    def training_rows_dropped(self) -> int:
        """Offered rows the fit could not use."""
        return self.training_rows_offered - self.training_rows

    @property
    def prediction_rows_dropped(self) -> int:
        """Offered rows that received no prediction, for want of a predictor."""
        return self.prediction_rows_offered - self.prediction_rows

    @classmethod
    def from_fill(cls, fit: FitReport, fill: FillReport) -> RowCounts:
        """Return the counts of an operational run from its two reports."""
        return cls(
            training_rows=int(fit.fitted_rows),
            training_rows_offered=int(fit.rows),
            # Candidates, not the whole frame: an observed row was never a
            # prediction the model declined to make.
            prediction_rows=int(fill.filled_rows),
            prediction_rows_offered=int(fill.candidate_rows),
        )

    @classmethod
    def from_validation(cls, fit: FitReport, features: ValidationFeatureSet) -> RowCounts:
        """Return the counts of an artificial-gap run from the fit and the feature set.

        The rows a validation run can predict are the withheld rows whose features
        are complete, which is exactly what
        :attr:`ValidationFeatureSet.complete_mask` marks - so a run blocked by
        ambiguity A4 records ``prediction_rows`` far below
        ``prediction_rows_offered`` rather than silently scoring a handful of rows.
        """
        if not isinstance(features, ValidationFeatureSet):
            raise ProvenanceError(
                f"features must be a ValidationFeatureSet, got {type(features).__name__}"
            )
        holdout = features.holdout_mask
        return cls(
            training_rows=int(fit.fitted_rows),
            training_rows_offered=int(fit.rows),
            prediction_rows=int((holdout & features.complete_mask).sum()),
            prediction_rows_offered=int(holdout.sum()),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary for the run manifest."""
        return {
            "training_rows": self.training_rows,
            "training_rows_offered": self.training_rows_offered,
            "training_rows_dropped": self.training_rows_dropped,
            "prediction_rows": self.prediction_rows,
            "prediction_rows_offered": self.prediction_rows_offered,
            "prediction_rows_dropped": self.prediction_rows_dropped,
        }


# ---------------------------------------------------------------------------
# Resolved ambiguities (method_spec.md, Known ambiguities)
# ---------------------------------------------------------------------------

#: Why the ambiguity block exists, carried in every manifest that has one.
AMBIGUITY_NOTE: Final = (
    "Points Zhu et al. (2022) leaves open. Each choice below is this package's "
    "documented default or the caller's override; none of them reproduces the "
    "paper exactly. See docs/method_spec.md, Known ambiguities."
)


def ambiguity_choices(config: RFRConfig) -> dict[str, Any]:
    """Return the choice ``config`` makes for each documented ambiguity A1-A12.

    ``docs/method_spec.md`` requires every ambiguity to be *reported in run
    output*, not merely exposed in configuration, so this belongs in the manifest
    rather than in a docstring. A8 and A10 are properties of optional reproduction
    modules rather than of a run's configuration, and are recorded as such.
    """
    if not isinstance(config, RFRConfig):
        raise ConfigError(f"config must be an RFRConfig, got {type(config).__name__}")
    features = config.features
    scenario = config.validation.gaps
    # In legacy_fluxlib mode fluxlib's own rules replace A2 and A4, so reporting the
    # (unused) paper_safe settings there would describe a run that did not happen.
    legacy = features.mode is FeatureMode.LEGACY_FLUXLIB
    grid = {key: list(values) for key, values in config.grid.items()}
    default_grid = {key: list(values) for key, values in DEFAULT_HYPERPARAMETER_GRID.items()}
    preset = config.preset

    choices: dict[str, dict[str, Any]] = {
        "A1": {
            "topic": "GridSearchCV hyperparameter grid not enumerated in the article",
            "settings": {
                "hyperparameter_grid": grid,
                "hyperparameter_preset": None if preset is None else preset.value,
                "is_package_default": grid == default_grid,
            },
        },
        "A2": {
            "topic": "radiation-category boundaries at exactly 10 and 100 W m-2",
            "settings": (
                {
                    "feature_mode": features.mode.value,
                    "radiation_thresholds": list(features.radiation_thresholds),
                    "rule": LEGACY_RADIATION_RULE,
                }
                if legacy
                else {
                    "radiation_thresholds": list(features.radiation_thresholds),
                    "boundary_convention": features.convention.value,
                }
            ),
        },
        "A3": {
            "topic": "the 20/30/50 mix as gap events or as withheld half-hours",
            "settings": {"allocation_basis": scenario.basis.value},
        },
        "A4": {
            "topic": "daily target statistics for a day with too few visible observations",
            "settings": (
                {"feature_mode": features.mode.value, "rule": LEGACY_DAILY_STATISTIC_RULE}
                if legacy
                else {
                    "daily_statistic_strategy": features.statistic_strategy.value,
                    "min_daily_observations": features.min_daily_observations,
                    "fallback_window_days": features.fallback_window,
                }
            ),
        },
        "A5": {
            "topic": "cross-validation details inside GridSearchCV",
            "settings": {
                "cv_strategy": config.cv.value,
                "cv_folds": config.cv_folds,
                "cv_shuffle": config.cv_shuffle,
                "is_paper_default": config.cv.is_paper_default and not config.cv_shuffle,
            },
        },
        "A6": {
            "topic": "whether the legacy fluxlib daily statistics were leakage safe",
            "settings": {
                "feature_mode": features.mode.value,
                "is_leakage_safe": not (legacy and features.use_receptive_limiter),
                "evidence": (
                    "fluxlib 0.0.23 computes the daily statistics before the artificial "
                    "gaps are applied, so held-out values reach them; see "
                    "docs/fluxlib_audit.md"
                ),
            },
        },
        "A7": {
            "topic": "the exact 25% withheld fraction is not always achievable",
            "settings": {
                "missing_fraction": scenario.missing_fraction,
                "fraction_tolerance": scenario.fraction_tolerance,
                "mix_tolerance": scenario.mix_tolerance,
            },
        },
        "A8": {
            "topic": "denominator of the supplement's normalized joint-uncertainty ratio",
            "settings": {
                "bias_iqr_by_gap_class": config.validation.bias_iqr_by_gap_class,
                "normalized_ratio": "experimental; denominator not reconstructed",
            },
        },
        "A9": {
            "topic": "hemisphere inference at or near the equator",
            "settings": {
                "hemisphere": (
                    config.resolve_hemisphere().value if config.hemisphere_source else None
                ),
                "hemisphere_source": config.hemisphere_source,
                "latitude": config.latitude,
                "rule": "latitude >= 0 -> north; an explicit hemisphere always wins",
            },
        },
        "A10": {
            "topic": "Table S3 NEE units (g C m-2 d-1) differ from the model's units",
            "settings": {
                "reported_in": "model units; benchmark conversion is explicit",
                "conversion": "rfrgapfill.benchmarks.convert_nee_to_carbon_units",
                "rule": "a rate conversion, not an aggregation of half hours into days",
            },
        },
        "A11": {
            "topic": "degrees of freedom of the daily standard deviation, quantile method",
            "settings": {
                "daily_std_ddof": features.daily_std_ddof,
                "quantile_interpolation": "linear",
            },
        },
        "A12": {
            "topic": "which coefficient of determination R2 names",
            "settings": {"r2_definition": config.validation.r2.value},
        },
    }
    return {"note": AMBIGUITY_NOTE, "paper_exact": False, "choices": choices}


# ---------------------------------------------------------------------------
# The manifest
# ---------------------------------------------------------------------------


class RunKind(str, Enum):
    """What kind of run a manifest describes."""

    #: An operational fill of a series' real gaps (``docs/method_spec.md`` 7).
    FILL = "fill"
    #: An artificial-gap validation run (``docs/method_spec.md`` 4).
    VALIDATION = "validation"

    @classmethod
    def coerce(cls, value: object) -> RunKind:
        """Return ``value`` as a :class:`RunKind`."""
        return coerce_enum(cls, value, field_name="kind")


#: The audit items ``docs/rfr_gapfill_ai_steps_v2.md`` Step 15 requires of every
#: manifest, mapped onto the path each occupies in the exported document.
REQUIRED_FIELDS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "versions.rfr_gapfill": "package version",
        "versions.python": "Python version",
        "versions.scikit_learn": "scikit-learn version",
        "run.target": "target",
        "run.mode": "RFR mode",
        "run.feature_mode": "feature mode",
        "column_map.variables": "column map",
        "run.time_resolution": "time resolution",
        "run.random_state": "random seed",
        "model.hyperparameter_grid": "hyperparameter grid",
        "model.best_params": "best parameters",
        "gaps.config": "gap configuration",
        "rows.training_rows": "training-row count",
        "rows.prediction_rows": "prediction-row count",
        "qc.observed_qc_values": "QC rules",
    }
)

#: Required in addition for an artificial-gap run, which has placed intervals to
#: account for. An operational fill has a gap *configuration* but no gap manifest.
VALIDATION_REQUIRED_FIELDS: Final[Mapping[str, str]] = MappingProxyType(
    {"gaps.manifest": "gap manifest"}
)

#: Required only when the run builds the season feature; the ORF benchmark does
#: not, and ``RFRConfig`` accepts no hemisphere for it (method_spec.md 3.3, A9).
_HEMISPHERE_FIELD: Final[Mapping[str, str]] = MappingProxyType(
    {"run.hemisphere": "hemisphere/latitude"}
)

#: Distinguishes "absent from the document" from a legitimately ``None`` value.
_ABSENT: Final = object()

#: Manifest fields that accept an object and are rendered at construction, read
#: back through :meth:`RunManifest.section`.
_SECTION_FIELDS: Final[tuple[str, ...]] = (
    "features",
    "model",
    "gap_manifest",
    "time_axis",
    "fill",
    "metrics",
    "validation",
)


def _lookup(document: Mapping[str, Any], path: str) -> Any:
    """Return the value at a dotted ``path`` in ``document``, or :data:`_ABSENT`."""
    current: Any = document
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return _ABSENT
        current = current[part]
    return current


def _is_missing(value: Any) -> bool:
    """Whether ``value`` fails to record anything.

    ``0`` and ``False`` are recorded values - a seed of 0 and a run that filled no
    row are both real answers - so only absence, ``None`` and empty containers
    count as missing.
    """
    if value is _ABSENT or value is None:
        return True
    if isinstance(value, (str, Mapping, Sequence, set, frozenset)):
        return len(value) == 0
    return False


@dataclass(frozen=True)
class RunManifest(FrozenRecord):
    """Everything that produced one result, in one JSON-serialisable record.

    Build one with :meth:`from_fill` or :meth:`from_validation` rather than by
    hand: both assemble the sections from the objects that already own them, so
    the manifest cannot describe a configuration the run did not use. Direct
    construction is supported for tests and for stages this package does not own,
    and :meth:`require_complete` is what tells such a manifest apart from a
    finished one.

    Sections handed in as objects (a :class:`~rfrgapfill.schema.ColumnMap`, a
    :class:`~rfrgapfill.time.TimeAxis`, a :class:`~rfrgapfill.gaps.GapManifest`,
    a report) are rendered to plain containers **at construction**, so a manifest
    that cannot be exported fails here rather than at the moment someone tries to
    write the audit file.
    """

    #: Whether this describes an operational fill or an artificial-gap run.
    kind: RunKind | str
    #: The target flux the run was for. Features and model are target-specific.
    target: str
    #: The settings the run used, live rather than rendered.
    config: RFRConfig
    #: Training and prediction row counts (Step 15).
    rows: RowCounts
    #: ISO-8601 UTC timestamp. Defaults to the moment the manifest is built.
    created_at: str | None = None
    #: Package, Python and scikit-learn versions. Defaults to this environment.
    versions: Mapping[str, str] | None = None
    #: Canonical-name to input-column mapping. Defaults to the configured one.
    column_map: ColumnMap | Mapping[str, str] | None = None
    #: The QC/provenance column separating measured from pre-filled values.
    qc_column: str | None = None
    #: The cadence the run ran at, ISO-8601. Defaults from ``time_axis``, then
    #: from ``config.frequency``.
    time_resolution: str | None = None
    #: Feature-stage description. Defaults to :func:`describe_features`.
    features: Section = None
    #: :meth:`RFRModel.to_dict` - grid, best parameters, seed, row accounting.
    model: Section = None
    #: :meth:`GapManifest.to_dict` - the placed artificial intervals.
    gap_manifest: Section = None
    #: :meth:`TimeAxis.to_dict` - cadence, coverage and span of the run's index.
    time_axis: Section = None
    #: :meth:`FillReport.to_dict` - what happened to the rows of an operational fill.
    fill: Section = None
    #: Scored metrics, in whatever shape the validation layer reports them.
    metrics: Section = None
    #: Leakage-safe validation row accounting, from :class:`ValidationFeatureSet`.
    validation: Section = None
    #: Anything the caller wants carried alongside: site metadata, a run label.
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", RunKind.coerce(self.kind))
        if not isinstance(self.target, str) or not self.target.strip():
            raise ProvenanceError(f"target must be a non-empty string, got {self.target!r}")
        if not isinstance(self.config, RFRConfig):
            raise ConfigError(f"config must be an RFRConfig, got {type(self.config).__name__}")
        if not isinstance(self.rows, RowCounts):
            raise ProvenanceError(f"rows must be a RowCounts, got {type(self.rows).__name__}")

        if self.created_at is None:
            stamp = datetime.now(tz=timezone.utc).isoformat(timespec="seconds")
            object.__setattr__(self, "created_at", stamp)
        elif not isinstance(self.created_at, str):
            raise ProvenanceError(f"created_at must be an ISO-8601 string, got {self.created_at!r}")

        versions = environment_versions() if self.versions is None else dict(self.versions)
        object.__setattr__(
            self, "versions", MappingProxyType({str(k): str(v) for k, v in versions.items()})
        )

        columns = ColumnMap.coerce(self.column_map)
        object.__setattr__(
            self, "column_map", self.config.columns if len(columns) == 0 else columns
        )

        if self.qc_column is not None and not isinstance(self.qc_column, str):
            raise ProvenanceError(f"qc_column must be a string or None, got {self.qc_column!r}")

        for name in _SECTION_FIELDS:
            object.__setattr__(self, name, _section(getattr(self, name), field_name=name))
        object.__setattr__(self, "extra", _section(self.extra, field_name="extra") or {})

        if self.features is None:
            described = describe_features(self.config, target=self.target)
            object.__setattr__(self, "features", _section(described, field_name="features"))

        if self.time_resolution is None:
            object.__setattr__(self, "time_resolution", self._infer_time_resolution())
        elif not isinstance(self.time_resolution, str):
            raise ProvenanceError(
                f"time_resolution must be an ISO-8601 duration string or None, "
                f"got {self.time_resolution!r}"
            )

    def _infer_time_resolution(self) -> str | None:
        """Return the run's cadence from the time axis, the gaps, or the configuration."""
        for name in ("time_axis", "gap_manifest"):
            section = self.section(name)
            if section is not None:
                step = section.get("time_step")
                if isinstance(step, str) and step:
                    return step
        step = self.config.time_step
        return None if step is None else iso_duration(step)

    # -- accessors -----------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover - diagnostic only
        return (
            f"{type(self).__name__}(kind={self.run_kind.value!r}, target={self.target!r}, "
            f"method={self.method!r}, digest={self.settings_digest[:12]})"
        )

    def section(self, name: str) -> dict[str, Any] | None:
        """Return the normalised manifest section ``name`` as a plain dictionary.

        The section fields accept objects and are rendered in ``__post_init__``,
        so this is where that narrowing is stated once instead of at every use.
        """
        if name not in _SECTION_FIELDS:
            raise ProvenanceError(
                f"{name!r} is not a manifest section; expected one of {', '.join(_SECTION_FIELDS)}"
            )
        section = getattr(self, name)
        assert section is None or isinstance(section, Mapping)
        return None if section is None else dict(section)

    @property
    def run_kind(self) -> RunKind:
        """The validated :class:`RunKind` (narrowed from the input union)."""
        assert isinstance(self.kind, RunKind)
        return self.kind

    @property
    def columns(self) -> ColumnMap:
        """The validated :class:`~rfrgapfill.schema.ColumnMap`."""
        assert isinstance(self.column_map, ColumnMap)
        return self.column_map

    @property
    def method(self) -> str:
        """The arm this run was: ``RFR3``, ``RFR10``, ``ORF3`` or ``ORF10``."""
        return method_label(self.config)

    @property
    def environment_drift(self) -> dict[str, list[str]]:
        """Packages whose version differs between the fit and this environment.

        Empty for a run whose model was fitted in the process that wrote the
        manifest. Non-empty means the model was loaded from disk under a different
        environment, which is worth seeing next to the numbers it produced.
        """
        fitted = (self.section("model") or {}).get("versions")
        if not isinstance(fitted, Mapping):
            return {}
        current = self.versions or {}
        return {
            str(package): [str(was), str(current[package])]
            for package, was in sorted(fitted.items())
            if package in current and str(was) != str(current[package])
        }

    @property
    def settings_digest(self) -> str:
        """A stable SHA-256 digest of the settings this run used.

        Covers the environment versions, the target and the whole validated
        configuration - and nothing that varies between two runs of the same
        design, so the timestamp, the row counts, the placed intervals and the
        scores are all excluded. Two manifests sharing a digest were configured
        identically; two that differ were not, and comparing their ``config``
        sections says where.
        """
        payload = {
            "versions": dict(self.versions or {}),
            "target": self.target,
            "config": self.config.to_dict(),
        }
        text = json.dumps(
            _jsonable(payload), sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    # -- completeness --------------------------------------------------------

    def required_fields(self) -> dict[str, str]:
        """Return the audit items this manifest must supply, as ``path -> item``.

        Step 15's list, narrowed to what applies: a gap manifest only for an
        artificial-gap run, a hemisphere only when the season feature is built
        (the ORF benchmark builds none, and its configuration accepts no
        hemisphere).
        """
        required = dict(REQUIRED_FIELDS)
        if self.run_kind is RunKind.VALIDATION:
            required.update(VALIDATION_REQUIRED_FIELDS)
        if self.config.features.requires_hemisphere:
            required.update(_HEMISPHERE_FIELD)
        return required

    def missing_fields(self) -> dict[str, str]:
        """Return the required audit items this manifest cannot supply."""
        document = self.to_dict()
        return {
            path: item
            for path, item in self.required_fields().items()
            if _is_missing(_lookup(document, path))
        }

    @property
    def is_complete(self) -> bool:
        """Whether every required audit item is recorded."""
        return not self.missing_fields()

    def require_complete(self) -> RunManifest:
        """Return ``self``, or raise :class:`ProvenanceError` naming what is missing.

        The exit criterion of Step 15 made checkable: a result whose manifest
        passes this can be traced back to the settings that produced it, and one
        whose manifest does not is told exactly which item is absent.
        """
        missing = self.missing_fields()
        if missing:
            detail = ", ".join(f"{item} ({path})" for path, item in sorted(missing.items()))
            raise ProvenanceError(
                f"run manifest for {self.target!r} is missing required provenance: {detail}. "
                "Build it with RunManifest.from_fill() or from_validation(), which assemble "
                "every section from the objects that own it."
            )
        return self

    # -- export --------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return the exported document: plain containers, JSON-ready throughout."""
        from rfrgapfill import PAPER_DOI

        config = self.config
        return {
            "manifest_format": MANIFEST_FORMAT,
            "manifest_version": MANIFEST_VERSION,
            "kind": self.run_kind.value,
            "created_at": self.created_at,
            "settings_digest": self.settings_digest,
            "reference": {
                "method": "Zhu et al. (2022), Agric. For. Meteorol. 314, 108777",
                "doi": PAPER_DOI,
                "specification": "docs/method_spec.md",
            },
            "versions": dict(self.versions or {}),
            "environment_drift": self.environment_drift,
            "run": {
                "site_id": config.site_id,
                "target": self.target,
                "method": self.method,
                "mode": config.rfr_mode.value,
                "is_orf": config.is_orf,
                "use_receptive_limiter": config.features.use_receptive_limiter,
                "feature_mode": config.features.mode.value,
                "random_state": config.random_state,
                "time_resolution": self.time_resolution,
                "hemisphere": (
                    config.resolve_hemisphere().value if config.hemisphere_source else None
                ),
                "hemisphere_source": config.hemisphere_source,
                "latitude": config.latitude,
                "is_paper_faithful": config.is_paper_faithful,
            },
            "qc": {
                "qc_column": self.qc_column,
                "observed_qc_values": list(config.observed_qc_values),
                "rule": (
                    "a target value counts as observed when it is present, finite and its QC "
                    "flag is one of observed_qc_values; without a qc_column every present "
                    "value counts as measured"
                ),
            },
            "column_map": self.columns.to_dict(),
            "rows": self.rows.to_dict(),
            "time_axis": self.section("time_axis"),
            "features": self.section("features"),
            "model": self.section("model"),
            "gaps": {
                "config": config.validation.gaps.to_dict(),
                "manifest": self.section("gap_manifest"),
            },
            "validation": self.section("validation"),
            "fill": self.section("fill"),
            "metrics": self.section("metrics"),
            "ambiguities": ambiguity_choices(config),
            "config": config.to_dict(),
            "extra": dict(self.extra),
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        """Return the manifest as a JSON document.

        Does not check completeness: an unfinished manifest is worth printing to
        see what it is short of. :meth:`save` does check, because a file outlives
        the session that could have explained it.
        """
        return json.dumps(self.to_dict(), indent=indent, allow_nan=False, default=str)

    def save(self, path: str | Path, *, indent: int | None = 2, check: bool = True) -> Path:
        """Write the manifest to ``path`` as JSON and return the path written.

        :param check: run :meth:`require_complete` first. On by default: the file
            is the artifact someone reads months later, and an audit record that
            silently omits the seed or the grid is worse than no file at all. Pass
            ``False`` deliberately, to archive a partial run.
        """
        if check:
            self.require_complete()
        destination = Path(path)
        destination.write_text(self.to_json(indent=indent) + "\n", encoding="utf-8")
        return destination

    # -- constructors --------------------------------------------------------

    @classmethod
    def from_fill(
        cls,
        filler: RFRGapFiller,
        result: FillResult,
        *,
        metrics: Section = None,
        extra: Mapping[str, Any] | None = None,
        created_at: str | None = None,
    ) -> RunManifest:
        """Return the manifest of an operational fill (``docs/method_spec.md`` 7).

        Every section is taken from the objects that already own it - the fitted
        filler's column mapping, QC column and model, and the result's time axis
        and row accounting - so the manifest describes the run that happened
        rather than the configuration it was asked for.

        :param filler: the fitted :class:`~rfrgapfill.fill.RFRGapFiller`.
        :param result: what its :meth:`~rfrgapfill.fill.RFRGapFiller.fill`
            returned. Must be that filler's own result: the target is checked.
        """
        if not isinstance(filler, RFRGapFiller):
            raise ProvenanceError(f"filler must be an RFRGapFiller, got {type(filler).__name__}")
        if not filler.is_fitted:
            raise ProvenanceError(
                "cannot describe an unfitted filler; call fit(data, target=...) first"
            )
        if not isinstance(result, FillResult):
            raise ProvenanceError(f"result must be a FillResult, got {type(result).__name__}")
        if result.target != filler.target:
            raise ProvenanceError(
                f"result fills {result.target!r} but the filler was fitted for "
                f"{filler.target!r}; a manifest may not pair a result with another run"
            )
        return cls(
            kind=RunKind.FILL,
            target=filler.target,
            config=filler.config,
            rows=RowCounts.from_fill(filler.model.fit_report, result.report),
            created_at=created_at,
            column_map=filler.column_map,
            qc_column=filler.qc_column,
            model=filler.model.to_dict(),
            time_axis=result.time_axis,
            fill=result.report.to_dict(),
            metrics=metrics,
            extra=extra or {},
        )

    @classmethod
    def from_validation(
        cls,
        *,
        config: RFRConfig,
        model: RFRModel,
        gaps: GapManifest,
        features: ValidationFeatureSet | None = None,
        target: str | None = None,
        column_map: ColumnMap | Mapping[str, str] | None = None,
        qc_column: str | None = None,
        time_axis: TimeAxis | Mapping[str, Any] | None = None,
        rows: RowCounts | None = None,
        metrics: Section = None,
        extra: Mapping[str, Any] | None = None,
        created_at: str | None = None,
    ) -> RunManifest:
        """Return the manifest of an artificial-gap validation run.

        :param model: the fitted model the withheld intervals were predicted with.
        :param gaps: the :class:`~rfrgapfill.gaps.GapManifest` the scenario placed,
            which is what makes the run repeatable: it carries the seed, the basis
            of ambiguity A3, and every interval.
        :param features: the leakage-safe
            :class:`~rfrgapfill.leakage.ValidationFeatureSet` the run was built
            from. Supplies the row accounting when ``rows`` is not given, and its
            own account of what the holdout cost.
        :param target: defaults to the model's target; a mismatch is rejected
            rather than silently preferred either way.
        """
        if not isinstance(config, RFRConfig):
            raise ConfigError(f"config must be an RFRConfig, got {type(config).__name__}")
        if not isinstance(model, RFRModel):
            raise ProvenanceError(f"model must be an RFRModel, got {type(model).__name__}")
        if not model.is_fitted:
            raise ProvenanceError("cannot describe an unfitted model; call fit(X, y) first")
        if not isinstance(gaps, GapManifest):
            raise ProvenanceError(f"gaps must be a GapManifest, got {type(gaps).__name__}")

        resolved = _resolve_validation_target(target, model=model, features=features)
        if features is not None and rows is None:
            rows = RowCounts.from_validation(model.fit_report, features)
        if rows is None:
            raise ProvenanceError(
                "row counts are required: pass rows=RowCounts(...) or the "
                "ValidationFeatureSet the run was built from"
            )

        section: Mapping[str, Any] | None = None
        if features is not None:
            # The feature description already appears under `features`; keep only
            # the row accounting here rather than writing it into the document twice.
            section = {key: value for key, value in features.to_dict().items() if key != "features"}
        return cls(
            kind=RunKind.VALIDATION,
            target=resolved,
            config=config,
            rows=rows,
            created_at=created_at,
            column_map=column_map,
            qc_column=qc_column,
            model=model.to_dict(),
            gap_manifest=gaps,
            time_axis=time_axis,
            validation=section,
            metrics=metrics,
            extra=extra or {},
        )


def _resolve_validation_target(
    target: str | None,
    *,
    model: RFRModel,
    features: ValidationFeatureSet | None,
) -> str:
    """Return the target a validation manifest is for, rejecting a disagreement."""
    candidates = {
        "model": model.target,
        "features": None if features is None else features.target,
        "target": target,
    }
    named = {source: value for source, value in candidates.items() if value is not None}
    if not named:
        raise ProvenanceError("no target available: pass target=..., or a model fitted for one")
    distinct = sorted(set(named.values()))
    if len(distinct) > 1:
        detail = ", ".join(f"{source}={value!r}" for source, value in sorted(named.items()))
        raise ProvenanceError(
            f"the run's target is ambiguous ({detail}); a manifest may not describe two "
            "targets, and the model, the features and the target must agree"
        )
    return distinct[0]


# ---------------------------------------------------------------------------
# Reading a manifest back
# ---------------------------------------------------------------------------


def load_manifest(path: str | Path) -> dict[str, Any]:
    """Return the manifest document at ``path``.

    Data, not a :class:`RunManifest`: an archived manifest is a record of a run
    that already happened, and rebuilding a live configuration from it would
    invite treating a hand-edited file as a validated one. Read the sections, or
    build a fresh :class:`~rfrgapfill.config.RFRConfig` from what they say.
    """
    source = Path(path)
    try:
        document = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ProvenanceError(f"cannot read a run manifest from {source}: {error}") from error
    if not isinstance(document, Mapping) or document.get("manifest_format") != MANIFEST_FORMAT:
        raise ProvenanceError(
            f"{source} is not an rfr-gapfill run manifest "
            f"(expected manifest_format={MANIFEST_FORMAT!r})"
        )
    return dict(document)
