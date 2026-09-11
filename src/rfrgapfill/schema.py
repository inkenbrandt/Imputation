"""Canonical variable names and the user column-mapping layer.

Defines the canonical driver names for RFR3 and RFR10 (``docs/method_spec.md``
section 2) and the :class:`ColumnMap` that resolves them against arbitrary station
column names. FLUXNET2015 names are a default mapping, never a hard-coded
requirement: nothing outside :data:`FLUXNET2015_COLUMNS` may mention a station
column name.

This is the lowest layer of the package. It owns the vocabulary (:class:`Mode`,
:class:`Hemisphere`, :class:`GapClass`, the canonical variables) and the error type
used by every validated configuration object, so :mod:`rfrgapfill.config` and the
science modules can import from here without a cycle.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field, fields
from enum import Enum
from types import MappingProxyType
from typing import Any, Final, TypeVar, cast

__all__ = [
    "AIR_TEMPERATURE",
    "CANONICAL_VARIABLES",
    "EBR_VARIABLES",
    "FLUXNET2015_COLUMNS",
    "NET_RADIATION",
    "RELATIVE_HUMIDITY",
    "RFR3_DRIVERS",
    "RFR10_ADDITIONAL_DRIVERS",
    "RFR10_DRIVERS",
    "SHORTWAVE",
    "SOIL_HEAT_FLUX",
    "SOIL_TEMPERATURE",
    "SOIL_WATER_CONTENT",
    "VARIABLE_UNITS",
    "VPD",
    "WIND_DIRECTION",
    "WIND_SPEED",
    "ColumnMap",
    "ColumnMapError",
    "ConfigError",
    "FrozenRecord",
    "GapClass",
    "Hemisphere",
    "Mode",
    "coerce_enum",
]


class ConfigError(ValueError):
    """Raised when a configuration object is invalid or internally inconsistent.

    Subclasses :class:`ValueError`, so ordinary ``except ValueError`` handling keeps
    working while callers that care can catch the package-specific type.
    """


class ColumnMapError(ConfigError):
    """Raised when required canonical variables are not mapped to input columns."""


# ---------------------------------------------------------------------------
# Serialisation support for frozen configuration objects
# ---------------------------------------------------------------------------


def _unfreeze(value: object) -> object:
    """Return a picklable copy of ``value``, unwrapping a read-only mapping."""
    return dict(value) if isinstance(value, MappingProxyType) else value


class FrozenRecord:
    """Pickle and copy support for this package's frozen dataclasses.

    The configuration objects and the fitted model's reports all normalise their
    mapping fields into a :class:`types.MappingProxyType`, so a validated object
    cannot be edited in place behind the validation that produced it. ``pickle``
    cannot serialise a mapping proxy, which would leave a ``joblib``-saved model
    unable to carry the configuration it was fitted under - and
    ``docs/method_spec.md`` section 5 requires exactly that.

    The pickled state is therefore the dataclass fields with the proxies copied
    back to plain dictionaries, and restoring re-runs ``__post_init__`` so a
    reloaded object is **revalidated** rather than trusted. A model file edited to
    carry an impossible configuration fails on load rather than predicting under
    settings the constructor would have rejected.
    """

    __slots__ = ()

    def __getstate__(self) -> dict[str, Any]:
        """Return the dataclass fields, with read-only mappings copied to dicts."""
        return {item.name: _unfreeze(getattr(self, item.name)) for item in fields(cast(Any, self))}

    def __setstate__(self, state: Mapping[str, Any]) -> None:
        """Restore ``state`` and re-run the dataclass validation over it."""
        for name, value in state.items():
            object.__setattr__(self, name, value)
        post_init = getattr(self, "__post_init__", None)
        if post_init is not None:
            post_init()


# ---------------------------------------------------------------------------
# Canonical variable vocabulary (method_spec.md section 2)
# ---------------------------------------------------------------------------

#: Downward shortwave radiation. Drives the radiation category and day/night split.
SHORTWAVE: Final = "shortwave"
#: Vapour pressure deficit.
VPD: Final = "vpd"
#: Air temperature.
AIR_TEMPERATURE: Final = "air_temperature"
#: Net radiation. Also required by the energy-balance ratio.
NET_RADIATION: Final = "net_radiation"
#: Horizontal wind speed.
WIND_SPEED: Final = "wind_speed"
#: Wind direction.
WIND_DIRECTION: Final = "wind_direction"
#: Soil heat flux. Also required by the energy-balance ratio.
SOIL_HEAT_FLUX: Final = "soil_heat_flux"
#: Soil temperature.
SOIL_TEMPERATURE: Final = "soil_temperature"
#: Relative humidity.
RELATIVE_HUMIDITY: Final = "relative_humidity"
#: Volumetric soil water content.
SOIL_WATER_CONTENT: Final = "soil_water_content"

#: The three MDS-equivalent drivers used by RFR3, in specification order.
RFR3_DRIVERS: Final[tuple[str, ...]] = (SHORTWAVE, VPD, AIR_TEMPERATURE)

#: The seven drivers RFR10 adds to :data:`RFR3_DRIVERS`, in specification order.
RFR10_ADDITIONAL_DRIVERS: Final[tuple[str, ...]] = (
    NET_RADIATION,
    WIND_SPEED,
    WIND_DIRECTION,
    SOIL_HEAT_FLUX,
    SOIL_TEMPERATURE,
    RELATIVE_HUMIDITY,
    SOIL_WATER_CONTENT,
)

#: All ten RFR10 drivers, numbered 1-10 exactly as in ``docs/method_spec.md``.
RFR10_DRIVERS: Final[tuple[str, ...]] = RFR3_DRIVERS + RFR10_ADDITIONAL_DRIVERS

#: Every canonical variable the package knows, in specification order.
CANONICAL_VARIABLES: Final[tuple[str, ...]] = RFR10_DRIVERS

#: Variables the energy-balance ratio needs regardless of the selected mode
#: (``EBR = sum(H + LE) / sum(NETRAD - G)``, method_spec.md section 6.4).
EBR_VARIABLES: Final[tuple[str, ...]] = (NET_RADIATION, SOIL_HEAT_FLUX)

#: Units each canonical variable is expected in. Recorded for provenance and for
#: error messages; the package does not convert units.
VARIABLE_UNITS: Final[Mapping[str, str]] = MappingProxyType(
    {
        SHORTWAVE: "W m-2",
        VPD: "hPa",
        AIR_TEMPERATURE: "degC",
        NET_RADIATION: "W m-2",
        WIND_SPEED: "m s-1",
        WIND_DIRECTION: "degrees",
        SOIL_HEAT_FLUX: "W m-2",
        SOIL_TEMPERATURE: "degC",
        RELATIVE_HUMIDITY: "%",
        SOIL_WATER_CONTENT: "%",
    }
)

#: Reference FLUXNET2015 column names: a convenience default for
#: :meth:`ColumnMap.fluxnet2015`, never an assumption made anywhere else.
FLUXNET2015_COLUMNS: Final[Mapping[str, str]] = MappingProxyType(
    {
        SHORTWAVE: "SW_IN_F",
        VPD: "VPD_F_MDS",
        AIR_TEMPERATURE: "TA_F_MDS",
        NET_RADIATION: "NETRAD",
        WIND_SPEED: "WS",
        WIND_DIRECTION: "WD",
        SOIL_HEAT_FLUX: "G_F_MDS",
        SOIL_TEMPERATURE: "TS_F_MDS",
        RELATIVE_HUMIDITY: "RH",
        SOIL_WATER_CONTENT: "SWC_F_MDS",
    }
)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

_E = TypeVar("_E", bound=Enum)


def coerce_enum(
    enum_cls: type[_E],
    value: object,
    *,
    field_name: str,
    aliases: Mapping[str, _E] | None = None,
) -> _E:
    """Return ``value`` as a member of ``enum_cls`` or raise :class:`ConfigError`.

    Strings match case-insensitively with ``-`` and spaces normalised to ``_``, so
    ``"rfr10"``, ``"RFR10"`` and ``Mode.RFR10`` are equivalent. An unknown value
    raises immediately, listing every accepted spelling, rather than failing later
    inside a transformer.
    """
    members = list(enum_cls.__members__.values())
    if isinstance(value, enum_cls):
        return value
    if isinstance(value, str):
        key = value.strip().lower().replace("-", "_").replace(" ", "_")
        if aliases is not None and key in aliases:
            return aliases[key]
        for member in members:
            if str(member.value).lower() == key:
                return member
    accepted = [str(member.value) for member in members]
    if aliases is not None:
        accepted += [alias for alias in aliases if alias not in accepted]
    raise ConfigError(
        f"{field_name}={value!r} is not a recognised {enum_cls.__name__}; "
        f"expected one of: {', '.join(sorted(accepted))}"
    )


class Mode(str, Enum):
    """Named RFR driver configuration: ``RFR3`` or ``RFR10``."""

    RFR3 = "RFR3"
    RFR10 = "RFR10"

    @property
    def drivers(self) -> tuple[str, ...]:
        """Canonical driver variables this mode requires, in specification order."""
        return RFR3_DRIVERS if self is Mode.RFR3 else RFR10_DRIVERS

    @classmethod
    def coerce(cls, value: object) -> Mode:
        """Return ``value`` as a :class:`Mode`, rejecting unknown modes."""
        return coerce_enum(cls, value, field_name="mode")


class Hemisphere(str, Enum):
    """Hemisphere used for the season feature (method_spec.md section 3.3)."""

    NORTH = "north"
    SOUTH = "south"

    @classmethod
    def coerce(cls, value: object) -> Hemisphere:
        """Return ``value`` as a :class:`Hemisphere`, accepting common spellings."""
        aliases = {
            "n": cls.NORTH,
            "northern": cls.NORTH,
            "s": cls.SOUTH,
            "southern": cls.SOUTH,
        }
        return coerce_enum(cls, value, field_name="hemisphere", aliases=aliases)

    @classmethod
    def from_latitude(cls, latitude: float) -> Hemisphere:
        """Infer the hemisphere from latitude using the documented rule (A9).

        ``latitude >= 0 -> north``. This is a declared tie-break for sites on the
        equator, not a scientific claim; an explicit hemisphere always overrides it.
        """
        if isinstance(latitude, bool) or not isinstance(latitude, (int, float)):
            raise ConfigError(f"latitude must be a number, got {latitude!r}")
        value = float(latitude)
        if value != value:  # NaN
            raise ConfigError("latitude must be a finite number, got NaN")
        if not -90.0 <= value <= 90.0:
            raise ConfigError(f"latitude must lie in [-90, 90], got {value!r}")
        return cls.NORTH if value >= 0.0 else cls.SOUTH


class GapClass(str, Enum):
    """Artificial-gap duration class (method_spec.md section 4.1)."""

    SHORT = "short"
    LONG = "long"
    VERY_LONG = "very_long"

    @classmethod
    def coerce(cls, value: object) -> GapClass:
        """Return ``value`` as a :class:`GapClass`.

        The nominal durations are accepted as aliases so the public API may be
        called with ``{"24h": 0.20, "7d": 0.30, "30d": 0.50}`` as well as with class
        names. An alias names the class, not the duration; the duration itself stays
        configurable through ``GapScenarioConfig.durations``.
        """
        aliases = {
            "24h": cls.SHORT,
            "1d": cls.SHORT,
            "7d": cls.LONG,
            "30d": cls.VERY_LONG,
            "verylong": cls.VERY_LONG,
        }
        return coerce_enum(cls, value, field_name="gap_class", aliases=aliases)


# ---------------------------------------------------------------------------
# Column mapping
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ColumnMap(FrozenRecord):
    """Mapping from canonical variable names to the input frame's column names.

    ``variables`` maps canonical names (:data:`CANONICAL_VARIABLES`) to arbitrary
    station column names; ``timestamp`` names the timestamp column when the frame
    carries no ``DatetimeIndex``. Unknown canonical names, blank column names, and
    two canonical variables pointing at the same input column are all rejected at
    construction.

    Instances are frozen and hold their mapping in canonical order, so
    :meth:`to_dict` output is stable enough to go straight into a run manifest.
    """

    variables: Mapping[str, str] = field(default_factory=dict)
    timestamp: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.variables, Mapping):
            raise ConfigError(
                "ColumnMap.variables must be a mapping of canonical name -> column name, "
                f"got {type(self.variables).__name__}"
            )
        unknown = [str(key) for key in self.variables if key not in CANONICAL_VARIABLES]
        if unknown:
            raise ConfigError(
                f"unknown canonical variable(s): {', '.join(sorted(unknown))}; "
                f"expected one of: {', '.join(CANONICAL_VARIABLES)}"
            )
        cleaned: dict[str, str] = {}
        for name in CANONICAL_VARIABLES:
            if name not in self.variables:
                continue
            column = self.variables[name]
            if not isinstance(column, str) or not column.strip():
                raise ConfigError(
                    f"column name for canonical variable {name!r} must be a non-empty "
                    f"string, got {column!r}"
                )
            cleaned[name] = column
        mapped = list(cleaned.values())
        duplicates = sorted({column for column in mapped if mapped.count(column) > 1})
        if duplicates:
            raise ConfigError(
                "the same input column is mapped to more than one canonical variable: "
                f"{', '.join(duplicates)}"
            )
        if self.timestamp is not None and (
            not isinstance(self.timestamp, str) or not self.timestamp.strip()
        ):
            raise ConfigError(
                f"timestamp column must be a non-empty string or None, got {self.timestamp!r}"
            )
        object.__setattr__(self, "variables", MappingProxyType(cleaned))

    # -- lookup --------------------------------------------------------------

    def __contains__(self, name: object) -> bool:
        return name in self.variables

    def __iter__(self) -> Iterator[str]:
        return iter(self.variables)

    def __len__(self) -> int:
        return len(self.variables)

    def has(self, name: str) -> bool:
        """Return whether canonical ``name`` is mapped."""
        return name in self.variables

    def column(self, name: str) -> str:
        """Return the input column mapped to canonical ``name``."""
        if name not in CANONICAL_VARIABLES:
            raise ConfigError(
                f"unknown canonical variable {name!r}; "
                f"expected one of: {', '.join(CANONICAL_VARIABLES)}"
            )
        try:
            return self.variables[name]
        except KeyError:
            raise ColumnMapError(f"canonical variable {name!r} is not mapped to a column") from None

    def columns(self, names: Iterable[str]) -> tuple[str, ...]:
        """Return the input columns for ``names``, in the order given."""
        return tuple(self.column(name) for name in names)

    # -- requirements --------------------------------------------------------

    def missing(self, required: Iterable[str]) -> tuple[str, ...]:
        """Return the canonical names in ``required`` that are not mapped."""
        return tuple(name for name in required if name not in self.variables)

    def require(self, required: Iterable[str], *, context: str = "") -> None:
        """Raise :class:`ColumnMapError` listing every unmapped name in ``required``."""
        missing = self.missing(required)
        if missing:
            where = f" for {context}" if context else ""
            raise ColumnMapError(
                f"missing column mapping{where}: {', '.join(missing)}. "
                "Map each required canonical variable to a column in your data."
            )

    def missing_columns(self, available: Iterable[str]) -> tuple[str, ...]:
        """Return mapped column names that are absent from ``available``.

        Checked before fitting so a mistyped column name fails with a clear message
        instead of silently producing an all-missing feature.
        """
        present = set(available)
        return tuple(column for column in self.variables.values() if column not in present)

    # -- construction and serialisation --------------------------------------

    @classmethod
    def coerce(cls, value: ColumnMap | Mapping[str, str] | None) -> ColumnMap:
        """Return ``value`` as a :class:`ColumnMap`, accepting a plain mapping."""
        if value is None:
            return cls()
        if isinstance(value, ColumnMap):
            return value
        if isinstance(value, Mapping):
            return cls(variables=dict(value))
        raise ConfigError(
            f"column_map must be a ColumnMap or a mapping, got {type(value).__name__}"
        )

    @classmethod
    def fluxnet2015(
        cls,
        mode: Mode | str | None = None,
        *,
        overrides: Mapping[str, str] | None = None,
        timestamp: str | None = None,
    ) -> ColumnMap:
        """Return the reference FLUXNET2015 mapping, optionally restricted to ``mode``.

        A convenience for FLUXNET inputs only. ``overrides`` replaces individual
        columns for sites that deviate from the reference names.
        """
        names = Mode.coerce(mode).drivers if mode is not None else CANONICAL_VARIABLES
        variables = {name: FLUXNET2015_COLUMNS[name] for name in names}
        if overrides:
            variables.update(overrides)
        return cls(variables=variables, timestamp=timestamp)

    def with_overrides(
        self,
        variables: Mapping[str, str] | None = None,
        *,
        timestamp: str | None = None,
    ) -> ColumnMap:
        """Return a copy with ``variables`` merged in and ``timestamp`` replaced."""
        merged = dict(self.variables)
        if variables:
            merged.update(variables)
        return ColumnMap(
            variables=merged,
            timestamp=self.timestamp if timestamp is None else timestamp,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation for the run manifest."""
        return {"variables": dict(self.variables), "timestamp": self.timestamp}
