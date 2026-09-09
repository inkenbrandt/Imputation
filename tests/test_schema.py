"""Canonical-vocabulary and column-mapping tests.

Covers the driver lists of ``docs/method_spec.md`` section 2, the ``ColumnMap``
layer that keeps station column names out of the code, and the documented
latitude-to-hemisphere rule (ambiguity A9).

See ``docs/method_spec.md`` for the contract and
``docs/supplement_benchmarks.md`` for benchmark provenance.
"""

from __future__ import annotations

import pytest

from rfrgapfill.schema import (
    CANONICAL_VARIABLES,
    EBR_VARIABLES,
    FLUXNET2015_COLUMNS,
    RFR3_DRIVERS,
    RFR10_DRIVERS,
    VARIABLE_UNITS,
    ColumnMap,
    ColumnMapError,
    ConfigError,
    GapClass,
    Hemisphere,
    Mode,
)

RFR3_MAPPING = {"shortwave": "SW_IN", "vpd": "VPD", "air_temperature": "TA"}


# ---------------------------------------------------------------------------
# Canonical variables
# ---------------------------------------------------------------------------


def test_rfr3_drivers_are_the_three_mds_equivalent_variables() -> None:
    assert RFR3_DRIVERS == ("shortwave", "vpd", "air_temperature")


def test_rfr10_extends_rfr3_with_seven_more_drivers_in_specification_order() -> None:
    assert RFR10_DRIVERS[:3] == RFR3_DRIVERS
    assert RFR10_DRIVERS[3:] == (
        "net_radiation",
        "wind_speed",
        "wind_direction",
        "soil_heat_flux",
        "soil_temperature",
        "relative_humidity",
        "soil_water_content",
    )
    assert len(RFR10_DRIVERS) == 10
    assert len(set(RFR10_DRIVERS)) == 10


def test_every_canonical_variable_has_units_and_a_fluxnet_reference_name() -> None:
    assert set(VARIABLE_UNITS) == set(CANONICAL_VARIABLES)
    assert set(FLUXNET2015_COLUMNS) == set(CANONICAL_VARIABLES)


def test_energy_balance_variables_are_net_radiation_and_soil_heat_flux() -> None:
    assert EBR_VARIABLES == ("net_radiation", "soil_heat_flux")


# ---------------------------------------------------------------------------
# Mode
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", ["RFR3", "rfr3", "  RFR3 ", Mode.RFR3])
def test_mode_accepts_documented_spellings(value: object) -> None:
    assert Mode.coerce(value) is Mode.RFR3


@pytest.mark.parametrize("value", ["RFR5", "RFR", "", "rfr 3", None, 3, ["RFR3"]])
def test_mode_rejects_unknown_modes(value: object) -> None:
    with pytest.raises(ConfigError, match="not a recognised Mode"):
        Mode.coerce(value)


def test_mode_error_lists_the_valid_modes() -> None:
    with pytest.raises(ConfigError, match="RFR10, RFR3"):
        Mode.coerce("RFR7")


def test_mode_exposes_its_driver_list() -> None:
    assert Mode.RFR3.drivers == RFR3_DRIVERS
    assert Mode.RFR10.drivers == RFR10_DRIVERS


# ---------------------------------------------------------------------------
# Hemisphere and the latitude rule (A9)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("latitude", "expected"),
    [
        (51.5, Hemisphere.NORTH),
        (0.1, Hemisphere.NORTH),
        (0.0, Hemisphere.NORTH),  # documented equator tie-break: latitude >= 0 -> north
        (-0.0, Hemisphere.NORTH),  # negative zero compares >= 0 in Python
        (-0.1, Hemisphere.SOUTH),
        (-33.9, Hemisphere.SOUTH),
        (90.0, Hemisphere.NORTH),
        (-90.0, Hemisphere.SOUTH),
    ],
)
def test_latitude_maps_to_hemisphere_with_the_documented_rule(
    latitude: float, expected: Hemisphere
) -> None:
    assert Hemisphere.from_latitude(latitude) is expected


@pytest.mark.parametrize("latitude", [90.5, -90.5, 1000.0])
def test_latitude_outside_the_globe_is_rejected(latitude: float) -> None:
    with pytest.raises(ConfigError, match=r"\[-90, 90\]"):
        Hemisphere.from_latitude(latitude)


@pytest.mark.parametrize("latitude", ["north", None, True, float("nan")])
def test_non_numeric_latitude_is_rejected(latitude: object) -> None:
    with pytest.raises(ConfigError, match="latitude"):
        Hemisphere.from_latitude(latitude)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("north", Hemisphere.NORTH),
        ("Northern", Hemisphere.NORTH),
        ("N", Hemisphere.NORTH),
        ("south", Hemisphere.SOUTH),
        ("S", Hemisphere.SOUTH),
        (Hemisphere.SOUTH, Hemisphere.SOUTH),
    ],
)
def test_hemisphere_accepts_documented_spellings(value: object, expected: Hemisphere) -> None:
    assert Hemisphere.coerce(value) is expected


def test_hemisphere_rejects_unknown_values() -> None:
    with pytest.raises(ConfigError, match="not a recognised Hemisphere"):
        Hemisphere.coerce("equator")


# ---------------------------------------------------------------------------
# GapClass
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("short", GapClass.SHORT),
        ("24h", GapClass.SHORT),
        ("long", GapClass.LONG),
        ("7d", GapClass.LONG),
        ("very_long", GapClass.VERY_LONG),
        ("very long", GapClass.VERY_LONG),
        ("30d", GapClass.VERY_LONG),
    ],
)
def test_gap_class_accepts_class_names_and_nominal_duration_aliases(
    value: str, expected: GapClass
) -> None:
    assert GapClass.coerce(value) is expected


def test_gap_class_rejects_unknown_classes() -> None:
    with pytest.raises(ConfigError, match="not a recognised GapClass"):
        GapClass.coerce("14d")


# ---------------------------------------------------------------------------
# ColumnMap
# ---------------------------------------------------------------------------


def test_column_map_resolves_canonical_names_to_station_columns() -> None:
    column_map = ColumnMap(RFR3_MAPPING)

    assert column_map.column("shortwave") == "SW_IN"
    assert column_map.columns(RFR3_DRIVERS) == ("SW_IN", "VPD", "TA")
    assert column_map.has("vpd")
    assert "net_radiation" not in column_map


def test_column_map_stores_variables_in_canonical_order() -> None:
    column_map = ColumnMap({"air_temperature": "TA", "shortwave": "SW_IN", "vpd": "VPD"})

    assert list(column_map.variables) == list(RFR3_DRIVERS)


def test_column_map_rejects_unknown_canonical_names() -> None:
    with pytest.raises(ConfigError, match="unknown canonical variable"):
        ColumnMap({"sunshine": "SW_IN"})


@pytest.mark.parametrize("column", ["", "   ", None, 3])
def test_column_map_rejects_blank_or_non_string_columns(column: object) -> None:
    with pytest.raises(ConfigError, match="non-empty string"):
        ColumnMap({"shortwave": column})  # type: ignore[dict-item]


def test_column_map_rejects_one_column_serving_two_variables() -> None:
    with pytest.raises(ConfigError, match="more than one canonical variable"):
        ColumnMap({"shortwave": "SW_IN", "net_radiation": "SW_IN"})


def test_column_map_reports_and_raises_on_missing_requirements() -> None:
    column_map = ColumnMap({"shortwave": "SW_IN"})

    assert column_map.missing(RFR3_DRIVERS) == ("vpd", "air_temperature")
    with pytest.raises(ColumnMapError, match="vpd, air_temperature"):
        column_map.require(RFR3_DRIVERS, context="mode=RFR3")


def test_column_map_reports_columns_absent_from_the_data() -> None:
    column_map = ColumnMap(RFR3_MAPPING)

    assert column_map.missing_columns(["SW_IN", "VPD", "NEE"]) == ("TA",)
    assert column_map.missing_columns(["SW_IN", "VPD", "TA"]) == ()


def test_requesting_an_unmapped_variable_raises_rather_than_returning_none() -> None:
    column_map = ColumnMap(RFR3_MAPPING)

    with pytest.raises(ColumnMapError, match="not mapped"):
        column_map.column("soil_water_content")
    with pytest.raises(ConfigError, match="unknown canonical variable"):
        column_map.column("sunshine")


def test_fluxnet_helper_is_a_convenience_default_restricted_to_the_mode() -> None:
    column_map = ColumnMap.fluxnet2015("RFR3")

    assert column_map.column("shortwave") == "SW_IN_F"
    assert column_map.missing(RFR3_DRIVERS) == ()
    assert not column_map.has("net_radiation")
    assert ColumnMap.fluxnet2015("RFR10").missing(RFR10_DRIVERS) == ()


def test_fluxnet_helper_accepts_site_specific_overrides() -> None:
    column_map = ColumnMap.fluxnet2015("RFR3", overrides={"shortwave": "SW_IN_F_MDS"})

    assert column_map.column("shortwave") == "SW_IN_F_MDS"
    assert column_map.column("vpd") == "VPD_F_MDS"


def test_column_map_coerces_plain_mappings_and_none() -> None:
    assert ColumnMap.coerce(RFR3_MAPPING) == ColumnMap(RFR3_MAPPING)
    assert ColumnMap.coerce(None) == ColumnMap()
    assert ColumnMap.coerce(ColumnMap(RFR3_MAPPING)).column("vpd") == "VPD"
    with pytest.raises(ConfigError, match="ColumnMap or a mapping"):
        ColumnMap.coerce(["shortwave"])  # type: ignore[arg-type]


def test_column_map_overrides_return_a_new_validated_map() -> None:
    original = ColumnMap(RFR3_MAPPING)
    updated = original.with_overrides({"vpd": "VPD_2"}, timestamp="TIMESTAMP_START")

    assert original.column("vpd") == "VPD"
    assert updated.column("vpd") == "VPD_2"
    assert updated.timestamp == "TIMESTAMP_START"


def test_column_map_is_immutable_and_serialisable() -> None:
    column_map = ColumnMap(RFR3_MAPPING, timestamp="TIMESTAMP_START")

    with pytest.raises(TypeError):
        column_map.variables["shortwave"] = "other"  # type: ignore[index]
    assert column_map.to_dict() == {
        "variables": {"shortwave": "SW_IN", "vpd": "VPD", "air_temperature": "TA"},
        "timestamp": "TIMESTAMP_START",
    }
