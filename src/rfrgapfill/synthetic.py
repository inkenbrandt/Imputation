"""Deterministic synthetic eddy-covariance site data (Step 16).

The test suite and the worked examples need a site whose structure is *known*:
a diurnal radiation cycle, a seasonal temperature swing, a VPD that follows both,
a soil temperature that lags the air, fluxes that are real functions of those
drivers, and measurement noise on top. This module builds that site, so nothing
in this package's tests depends on downloading FLUXNET2015 or any other external
dataset.

Everything here is a **fixture generator**, never part of the method. No model
learns from these equations and nothing else in :mod:`rfrgapfill` imports this
module; it only produces frames the rest of the package consumes like any other
site.

What the generator guarantees
-----------------------------

The value of a synthetic site is what can be asserted about it:

- **it is deterministic.** The same arguments give bit-identical frames. Each
  stochastic component draws from its own stream (``SeedSequence.spawn``), so
  changing one parameter does not reshuffle the others;
- **the drivers are complete.** The paper's meteorological drivers arrive
  pre-filled (method_spec.md section 7), so only the target fluxes carry gaps
  and QC flags, and RFR10 is runnable on every row;
- **the noise-free truth is kept.** :attr:`SyntheticSite.truth` holds the fluxes
  before measurement noise, so a test can measure how close a fit gets to the
  signal rather than to the noise floor;
- **energy balance closes at a known ratio.** Sensible heat is the residual of
  available energy after latent heat, so ``sum(H + LE) / sum(NETRAD - G)`` over
  :attr:`SyntheticSite.truth` is exactly
  :attr:`SyntheticSite.energy_balance_closure` - the fixture the EBR metric of
  method_spec.md section 6.4 can be checked against;
- **the gaps are known.** :func:`known_gap_manifest` places 24-hour, 7-day and
  30-day intervals at fixed offsets, hand-checkable and identical run to run.

The physics is deliberately simple but not arbitrary: clear-sky geometry from
the solar declination, Tetens saturation vapour pressure, a first-order soil
thermal lag, a Priestley-Taylor evaporative demand drawn from a bucket soil
moisture store, a rectangular-hyperbola light response for GPP and a Q10
respiration on soil temperature. That is enough for a Random Forest to have
something real to learn, and for a reader to predict the sign of every effect.

Hemisphere handling is a property of the geometry rather than a switch: the
seasonal index is derived from daily potential radiation, so a negative
``latitude`` flips the seasons, the phenology and the temperature cycle together
without a calendar-month branch anywhere in this module.

The known gap plan is **not** the paper's 25% / 20-30-50 scenario - that is
:class:`rfrgapfill.gaps.GapScenarioGenerator`'s job, and it should be run against
this site's frame like any other. The plan here is a handful of fixed intervals
whose row counts a test can state in advance.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import timedelta
from itertools import pairwise
from types import MappingProxyType
from typing import Any, Final

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from rfrgapfill.config import (
    DEFAULT_OBSERVED_QC_VALUES,
    AllocationBasis,
    GapScenarioConfig,
    RFRConfig,
)
from rfrgapfill.gaps import ArtificialGap, GapAllocation, GapError, GapManifest
from rfrgapfill.leakage import observed_target_mask
from rfrgapfill.schema import (
    AIR_TEMPERATURE,
    FLUXNET2015_COLUMNS,
    NET_RADIATION,
    RELATIVE_HUMIDITY,
    SHORTWAVE,
    SOIL_HEAT_FLUX,
    SOIL_TEMPERATURE,
    SOIL_WATER_CONTENT,
    VPD,
    WIND_DIRECTION,
    WIND_SPEED,
    ColumnMap,
    ConfigError,
    FrozenRecord,
    GapClass,
    Hemisphere,
    Mode,
)
from rfrgapfill.time import (
    as_datetime_index,
    duration_to_periods,
    elapsed_hours,
    interval_mask,
    to_timedelta,
)

__all__ = [
    "DEFAULT_KNOWN_GAPS",
    "DEFAULT_LATITUDE",
    "DEFAULT_SEED",
    "DEFAULT_SITE_ID",
    "DEFAULT_START",
    "MISSING_QC",
    "OBSERVED_QC",
    "PRE_FILLED_QC",
    "TARGETS",
    "KnownGap",
    "SyntheticSite",
    "known_gap_manifest",
    "synthetic_site",
]


# ---------------------------------------------------------------------------
# Fixture identity
# ---------------------------------------------------------------------------

#: First timestamp of the default record. 2018 is not a leap year, so a
#: 365-day request at 30 minutes is exactly one calendar year.
DEFAULT_START: Final = "2018-01-01"
#: Site identifier carried into configurations and run manifests.
DEFAULT_SITE_ID: Final = "SYN-Ref"
#: Default latitude: a mid-latitude northern site with a clear growing season.
DEFAULT_LATITUDE: Final = 45.0
#: Default seed. The default arguments together define *the* reference site.
DEFAULT_SEED: Final = 20220101
#: The three target fluxes, in the paper's order.
TARGETS: Final[tuple[str, ...]] = ("NEE", "H", "LE")

#: QC flag of a genuinely observed value, matching FLUXNET's convention.
OBSERVED_QC: Final = 0
#: QC flag of a value that was already gap-filled before ingestion.
PRE_FILLED_QC: Final = 1
#: QC flag carried by a row whose flux is missing outright.
MISSING_QC: Final = 3


# ---------------------------------------------------------------------------
# Physical constants and site parameters
# ---------------------------------------------------------------------------

_SOLAR_CONSTANT: Final = 1361.0  # W m-2
_CLEAR_SKY_TRANSMISSIVITY: Final = 0.75
_OBLIQUITY_DEGREES: Final = 23.44
_DAYS_PER_YEAR: Final = 365.0
_HOURS_PER_YEAR: Final = 24.0 * 365.25

_ALBEDO: Final = 0.20
_NET_LONGWAVE_CLEAR: Final = 75.0  # W m-2 lost under a clear sky
_NET_LONGWAVE_CLOUDY: Final = 20.0  # W m-2 lost under full cloud
_SOIL_HEAT_DAY_FRACTION: Final = 0.10
_SOIL_HEAT_NIGHT_FRACTION: Final = 0.45
_SOIL_HEAT_LAG_HOURS: Final = 1.0

_MEAN_AIR_TEMPERATURE: Final = 9.0  # degC
_SEASONAL_AMPLITUDE: Final = 11.0  # degC, half the winter-to-summer swing
_DIURNAL_AMPLITUDE: Final = 8.0  # degC at midsummer under a clear sky
_SYNOPTIC_AMPLITUDE: Final = 3.5  # degC, weather-scale anomalies
_THERMAL_LAG_HOURS: Final = 3.0  # air temperature peaks after solar noon
_SEASONAL_LAG_DAYS: Final = 20.0  # the warmest weeks trail the solstice
_SYNOPTIC_TAU_DAYS: Final = 3.0
_SKY_TAU_HOURS: Final = 10.0

_SOIL_LAG_HOURS: Final = 36.0  # damps the diurnal cycle, lags it about 6 h
_SOIL_TEMPERATURE_OFFSET: Final = 1.0  # degC warmer than the lagged air

_DEWPOINT_DEPRESSION: Final = 1.5  # degC below the daily minimum temperature
_MOISTURE_AMPLITUDE: Final = 2.0  # degC of air-mass dewpoint variation
_MOISTURE_TAU_DAYS: Final = 2.0
_PSYCHROMETRIC_CONSTANT: Final = 0.665  # hPa K-1 at sea level
_LATENT_HEAT_VAPORISATION: Final = 2.45e6  # J kg-1
_PRIESTLEY_TAYLOR_ALPHA: Final = 1.26

_MEAN_WIND_SPEED: Final = 2.6  # m s-1
_WIND_VARIABILITY: Final = 1.4  # m s-1
_WIND_DAYTIME_BOOST: Final = 1.2  # m s-1 at peak radiation
_WIND_TAU_HOURS: Final = 8.0
_PREVAILING_WIND_DIRECTION: Final = 225.0  # degrees, south-westerly
_WIND_DIRECTION_SPREAD: Final = 55.0  # degrees
_WIND_DIRECTION_TAU_HOURS: Final = 18.0

_ROOT_DEPTH_MM: Final = 800.0  # rooting depth of the bucket store
_FIELD_CAPACITY: Final = 34.0  # % volumetric
_WILTING_POINT: Final = 12.0  # % volumetric
_RESIDUAL_WATER: Final = 8.0  # % volumetric, the floor of the store
_ANNUAL_RAINFALL_MM: Final = 1050.0
_RAIN_EVENT_MM: Final = 3.5  # mean depth of one wet half hour

_GREENUP_THRESHOLD: Final = 0.28  # seasonal index at which the canopy starts
_GREENUP_RANGE: Final = 0.40  # seasonal index span from bare to full canopy
_CANOPY_TAU_DAYS: Final = 6.0
_SOIL_EVAPORATION_FRACTION: Final = 0.40  # evaporation that happens without leaves

_GPP_MAX: Final = 30.0  # umol m-2 s-1 at full canopy, unstressed
_QUANTUM_YIELD: Final = 0.05  # umol CO2 per umol photon
_PAR_PER_WATT: Final = 2.1  # umol photons per J of global shortwave
_VPD_THRESHOLD: Final = 10.0  # hPa below which stomata are unaffected
_VPD_SENSITIVITY: Final = 0.030  # per hPa above the threshold
_RECO_REFERENCE: Final = 2.0  # umol m-2 s-1 at the reference temperature
_RECO_REFERENCE_TEMPERATURE: Final = 10.0  # degC
_RECO_Q10: Final = 2.0
_RECO_BARE_FRACTION: Final = 0.35  # respiration independent of the canopy

_NEE_NOISE: Final = 0.7  # umol m-2 s-1
_NEE_NOISE_RELATIVE: Final = 0.05
_ENERGY_NOISE: Final = 9.0  # W m-2
_ENERGY_NOISE_RELATIVE: Final = 0.07
_SHORTWAVE_NOISE: Final = 3.0  # W m-2, daylight only

_OUTAGE_MEAN_DAYS: Final = 0.5  # mean instrument outage, in days
_PRE_FILLED_MEAN_HOURS: Final = 4.0
_DAYTIME_THRESHOLD: Final = 20.0  # W m-2, the paper's day/night split

_EPS: Final = 1e-12


# ---------------------------------------------------------------------------
# Deterministic building blocks
# ---------------------------------------------------------------------------


def _ar1(rng: np.random.Generator, n: int, tau_steps: float) -> NDArray[np.float64]:
    """Return a zero-mean, unit-variance AR(1) series with e-folding ``tau_steps``.

    Every slowly varying anomaly in this module - cloudiness, weather, wind -
    comes from here, so a synthetic year contains spells rather than white noise
    and a daily statistic computed from it means something.
    """
    if n <= 0:
        return np.zeros(0, dtype=float)
    phi = float(np.exp(-1.0 / max(float(tau_steps), _EPS)))
    innovations = rng.standard_normal(n) * math.sqrt(max(1.0 - phi * phi, 0.0))
    out = np.empty(n, dtype=float)
    value = float(rng.standard_normal())
    for position in range(n):
        value = phi * value + float(innovations[position])
        out[position] = value
    return out


def _lag(values: NDArray[np.float64], tau_steps: float) -> NDArray[np.float64]:
    """Return ``values`` through a first-order lag of time constant ``tau_steps``.

    A damped, delayed copy of its input. The soil temperature's response to air
    temperature, the afternoon peak of air temperature after solar noon and the
    smoothing of the phenology curve are all this one operator.
    """
    alpha = float(1.0 - np.exp(-1.0 / max(float(tau_steps), _EPS)))
    out = np.empty(values.size, dtype=float)
    value = float(values[0]) if values.size else 0.0
    for position in range(values.size):
        value += alpha * (float(values[position]) - value)
        out[position] = value
    return out


def _daily(values: NDArray[np.float64], index: pd.DatetimeIndex, how: str) -> NDArray[np.float64]:
    """Return the per-calendar-day ``how`` of ``values``, broadcast back to every row."""
    series = pd.Series(values, index=index)
    return np.asarray(series.groupby(index.normalize()).transform(how).to_numpy(), dtype=float)


def _normalise(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return ``values`` rescaled to [0, 1], or a flat 0.5 when they do not vary.

    The flat case is the equator, where daily potential radiation barely changes
    across the year: there is no seasonal index to build, and dividing through a
    near-zero range would manufacture a season out of rounding error.
    """
    low = float(np.min(values))
    high = float(np.max(values))
    if high - low <= _EPS:
        return np.full(values.size, 0.5, dtype=float)
    return (values - low) / (high - low)


def _saturation_vapour_pressure(temperature: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return saturation vapour pressure in hPa (Tetens, over water)."""
    return 6.1078 * np.exp(17.27 * temperature / (temperature + 237.3))


def _saturation_slope(
    temperature: NDArray[np.float64], saturation: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Return d(es)/dT in hPa K-1, the slope Priestley-Taylor needs."""
    return 4098.0 * saturation / np.square(temperature + 237.3)


def _burst_mask(
    rng: np.random.Generator,
    n: int,
    fraction: float,
    *,
    mean_length: int,
    candidates: NDArray[np.bool_] | None = None,
) -> NDArray[np.bool_]:
    """Return a bursty boolean mask covering roughly ``fraction`` of ``n`` rows.

    Real flux gaps arrive as instrument outages, not as scattered rows, and the
    artificial-gap generator's 50%-observed rule (method_spec.md section 4.4) is
    only exercised by data whose missing values clump. Bursts may overlap and may
    run past the end, so the achieved share sits at or just below the request -
    which is why nothing here promises an exact count.
    """
    mask = np.zeros(n, dtype=bool)
    if n <= 0 or fraction <= 0.0:
        return mask
    length = max(int(mean_length), 1)
    bursts = max(round(fraction * n / length), 1)
    if candidates is None:
        starts = rng.integers(0, n, size=bursts)
    else:
        pool = np.flatnonzero(candidates)
        if pool.size == 0:
            return mask
        starts = pool[rng.integers(0, pool.size, size=bursts)]
    spans = 1 + rng.poisson(max(length - 1, 0), size=bursts)
    for start, span in zip(starts, spans, strict=True):
        mask[int(start) : int(start) + int(span)] = True
    return mask


def _fraction(value: object, *, field_name: str) -> float:
    """Return ``value`` as a fraction in [0, 1), rejecting anything else."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{field_name} must be a number, got {value!r}")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number < 1.0:
        raise ConfigError(f"{field_name} must lie in [0, 1), got {value!r}")
    return number


# ---------------------------------------------------------------------------
# The site
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class SyntheticSite:
    """One synthetic site: the frame a workflow sees, plus what generated it.

    :attr:`frame` is an ordinary site frame - ten drivers under their
    FLUXNET2015 names, three flux targets and their QC flags - and is the only
    part of this object the package's own modules ever need. Everything else is
    the answer key: the noise-free fluxes, the seed, and the closure ratio the
    energy balance was built to.
    """

    #: Drivers, targets and QC flags on the requested time axis.
    frame: pd.DataFrame
    #: Noise-free ``NEE``, ``H``, ``LE`` and the ``GPP``/``RECO`` behind NEE.
    truth: pd.DataFrame
    #: Canonical-name mapping for :attr:`frame`; all ten drivers are present.
    column_map: ColumnMap
    #: Site identifier, recorded in configurations and manifests.
    site_id: str
    #: Site latitude in degrees; a negative value is a southern-hemisphere site.
    latitude: float
    #: Cadence of the time axis.
    time_step: timedelta
    #: Seed the whole site was generated from.
    seed: int
    #: ``sum(H + LE) / sum(NETRAD - G)`` over :attr:`truth`, exactly.
    energy_balance_closure: float
    #: Target flux columns, in the paper's order.
    targets: tuple[str, ...] = TARGETS

    # -- accessors -----------------------------------------------------------

    @property
    def index(self) -> pd.DatetimeIndex:
        """The site's time axis."""
        return as_datetime_index(self.frame.index, field_name="the synthetic site index")

    @property
    def hemisphere(self) -> Hemisphere:
        """Hemisphere implied by :attr:`latitude` under the documented rule (A9)."""
        return Hemisphere.from_latitude(self.latitude)

    def qc_column(self, target: str) -> str:
        """Return the QC column name for ``target``."""
        if target not in self.targets:
            raise ConfigError(
                f"unknown target {target!r}; this site carries {', '.join(self.targets)}"
            )
        return f"{target}_QC"

    def qc_columns(self) -> Mapping[str, str]:
        """Return the QC column of every target."""
        return MappingProxyType({target: self.qc_column(target) for target in self.targets})

    def observed(self, target: str) -> pd.Series:
        """Return the genuinely observed rows of ``target``, QC flags applied."""
        return observed_target_mask(
            self.frame,
            target,
            qc_column=self.qc_column(target),
            observed_qc_values=DEFAULT_OBSERVED_QC_VALUES,
        )

    # -- ready-made objects for a run ----------------------------------------

    def config(self, mode: Mode | str = Mode.RFR3, **changes: Any) -> RFRConfig:
        """Return an :class:`~rfrgapfill.config.RFRConfig` for this site.

        The site's cadence, latitude, identifier, column mapping and seed are
        filled in; ``changes`` overrides any of them and supplies the rest.
        """
        settings: dict[str, Any] = {
            "mode": mode,
            "frequency": self.time_step,
            "latitude": self.latitude,
            "site_id": self.site_id,
            "random_state": self.seed,
            "column_map": self.column_map,
        }
        settings.update(changes)
        return RFRConfig(**settings)

    def known_gaps(
        self,
        *,
        targets: Sequence[str] | str | None = None,
        plan: Sequence[KnownGap] | None = None,
        scenario: GapScenarioConfig | None = None,
    ) -> GapManifest:
        """Return the fixed artificial-gap manifest for this site.

        ``targets`` defaults to all three, so the withheld rows are the ones
        every flux observes - the paper's shared-gap rule (method_spec.md
        section 4.4) applied to the fixture. ``plan`` defaults to
        :data:`DEFAULT_KNOWN_GAPS`.
        """
        chosen = self.targets if targets is None else targets
        names = (chosen,) if isinstance(chosen, str) else tuple(chosen)
        return known_gap_manifest(
            self.frame,
            time_step=self.time_step,
            targets=names,
            qc_columns={target: self.qc_column(target) for target in names},
            plan=DEFAULT_KNOWN_GAPS if plan is None else plan,
            scenario=scenario,
            random_state=self.seed,
        )


def synthetic_site(
    *,
    start: pd.Timestamp | str = DEFAULT_START,
    days: int = 365,
    frequency: timedelta | str = "30min",
    latitude: float = DEFAULT_LATITUDE,
    site_id: str = DEFAULT_SITE_ID,
    seed: int = DEFAULT_SEED,
    noise_scale: float = 1.0,
    real_gap_fraction: float = 0.05,
    pre_filled_fraction: float = 0.10,
    growth_per_year: float = 0.05,
    energy_balance_closure: float = 0.85,
) -> SyntheticSite:
    """Build a deterministic synthetic flux site with known structure.

    The defaults are the reference fixture: one non-leap calendar year of
    half-hourly data at 45 degrees north, seeded, with 5% of the fluxes missing
    outright and 10% flagged as already gap-filled. Called with no arguments it
    returns the same site every time.

    :param days: length of the record in days. The default 365 satisfies Step
        16's "at least one year"; longer records give the growth trend and the
        30-day gap class more room.
    :param frequency: cadence of the time axis. Must divide a day exactly.
    :param latitude: site latitude. A negative value moves the site to the
        southern hemisphere and flips the seasons, the phenology and the
        temperature cycle together, because all three are derived from the same
        solar geometry rather than from the calendar month.
    :param noise_scale: multiplies every measurement-noise term. ``0.0`` makes
        :attr:`SyntheticSite.frame` carry the noise-free
        :attr:`SyntheticSite.truth` exactly, which is what a test asserting an
        exact relationship wants.
    :param real_gap_fraction: approximate share of rows whose fluxes are missing
        outright, as bursty instrument outages shared by all three targets.
    :param pre_filled_fraction: approximate share of rows whose flux is present
        but flagged as gap-filled before ingestion. NEE's flagged blocks start at
        night, the way friction-velocity filtering removes calm nights.
    :param growth_per_year: fractional increase in photosynthetic capacity per
        year of record - the gradual change ``time_distance_hours`` exists to
        represent (method_spec.md section 3.2).
    :param energy_balance_closure: the ratio ``sum(H + LE) / sum(NETRAD - G)``
        the noise-free fluxes are built to. Real eddy-covariance sites
        under-close; ``1.0`` gives perfect closure.

    :returns: a :class:`SyntheticSite` holding the frame, the noise-free truth
        and the metadata needed to configure a run against it.
    """
    if isinstance(days, bool) or not isinstance(days, int) or days < 1:
        raise ConfigError(f"days must be a positive integer, got {days!r}")
    step = to_timedelta(frequency, field_name="frequency")
    steps_per_day = duration_to_periods(timedelta(days=1), step, field_name="one day")
    if isinstance(noise_scale, bool) or not isinstance(noise_scale, (int, float)):
        raise ConfigError(f"noise_scale must be a number, got {noise_scale!r}")
    noise_scale = float(noise_scale)
    if not math.isfinite(noise_scale) or noise_scale < 0.0:
        raise ConfigError(f"noise_scale must be non-negative and finite, got {noise_scale!r}")
    real_gap_fraction = _fraction(real_gap_fraction, field_name="real_gap_fraction")
    pre_filled_fraction = _fraction(pre_filled_fraction, field_name="pre_filled_fraction")
    closure = float(energy_balance_closure)
    if not math.isfinite(closure):
        raise ConfigError(f"energy_balance_closure must be finite, got {energy_balance_closure!r}")
    Hemisphere.from_latitude(latitude)  # validates the latitude range and its type

    n = days * steps_per_day
    index = pd.date_range(pd.Timestamp(start), periods=n, freq=pd.Timedelta(step))
    steps_per_hour = 3600.0 / step.total_seconds()

    sky, weather, moisture, wind, rain, noise, outage, flagged = (
        np.random.default_rng(child) for child in np.random.SeedSequence(seed).spawn(8)
    )

    # -- radiation -----------------------------------------------------------
    day_of_year = index.dayofyear.to_numpy(dtype=float)
    hour_of_day = (
        index.hour.to_numpy(dtype=float)
        + index.minute.to_numpy(dtype=float) / 60.0
        + index.second.to_numpy(dtype=float) / 3600.0
    )
    declination = np.radians(_OBLIQUITY_DEGREES) * np.sin(
        2.0 * np.pi * (284.0 + day_of_year) / _DAYS_PER_YEAR
    )
    latitude_radians = np.radians(float(latitude))
    hour_angle = np.radians(15.0 * (hour_of_day - 12.0))
    upright = np.sin(latitude_radians) * np.sin(declination)
    swinging = np.cos(latitude_radians) * np.cos(declination) * np.cos(hour_angle)
    potential = np.clip(
        _SOLAR_CONSTANT * _CLEAR_SKY_TRANSMISSIVITY * (upright + swinging), 0.0, None
    )

    clearness = np.clip(0.88 + 0.28 * _ar1(sky, n, _SKY_TAU_HOURS * steps_per_hour), 0.12, 1.0)
    sensor_noise = np.where(
        potential > 0.0, noise_scale * _SHORTWAVE_NOISE * noise.standard_normal(n), 0.0
    )
    shortwave = np.clip(potential * clearness + sensor_noise, 0.0, None)
    daily_clearness = _daily(clearness, index, "mean")

    # -- seasonality ---------------------------------------------------------
    # Daily potential radiation is the seasonal clock, so the southern
    # hemisphere flips without a calendar-month branch anywhere below.
    season = _normalise(_daily(potential, index, "mean"))
    season_lagged = _normalise(_lag(season, _SEASONAL_LAG_DAYS * steps_per_day))

    # -- air temperature -----------------------------------------------------
    solar_forcing = _lag(_normalise(potential), _THERMAL_LAG_HOURS * steps_per_hour)
    diurnal = (
        _DIURNAL_AMPLITUDE
        * (solar_forcing - _daily(solar_forcing, index, "mean"))
        * (0.6 + 0.4 * daily_clearness)
    )
    air_temperature = (
        _MEAN_AIR_TEMPERATURE
        + _SEASONAL_AMPLITUDE * (2.0 * season_lagged - 1.0)
        + diurnal
        + _SYNOPTIC_AMPLITUDE * _ar1(weather, n, _SYNOPTIC_TAU_DAYS * steps_per_day)
    )

    # -- humidity ------------------------------------------------------------
    # Dewpoint tracks the daily minimum temperature (the FAO-56 approximation),
    # so relative humidity peaks before dawn and VPD peaks in the afternoon.
    saturation = _saturation_vapour_pressure(air_temperature)
    dewpoint = np.minimum(
        _daily(air_temperature, index, "min")
        - _DEWPOINT_DEPRESSION
        + _MOISTURE_AMPLITUDE * _ar1(moisture, n, _MOISTURE_TAU_DAYS * steps_per_day),
        air_temperature - 0.2,
    )
    relative_humidity = np.clip(
        100.0 * _saturation_vapour_pressure(dewpoint) / saturation, 5.0, 100.0
    )
    vapour_pressure_deficit = saturation * (1.0 - relative_humidity / 100.0)

    # -- soil temperature ----------------------------------------------------
    soil_temperature = (
        _lag(air_temperature, _SOIL_LAG_HOURS * steps_per_hour) + _SOIL_TEMPERATURE_OFFSET
    )

    # -- wind ----------------------------------------------------------------
    wind_speed = np.clip(
        _MEAN_WIND_SPEED
        + _WIND_VARIABILITY * _ar1(wind, n, _WIND_TAU_HOURS * steps_per_hour)
        + _WIND_DAYTIME_BOOST * _normalise(potential),
        0.05,
        None,
    )
    wind_direction = (
        _PREVAILING_WIND_DIRECTION
        + _WIND_DIRECTION_SPREAD * _ar1(wind, n, _WIND_DIRECTION_TAU_HOURS * steps_per_hour)
    ) % 360.0

    # -- energy available at the surface -------------------------------------
    net_longwave = _NET_LONGWAVE_CLOUDY + (_NET_LONGWAVE_CLEAR - _NET_LONGWAVE_CLOUDY) * clearness
    net_radiation = (1.0 - _ALBEDO) * shortwave - net_longwave
    soil_heat_flux = _lag(
        np.where(net_radiation > 0.0, _SOIL_HEAT_DAY_FRACTION, _SOIL_HEAT_NIGHT_FRACTION)
        * net_radiation,
        _SOIL_HEAT_LAG_HOURS * steps_per_hour,
    )
    available_energy = net_radiation - soil_heat_flux

    # -- canopy --------------------------------------------------------------
    canopy = _lag(
        np.clip((season_lagged - _GREENUP_THRESHOLD) / _GREENUP_RANGE, 0.0, 1.0),
        _CANOPY_TAU_DAYS * steps_per_day,
    )
    growth = 1.0 + float(growth_per_year) * (
        elapsed_hours(index).to_numpy(dtype=float) / _HOURS_PER_YEAR
    )

    # -- soil water and latent heat ------------------------------------------
    slope = _saturation_slope(air_temperature, saturation)
    demand = (
        _PRIESTLEY_TAYLOR_ALPHA
        * slope
        / (slope + _PSYCHROMETRIC_CONSTANT)
        * np.clip(available_energy, 0.0, None)
    )
    soil_water_content, latent_heat, water_stress = _soil_water_balance(
        demand=demand,
        rainfall=_rainfall(rain, clearness, n=n, days=days),
        canopy=canopy,
        step_seconds=step.total_seconds(),
    )
    # Sensible heat is the residual of available energy, which is what makes the
    # energy-balance ratio of the noise-free fluxes exactly ``closure``.
    sensible_heat = closure * available_energy - latent_heat

    # -- carbon --------------------------------------------------------------
    capacity = (
        _GPP_MAX
        * canopy
        * water_stress
        * growth
        * np.exp(-_VPD_SENSITIVITY * np.clip(vapour_pressure_deficit - _VPD_THRESHOLD, 0.0, None))
    )
    light = _QUANTUM_YIELD * _PAR_PER_WATT * shortwave
    denominator = light + capacity
    gross_primary_production = np.where(
        denominator > 0.0, capacity * light / np.where(denominator > 0.0, denominator, 1.0), 0.0
    )
    respiration = (
        _RECO_REFERENCE
        * _RECO_Q10 ** ((soil_temperature - _RECO_REFERENCE_TEMPERATURE) / 10.0)
        * (_RECO_BARE_FRACTION + (1.0 - _RECO_BARE_FRACTION) * canopy)
    )
    net_ecosystem_exchange = respiration - gross_primary_production

    truth = pd.DataFrame(
        {
            "NEE": net_ecosystem_exchange,
            "H": sensible_heat,
            "LE": latent_heat,
            "GPP": gross_primary_production,
            "RECO": respiration,
        },
        index=index,
    )

    # -- measurement noise, outages and QC flags -----------------------------
    scales = {
        "NEE": _NEE_NOISE + _NEE_NOISE_RELATIVE * np.abs(net_ecosystem_exchange),
        "H": _ENERGY_NOISE + _ENERGY_NOISE_RELATIVE * np.abs(sensible_heat),
        "LE": _ENERGY_NOISE + _ENERGY_NOISE_RELATIVE * np.abs(latent_heat),
    }
    # One shared outage mask: the three fluxes come off one instrument system,
    # so a row missing from NEE is usually missing from H and LE as well.
    missing = _burst_mask(
        outage,
        n,
        real_gap_fraction,
        mean_length=max(int(_OUTAGE_MEAN_DAYS * steps_per_day), 1),
    )
    night = shortwave <= _DAYTIME_THRESHOLD
    fluxes: dict[str, NDArray[Any]] = {}
    for target in TARGETS:
        values = truth[target].to_numpy(dtype=float) + noise_scale * noise.normal(
            0.0, scales[target]
        )
        blocks = _burst_mask(
            flagged,
            n,
            pre_filled_fraction,
            mean_length=max(int(_PRE_FILLED_MEAN_HOURS * steps_per_hour), 1),
            candidates=night if target == "NEE" else None,
        )
        pre_filled = blocks & ~missing
        fluxes[target] = np.where(missing, np.nan, values)
        fluxes[f"{target}_QC"] = np.where(
            missing, MISSING_QC, np.where(pre_filled, PRE_FILLED_QC, OBSERVED_QC)
        ).astype(np.int8)

    drivers = {
        SHORTWAVE: shortwave,
        VPD: vapour_pressure_deficit,
        AIR_TEMPERATURE: air_temperature,
        NET_RADIATION: net_radiation,
        WIND_SPEED: wind_speed,
        WIND_DIRECTION: wind_direction,
        SOIL_HEAT_FLUX: soil_heat_flux,
        SOIL_TEMPERATURE: soil_temperature,
        RELATIVE_HUMIDITY: relative_humidity,
        SOIL_WATER_CONTENT: soil_water_content,
    }
    frame = pd.DataFrame(
        {FLUXNET2015_COLUMNS[name]: values for name, values in drivers.items()} | fluxes,
        index=index,
    )
    return SyntheticSite(
        frame=frame,
        truth=truth,
        column_map=ColumnMap.fluxnet2015(),
        site_id=site_id,
        latitude=float(latitude),
        time_step=step,
        seed=int(seed),
        energy_balance_closure=closure,
    )


def _rainfall(
    rng: np.random.Generator,
    clearness: NDArray[np.float64],
    *,
    n: int,
    days: int,
) -> NDArray[np.float64]:
    """Return rainfall in mm per step, falling preferentially under cloud."""
    weight = np.clip(1.0 - clearness, 0.0, None)
    total = float(weight.sum())
    if n <= 0 or total <= _EPS:
        return np.zeros(max(n, 0), dtype=float)
    events = max(round(_ANNUAL_RAINFALL_MM * days / _DAYS_PER_YEAR / _RAIN_EVENT_MM), 1)
    wet = rng.random(n) < np.clip(weight / total * events, 0.0, 1.0)
    return np.where(wet, rng.exponential(_RAIN_EVENT_MM, size=n), 0.0)


def _soil_water_balance(
    *,
    demand: NDArray[np.float64],
    rainfall: NDArray[np.float64],
    canopy: NDArray[np.float64],
    step_seconds: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Run the bucket store and return soil water, latent heat and water stress.

    One sequential pass, because each is a function of the last: the store sets
    the stress, the stress and the canopy scale the evaporative demand into
    actual latent heat, and that latent heat empties the store. Latent heat is
    therefore never a free parameter - it is what the water balance allows, which
    is what gives ``soil_water_content`` genuine predictive power over ``LE``.
    """
    n = demand.size
    percent_per_mm = 100.0 / _ROOT_DEPTH_MM
    mm_per_watt = step_seconds / _LATENT_HEAT_VAPORISATION
    water = np.empty(n, dtype=float)
    latent = np.empty(n, dtype=float)
    stress = np.empty(n, dtype=float)
    store = _FIELD_CAPACITY
    span = _FIELD_CAPACITY - _WILTING_POINT
    for position in range(n):
        available = min(max((store - _WILTING_POINT) / span, 0.0), 1.0)
        leaves = float(canopy[position])
        surface = _SOIL_EVAPORATION_FRACTION + (1.0 - _SOIL_EVAPORATION_FRACTION) * leaves
        evaporation = float(demand[position]) * available * surface
        store += float(rainfall[position]) * percent_per_mm
        store -= evaporation * mm_per_watt * percent_per_mm
        store = min(max(store, _RESIDUAL_WATER), _FIELD_CAPACITY)
        water[position] = store
        latent[position] = evaporation
        stress[position] = available
    return water, latent, stress


# ---------------------------------------------------------------------------
# Known gaps
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KnownGap(FrozenRecord):
    """One planned artificial gap: a duration class and where it starts.

    ``start_day`` is elapsed days from the first timestamp, so a plan is
    independent of the record's start date and of its cadence. The duration
    itself comes from the scenario configuration, which is where the paper's
    24 h / 7 d / 30 d live (method_spec.md section 4.1).
    """

    gap_class: GapClass | str
    start_day: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "gap_class", GapClass.coerce(self.gap_class))
        if isinstance(self.start_day, bool) or not isinstance(self.start_day, (int, float)):
            raise ConfigError(f"KnownGap.start_day must be a number, got {self.start_day!r}")
        offset = float(self.start_day)
        if not math.isfinite(offset) or offset < 0.0:
            raise ConfigError(
                f"KnownGap.start_day must be a non-negative number of days, got {self.start_day!r}"
            )
        object.__setattr__(self, "start_day", offset)

    @property
    def duration_class(self) -> GapClass:
        """The validated :class:`~rfrgapfill.schema.GapClass`."""
        assert isinstance(self.gap_class, GapClass)
        return self.gap_class


#: The reference plan: one 30-day gap, two 7-day gaps and three 24-hour gaps,
#: spread across the seasons of a one-year record and never overlapping. Needs a
#: record of at least 336 days; pass your own plan for anything shorter.
#:
#: The offsets are chosen so that every interval clears the scenario's
#: ``min_observed_fraction`` on the reference site. That is a property of this
#: plan against *those* outages, not a guarantee: a different ``seed`` or a
#: larger ``real_gap_fraction`` can drop an interval below it, which the
#: manifest reports per gap as ``observed_fraction`` rather than rejecting.
DEFAULT_KNOWN_GAPS: Final[tuple[KnownGap, ...]] = (
    KnownGap(GapClass.VERY_LONG, 45.0),
    KnownGap(GapClass.LONG, 120.0),
    KnownGap(GapClass.SHORT, 200.0),
    KnownGap(GapClass.LONG, 250.0),
    KnownGap(GapClass.SHORT, 300.0),
    KnownGap(GapClass.SHORT, 335.0),
)


def known_gap_manifest(
    data: pd.DataFrame,
    *,
    time_step: timedelta | str,
    targets: Sequence[str] = TARGETS,
    qc_columns: Mapping[str, str] | None = None,
    plan: Sequence[KnownGap] = DEFAULT_KNOWN_GAPS,
    scenario: GapScenarioConfig | None = None,
    observed_qc_values: Sequence[int] = DEFAULT_OBSERVED_QC_VALUES,
    random_state: int = DEFAULT_SEED,
) -> GapManifest:
    """Return a :class:`~rfrgapfill.gaps.GapManifest` for a fixed set of intervals.

    Nothing is sampled. Each :class:`KnownGap` becomes the half-open interval
    ``[start, start + duration)`` at the duration its class carries in
    ``scenario``, so a 24-hour gap spans 24 elapsed hours, a 7-day gap 7 days and
    a 30-day gap 30 days, whatever rows the frame happens to hold there. The
    result is an ordinary manifest: it produces the holdout mask for
    :mod:`rfrgapfill.leakage`, groups metrics by gap class, and goes into a run
    manifest unchanged.

    Availability follows the generator's rule - a row counts only where **every**
    target in ``targets`` is genuinely observed - so one set of locations is a
    fair test set for NEE, H and LE together (method_spec.md section 4.4).

    Because the design is stated as intervals rather than as a withheld fraction,
    the recorded scenario carries the fraction and mix these intervals actually
    withhold, and ``fraction_error`` and ``mix_errors`` are zero here by
    construction: there was no request to fall short of. A manifest from
    :class:`~rfrgapfill.gaps.GapScenarioGenerator` is the one whose A7 tolerances
    mean something.

    :raises GapError: if an interval runs past the end of the record, if two
        intervals overlap while the scenario forbids it, or if the frame holds no
        observation for the plan to withhold.
    """
    index = as_datetime_index(data.index, field_name="the data index")
    if index.size == 0:
        raise GapError("cannot place known gaps on an empty frame")
    step = to_timedelta(time_step, field_name="time_step")
    base = GapScenarioConfig() if scenario is None else scenario
    names = tuple(targets)
    if not names:
        raise GapError("at least one target is required to measure what a gap withholds")
    if not plan:
        raise GapError("the known-gap plan is empty; give at least one KnownGap")

    lookup = dict(qc_columns or {})
    available = pd.Series(np.ones(index.size, dtype=bool), index=index)
    for target in names:
        available &= observed_target_mask(
            data, target, qc_column=lookup.get(target), observed_qc_values=observed_qc_values
        ).to_numpy()
    n_available = int(available.sum())
    if n_available == 0:
        raise GapError(
            f"no genuinely observed value is common to targets {', '.join(names)}, "
            "so a known gap would withhold nothing"
        )

    origin = index[0]
    horizon = index[-1] + pd.Timedelta(step)
    placed: list[tuple[pd.Timestamp, pd.Timestamp, GapClass, int, int, int]] = []
    for entry in plan:
        gap_class = entry.duration_class
        duration = base.duration(gap_class)
        start = origin + pd.Timedelta(days=entry.start_day)
        end = start + pd.Timedelta(duration)
        if end > horizon:
            raise GapError(
                f"the {gap_class.value} gap planned at day {entry.start_day:g} ends at {end}, "
                f"past the end of a record that runs to {horizon}; move the gap or generate "
                "a longer record"
            )
        rows = interval_mask(index, start, duration).to_numpy()
        placed.append(
            (
                start,
                end,
                gap_class,
                duration_to_periods(duration, step, field_name=f"durations[{gap_class.value!r}]"),
                int(rows.sum()),
                int((rows & available.to_numpy()).sum()),
            )
        )

    placed.sort(key=lambda item: (item[0], item[1]))
    if not base.allow_overlap:
        for earlier, later in pairwise(placed):
            if later[0] < earlier[1]:
                raise GapError(
                    f"known gaps overlap: the {earlier[2].value} gap [{earlier[0]}, {earlier[1]}) "
                    f"runs into the {later[2].value} gap starting {later[0]}; move one, or set "
                    "allow_overlap=True on the scenario"
                )

    counter: dict[GapClass, int] = dict.fromkeys(GapClass, 0)
    gaps: list[ArtificialGap] = []
    for start, end, gap_class, expected, rows_present, observed in placed:
        counter[gap_class] += 1
        gaps.append(
            ArtificialGap(
                gap_id=f"{gap_class.value}-{counter[gap_class]}",
                gap_class=gap_class,
                start=start,
                end=end,
                n_expected=expected,
                n_rows=rows_present,
                n_observed_before_masking=observed,
            )
        )

    withheld = np.zeros(index.size, dtype=bool)
    for gap in gaps:
        withheld |= interval_mask(index, gap.start, end=gap.end).to_numpy()
    n_withheld_rows = int(withheld.sum())
    n_withheld_observed = int((withheld & available.to_numpy()).sum())
    if n_withheld_observed == 0:
        raise GapError(
            "the known-gap plan withholds no observed value; every planned interval falls on "
            "rows that are missing or flagged as already gap-filled"
        )

    observed_by_class = {
        gap_class: sum(gap.n_observed_before_masking for gap in gaps if gap.gap_class is gap_class)
        for gap_class in GapClass
    }
    basis = base.basis
    allocation = GapAllocation(
        basis=basis,
        n_available=n_available,
        target_records=n_withheld_observed,
        events=dict(counter),
        expected_records_per_event={
            gap_class: duration_to_periods(
                base.duration(gap_class), step, field_name=f"durations[{gap_class.value!r}]"
            )
            for gap_class in GapClass
        },
    )
    return GapManifest(
        gaps=tuple(gaps),
        allocation=allocation,
        scenario=base.replace(
            missing_fraction=n_withheld_observed / n_available,
            gap_mix=_exact_shares(
                dict(counter) if basis is AllocationBasis.GAP_EVENTS else observed_by_class
            ),
            allocation_basis=basis,
        ),
        n_available=n_available,
        n_withheld_rows=n_withheld_rows,
        n_withheld_observed=n_withheld_observed,
        time_step=step,
        random_state=int(random_state),
        targets=names,
    )


def _exact_shares(counts: Mapping[GapClass, int]) -> dict[GapClass, float]:
    """Return ``counts`` as shares summing to exactly 1.0.

    ``GapScenarioConfig`` rejects a mix that does not sum to one, and three
    independently rounded ratios need not. The largest share absorbs the
    remainder, so the correction lands where it is proportionally smallest.
    """
    total = sum(counts.values())
    if total <= 0:
        raise GapError("cannot express a gap mix: the plan withholds nothing")
    shares = {gap_class: count / total for gap_class, count in counts.items()}
    largest = max(shares, key=lambda gap_class: (shares[gap_class], gap_class.value))
    shares[largest] += 1.0 - math.fsum(shares.values())
    return shares
