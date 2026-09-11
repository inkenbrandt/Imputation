"""Shared synthetic fixtures for the test suite.

The generators here produce half-hourly flux data with real diurnal and seasonal
structure, so an end-to-end test exercises a model that has something learnable
to find rather than noise. Everything is seeded: the same call gives the same
frame, which is what lets a test assert on a metric at all.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["FLUXNET_TARGETS", "synthetic_site"]

#: Target columns the synthetic site carries, matching the paper's three fluxes.
FLUXNET_TARGETS = ("NEE", "H", "LE")


def synthetic_site(
    *,
    days: int = 400,
    seed: int = 0,
    missing_fraction: float = 0.12,
    start: str = "2019-01-01",
    freq: str = "30min",
) -> pd.DataFrame:
    """Return a synthetic half-hourly site with FLUXNET2015 column names.

    Drivers carry a diurnal cycle and a seasonal cycle; the three fluxes are
    smooth functions of them plus noise, so a Random Forest can learn them and a
    validation run produces metrics that mean something. ``missing_fraction`` of
    each target is then removed and flagged ``1`` in its ``<target>_QC`` column,
    standing in for the pre-existing real gaps every eddy-covariance series has.
    """
    rng = np.random.default_rng(seed)
    index = pd.date_range(start, periods=days * 48, freq=freq, name="timestamp")
    n = len(index)
    hour = index.hour + index.minute / 60
    day_of_year = index.dayofyear

    shortwave = np.clip(800 * np.sin(np.pi * (hour - 5) / 14), 0, None) * (
        0.7 + 0.3 * np.sin(2 * np.pi * day_of_year / 365)
    )
    shortwave = np.where((hour < 5) | (hour > 19), 0.0, shortwave)
    shortwave = shortwave + rng.normal(0, 5, n).clip(0)

    air_temperature = (
        12
        + 10 * np.sin(2 * np.pi * (day_of_year - 100) / 365)
        + 5 * np.sin(np.pi * (hour - 6) / 12)
        + rng.normal(0, 1, n)
    )
    humidity = np.clip(
        80 - 0.03 * shortwave - 0.8 * (air_temperature - 12) + rng.normal(0, 4, n), 10, 100
    )
    vpd = np.clip(
        0.61 * np.exp(17.5 * air_temperature / (air_temperature + 241)) * (1 - humidity / 100) * 10,
        0,
        None,
    )
    net_radiation = 0.75 * shortwave - 40 + rng.normal(0, 8, n)
    soil_heat = 0.08 * net_radiation + rng.normal(0, 4, n)
    soil_temperature = (
        12 + 8 * np.sin(2 * np.pi * (day_of_year - 120) / 365) + rng.normal(0, 0.5, n)
    )

    latent = np.clip(
        0.45 * (net_radiation - soil_heat) * (1 - np.exp(-vpd / 8)) + rng.normal(0, 12, n),
        -20,
        None,
    )
    sensible = (
        0.30 * (net_radiation - soil_heat) + 1.5 * (air_temperature - 12) + rng.normal(0, 15, n)
    )
    uptake = 0.04 * shortwave * np.clip(1 - ((air_temperature - 22) / 18) ** 2, 0, 1)
    respiration = 1.2 * np.exp(0.07 * (soil_temperature - 10))
    net_exchange = respiration - uptake + rng.normal(0, 1.2, n)

    frame = pd.DataFrame(
        {
            "SW_IN_F": shortwave,
            "VPD_F_MDS": vpd,
            "TA_F_MDS": air_temperature,
            "NETRAD": net_radiation,
            "WS": np.clip(2 + rng.normal(0, 1, n), 0.1, None),
            "WD": rng.uniform(0, 360, n),
            "G_F_MDS": soil_heat,
            "TS_F_MDS": soil_temperature,
            "RH": humidity,
            "SWC_F_MDS": np.clip(
                30 + 5 * np.sin(2 * np.pi * (day_of_year - 60) / 365) + rng.normal(0, 1, n), 5, 60
            ),
            "NEE": net_exchange,
            "H": sensible,
            "LE": latent,
        },
        index=index,
    )

    for target in FLUXNET_TARGETS:
        flags = np.zeros(n, dtype=int)
        dropped = rng.random(n) < missing_fraction
        flags[dropped] = 1
        frame.loc[dropped, target] = np.nan
        frame[f"{target}_QC"] = flags
    return frame
