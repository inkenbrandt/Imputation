"""Historical ``fluxlib`` compatibility: ``feature_mode="legacy_fluxlib"``.

Reproduces the receptive-limiter feature derivation of the paper-era ``fluxlib``
code that Zhu et al. (2022) cite as their implementation, so that a run can
measure how far that historical setup and this package's leakage-safe default
lie apart. It is **not** the paper-faithful default and never becomes one:
``docs/fluxlib_audit.md`` records what was found, how each difference is
classified, and why ``paper_safe`` stays the default.

Source: https://github.com/soonyenju/fluxlib, release 0.0.23, commit ``da53256``
(2021-08-04), ``fluxlib/gapfill/ggapfill.py`` - ``GFiller.set_stats``,
``set_season_tag``, ``set_rg_tag`` and ``set_doy_year_tag``, in the order
``run_filling_pipeline`` calls them. It is the last change to the gap-filling code
before the article appeared.

What the historical derivation does, all of it reproduced here:

* **Daily statistics before the split.** ``set_stats`` runs on the whole frame
  before the artificial-gap train/test indices are applied, so held-out values
  contribute to the daily statistics of their own days. This answers ambiguity
  A6; :mod:`rfrgapfill.leakage` reproduces it in this mode only, and warns.
* **Seven daily statistics, not four:** max, min, mean, standard deviation
  (``ddof=1``) and the 25th, 50th and 75th percentiles (linear interpolation).
* **Linear interpolation of the target first.** Every gap in the target - real
  ones and QC-rejected values alike - is linearly interpolated before the daily
  statistics are taken, so a day inside a gap is never without statistics.
* **A one-day look-ahead in the join.** The daily values are upsampled with
  ``resample(...).bfill()``: the row at exactly midnight carries its own day's
  statistics and every other row carries the *next* day's. Rows after the last
  midnight of the series are filled by the closing interpolation, which repeats
  the last day's values.
* **Calendar features instead of elapsed hours:** day of year and year.
  ``set_hour_diff`` existed but ``run_filling_pipeline`` never called it.
* **Integer tags:** a season code 1-4 (winter, spring, summer, autumn of the
  site's hemisphere) and a radiation rank 1-3 (weak, medium, strong), with ``0``
  for a value lying *exactly* on a threshold and for a missing one.

Two parts of the historical pipeline are deliberately **not** reproduced, because
they fabricate predictors, which ``docs/method_spec.md`` section 7 forbids in every
mode. The closing ``df.interpolate()`` of ``set_stats`` also interpolated the
*drivers*, and prediction rows were ``.interpolate().bfill()``-ed; on the
pre-filled FLUXNET2015 drivers the paper used, the first changes nothing. Rows
before the first visible target value therefore keep missing daily statistics
here, where ``fluxlib`` back-filled them at prediction time.

The feature columns carry their own names (``legacy_season``, ``<target>_legacy_p50``
and so on), disjoint from the ``paper_safe`` ones, so a model fitted in one mode
rejects a matrix built in the other instead of silently mixing them.
"""

from __future__ import annotations

from typing import Any, Final

import numpy as np
import pandas as pd

from rfrgapfill.config import DEFAULT_RADIATION_THRESHOLDS, RFRConfig
from rfrgapfill.features import _as_boolean_mask, _as_float_series, _time_index
from rfrgapfill.schema import ConfigError, Hemisphere

__all__ = [
    "LEGACY_CALENDAR_FEATURES",
    "LEGACY_DAILY_STATISTIC_RULE",
    "LEGACY_DAY_OF_YEAR",
    "LEGACY_FLUXLIB_SOURCE",
    "LEGACY_RADIATION_RANK",
    "LEGACY_RADIATION_RULE",
    "LEGACY_SEASON",
    "LEGACY_STATISTIC_SUFFIXES",
    "LEGACY_YEAR",
    "LegacyFluxlibWarning",
    "build_legacy_features",
    "describe_legacy_features",
    "legacy_calendar",
    "legacy_daily_statistics",
    "legacy_feature_names",
    "legacy_radiation_rank",
    "legacy_season_code",
    "legacy_statistic_names",
]


class LegacyFluxlibWarning(UserWarning):
    """Emitted when validation features are built in ``legacy_fluxlib`` mode.

    The mode reproduces a derivation in which held-out truth reaches the daily
    statistics used to predict it, so every score it produces is optimistic by
    construction. The warning is the "never silently" half of method_spec.md 3.5.
    """


#: Where the reproduced behaviour comes from, recorded in every description.
LEGACY_FLUXLIB_SOURCE: Final = {
    "repository": "https://github.com/soonyenju/fluxlib",
    "release": "0.0.23",
    "commit": "da53256",
    "date": "2021-08-04",
    "module": "fluxlib/gapfill/ggapfill.py",
    "functions": ["set_stats", "set_season_tag", "set_rg_tag", "set_doy_year_tag"],
    "audit": "docs/fluxlib_audit.md",
}

#: Suffixes of the seven daily statistics, in ``set_stats`` order, each paired
#: with the pandas reduction behind it.
_LEGACY_STATISTICS: Final[tuple[tuple[str, str], ...]] = (
    ("_legacy_max", "max"),
    ("_legacy_min", "min"),
    ("_legacy_mean", "mean"),
    ("_legacy_std", "std"),
    ("_legacy_p25", "p25"),
    ("_legacy_p50", "p50"),
    ("_legacy_p75", "p75"),
)

#: Suffixes of the seven legacy daily statistics, in ``set_stats`` order.
LEGACY_STATISTIC_SUFFIXES: Final[tuple[str, ...]] = tuple(
    suffix for suffix, _ in _LEGACY_STATISTICS
)

#: Column name of the legacy season code (``set_season_tag``).
LEGACY_SEASON: Final = "legacy_season"
#: Column name of the legacy radiation rank (``set_rg_tag``).
LEGACY_RADIATION_RANK: Final = "legacy_rg_rank"
#: Column name of the legacy day of year (``set_doy_year_tag``).
LEGACY_DAY_OF_YEAR: Final = "legacy_doy"
#: Column name of the legacy calendar year (``set_doy_year_tag``).
LEGACY_YEAR: Final = "legacy_year"

#: The non-target legacy features, in ``run_filling_pipeline`` order.
LEGACY_CALENDAR_FEATURES: Final[tuple[str, ...]] = (
    LEGACY_SEASON,
    LEGACY_RADIATION_RANK,
    LEGACY_DAY_OF_YEAR,
    LEGACY_YEAR,
)

#: ``set_rg_tag`` in words, for manifests (ambiguity A2).
LEGACY_RADIATION_RULE: Final = (
    "fluxlib set_rg_tag: x < 10 -> 1, 10 < x < 100 -> 2, x > 100 -> 3; a value exactly "
    "on 10 or 100, or missing, -> 0"
)

#: ``set_stats`` in words, for manifests (ambiguities A4 and A6).
LEGACY_DAILY_STATISTIC_RULE: Final = (
    "fluxlib set_stats: the target is linearly interpolated across every gap, the daily "
    "max, min, mean, std, p25, p50 and p75 are taken per calendar day, and each row "
    "receives the statistics of the day its timestamp rounds up to (midnight keeps its "
    "own day, every other row takes the next day's). In validation the statistics are "
    "computed before the holdout is hidden, so held-out values reach them."
)


def legacy_statistic_names(target: str) -> tuple[str, ...]:
    """Return the seven legacy daily-statistic column names for ``target``, in order."""
    if not isinstance(target, str) or not target.strip():
        raise ConfigError(f"target must be a non-empty string, got {target!r}")
    return tuple(f"{target}{suffix}" for suffix in LEGACY_STATISTIC_SUFFIXES)


def legacy_feature_names(target: str | None) -> tuple[str, ...]:
    """Return the legacy receptive-limiter columns, in ``run_filling_pipeline`` order.

    ``stat_tags + season_tag + rg_tag + doy_year_tag``: the daily statistics first,
    then the calendar features. Without a target only the calendar features can be
    named, mirroring :func:`rfrgapfill.features.receptive_limiter_features`.
    """
    statistics = () if target is None else legacy_statistic_names(target)
    return statistics + LEGACY_CALENDAR_FEATURES


# ---------------------------------------------------------------------------
# The transformers
# ---------------------------------------------------------------------------


def legacy_daily_statistics(
    target: object,
    *,
    available_mask: object | None = None,
    target_name: str | None = None,
) -> pd.DataFrame:
    """Return ``fluxlib``'s seven daily target statistics, joined back to every row.

    A transcription of ``GFiller.set_stats`` (release 0.0.23):

    1. values outside ``available_mask`` are treated as missing, then the whole
       series is linearly interpolated over row position - ``Series.interpolate()``,
       which leaves leading gaps missing and repeats the last value over trailing
       ones;
    2. per calendar day, the max, min, mean, sample standard deviation and the
       linear 25th, 50th and 75th percentiles are taken;
    3. each row receives the statistics of the day its timestamp rounds *up* to,
       which is what ``resample(scale).bfill()`` of the daily values amounts to on
       any timestamp that falls on the quarter hour: a row at exactly midnight
       keeps its own day, every other row takes the following day's;
    4. the joined columns are interpolated once more, as the closing
       ``df.interpolate()`` of ``set_stats`` did.

    ``available_mask`` decides what the statistics may read. **This transformer
    does not protect against leakage** - in ``legacy_fluxlib`` mode the validation
    workflow deliberately passes the observed mask rather than the available one,
    because that is what the historical code did.
    """
    series = _as_float_series(target, field_name="target")
    index = _time_index(series, field_name="target")
    name = target_name if target_name is not None else series.name
    if not isinstance(name, str) or not name.strip():
        raise ConfigError(
            "the target's name is needed to name its daily-statistic features; pass "
            "target_name= or give the Series a name"
        )
    columns = legacy_statistic_names(name)

    visible = pd.Series(series.to_numpy(dtype=float), index=index)
    if available_mask is not None:
        mask = _as_boolean_mask(available_mask, index, field_name="available_mask")
        visible = visible.where(mask)
    # Step 1. Row position, not time: pandas' default linear method, as fluxlib ran it.
    interpolated = visible.interpolate()

    # Step 2.
    grouped = interpolated.groupby(index.normalize(), sort=True)
    reductions = {
        "max": grouped.max(),
        "min": grouped.min(),
        "mean": grouped.mean(),
        "std": grouped.std(ddof=1),
        "p25": grouped.quantile(0.25),
        "p50": grouped.quantile(0.50),
        "p75": grouped.quantile(0.75),
    }
    per_day = pd.DataFrame(
        {
            column: reductions[statistic]
            for column, (_, statistic) in zip(columns, _LEGACY_STATISTICS, strict=True)
        }
    )

    # Step 3: the one-day look-ahead. A day the series does not reach is missing.
    joined: pd.DataFrame = per_day.reindex(index.ceil("D"))
    joined.index = index
    # Step 4.
    joined = joined.interpolate()
    joined.index.name = series.index.name
    return joined


def legacy_season_code(timestamps: object, *, hemisphere: Hemisphere | str) -> pd.Series:
    """Return ``set_season_tag``'s season code: 1 winter, 2 spring, 3 summer, 4 autumn.

    ``(month % 12 + 3) // 3`` in the northern hemisphere and
    ``((month + 6) % 12 + 3) // 3`` in the southern (``isnorth=False``, added in
    release 0.0.23). The months each code covers are those of
    :func:`rfrgapfill.features.season_tag`, and the codes are its ordinal codes
    plus one.
    """
    resolved = Hemisphere.coerce(hemisphere)
    index = _time_index(timestamps, field_name="timestamps")
    months = np.asarray(index.month, dtype=int)
    shifted = months if resolved is Hemisphere.NORTH else months + 6
    codes = (shifted % 12 + 3) // 3
    tagged: pd.Series = pd.Series(codes.astype(float), index=index, name=LEGACY_SEASON)
    return tagged


def legacy_radiation_rank(
    shortwave: object,
    *,
    thresholds: tuple[float, float] = DEFAULT_RADIATION_THRESHOLDS,
) -> pd.Series:
    """Return ``set_rg_tag``'s radiation rank.

    ``x < 10`` is 1, ``10 < x < 100`` is 2 and ``x > 100`` is 3. Every comparison
    is strict, so a value exactly on 10 or 100 satisfies none of them and falls to
    ``np.select``'s default of 0 - as does a missing value, since ``NaN`` compares
    false against everything. Neither of this package's boundary conventions
    (ambiguity A2) behaves this way, which is why the rank is its own feature.
    """
    series = _as_float_series(shortwave, field_name="shortwave")
    values = series.to_numpy(dtype=float)
    low, high = thresholds
    ranks = np.select(
        [values < low, (values > low) & (values < high), values > high],
        [1.0, 2.0, 3.0],
        default=0.0,
    )
    ranked: pd.Series = pd.Series(ranks, index=series.index, name=LEGACY_RADIATION_RANK)
    return ranked


def legacy_calendar(timestamps: object) -> pd.DataFrame:
    """Return ``set_doy_year_tag``'s day of year and calendar year, as floats."""
    index = _time_index(timestamps, field_name="timestamps")
    calendar: pd.DataFrame = pd.DataFrame(
        {
            LEGACY_DAY_OF_YEAR: np.asarray(index.dayofyear, dtype=float),
            LEGACY_YEAR: np.asarray(index.year, dtype=float),
        },
        index=index,
    )
    return calendar


def build_legacy_features(
    target: pd.Series,
    shortwave: pd.Series,
    *,
    hemisphere: Hemisphere | str,
    target_name: str,
    available_mask: object | None = None,
    thresholds: tuple[float, float] = DEFAULT_RADIATION_THRESHOLDS,
) -> pd.DataFrame:
    """Return every legacy receptive-limiter feature, in :func:`legacy_feature_names` order.

    Called by :func:`rfrgapfill.features.build_feature_matrix` when
    ``feature_mode="legacy_fluxlib"``; the drivers themselves are added there.
    """
    index = _time_index(target, field_name="target")
    statistics = legacy_daily_statistics(
        target, available_mask=available_mask, target_name=target_name
    )
    calendar = legacy_calendar(index)
    features = pd.DataFrame(index=index)
    for name in statistics.columns:
        features[name] = statistics[name].to_numpy(dtype=float)
    features[LEGACY_SEASON] = legacy_season_code(index, hemisphere=hemisphere).to_numpy()
    features[LEGACY_RADIATION_RANK] = legacy_radiation_rank(
        shortwave, thresholds=thresholds
    ).to_numpy()
    for name in calendar.columns:
        features[name] = calendar[name].to_numpy()
    assert tuple(features.columns) == legacy_feature_names(target_name)
    assembled: pd.DataFrame = features
    return assembled


def describe_legacy_features(config: RFRConfig) -> dict[str, Any]:
    """Return the manifest description of the legacy feature stage."""
    return {
        "source": dict(LEGACY_FLUXLIB_SOURCE),
        "hemisphere": config.resolve_hemisphere().value,
        "hemisphere_source": config.hemisphere_source,
        "radiation_thresholds": list(config.features.radiation_thresholds),
        "radiation_rule": LEGACY_RADIATION_RULE,
        "radiation_rank_codes": {"unclassified": 0, "weak": 1, "medium": 2, "strong": 3},
        "season_codes": {"winter": 1, "spring": 2, "summer": 3, "autumn": 4},
        "daily_statistics": [statistic for _, statistic in _LEGACY_STATISTICS],
        "daily_statistic_rule": LEGACY_DAILY_STATISTIC_RULE,
        "daily_std_ddof": 1,
        "quantile_interpolation": "linear",
        "time_features": [LEGACY_DAY_OF_YEAR, LEGACY_YEAR],
        "held_out_truth_in_features": True,
        "not_reproduced": [
            "linear interpolation of the drivers",
            "back-filling of prediction rows before the first visible target value",
        ],
    }
