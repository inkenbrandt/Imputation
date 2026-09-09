"""Receptive-limiter feature engineering.

The receptive limiter is the paper's feature-engineering stage and the only thing
that distinguishes RFR from the ORF benchmark (``docs/method_spec.md`` section 3,
Supplementary Figure S1). It is implemented here as four independent, pure
transformers plus one assembler:

============================  ===========================================
:func:`radiation_tag`         shortwave radiation -> weak/medium/strong
:func:`time_distance_hours`   elapsed hours since the series origin
:func:`season_tag`            calendar month + hemisphere -> season
:func:`daily_flux_statistics` per-day target Q1/Q2/Q3/std, joined back
:func:`build_feature_matrix`  drivers + the above, in a deterministic order
============================  ===========================================

Every transformer is a function of its inputs alone: no fitted state, no global
configuration, no hidden imputation. Missing input yields missing output rather
than a default class, so a row with an incomplete predictor is visibly
incomplete and can be excluded from training by the model layer instead of being
silently filled with a fabricated value (method_spec.md section 7).

Two properties this module must keep, both of which are pinned by tests:

* **Deterministic feature order.** :func:`feature_names` returns the column order
  for a configuration and target without needing any data, and
  :func:`build_feature_matrix` is required to reproduce it exactly. The order is
  part of the run manifest and of every serialised model.
* **Leakage safety.** :func:`daily_flux_statistics` derives features *from the
  target*, so it takes an ``available_mask`` naming the observations visible to
  the model. Held-out truth never reaches a feature used to predict it
  (method_spec.md section 3.5). Passing the mask is the caller's job;
  :mod:`rfrgapfill.leakage` is the workflow that wires it up for artificial-gap
  validation, and the one place that should be building validation features.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import Enum
from typing import Any, Final

import numpy as np
import pandas as pd

from rfrgapfill.config import (
    BoundaryConvention,
    DailyStatisticStrategy,
    FeatureConfig,
    RFRConfig,
)
from rfrgapfill.schema import SHORTWAVE, ColumnMap, ConfigError, Hemisphere
from rfrgapfill.time import TIME_DISTANCE_HOURS, as_datetime_index, elapsed_hours

__all__ = [
    "DAILY_STATISTIC_SUFFIXES",
    "RADIATION_CATEGORY",
    "SEASON",
    "SEASONS_BY_HEMISPHERE",
    "TIME_DISTANCE_HOURS",
    "FeatureError",
    "RadiationClass",
    "Season",
    "build_feature_matrix",
    "daily_flux_statistics",
    "daily_statistic_names",
    "describe_features",
    "feature_names",
    "radiation_tag",
    "receptive_limiter_features",
    "season_tag",
    "time_distance_hours",
]


class FeatureError(ValueError):
    """Raised when input data cannot support a feature transformation.

    Reports invalid *data* - a non-numeric driver column, a target that is not on
    a time axis, a mask that does not line up with its series. An invalid
    *configuration* raises :class:`~rfrgapfill.schema.ConfigError` instead.
    """


# ---------------------------------------------------------------------------
# Feature vocabulary
# ---------------------------------------------------------------------------

#: Column name of the radiation-category feature (method_spec.md 3.1).
RADIATION_CATEGORY: Final = "radiation_category"

#: Column name of the hemisphere-aware season feature (method_spec.md 3.3).
SEASON: Final = "season"

#: Suffixes of the four daily target statistics, in specification order
#: (method_spec.md 3.4). Prefixed with the target name to form column names.
DAILY_STATISTIC_SUFFIXES: Final[tuple[str, ...]] = (
    "_daily_q1",
    "_daily_q2",
    "_daily_q3",
    "_daily_std",
)

#: Quantiles behind ``_daily_q1``, ``_daily_q2`` and ``_daily_q3``.
_DAILY_QUANTILES: Final[tuple[float, float, float]] = (0.25, 0.50, 0.75)


class RadiationClass(str, Enum):
    """Shortwave radiation category (method_spec.md 3.1).

    Ordered weak < medium < strong, which is the order the ordinal encoding in
    :func:`build_feature_matrix` uses (0, 1, 2).
    """

    WEAK = "weak"
    MEDIUM = "medium"
    STRONG = "strong"


class Season(str, Enum):
    """Season tag (method_spec.md 3.3).

    The declaration order below is the fixed categorical order and therefore the
    ordinal encoding (winter 0, spring 1, summer 2, autumn 3). It is a stable
    labelling, not a claim that seasons are ordered: it exists so that a
    serialised model and a later prediction frame agree on the codes.
    """

    WINTER = "winter"
    SPRING = "spring"
    SUMMER = "summer"
    AUTUMN = "autumn"


#: Ordered categories of the radiation feature, weak -> strong.
_RADIATION_CATEGORIES: Final[tuple[str, ...]] = tuple(member.value for member in RadiationClass)

#: Categories of the season feature, in fixed encoding order.
_SEASON_CATEGORIES: Final[tuple[str, ...]] = tuple(member.value for member in Season)

#: Dtypes the two categorical features always carry, whichever values appear in
#: the data: a summer-only frame still declares all four seasons, so its ordinal
#: codes agree with those a fitted model was trained on.
_RADIATION_DTYPE: Final = pd.CategoricalDtype(list(_RADIATION_CATEGORIES), ordered=True)
_SEASON_DTYPE: Final = pd.CategoricalDtype(list(_SEASON_CATEGORIES), ordered=False)


def _season_months(**groups: tuple[int, ...]) -> Mapping[int, Season]:
    """Return a month -> season lookup from three-month groups keyed by season name.

    The assertion is a spelling guard on the tables below: every calendar month
    must be claimed exactly once, so a duplicated or dropped month is a startup
    failure rather than a silently mislabelled season.
    """
    lookup: dict[int, Season] = {}
    for name, months in groups.items():
        for month in months:
            assert month not in lookup, f"month {month} is claimed twice"
            lookup[month] = Season(name)
    assert sorted(lookup) == list(range(1, 13)), "every calendar month must map to a season"
    return lookup


#: Month -> season per hemisphere, written out rather than derived arithmetically
#: so it can be read straight off the table in method_spec.md section 3.3.
SEASONS_BY_HEMISPHERE: Final[Mapping[Hemisphere, Mapping[int, Season]]] = {
    Hemisphere.NORTH: _season_months(
        winter=(12, 1, 2),
        spring=(3, 4, 5),
        summer=(6, 7, 8),
        autumn=(9, 10, 11),
    ),
    Hemisphere.SOUTH: _season_months(
        winter=(6, 7, 8),
        spring=(9, 10, 11),
        summer=(12, 1, 2),
        autumn=(3, 4, 5),
    ),
}


# ---------------------------------------------------------------------------
# Input coercion helpers
# ---------------------------------------------------------------------------


def _as_float_series(values: object, *, field_name: str) -> pd.Series:
    """Return ``values`` as a float :class:`pandas.Series`, preserving any index."""
    if isinstance(values, pd.Series):
        series = values
    elif isinstance(values, pd.Index):
        series = pd.Series(values.to_numpy(), index=values)
    elif isinstance(values, (np.ndarray, Sequence)) and not isinstance(values, (str, bytes)):
        series = pd.Series(list(values))
    else:
        raise FeatureError(
            f"{field_name} must be a Series, Index, array or sequence of numbers, "
            f"got {type(values).__name__}"
        )
    # Converted unconditionally rather than dtype-tested first: `np.issubdtype`
    # raises on pandas extension dtypes, and a list of strings arrives as
    # StringDtype under pandas 3. The conversion itself is the check.
    try:
        numeric: pd.Series = series.astype(float)
    except (TypeError, ValueError) as exc:
        raise FeatureError(
            f"{field_name} must be numeric; it could not be read as floats ({exc})"
        ) from None
    return numeric


def _as_boolean_mask(mask: object, index: pd.Index, *, field_name: str) -> np.ndarray:
    """Return ``mask`` as a boolean array aligned to ``index``.

    A :class:`pandas.Series` is reindexed onto ``index`` so alignment is by label,
    not by position; a label the mask does not cover is treated as *not
    available*, which is the conservative direction for a leakage guard. A plain
    array must already match ``index`` in length.
    """
    if isinstance(mask, pd.Series):
        if not pd.api.types.is_bool_dtype(mask.dtype):
            raise FeatureError(f"{field_name} must be boolean, got dtype {mask.dtype}")
        # `.eq(True)` rather than `.fillna(False)`: reindexing onto a label the
        # mask does not cover leaves NaN, and comparing is both simpler and free
        # of pandas' downcasting behaviour on object columns.
        return np.asarray(mask.reindex(index).eq(True).to_numpy(), dtype=bool)
    array = np.asarray(mask)
    if array.dtype != np.bool_:
        raise FeatureError(f"{field_name} must be boolean, got dtype {array.dtype}")
    if array.shape != (len(index),):
        raise FeatureError(
            f"{field_name} has shape {array.shape} but the series has {len(index)} rows; "
            "pass a boolean Series to align by timestamp instead"
        )
    return array


def _time_index(values: object, *, field_name: str) -> pd.DatetimeIndex:
    """Return the :class:`pandas.DatetimeIndex` behind ``values``.

    Accepts timestamps directly or an object carrying them on its index, so
    ``season_tag(df.index)`` and ``season_tag(series)`` both work.
    """
    if isinstance(values, (pd.Series, pd.DataFrame)):
        return as_datetime_index(values.index, field_name=f"{field_name} index")
    return as_datetime_index(values, field_name=field_name)


# ---------------------------------------------------------------------------
# 3.1 Radiation category
# ---------------------------------------------------------------------------


def radiation_tag(
    shortwave: object,
    *,
    thresholds: tuple[float, float] | None = None,
    convention: BoundaryConvention | str | None = None,
    config: FeatureConfig | None = None,
) -> pd.Series:
    """Return the radiation category of each shortwave value (method_spec.md 3.1).

    The paper gives the thresholds - weak below 10 W m-2, medium 10-100 W m-2,
    strong above 100 W m-2 - but its prose does not settle what happens *at*
    exactly 10 and 100 (ambiguity A2). The convention is therefore explicit and
    configurable, and the bins are inclusive and exhaustive over the reals:

    ``medium_inclusive`` (the documented default)
        ``x < 10`` weak, ``10 <= x <= 100`` medium, ``100 < x`` strong.
    ``medium_exclusive``
        ``x <= 10`` weak, ``10 < x < 100`` medium, ``100 <= x`` strong.

    Missing shortwave yields a **missing category**, never a default class: a row
    whose radiation is unknown is not quietly declared weak.

    Thresholds and convention come from ``config`` when given, and otherwise from
    the explicit arguments, and otherwise from :class:`FeatureConfig` defaults.
    Passing both ``config`` and an explicit override is rejected rather than
    silently resolved, so no result can disagree with the manifest that describes
    it.

    :returns: an ordered categorical Series (``weak`` < ``medium`` < ``strong``)
        carrying the index of ``shortwave`` and named
        :data:`RADIATION_CATEGORY`.
    """
    if config is not None and (thresholds is not None or convention is not None):
        raise ConfigError(
            "pass either config= or explicit thresholds/convention, not both: "
            "two sources of the same setting cannot be reconciled in the run manifest"
        )
    if config is None:
        config = FeatureConfig(
            radiation_thresholds=(
                FeatureConfig().radiation_thresholds if thresholds is None else thresholds
            ),
            boundary_convention=(
                FeatureConfig().boundary_convention if convention is None else convention
            ),
        )
    low, high = config.radiation_thresholds

    series = _as_float_series(shortwave, field_name="shortwave")
    values = series.to_numpy(dtype=float)

    if config.convention is BoundaryConvention.MEDIUM_INCLUSIVE:
        is_weak = values < low
        is_strong = values > high
    else:
        is_weak = values <= low
        is_strong = values >= high
    # NaN compares False against every bound, so it falls through both branches
    # and is excluded explicitly rather than landing in `medium`.
    known = ~np.isnan(values)

    codes = np.full(values.shape, -1, dtype=np.int8)
    codes[known & is_weak] = 0
    codes[known & ~is_weak & ~is_strong] = 1
    codes[known & is_strong] = 2

    categorical = pd.Categorical.from_codes(codes, dtype=_RADIATION_DTYPE)
    tagged: pd.Series = pd.Series(categorical, index=series.index, name=RADIATION_CATEGORY)
    return tagged


# ---------------------------------------------------------------------------
# 3.2 Time distance
# ---------------------------------------------------------------------------


def time_distance_hours(
    timestamps: object,
    *,
    origin: pd.Timestamp | str | None = None,
) -> pd.Series:
    """Return hours elapsed since the start of the series (method_spec.md 3.2).

    ``time_distance_hours = (timestamp - first_timestamp) / 1 hour``, computed
    from timestamp differences and never from row position, so a row after a
    week-long gap carries its true elapsed distance. At 30-minute cadence a
    complete series yields 0.0, 0.5, 1.0, ... The feature's stated purpose is to
    represent gradual ecosystem growth, degradation and other long-term trends.

    ``origin`` defaults to the earliest timestamp given. **Pass the origin of the
    full site series whenever a subset is transformed on its own** - training
    rows, a prediction frame and an artificial-gap interval must all measure from
    the same zero, or the same timestamp would carry different feature values in
    each. :func:`build_feature_matrix` threads this through for you.

    A thin wrapper over :func:`rfrgapfill.time.elapsed_hours`, re-exported here so
    that the four receptive-limiter transformers can be read side by side; the
    elapsed-time arithmetic itself lives in one place.
    """
    index = _time_index(timestamps, field_name="timestamps")
    return elapsed_hours(index, origin=origin)


# ---------------------------------------------------------------------------
# 3.3 Season
# ---------------------------------------------------------------------------


def season_tag(
    timestamps: object,
    *,
    hemisphere: Hemisphere | str | None = None,
    latitude: float | None = None,
) -> pd.Series:
    """Return the season of each timestamp for the site's hemisphere (3.3).

    Seasons are three-month calendar groups, mirrored between hemispheres:

    ======================  ==========  ==========
    Months                  Northern    Southern
    ======================  ==========  ==========
    Dec, Jan, Feb           ``winter``  ``summer``
    Mar, Apr, May           ``spring``  ``autumn``
    Jun, Jul, Aug           ``summer``  ``winter``
    Sep, Oct, Nov           ``autumn``  ``spring``
    ======================  ==========  ==========

    The hemisphere must be supplied: pass ``hemisphere="north"|"south"``, or a
    ``latitude`` to infer it by the documented rule ``latitude >= 0 -> north``
    (ambiguity A9 - a declared tie-break for equatorial sites, not a scientific
    claim). There is no silent default, because guessing it would mirror the
    seasonal signal of the model.

    :returns: a categorical Series named :data:`SEASON`, with the fixed category
        order of :class:`Season`.
    """
    if hemisphere is not None and latitude is not None:
        raise ConfigError(
            "pass either hemisphere= or latitude=, not both: an explicit hemisphere "
            "always overrides latitude, so supplying both hides which one applied"
        )
    if hemisphere is not None:
        resolved = Hemisphere.coerce(hemisphere)
    elif latitude is not None:
        resolved = Hemisphere.from_latitude(latitude)
    else:
        raise ConfigError(
            "season_tag needs a hemisphere: pass hemisphere='north'|'south' or "
            "latitude=<degrees>. The season feature is hemisphere-specific and has "
            "no defensible default (docs/method_spec.md section 3.3)."
        )

    index = _time_index(timestamps, field_name="timestamps")
    lookup = SEASONS_BY_HEMISPHERE[resolved]
    months = np.asarray(index.month, dtype=int)
    codes = np.asarray(
        [_SEASON_CATEGORIES.index(lookup[month].value) for month in range(1, 13)], dtype=np.int8
    )[months - 1]

    categorical = pd.Categorical.from_codes(codes, dtype=_SEASON_DTYPE)
    tagged: pd.Series = pd.Series(categorical, index=index, name=SEASON)
    return tagged


# ---------------------------------------------------------------------------
# 3.4 Daily target statistics
# ---------------------------------------------------------------------------


def daily_statistic_names(target: str) -> tuple[str, ...]:
    """Return the four daily-statistic column names for ``target``, in order."""
    if not isinstance(target, str) or not target.strip():
        raise ConfigError(f"target must be a non-empty string, got {target!r}")
    return tuple(f"{target}{suffix}" for suffix in DAILY_STATISTIC_SUFFIXES)


def daily_flux_statistics(
    target: object,
    *,
    available_mask: object | None = None,
    target_name: str | None = None,
    min_observations: int | None = None,
    strategy: DailyStatisticStrategy | str | None = None,
    fallback_window_days: int | None = None,
    ddof: int | None = None,
    config: FeatureConfig | None = None,
) -> pd.DataFrame:
    """Return per-day Q1, Q2, Q3 and std of the target, joined back to every row.

    Statistics are computed per **calendar day** from the target observations
    *visible to the model* and then broadcast to all timestamps of that day
    (method_spec.md 3.4). The paper's stated purpose is to reduce the effect of
    potential outliers. Because they are derived from the target, they are
    target-specific: there is a separate feature matrix per target.

    **This is the leakage-critical transformer** (method_spec.md 3.5, ambiguity
    A6). ``available_mask`` is a boolean Series or array marking the rows whose
    target value the model is allowed to see - quality-controlled observations,
    minus anything hidden by an artificial gap. Masked-out rows are excluded from
    the statistics exactly as if they were missing, so hidden truth cannot reach a
    feature used to predict it. With no mask the whole target is treated as
    visible, which is correct for operational filling of real gaps (where the
    missing values are genuinely absent) and wrong for artificial-gap validation
    (where the mask must be built first).

    A day with fewer than ``min_observations`` visible values is handled by the
    chosen :class:`~rfrgapfill.config.DailyStatisticStrategy`, since the paper
    does not say what it did with one (ambiguity A4):

    ``missing`` (the documented default)
        All four statistics are missing for that day. Nothing is imputed and
        nothing is borrowed; the affected rows are excluded from training and
        flagged at prediction time by the model layer.
    ``within_day_available``
        Whatever the day itself has is used, ignoring the minimum. Never looks
        outside the calendar day.
    ``neighbor_day_fallback``
        The day takes the statistics of the nearest day that meets the minimum,
        up to ``fallback_window_days`` away; ties resolve to the earlier day.
    ``rolling_available``
        The day is recomputed from the visible observations within
        ``+/- fallback_window_days`` calendar days, and stays missing if that
        pool is still below the minimum.

    All four are leakage safe: each of them draws only on values ``available_mask``
    admits, so the two that reach into neighbouring days reach into *visible*
    neighbouring days. They differ in what the feature means, not in what it is
    allowed to see, which is why the choice is recorded in the run manifest.

    Note that ``ddof=1`` additionally leaves the standard deviation undefined for
    a day with a single visible observation, while its quartiles are all defined
    and equal to that value.

    :param strategy: the below-minimum behaviour above. Defaults to the
        configured ``daily_statistic_strategy`` (``missing``).
    :param fallback_window_days: reach of the two fallback strategies, in
        calendar days; rejected for the strategies that never leave the day.
    :param ddof: delta degrees of freedom for the standard deviation. Defaults to
        the configured ``daily_std_ddof`` (1, the sample standard deviation;
        ambiguity A11 - the paper does not state which convention it used).
    :returns: a frame indexed like ``target`` whose columns are
        :func:`daily_statistic_names`.
    """
    overrides = (min_observations, strategy, fallback_window_days, ddof)
    if config is not None and any(override is not None for override in overrides):
        raise ConfigError(
            "pass either config= or explicit min_observations/strategy/"
            "fallback_window_days/ddof, not both: two sources of the same setting "
            "cannot be reconciled in the run manifest"
        )
    if config is None:
        # Built rather than checked field by field, so an explicit call and a
        # configured run are validated by exactly the same rules.
        base = FeatureConfig()
        config = FeatureConfig(
            min_daily_observations=(
                base.min_daily_observations if min_observations is None else min_observations
            ),
            daily_statistic_strategy=(base.statistic_strategy if strategy is None else strategy),
            fallback_window_days=fallback_window_days,
            daily_std_ddof=base.daily_std_ddof if ddof is None else ddof,
        )
    minimum = config.min_daily_observations
    degrees = config.daily_std_ddof
    chosen = config.statistic_strategy
    window = config.fallback_window

    series = _as_float_series(target, field_name="target")
    index = _time_index(series, field_name="target")
    name = target_name if target_name is not None else series.name
    if not isinstance(name, str) or not name.strip():
        raise ConfigError(
            "the target's name is needed to name its daily-statistic features; pass "
            "target_name= or give the Series a name"
        )
    columns = daily_statistic_names(name)

    visible = pd.Series(series.to_numpy(dtype=float), index=index)
    if available_mask is not None:
        mask = _as_boolean_mask(available_mask, index, field_name="available_mask")
        visible = visible.where(mask)

    day = index.normalize()
    grouped = visible.groupby(day, sort=True)
    # pandas skips NaN in every one of these, so masked-out and genuinely missing
    # values are excluded identically.
    per_day = pd.DataFrame(
        {
            columns[0]: grouped.quantile(_DAILY_QUANTILES[0]),
            columns[1]: grouped.quantile(_DAILY_QUANTILES[1]),
            columns[2]: grouped.quantile(_DAILY_QUANTILES[2]),
            columns[3]: grouped.std(ddof=degrees),
        }
    )
    counts = grouped.count().to_numpy()

    if chosen is not DailyStatisticStrategy.WITHIN_DAY_AVAILABLE:
        # Every other strategy starts from "a day below the minimum has no
        # statistics of its own"; the two reaching strategies then look further.
        deficient = counts < minimum
        per_day.loc[deficient, :] = np.nan
        if chosen is DailyStatisticStrategy.NEIGHBOR_DAY_FALLBACK:
            assert window is not None
            _fill_from_nearest_day(per_day, deficient=deficient, window_days=window)
        elif chosen is DailyStatisticStrategy.ROLLING_AVAILABLE:
            assert window is not None
            _fill_from_rolling_window(
                per_day,
                visible=visible,
                deficient=deficient,
                window_days=window,
                minimum=minimum,
                degrees=degrees,
            )

    joined: pd.DataFrame = per_day.reindex(day)
    joined.index = index
    joined.index.name = series.index.name
    return joined


def _fill_from_nearest_day(
    per_day: pd.DataFrame,
    *,
    deficient: np.ndarray,
    window_days: int,
) -> None:
    """Give each deficient day the statistics of the nearest day that qualifies.

    In place. Distance is measured in calendar days between the day labels, not
    in row positions, so a day on the far side of a missing month is correctly
    seen as a month away. Ties resolve to the **earlier** day, an arbitrary but
    fixed rule that keeps the result reproducible. A day with no qualifying day
    within ``window_days`` keeps its missing statistics.
    """
    qualifying = np.flatnonzero(~deficient)
    wanted = np.flatnonzero(deficient)
    if qualifying.size == 0 or wanted.size == 0:
        return

    days = per_day.index.to_numpy(dtype="datetime64[ns]").astype("int64")
    limit = int(pd.Timedelta(days=window_days).value)
    donor_days = days[qualifying]

    insert = np.searchsorted(donor_days, days[wanted])
    left = np.clip(insert - 1, 0, donor_days.size - 1)
    right = np.clip(insert, 0, donor_days.size - 1)
    # `insert == 0` means there is no earlier donor and `insert == size` none
    # later; the sentinel distance keeps those candidates from ever winning.
    unreachable = np.iinfo(np.int64).max
    distance_left = np.where(insert > 0, days[wanted] - donor_days[left], unreachable)
    distance_right = np.where(
        insert < donor_days.size, donor_days[right] - days[wanted], unreachable
    )

    take_left = distance_left <= distance_right
    donor = np.where(take_left, qualifying[left], qualifying[right])
    distance = np.minimum(distance_left, distance_right)

    reachable = distance <= limit
    if not reachable.any():
        return
    # Snapshot first: the rows being written are all deficient and the rows being
    # read are all qualifying, but reading from an array that is not being
    # mutated makes that independence explicit rather than incidental.
    values = per_day.to_numpy(dtype=float, copy=True)
    per_day.iloc[wanted[reachable], :] = values[donor[reachable]]


def _fill_from_rolling_window(
    per_day: pd.DataFrame,
    *,
    visible: pd.Series,
    deficient: np.ndarray,
    window_days: int,
    minimum: int,
    degrees: int,
) -> None:
    """Recompute each deficient day from a centred window of visible observations.

    In place. The pool is every visible target observation whose calendar day
    lies within ``window_days`` days of the deficient day, inclusive on both
    sides and including the day itself. A pool that is still below ``minimum``
    leaves the day missing rather than reporting a statistic of one or two
    values borrowed from a week away.
    """
    wanted = np.flatnonzero(deficient)
    if wanted.size == 0:
        return

    observed = visible.dropna()
    if observed.empty:
        return
    observation_days = (
        as_datetime_index(observed.index, field_name="target")
        .normalize()
        .to_numpy(dtype="datetime64[ns]")
        .astype("int64")
    )
    order = np.argsort(observation_days, kind="stable")
    observation_days = observation_days[order]
    observation_values = observed.to_numpy(dtype=float)[order]

    days = per_day.index.to_numpy(dtype="datetime64[ns]").astype("int64")
    reach = int(pd.Timedelta(days=window_days).value)

    for position in wanted:
        centre = days[position]
        start = np.searchsorted(observation_days, centre - reach, side="left")
        stop = np.searchsorted(observation_days, centre + reach, side="right")
        pool = observation_values[start:stop]
        if pool.size < minimum:
            continue
        quantiles = np.quantile(pool, _DAILY_QUANTILES)
        deviation = float(pool.std(ddof=degrees)) if pool.size > degrees else np.nan
        per_day.iloc[position, :] = [*quantiles, deviation]


# ---------------------------------------------------------------------------
# Feature matrix assembly
# ---------------------------------------------------------------------------


def feature_names(
    config: RFRConfig | FeatureConfig,
    *,
    target: str | None = None,
    drivers: Sequence[str] | None = None,
) -> tuple[str, ...]:
    """Return the feature columns for ``config``, in their deterministic order.

    The order is: the canonical drivers of the selected mode, in specification
    order, followed - when the receptive limiter is enabled - by the radiation
    category, elapsed hours, the season tag and the four daily target statistics.
    :func:`build_feature_matrix` reproduces it exactly, and acceptance test 6
    pins the two together.

    Computable without any data, so it can be recorded in the run manifest and
    checked against a reloaded model before predicting. ``use_receptive_limiter =
    False`` yields the ORF benchmark's columns: the same drivers, nothing else
    (method_spec.md 3.6).
    """
    if isinstance(config, RFRConfig):
        features = config.features
        names = list(config.drivers) if drivers is None else list(drivers)
    else:
        features = config
        if drivers is None:
            raise ConfigError(
                "drivers= is required when feature_names is called with a FeatureConfig; "
                "pass an RFRConfig to take them from its mode"
            )
        names = list(drivers)
    return tuple(names) + receptive_limiter_features(features, target=target)


def receptive_limiter_features(
    config: RFRConfig | FeatureConfig,
    *,
    target: str | None = None,
) -> tuple[str, ...]:
    """Return the feature columns the receptive limiter contributes, in order.

    Empty for an ORF configuration, which is the whole of the difference between
    the two arms of the Supplementary Figure S1 comparison. :func:`feature_names`
    is built from this, so the identity

    ``feature_names(cfg, target=t) == cfg.drivers + receptive_limiter_features(cfg, target=t)``

    holds by construction rather than by two lists being kept in step by hand.
    The ORF benchmark tests assert exactly that ORF drops this tuple and keeps
    everything else (method_spec.md 3.6).
    """
    features = config.features if isinstance(config, RFRConfig) else config
    if not features.use_receptive_limiter:
        return ()
    names = [RADIATION_CATEGORY, TIME_DISTANCE_HOURS, SEASON]
    if target is not None:
        names += list(daily_statistic_names(target))
    return tuple(names)


def build_feature_matrix(
    data: pd.DataFrame,
    *,
    config: RFRConfig,
    target: str | None = None,
    column_map: ColumnMap | Mapping[str, str] | None = None,
    available_mask: object | None = None,
    origin: pd.Timestamp | str | None = None,
    encode: bool = True,
) -> pd.DataFrame:
    """Assemble the receptive-limiter feature matrix for one site and one target.

    ``data`` must already sit on a validated time axis - call
    :func:`rfrgapfill.time.prepare_time_index` first - and carry the driver
    columns named by ``column_map`` (or by ``config.column_map``). Drivers are
    copied into the matrix under their **canonical** names, so a matrix built
    from FLUXNET columns and one built from a station's own names are
    interchangeable.

    Column order is exactly :func:`feature_names`. Rows are the rows of ``data``,
    in order, with missing inputs carried through as missing features; no row is
    dropped and no value is imputed here. Deciding which rows are usable belongs
    to the model layer, which reports what it excluded.

    :param target: the target flux column in ``data``. Required while the
        receptive limiter is on, since the daily statistics are derived from it;
        ignored for the ORF benchmark, which has no target-derived features.
    :param available_mask: rows whose target value the model may see, passed
        straight to :func:`daily_flux_statistics`. **Required for artificial-gap
        validation**; see that function for what omitting it means.
    :param origin: zero point for :func:`time_distance_hours`. Defaults to the
        first timestamp of ``data``; pass the full series' origin whenever
        ``data`` is a subset, or the same timestamp will get different values in
        the subset than in the whole series.
    :param encode: when true (the default) the two categorical features are
        returned as float ordinal codes - ``weak``/``medium``/``strong`` as
        0/1/2 and ``winter``/``spring``/``summer``/``autumn`` as 0/1/2/3, with
        missing categories as ``NaN`` - which is what the Random Forest consumes.
        Pass ``encode=False`` to keep them as pandas categoricals for inspection
        and testing. The codes are fixed by the declaration order of
        :class:`RadiationClass` and :class:`Season`, so a model and a later
        prediction frame always agree.
    """
    if not isinstance(data, pd.DataFrame):
        raise FeatureError(f"data must be a pandas DataFrame, got {type(data).__name__}")
    if not isinstance(config, RFRConfig):
        raise ConfigError(f"config must be an RFRConfig, got {type(config).__name__}")

    index = as_datetime_index(data.index, field_name="the data index")
    features = config.features
    columns = config.require_column_map(column_map)
    missing_columns = columns.missing_columns(data.columns)
    if missing_columns:
        raise FeatureError(
            f"mapped driver column(s) absent from the data: {', '.join(missing_columns)}"
        )

    matrix = pd.DataFrame(index=index)
    matrix.index.name = data.index.name
    for canonical in config.drivers:
        matrix[canonical] = _as_float_series(
            data[columns.column(canonical)], field_name=f"driver {canonical!r}"
        ).to_numpy(dtype=float)

    if features.use_receptive_limiter:
        if target is None:
            raise ConfigError(
                "target= is required when the receptive limiter is enabled: the daily "
                "statistics of method_spec.md 3.4 are target-specific. For the ORF "
                "benchmark set FeatureConfig(use_receptive_limiter=False)."
            )
        if target not in data.columns:
            raise FeatureError(f"target column {target!r} is not in the data")

        matrix[RADIATION_CATEGORY] = radiation_tag(
            data[columns.column(SHORTWAVE)], config=features
        ).to_numpy()
        matrix[TIME_DISTANCE_HOURS] = time_distance_hours(
            index, origin=index.min() if origin is None else origin
        ).to_numpy(dtype=float)
        matrix[SEASON] = season_tag(index, hemisphere=config.resolve_hemisphere()).to_numpy()
        statistics = daily_flux_statistics(
            data[target],
            available_mask=available_mask,
            target_name=target,
            config=features,
        )
        for name in statistics.columns:
            matrix[name] = statistics[name].to_numpy(dtype=float)

        if encode:
            matrix[RADIATION_CATEGORY] = _encode(matrix[RADIATION_CATEGORY], _RADIATION_DTYPE)
            matrix[SEASON] = _encode(matrix[SEASON], _SEASON_DTYPE)
        else:
            matrix[RADIATION_CATEGORY] = pd.Categorical(
                matrix[RADIATION_CATEGORY], dtype=_RADIATION_DTYPE
            )
            matrix[SEASON] = pd.Categorical(matrix[SEASON], dtype=_SEASON_DTYPE)

    expected = feature_names(config, target=target if features.use_receptive_limiter else None)
    assert tuple(matrix.columns) == expected, (
        f"feature order drifted from feature_names(): {tuple(matrix.columns)} != {expected}"
    )
    assembled: pd.DataFrame = matrix
    return assembled


def _encode(values: pd.Series, dtype: pd.CategoricalDtype) -> pd.Series:
    """Return ordinal float codes for a categorical column, missing as ``NaN``."""
    categorical = pd.Categorical(values, dtype=dtype)
    codes = np.asarray(categorical.codes, dtype=float)
    codes[codes < 0] = np.nan
    encoded: pd.Series = pd.Series(codes, index=values.index, name=values.name)
    return encoded


def describe_features(
    config: RFRConfig,
    *,
    target: str | None = None,
) -> dict[str, Any]:
    """Return a manifest-ready description of the feature stage.

    Records the column order together with every choice that determined it, so a
    result can be traced back to the conventions that produced it (method_spec.md
    section 7).
    """
    features = config.features
    described: dict[str, Any] = {
        "use_receptive_limiter": features.use_receptive_limiter,
        "feature_mode": features.mode.value,
        "drivers": list(config.drivers),
        "feature_names": list(feature_names(config, target=target)),
        "target": target,
    }
    if features.use_receptive_limiter:
        described.update(
            {
                "radiation_thresholds": list(features.radiation_thresholds),
                "boundary_convention": features.convention.value,
                "hemisphere": config.resolve_hemisphere().value,
                "hemisphere_source": config.hemisphere_source,
                "min_daily_observations": features.min_daily_observations,
                "daily_statistic_strategy": features.statistic_strategy.value,
                "fallback_window_days": features.fallback_window,
                "daily_std_ddof": features.daily_std_ddof,
                "radiation_category_codes": {
                    name: code for code, name in enumerate(_RADIATION_CATEGORIES)
                },
                "season_codes": {name: code for code, name in enumerate(_SEASON_CATEGORIES)},
            }
        )
    return described
