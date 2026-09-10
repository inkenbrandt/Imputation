"""Receptive-limiter feature transformers.

The paper's feature-engineering stage (``docs/method_spec.md`` section 3) and the
distinguishing component of RFR against the ORF benchmark. Each of the four
groups is a small pure function that takes array-likes and returns a pandas
object, so every one of them can be hand-checked against a fixture without a
Random Forest anywhere in sight:

* :func:`radiation_tag` - the weak/medium/strong shortwave category (3.1);
* :func:`time_distance_hours` - elapsed hours since the series origin (3.2);
* :func:`season_tag` - hemisphere-aware season (3.3);
* :func:`daily_flux_statistics` - daily target quartiles and standard deviation (3.4).

:func:`build_feature_matrix` assembles them into the model's design matrix in a
fixed, documented column order.

**The leakage rule (3.5) lives here.** The daily statistics are derived from the
target, so they are the one feature group that can smuggle held-out truth into
the predictors built to predict it. Every entry point that touches the target
therefore takes an explicit ``available`` mask naming the observations the model
is allowed to see, and nothing in this module ever reads a target value outside
it. In the artificial-gap workflow the mask is built *before* the features are,
which is what makes the ordering in :mod:`rfrgapfill.validation` load-bearing
rather than stylistic.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, cast

import numpy as np
import pandas as pd

from rfrgapfill.config import (
    DEFAULT_RADIATION_THRESHOLDS,
    BoundaryConvention,
    ColumnMap,
    DailyStatisticStrategy,
    FeatureConfig,
    Hemisphere,
    Mode,
    RFRConfig,
)
from rfrgapfill.schema import CANONICAL_VARIABLES, SHORTWAVE
from rfrgapfill.time import TIME_DISTANCE_HOURS, as_datetime_index, elapsed_hours

__all__ = [
    "DAILY_STATISTIC_SUFFIXES",
    "RADIATION_CATEGORIES",
    "RADIATION_CLASS",
    "SEASONS",
    "SEASON_PREFIX",
    "TIME_DISTANCE_HOURS",
    "DailyStatistics",
    "FeatureError",
    "FeatureMatrix",
    "build_feature_matrix",
    "daily_flux_statistics",
    "daily_statistic_names",
    "feature_names",
    "radiation_tag",
    "season_names",
    "season_tag",
    "time_distance_hours",
]


class FeatureError(ValueError):
    """Raised when features cannot be built from the data as given.

    Reports invalid *data* or an impossible request - an unmapped driver column, a
    hemisphere that is needed but absent, a target the frame does not carry -
    rather than an invalid configuration, which is
    :class:`~rfrgapfill.schema.ConfigError`. Both subclass :class:`ValueError`.
    """


# ---------------------------------------------------------------------------
# Feature vocabulary
# ---------------------------------------------------------------------------

#: Name of the radiation-category feature (method_spec.md 3.1).
RADIATION_CLASS: Final = "radiation_class"

#: Radiation categories in increasing order of radiation. The order is the
#: encoding: ``weak`` -> 0, ``medium`` -> 1, ``strong`` -> 2.
RADIATION_CATEGORIES: Final[tuple[str, ...]] = ("weak", "medium", "strong")

#: Prefix of the one-hot season indicator columns (method_spec.md 3.3).
SEASON_PREFIX: Final = "season"

#: Seasons in calendar order from the northern winter. Nominal, not ordered.
SEASONS: Final[tuple[str, ...]] = ("winter", "spring", "summer", "autumn")

#: Suffixes of the four daily target statistics, in specification order (3.4).
DAILY_STATISTIC_SUFFIXES: Final[tuple[str, ...]] = (
    "daily_q1",
    "daily_q2",
    "daily_q3",
    "daily_std",
)

#: Northern-hemisphere season of each calendar month, indexed 1-12. The southern
#: hemisphere is this list shifted by two seasons, which :func:`season_tag` applies.
_NORTHERN_SEASON_BY_MONTH: Final[Mapping[int, str]] = {
    1: "winter",
    2: "winter",
    3: "spring",
    4: "spring",
    5: "spring",
    6: "summer",
    7: "summer",
    8: "summer",
    9: "autumn",
    10: "autumn",
    11: "autumn",
    12: "winter",
}

#: Season opposite each season, used to flip the northern table southwards.
_OPPOSITE_SEASON: Final[Mapping[str, str]] = {
    "winter": "summer",
    "spring": "autumn",
    "summer": "winter",
    "autumn": "spring",
}


def season_names() -> tuple[str, ...]:
    """Return the one-hot season column names in their fixed order."""
    return tuple(f"{SEASON_PREFIX}_{season}" for season in SEASONS)


def daily_statistic_names(target: str) -> tuple[str, ...]:
    """Return the four daily-statistic column names for ``target``, in order.

    They carry the target's name because the statistics are target-specific: a
    separate feature matrix exists per target (method_spec.md 3.4).
    """
    return tuple(f"{target}_{suffix}" for suffix in DAILY_STATISTIC_SUFFIXES)


def feature_names(
    target: str,
    *,
    mode: Mode | str,
    features: FeatureConfig | None = None,
) -> tuple[str, ...]:
    """Return the feature column names a run produces, in matrix order.

    The order is part of the contract: drivers first in the canonical order of
    method_spec.md section 2, then the receptive-limiter groups in the order of
    section 3. Computable without any data, so a caller can check a saved model's
    feature list against the configuration that is about to be used.
    """
    settings = FeatureConfig() if features is None else features
    names = list(Mode.coerce(mode).drivers)
    if settings.use_receptive_limiter:
        names.append(RADIATION_CLASS)
        names.append(TIME_DISTANCE_HOURS)
        names.extend(season_names())
        names.extend(daily_statistic_names(target))
    return tuple(names)


# ---------------------------------------------------------------------------
# 3.1 Radiation category
# ---------------------------------------------------------------------------


def radiation_tag(
    shortwave: object,
    *,
    thresholds: tuple[float, float] = DEFAULT_RADIATION_THRESHOLDS,
    convention: BoundaryConvention | str = BoundaryConvention.MEDIUM_INCLUSIVE,
) -> pd.Series:
    """Return the weak/medium/strong radiation category of ``shortwave``.

    The bins are exhaustive over the reals. Under the default
    ``medium_inclusive`` convention (ambiguity A2) a value of exactly 10 or
    exactly 100 W m-2 is ``medium``; ``medium_exclusive`` puts 10 in ``weak`` and
    100 in ``strong``.

    Missing radiation yields a missing category, never a default class: guessing
    one would move a row into a bin the measurement does not support. The result
    is an ordered :class:`~pandas.Categorical`, so ``weak < medium < strong``
    holds for anything that wants to compare them.
    """
    low, high = _check_thresholds(thresholds)
    rule = BoundaryConvention.coerce(convention)
    values = _as_float_series(shortwave, name=SHORTWAVE)

    if rule is BoundaryConvention.MEDIUM_INCLUSIVE:
        weak = values < low
        strong = values > high
    else:
        weak = values <= low
        strong = values >= high

    codes = np.where(weak, 0, np.where(strong, 2, 1))
    codes = np.where(values.isna().to_numpy(), -1, codes)
    tag = pd.Categorical.from_codes(
        codes.astype(np.int8),
        dtype=pd.CategoricalDtype(list(RADIATION_CATEGORIES), ordered=True),
    )
    result: pd.Series = pd.Series(tag, index=values.index, name=RADIATION_CLASS)
    return result


def radiation_code(tag: pd.Series) -> pd.Series:
    """Return :func:`radiation_tag` output as the ordinal codes the model sees.

    ``weak`` -> 0.0, ``medium`` -> 1.0, ``strong`` -> 2.0, missing -> NaN. The
    category is a binned continuous variable, so the ordinal encoding preserves
    the ordering the bins already carry and costs the forest no extra splits;
    contrast :func:`season_tag`, which is nominal and one-hot encoded.
    """
    codes = np.asarray(tag.cat.codes, dtype=float)
    codes[codes < 0] = np.nan
    result: pd.Series = pd.Series(codes, index=tag.index, name=RADIATION_CLASS)
    return result


# ---------------------------------------------------------------------------
# 3.2 Elapsed hours
# ---------------------------------------------------------------------------


def time_distance_hours(
    values: object,
    *,
    origin: pd.Timestamp | str | None = None,
) -> pd.Series:
    """Return elapsed hours since ``origin`` (method_spec.md 3.2).

    A thin alias for :func:`rfrgapfill.time.elapsed_hours`, kept here so the four
    receptive-limiter groups can be imported from one place. ``origin`` defaults
    to the earliest timestamp given; pass the site series' first timestamp
    whenever a subset is transformed on its own, or the same row would receive
    different values in the subset than in the whole series.
    """
    return elapsed_hours(values, origin=origin)


# ---------------------------------------------------------------------------
# 3.3 Hemisphere-aware season
# ---------------------------------------------------------------------------


def season_tag(values: object, hemisphere: Hemisphere | str) -> pd.Series:
    """Return the meteorological season of each timestamp for ``hemisphere``.

    December-February is ``winter`` in the north and ``summer`` in the south, and
    so on around the year (method_spec.md 3.3). The hemisphere is required: there
    is no silent default, because getting it wrong inverts the seasonal signal at
    every southern site.
    """
    index = as_datetime_index(values)
    half = Hemisphere.coerce(hemisphere)
    table = _NORTHERN_SEASON_BY_MONTH
    if half is Hemisphere.SOUTH:
        table = {month: _OPPOSITE_SEASON[season] for month, season in table.items()}
    seasons = [table[int(month)] for month in index.month]
    tag = pd.Categorical(seasons, categories=list(SEASONS), ordered=False)
    result: pd.Series = pd.Series(tag, index=index, name=SEASON_PREFIX)
    return result


def season_indicators(tag: pd.Series) -> pd.DataFrame:
    """Return the one-hot encoding of :func:`season_tag`, in :data:`SEASONS` order.

    Season is nominal - the year is a cycle, not a ladder - so one-hot indicators
    are used rather than the ordinal encoding applied to the radiation category.
    Columns are always all four seasons, even when the series covers only part of
    a year, so the feature matrix has the same shape for every site.
    """
    codes = np.asarray(tag.cat.codes)
    columns = {
        f"{SEASON_PREFIX}_{season}": (codes == position).astype(float)
        for position, season in enumerate(SEASONS)
    }
    indicators: pd.DataFrame = pd.DataFrame(columns, index=tag.index)
    return indicators


# ---------------------------------------------------------------------------
# 3.4 Daily target statistics, and the leakage rule of 3.5
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DailyStatistics:
    """Daily target statistics joined back to every timestamp, with their provenance.

    :attr:`frame` is what the model consumes. The counts beside it say how each
    day got its numbers, so a report can state how much of a long gap was covered
    by a neighbouring day's statistics rather than its own - the substitution is
    leakage-safe but it is still a substitution, and it stays visible.
    """

    #: One row per input timestamp, four columns named for the target.
    frame: pd.DataFrame
    #: The strategy that produced it.
    strategy: DailyStatisticStrategy
    #: Minimum visible observations a day needed to compute its own statistics.
    min_observations: int
    #: Calendar days spanned by the input.
    n_days: int
    #: Days with enough visible target observations of their own.
    n_days_observed: int
    #: Days that took a neighbouring day's statistics instead.
    n_days_from_neighbour: int
    #: Days left with missing statistics; their rows cannot be trained or predicted.
    n_days_missing: int
    #: Timestamps carrying a neighbouring day's statistics.
    n_rows_from_neighbour: int
    #: Timestamps left with missing statistics.
    n_rows_missing: int

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary for the run manifest."""
        return {
            "strategy": self.strategy.value,
            "min_observations": self.min_observations,
            "n_days": self.n_days,
            "n_days_observed": self.n_days_observed,
            "n_days_from_neighbour": self.n_days_from_neighbour,
            "n_days_missing": self.n_days_missing,
            "n_rows_from_neighbour": self.n_rows_from_neighbour,
            "n_rows_missing": self.n_rows_missing,
        }


def daily_flux_statistics(
    target: object,
    *,
    available: object | None = None,
    name: str | None = None,
    min_observations: int = 2,
    strategy: DailyStatisticStrategy | str = DailyStatisticStrategy.NEAREST_VISIBLE_DAY,
) -> DailyStatistics:
    """Return per-day Q1, Q2, Q3 and standard deviation of ``target``, joined to every row.

    ``available`` is the leakage boundary (method_spec.md 3.5): a boolean mask
    over the same index naming the target observations the model may see. Values
    outside it are dropped before anything is computed, so held-out truth cannot
    reach a feature built to predict it. Omitting it uses every finite target
    value, which is correct for operational filling - where nothing is held out -
    and wrong for artificial-gap validation, where the mask is the point.

    A day needs ``min_observations`` visible values before it has statistics of
    its own; the default of 2 is the fewest the sample standard deviation is
    defined on. What a day below that gets instead is ``strategy`` (ambiguity
    A4): :attr:`~rfrgapfill.config.DailyStatisticStrategy.WITHIN_DAY` leaves it
    missing, and the default
    :attr:`~rfrgapfill.config.DailyStatisticStrategy.NEAREST_VISIBLE_DAY` copies
    the nearest day that does qualify, ties going to the earlier day. Neither
    reads a value outside ``available``.

    Quartiles use pandas' default linear interpolation and the standard deviation
    uses ``ddof=1``.
    """
    rule = DailyStatisticStrategy.coerce(strategy)
    if min_observations < 1:
        raise FeatureError(f"min_observations must be >= 1, got {min_observations}")

    values = _as_float_series(target, name=name or "target")
    index = as_datetime_index(values.index)
    label = name or str(values.name)
    columns = list(daily_statistic_names(label))

    visible = values.copy()
    if available is not None:
        mask = _as_bool_array(available, length=len(values), field_name="available")
        visible = visible.where(mask)

    days = index.normalize()
    grouped = visible.groupby(days, sort=True)
    counts = grouped.count()
    per_day = pd.DataFrame(
        {
            columns[0]: grouped.quantile(0.25),
            columns[1]: grouped.quantile(0.50),
            columns[2]: grouped.quantile(0.75),
            columns[3]: grouped.std(ddof=1),
        }
    )
    qualifies = (counts >= min_observations).to_numpy() & per_day.notna().all(axis=1).to_numpy()
    per_day = per_day.where(pd.Series(qualifies, index=per_day.index), other=np.nan)

    source = np.where(qualifies, "day", "missing")
    if rule is DailyStatisticStrategy.NEAREST_VISIBLE_DAY:
        per_day, source = _fill_from_nearest_day(per_day, qualifies)

    joined = per_day.reindex(pd.DatetimeIndex(days))
    joined.index = values.index
    row_source = pd.Series(source, index=per_day.index).reindex(pd.DatetimeIndex(days)).to_numpy()

    return DailyStatistics(
        frame=joined,
        strategy=rule,
        min_observations=min_observations,
        n_days=len(per_day),
        n_days_observed=int(np.sum(source == "day")),
        n_days_from_neighbour=int(np.sum(source == "neighbour")),
        n_days_missing=int(np.sum(source == "missing")),
        n_rows_from_neighbour=int(np.sum(row_source == "neighbour")),
        n_rows_missing=int(np.sum(row_source == "missing")),
    )


def _fill_from_nearest_day(
    per_day: pd.DataFrame,
    qualifies: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Return ``per_day`` with unqualified days taking the nearest qualified day's row.

    Distance is measured between calendar days, so a day inside a 30-day gap
    takes the statistics of the closest day outside it. Ties go to the earlier
    day, which makes the result independent of iteration order.
    """
    source = np.where(qualifies, "day", "missing").astype(object)
    donors = np.flatnonzero(qualifies)
    if donors.size == 0 or donors.size == len(per_day):
        return per_day, source

    dates = per_day.index.to_numpy(dtype="datetime64[D]").astype(np.int64)
    positions = np.arange(len(per_day))
    after = np.searchsorted(donors, positions, side="left")
    previous = np.clip(after - 1, 0, donors.size - 1)
    following = np.clip(after, 0, donors.size - 1)

    distance_previous = np.abs(dates - dates[donors[previous]])
    distance_following = np.abs(dates - dates[donors[following]])
    # `<=` keeps the earlier donor on a tie.
    chosen = np.where(distance_previous <= distance_following, donors[previous], donors[following])

    filled = per_day.iloc[chosen].copy()
    filled.index = per_day.index
    source[~qualifies] = "neighbour"
    return filled, source


# ---------------------------------------------------------------------------
# Feature matrix
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FeatureMatrix:
    """A built design matrix together with what it is safe to use it for.

    :attr:`complete` is the part callers must not ignore: a row whose drivers or
    daily statistics are missing is not predictable, and the package neither
    imputes it nor lets it through (method_spec.md section 7). Training selects
    on it and prediction leaves the rest missing.
    """

    #: One row per input timestamp, columns in :func:`feature_names` order.
    frame: pd.DataFrame
    #: The target these features were built for.
    target: str
    #: Whether the receptive-limiter groups are present (False is the ORF benchmark).
    use_receptive_limiter: bool
    #: Rows whose every feature is finite, so the model can use them.
    complete: pd.Series
    #: Provenance of the daily statistics; ``None`` for ORF, which has none.
    daily_statistics: DailyStatistics | None

    @property
    def names(self) -> tuple[str, ...]:
        """The feature column names, in matrix order."""
        return tuple(self.frame.columns)

    @property
    def n_complete(self) -> int:
        """How many rows carry a complete feature vector."""
        return int(self.complete.sum())

    @property
    def n_incomplete(self) -> int:
        """How many rows are missing at least one feature."""
        return int(len(self.complete) - self.complete.sum())

    def incomplete_by_feature(self) -> pd.Series:
        """Return, per feature, how many rows it is missing on.

        The diagnostic to reach for when a run trains on far fewer rows than
        expected: it names the driver responsible instead of leaving a bare count.
        """
        counts: pd.Series = self.frame.isna().sum()
        return counts

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary for the run manifest."""
        return {
            "target": self.target,
            "use_receptive_limiter": self.use_receptive_limiter,
            "feature_names": list(self.names),
            "n_rows": len(self.frame),
            "n_complete": self.n_complete,
            "n_incomplete": self.n_incomplete,
            "incomplete_by_feature": {
                name: int(count)
                for name, count in self.incomplete_by_feature().items()
                if int(count) > 0
            },
            "daily_statistics": (
                None if self.daily_statistics is None else self.daily_statistics.to_dict()
            ),
        }


def build_feature_matrix(
    data: pd.DataFrame,
    *,
    target: str,
    config: RFRConfig,
    column_map: ColumnMap | Mapping[str, str] | None = None,
    available: object | None = None,
    origin: pd.Timestamp | None = None,
) -> FeatureMatrix:
    """Return the design matrix for one target (method_spec.md sections 2-3).

    ``data`` must already carry a :class:`~pandas.DatetimeIndex`; run it through
    :func:`rfrgapfill.time.prepare_time_index` first. Drivers are read through
    ``config``'s column map and renamed to their canonical names, so nothing
    downstream sees a station column name.

    ``available`` is passed straight to :func:`daily_flux_statistics` and is the
    leakage boundary. In artificial-gap validation it must exclude every withheld
    observation, and it must be built before this call - which is why
    :mod:`rfrgapfill.validation` generates the gap mask first.

    With ``config.features.use_receptive_limiter=False`` this returns the drivers
    alone: the ORF benchmark of Supplementary Figure S1, same drivers and same
    estimator, feature engineering removed.
    """
    if not isinstance(data.index, pd.DatetimeIndex):
        raise FeatureError(
            "build_feature_matrix needs a DatetimeIndex; call prepare_time_index() first"
        )
    if target not in data.columns:
        raise FeatureError(f"target column {target!r} is not in the data")

    settings = config.features
    mapping = config.require_column_map(column_map)
    frame = _driver_frame(data, config=config, column_map=mapping)

    daily: DailyStatistics | None = None
    if settings.use_receptive_limiter:
        tag = radiation_tag(
            frame[SHORTWAVE],
            thresholds=settings.radiation_thresholds,
            convention=settings.convention,
        )
        frame[RADIATION_CLASS] = radiation_code(tag)
        frame[TIME_DISTANCE_HOURS] = time_distance_hours(
            data.index, origin=data.index.min() if origin is None else origin
        )
        for column, values in season_indicators(
            season_tag(data.index, config.resolve_hemisphere())
        ).items():
            frame[column] = values
        daily = daily_flux_statistics(
            data[target],
            available=available,
            name=target,
            min_observations=settings.min_daily_observations,
            strategy=settings.daily_strategy,
        )
        for column in daily.frame.columns:
            frame[column] = daily.frame[column]

    expected = feature_names(target, mode=config.rfr_mode, features=settings)
    ordered = cast("pd.DataFrame", frame.loc[:, list(expected)])
    complete = ordered.notna().all(axis=1)
    complete.name = "complete"

    return FeatureMatrix(
        frame=ordered,
        target=target,
        use_receptive_limiter=settings.use_receptive_limiter,
        complete=complete,
        daily_statistics=daily,
    )


def _driver_frame(
    data: pd.DataFrame,
    *,
    config: RFRConfig,
    column_map: ColumnMap,
) -> pd.DataFrame:
    """Return the mode's drivers as float columns under their canonical names."""
    missing = column_map.missing_columns(data.columns)
    needed = set(column_map.columns(config.drivers))
    absent = sorted(column for column in missing if column in needed)
    if absent:
        raise FeatureError(
            f"the data is missing mapped driver column(s): {', '.join(absent)}. "
            "Check the column map against the frame's columns."
        )
    columns = {
        name: _as_float_series(data[column_map.column(name)], name=name)
        for name in CANONICAL_VARIABLES
        if name in config.drivers
    }
    drivers: pd.DataFrame = pd.DataFrame(columns, index=data.index)
    return drivers


# ---------------------------------------------------------------------------
# Coercion helpers
# ---------------------------------------------------------------------------


def _as_float_series(values: object, *, name: str) -> pd.Series:
    """Return ``values`` as a float Series, keeping a pandas index when given."""
    if isinstance(values, pd.Series):
        try:
            converted = values.astype(float)
        except (TypeError, ValueError) as error:
            raise FeatureError(f"{name} must be numeric: {error}") from error
        converted.name = name
        return converted
    array = np.asarray(values, dtype=object)
    try:
        numeric = array.astype(float)
    except (TypeError, ValueError) as error:
        raise FeatureError(f"{name} must be numeric: {error}") from error
    result: pd.Series = pd.Series(numeric, name=name)
    return result


def _as_bool_array(values: object, *, length: int, field_name: str) -> np.ndarray:
    """Return ``values`` as a boolean array of ``length``, treating NaN as False."""
    array = values.to_numpy() if isinstance(values, pd.Series) else np.asarray(values)
    if array.ndim != 1 or array.size != length:
        raise FeatureError(
            f"{field_name} must be a 1-D mask of length {length}, got shape {array.shape}"
        )
    if array.dtype == bool:
        return array
    filled = pd.Series(array).fillna(False)
    return filled.astype(bool).to_numpy()


def _check_thresholds(thresholds: Sequence[float]) -> tuple[float, float]:
    """Return ``thresholds`` as a strictly increasing pair of finite floats."""
    if len(thresholds) != 2:
        raise FeatureError(f"radiation thresholds must be a pair, got {thresholds!r}")
    low, high = float(thresholds[0]), float(thresholds[1])
    if not (np.isfinite(low) and np.isfinite(high)):
        raise FeatureError(f"radiation thresholds must be finite, got ({low}, {high})")
    if not low < high:
        raise FeatureError(f"radiation thresholds must be increasing, got ({low}, {high})")
    return low, high
