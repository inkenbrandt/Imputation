"""Validation metrics.

The quantities the paper reports for artificial-gap validation
(``docs/method_spec.md`` section 6): the coefficient of determination, the
regression slope of filled on measured, RMSE, the paper's bias definition, the
energy-balance ratio, and the all/daytime/nighttime split those core metrics are
reported over.

Two rules shape the module.

* **Orientation is fixed.** Every core metric takes ``measured`` first and
  ``filled`` second, and the regression that produces the slope puts measured on
  x and filled on y (method_spec.md 6.1). Swapping them silently would change
  published numbers, so the argument order is part of the contract.
* **Aggregate-only reporting is not acceptable.** Nighttime skill is
  substantially weaker than daytime skill (``docs/supplement_benchmarks.md``),
  so :func:`metrics_by_subset` reports ``all``, ``daytime`` and ``nighttime``
  together and no caller has to remember to ask for the split.

The functions are pure: they take array-likes, never a fitted model or a
configuration object, and they do not know where the predictions came from. The
validation workflow assembles them.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Final

import numpy as np
import pandas as pd

from rfrgapfill.config import DEFAULT_DAYTIME_THRESHOLD, MetricSubset, ValidationConfig

__all__ = [
    "DEFAULT_SUBSETS",
    "CoreMetrics",
    "MetricError",
    "MetricSubset",
    "SubsetMetrics",
    "bias",
    "core_metrics",
    "daytime_mask",
    "energy_balance_ratio",
    "metrics_by_subset",
    "metrics_from_config",
    "nighttime_mask",
    "r2",
    "regression_slope",
    "rmse",
    "subset_mask",
]


class MetricError(ValueError):
    """Raised when values cannot be scored.

    Reports invalid *data* - misaligned or non-numeric inputs, a day/night split
    requested without radiation - as opposed to
    :class:`~rfrgapfill.schema.ConfigError`, which reports an invalid
    configuration. Both subclass :class:`ValueError`.

    An empty subset is not an error: it scores as ``n=0`` with missing metrics,
    because a site with no nighttime observations is a real and reportable
    outcome rather than a failure.
    """


#: Subsets reported by default, matching :class:`~rfrgapfill.config.ValidationConfig`.
DEFAULT_SUBSETS: Final[tuple[MetricSubset, ...]] = (
    MetricSubset.ALL,
    MetricSubset.DAYTIME,
    MetricSubset.NIGHTTIME,
)


# ---------------------------------------------------------------------------
# Input coercion
# ---------------------------------------------------------------------------


def _reference_index(values: Sequence[object]) -> pd.Index | None:
    """Return the index of the first pandas input, or ``None`` if there is none."""
    for value in values:
        if isinstance(value, pd.Series):
            return value.index
    return None


def _as_float_array(
    values: object,
    *,
    field_name: str,
    index: pd.Index | None = None,
    length: int | None = None,
) -> np.ndarray:
    """Return ``values`` as a 1-D float array, checked against its companions.

    Pandas inputs are checked for index equality rather than merely for matching
    length: two series of the same length covering different timestamps would
    otherwise be scored against each other row by row.
    """
    if isinstance(values, pd.Series):
        if index is not None and not values.index.equals(index):
            raise MetricError(
                f"{field_name} does not share an index with the other inputs; "
                "align the series (for example with .reindex) before scoring"
            )
        array = values.to_numpy(dtype=float, na_value=np.nan)
    else:
        if isinstance(values, pd.DataFrame):
            raise MetricError(f"{field_name} must be one-dimensional, got a DataFrame")
        try:
            array = np.asarray(values, dtype=float)
        except (TypeError, ValueError) as error:
            raise MetricError(f"{field_name} is not numeric: {error}") from error
    array = np.atleast_1d(array)
    if array.ndim != 1:
        raise MetricError(f"{field_name} must be one-dimensional, got {array.ndim} dimensions")
    if length is not None and array.size != length:
        raise MetricError(
            f"{field_name} has {array.size} value(s) but the other inputs have {length}"
        )
    return array


def _pair(measured: object, filled: object) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(measured, filled)`` as aligned float arrays of equal length."""
    index = _reference_index((measured, filled))
    x = _as_float_array(measured, field_name="measured", index=index)
    y = _as_float_array(filled, field_name="filled", index=index, length=x.size)
    return x, y


def _scored(measured: object, filled: object) -> tuple[np.ndarray, np.ndarray]:
    """Return the pairs that can actually be scored: both values finite.

    A prediction is missing wherever the model had incomplete features
    (method_spec.md section 7), and the measured side is missing wherever the
    observation was not genuinely observed. Such pairs carry no information
    about skill, so they are dropped rather than counted as zero error.
    """
    x, y = _pair(measured, filled)
    valid = np.isfinite(x) & np.isfinite(y)
    return x[valid], y[valid]


# ---------------------------------------------------------------------------
# Core metrics (method_spec.md 6.1)
# ---------------------------------------------------------------------------


def r2(measured: object, filled: object) -> float:
    """Return the coefficient of determination of ``filled`` against ``measured``.

    ``1 - SS_res / SS_tot``, with residuals taken about the measured values and
    ``SS_tot`` about their mean - the usual regression-free definition, which
    (unlike a squared correlation) penalises a prediction that is well
    correlated but biased or mis-scaled. Returns NaN when fewer than two pairs
    can be scored or when the measured values do not vary, because the ratio is
    then undefined rather than zero.
    """
    x, y = _scored(measured, filled)
    if x.size < 2:
        return float("nan")
    total = float(np.sum((x - x.mean()) ** 2))
    if total == 0.0:
        return float("nan")
    residual = float(np.sum((x - y) ** 2))
    return 1.0 - residual / total


def regression_slope(measured: object, filled: object) -> float:
    """Return the ordinary least-squares slope of ``filled`` on ``measured``.

    Orientation is the paper's and is fixed: ``x = measured``, ``y = filled``,
    fitted with an intercept, so 1.0 means the filled values track the measured
    amplitude and a slope below 1.0 means the model under-predicts the range.
    Returns NaN when fewer than two pairs can be scored or when the measured
    values do not vary.
    """
    x, y = _scored(measured, filled)
    if x.size < 2:
        return float("nan")
    centred_x = x - x.mean()
    variance = float(np.dot(centred_x, centred_x))
    if variance == 0.0:
        return float("nan")
    return float(np.dot(centred_x, y - y.mean()) / variance)


def rmse(measured: object, filled: object) -> float:
    """Return the root mean squared error of ``filled`` against ``measured``.

    In the units of the target. Returns NaN when no pair can be scored.
    """
    x, y = _scored(measured, filled)
    if x.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((y - x) ** 2)))


def bias(measured: object, filled: object) -> float:
    """Return the paper's bias, ``(sum(filled) - sum(measured)) / n``.

    Equivalent to the mean prediction error for equal-weighted observations, and
    signed: positive means the filled values overestimate. Returns NaN when no
    pair can be scored.
    """
    x, y = _scored(measured, filled)
    if x.size == 0:
        return float("nan")
    return float((y.sum() - x.sum()) / x.size)


def energy_balance_ratio(
    sensible_heat: object,
    latent_heat: object,
    net_radiation: object,
    soil_heat_flux: object,
) -> float:
    """Return ``sum(H + LE) / sum(NETRAD - G)`` (method_spec.md 6.4).

    Summed over the rows where all four fluxes are finite, so the numerator and
    the denominator always cover the same rows. Returns NaN when no such row
    exists or when the available energy sums to exactly zero, rather than
    reporting an infinite ratio.

    Called twice over the same artificial-gap intervals - once with measured H
    and LE, once with filled - the two ratios and their difference say whether
    gap filling distorted the site's energy balance.
    """
    index = _reference_index((sensible_heat, latent_heat, net_radiation, soil_heat_flux))
    h = _as_float_array(sensible_heat, field_name="sensible_heat", index=index)
    le = _as_float_array(latent_heat, field_name="latent_heat", index=index, length=h.size)
    rn = _as_float_array(net_radiation, field_name="net_radiation", index=index, length=h.size)
    g = _as_float_array(soil_heat_flux, field_name="soil_heat_flux", index=index, length=h.size)
    valid = np.isfinite(h) & np.isfinite(le) & np.isfinite(rn) & np.isfinite(g)
    if not valid.any():
        return float("nan")
    available = float(np.sum(rn[valid] - g[valid]))
    if available == 0.0:
        return float("nan")
    return float(np.sum(h[valid] + le[valid]) / available)


# ---------------------------------------------------------------------------
# Day/night masks (method_spec.md 6.2)
# ---------------------------------------------------------------------------


def _shortwave_series(shortwave: object) -> pd.Series:
    """Return ``shortwave`` as a float Series, keeping a pandas index when given."""
    index = shortwave.index if isinstance(shortwave, pd.Series) else None
    values = _as_float_array(shortwave, field_name="shortwave")
    series: pd.Series = pd.Series(values, index=index, name="shortwave")
    return series


def daytime_mask(
    shortwave: object,
    *,
    threshold: float = DEFAULT_DAYTIME_THRESHOLD,
) -> pd.Series:
    """Return ``shortwave > threshold``: the paper's daytime definition.

    The threshold is downward shortwave radiation in W m-2 and defaults to the
    paper's 20; pass :attr:`~rfrgapfill.config.ValidationConfig.daytime_threshold`
    to honour a run's configuration. The comparison is strict, so a value of
    exactly the threshold is nighttime.

    Rows with missing radiation are neither daytime nor nighttime: the class is
    unknown, and guessing one would move real observations into the wrong
    subset. They still count towards the ``all`` subset, so day and night need
    not sum to it - :attr:`SubsetMetrics.n_missing_shortwave` records how many.
    """
    mask: pd.Series = _shortwave_series(shortwave) > _check_threshold(threshold)
    return mask


def nighttime_mask(
    shortwave: object,
    *,
    threshold: float = DEFAULT_DAYTIME_THRESHOLD,
) -> pd.Series:
    """Return ``shortwave <= threshold``: the complement of :func:`daytime_mask`.

    Complementary only over rows with observed radiation; see
    :func:`daytime_mask` for missing values.
    """
    mask: pd.Series = _shortwave_series(shortwave) <= _check_threshold(threshold)
    return mask


def subset_mask(
    shortwave: object,
    subset: MetricSubset | str,
    *,
    threshold: float = DEFAULT_DAYTIME_THRESHOLD,
) -> pd.Series:
    """Return the boolean mask selecting ``subset``.

    ``all`` selects every row, including rows with missing radiation; ``daytime``
    and ``nighttime`` apply :func:`daytime_mask` and :func:`nighttime_mask`.
    """
    chosen = MetricSubset.coerce(subset)
    if chosen is MetricSubset.DAYTIME:
        return daytime_mask(shortwave, threshold=threshold)
    if chosen is MetricSubset.NIGHTTIME:
        return nighttime_mask(shortwave, threshold=threshold)
    series = _shortwave_series(shortwave)
    every_row: pd.Series = pd.Series(
        np.ones(len(series), dtype=bool), index=series.index, name="shortwave"
    )
    return every_row


def _check_threshold(threshold: float) -> float:
    """Return ``threshold`` as a finite float or raise :class:`MetricError`."""
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise MetricError(f"daytime threshold must be a number, got {threshold!r}")
    value = float(threshold)
    if not np.isfinite(value):
        raise MetricError(f"daytime threshold must be finite, got {threshold!r}")
    return value


# ---------------------------------------------------------------------------
# Reported results
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CoreMetrics:
    """The four core metrics of method_spec.md 6.1 over one set of pairs."""

    #: Pairs actually scored: both the measurement and the prediction finite.
    n: int
    #: Coefficient of determination.
    r2: float
    #: Regression slope, measured on x and filled on y.
    slope: float
    #: Root mean squared error, in target units.
    rmse: float
    #: ``(sum(filled) - sum(measured)) / n``.
    bias: float

    @property
    def is_empty(self) -> bool:
        """Whether no pair could be scored, so every metric is missing."""
        return self.n == 0

    def to_dict(self) -> dict[str, Any]:
        """Return a serialisable representation for the run report.

        Metrics that are undefined stay NaN rather than becoming ``0.0``: a
        subset that could not be scored must not read as a perfect or a failed
        one.
        """
        return {
            "n": self.n,
            "r2": self.r2,
            "slope": self.slope,
            "rmse": self.rmse,
            "bias": self.bias,
        }


def core_metrics(measured: object, filled: object) -> CoreMetrics:
    """Return :class:`CoreMetrics` for one set of measured/filled pairs.

    Computed over the pairs where both sides are finite; ``n`` reports how many
    that was, so a metric can never be read without its sample size.
    """
    x, y = _scored(measured, filled)
    return CoreMetrics(
        n=int(x.size),
        r2=r2(x, y),
        slope=regression_slope(x, y),
        rmse=rmse(x, y),
        bias=bias(x, y),
    )


@dataclass(frozen=True)
class SubsetMetrics:
    """Core metrics reported over the all/daytime/nighttime subsets.

    The result of one validation run's scoring step. It always carries the
    threshold it used and how many rows had no radiation to classify, so a
    reported nighttime number can be traced back to the split that produced it.
    """

    #: Metrics keyed by subset, in the order they were requested.
    metrics: Mapping[MetricSubset, CoreMetrics]
    #: The daytime threshold in W m-2 that produced the split.
    daytime_threshold: float
    #: Scored pairs whose radiation was missing, so they joined neither day nor
    #: night. It is exactly the shortfall in ``daytime.n + nighttime.n`` against
    #: ``all.n``, which is why the count is reported next to the metrics.
    n_missing_shortwave: int

    def __getitem__(self, subset: MetricSubset | str) -> CoreMetrics:
        """Return the metrics for ``subset``, accepting ``"day"``/``"night"``."""
        chosen = MetricSubset.coerce(subset)
        try:
            return self.metrics[chosen]
        except KeyError:
            raise KeyError(
                f"{chosen.value} was not reported; this run scored: "
                f"{', '.join(subset.value for subset in self.metrics)}"
            ) from None

    def __contains__(self, subset: object) -> bool:
        try:
            return MetricSubset.coerce(subset) in self.metrics
        except ValueError:
            return False

    @property
    def all(self) -> CoreMetrics:
        """Metrics over every scored pair."""
        return self[MetricSubset.ALL]

    @property
    def daytime(self) -> CoreMetrics:
        """Metrics over pairs with ``shortwave > daytime_threshold``."""
        return self[MetricSubset.DAYTIME]

    @property
    def nighttime(self) -> CoreMetrics:
        """Metrics over pairs with ``shortwave <= daytime_threshold``."""
        return self[MetricSubset.NIGHTTIME]

    def to_frame(self) -> pd.DataFrame:
        """Return one row per subset, for tables and multi-site reports."""
        frame: pd.DataFrame = pd.DataFrame(
            [metrics.to_dict() for metrics in self.metrics.values()],
            index=pd.Index([subset.value for subset in self.metrics], name="subset"),
        )
        return frame

    def to_dict(self) -> dict[str, Any]:
        """Return a serialisable representation for the run report."""
        return {
            "daytime_threshold": self.daytime_threshold,
            "n_missing_shortwave": self.n_missing_shortwave,
            "subsets": {
                subset.value: metrics.to_dict() for subset, metrics in self.metrics.items()
            },
        }


# ---------------------------------------------------------------------------
# Subset reporting (method_spec.md 6.2)
# ---------------------------------------------------------------------------


def metrics_by_subset(
    measured: object,
    filled: object,
    shortwave: object | None = None,
    *,
    threshold: float = DEFAULT_DAYTIME_THRESHOLD,
    subsets: Sequence[MetricSubset | str] = DEFAULT_SUBSETS,
) -> SubsetMetrics:
    """Return core metrics for the ``all``, ``daytime`` and ``nighttime`` subsets.

    ``measured`` and ``filled`` are the known measurements inside the artificial
    gaps and the model's predictions for them; ``shortwave`` is the downward
    shortwave radiation on the same rows, which splits day from night. Pandas
    inputs must share an index - a mismatch raises rather than being scored row
    by row.

    ``shortwave`` may be omitted only when ``subsets`` asks for ``all`` alone;
    otherwise the split cannot be made, and the omission is an error rather than
    a silently missing subset.

    The scored counts satisfy ``all.n == daytime.n + nighttime.n +
    n_missing_shortwave`` whenever all three subsets are reported.
    """
    limit = _check_threshold(threshold)
    requested = _requested_subsets(subsets)
    index = _reference_index((measured, filled, shortwave))
    x = _as_float_array(measured, field_name="measured", index=index)
    y = _as_float_array(filled, field_name="filled", index=index, length=x.size)

    if shortwave is None:
        unclassifiable = [subset for subset in requested if subset is not MetricSubset.ALL]
        if unclassifiable:
            raise MetricError(
                "shortwave radiation is required to report the "
                f"{', '.join(subset.value for subset in unclassifiable)} subset(s); "
                "pass the shortwave driver or request only the 'all' subset"
            )
        radiation = np.full(x.size, np.nan)
    else:
        radiation = _as_float_array(shortwave, field_name="shortwave", index=index, length=x.size)

    results: dict[MetricSubset, CoreMetrics] = {}
    for subset in requested:
        selected = subset_mask(radiation, subset, threshold=limit).to_numpy(dtype=bool)
        results[subset] = core_metrics(x[selected], y[selected])

    scored = np.isfinite(x) & np.isfinite(y)
    return SubsetMetrics(
        metrics=MappingProxyType(results),
        daytime_threshold=limit,
        n_missing_shortwave=int(np.sum(scored & ~np.isfinite(radiation))),
    )


def metrics_from_config(
    measured: object,
    filled: object,
    shortwave: object | None = None,
    *,
    config: ValidationConfig,
) -> SubsetMetrics:
    """Return :func:`metrics_by_subset` using a run's validation configuration.

    The threshold and the reported subsets come from ``config``, so a run cannot
    report one split while its manifest records another.
    """
    return metrics_by_subset(
        measured,
        filled,
        shortwave,
        threshold=config.daytime_threshold,
        subsets=config.metric_subsets,
    )


def _requested_subsets(subsets: Sequence[MetricSubset | str]) -> tuple[MetricSubset, ...]:
    """Return the requested subsets, validated, deduplicated and order-preserving."""
    if isinstance(subsets, str) or not isinstance(subsets, Sequence):
        raise MetricError(f"subsets must be a sequence of metric subsets, got {subsets!r}")
    if not subsets:
        raise MetricError("subsets must name at least one metric subset")
    ordered: list[MetricSubset] = []
    for entry in subsets:
        subset = MetricSubset.coerce(entry)
        if subset not in ordered:
            ordered.append(subset)
    return tuple(ordered)
