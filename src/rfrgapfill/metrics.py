"""Validation metrics (``docs/method_spec.md`` section 6).

The paper's evaluation quantities, and only those: the four core metrics scored
on paired measured and predicted values - ``R2``, the regression slope, ``RMSE``
and the paper's ``bias`` - plus the energy-balance ratio for H and LE. Each is an
independent function of the values handed to it. Nothing here knows about gaps,
models, or where a subset came from, so the same functions score a whole
validation run, one gap class, one daytime subset, or a hand-built fixture.

What this module deliberately does not do: choose the subsets. Day/night
splitting at ``daytime_threshold`` and grouping by gap class are the validation
layer's job, which hands the resulting slices here one at a time.

Pairing
-------

Every paired metric applies the same two rules before it computes anything, and
:class:`CoreMetrics` reports what they cost:

* **Alignment is checked, never performed.** Two :class:`pandas.Series` with
  different indexes raise :class:`MetricError` rather than being aligned into a
  quiet union of missing values. Silent realignment is how a metric ends up
  comparing a prediction against the wrong timestamp's measurement.
* **Incomplete pairs are dropped, both sides at once.** A row is scored only when
  measured *and* predicted are finite - so a row the model left unfilled because
  its predictors were incomplete (``docs/method_spec.md`` section 5) removes
  itself from the score instead of poisoning it. ``n`` is the number of pairs
  that survived, and it is the ``n`` in the bias denominator.

Undefined is ``None``, not ``NaN``
----------------------------------

An empty subset has no ``RMSE``; a subset whose measurements are all identical
has no ``R2`` and no slope; an interval whose available energy sums to zero has
no ``EBR``. Every such case returns ``None``. That keeps an undefined metric
distinguishable from a computed one, survives the trip into a JSON run manifest
as ``null``, and refuses to propagate silently the way ``NaN`` does through a
later mean or comparison.

The ``R2`` ambiguity (A12)
--------------------------

The article reports ``R2`` next to a regression slope and does not say which of
the two standard quantities it means. :class:`~rfrgapfill.config.R2Definition`
exposes both. The default is ``residual`` - ``1 - SS_res / SS_tot``, scikit-learn's
``r2_score`` - because that is the unqualified meaning of "coefficient of
determination", and because the alternative, the squared Pearson correlation, is
invariant to any affine rescaling of the predictions and so cannot see a
systematic offset at all. A package that reports bias in the next column should
not report an ``R2`` that ignores it.

The two coincide exactly when the predictions are unbiased and their spread has
shrunk to ``r`` times the measured spread, which is the ordinary behaviour of a
forest regressing toward the mean. Supplementary Table S3 sits close to that
regime - its median ``R2`` and median slope agree to about 0.01 in every RFR row
- so the published numbers do not settle the question either way, and neither
reading may be called paper exact.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Union

import numpy as np
import pandas as pd

from rfrgapfill.config import R2Definition
from rfrgapfill.schema import FrozenRecord

__all__ = [
    "CoreMetrics",
    "EnergyBalanceComparison",
    "MetricError",
    "MetricInput",
    "R2Definition",
    "bias",
    "compare_energy_balance",
    "core_metrics",
    "energy_balance_ratio",
    "r2",
    "regression_slope",
    "rmse",
]

#: What a metric accepts for one series of values.
MetricInput = Union[pd.Series, "np.ndarray[Any, Any]", Sequence[float]]


class MetricError(ValueError):
    """Raised when values cannot be scored as given.

    Reports invalid *data* - series of different lengths, two series indexed
    differently, values that are not numbers. An invalid *configuration* still
    raises :class:`~rfrgapfill.schema.ConfigError`.
    """


# ---------------------------------------------------------------------------
# Pairing
# ---------------------------------------------------------------------------


def _as_array(values: MetricInput, *, name: str) -> np.ndarray[Any, Any]:
    """Return ``values`` as a one-dimensional float array."""
    if isinstance(values, pd.DataFrame):
        raise MetricError(f"{name} must be a single series of values, got a DataFrame")
    data = values.to_numpy() if isinstance(values, pd.Series) else values
    try:
        array = np.asarray(data, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise MetricError(f"{name} must hold numeric values: {error}") from error
    if array.ndim != 1:
        raise MetricError(f"{name} must be one-dimensional, got {array.ndim} dimensions")
    return array


def _aligned(**named: MetricInput) -> tuple[np.ndarray[Any, Any], ...]:
    """Return the named inputs as float arrays, refusing to align them silently.

    Series are required to carry the *same* index, not merely the same length: a
    metric is meaningful only when row ``i`` of each input describes the same half
    hour, and pandas would otherwise be happy to align two different time axes
    into a union of missing values.
    """
    arrays: dict[str, np.ndarray[Any, Any]] = {}
    reference: tuple[str, pd.Index] | None = None
    for name, values in named.items():
        if isinstance(values, pd.Series):
            if reference is None:
                reference = (name, values.index)
            elif not values.index.equals(reference[1]):
                raise MetricError(
                    f"{name} and {reference[0]} are indexed differently; metrics compare "
                    "values row by row and will not align two series for you"
                )
        arrays[name] = _as_array(values, name=name)
    lengths = {name: array.size for name, array in arrays.items()}
    if len(set(lengths.values())) > 1:
        detail = ", ".join(f"{name}={length}" for name, length in lengths.items())
        raise MetricError(f"every series must be the same length, got {detail}")
    return tuple(arrays.values())


def _complete(*arrays: np.ndarray[Any, Any]) -> tuple[np.ndarray[Any, Any], ...]:
    """Return ``arrays`` restricted to the rows where every one of them is finite."""
    keep = np.ones(arrays[0].size, dtype=bool)
    for array in arrays:
        keep &= np.isfinite(array)
    return tuple(array[keep] for array in arrays)


def _paired(
    measured: MetricInput, predicted: MetricInput
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Return the complete (measured, predicted) pairs, in order."""
    pair = _complete(*_aligned(measured=measured, predicted=predicted))
    return pair[0], pair[1]


def _defined(value: float) -> float | None:
    """Return ``value`` when it is a real number, otherwise ``None``."""
    return value if math.isfinite(value) else None


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------


def _r2(
    measured: np.ndarray[Any, Any],
    predicted: np.ndarray[Any, Any],
    definition: R2Definition,
) -> float | None:
    """Compute ``R2`` over complete pairs."""
    if measured.size == 0:
        return None
    centred = measured - measured.mean()
    total = float(centred @ centred)
    if total == 0.0:
        # Every measurement is identical: there is no variance to account for, so
        # neither definition has a value. A single pair lands here too.
        return None
    if definition is R2Definition.RESIDUAL:
        residual = predicted - measured
        return _defined(1.0 - float(residual @ residual) / total)
    spread = predicted - predicted.mean()
    predicted_total = float(spread @ spread)
    if predicted_total == 0.0:
        # A constant prediction correlates with nothing.
        return None
    covariance = float(centred @ spread)
    return _defined(covariance * covariance / (total * predicted_total))


def r2(
    measured: MetricInput,
    predicted: MetricInput,
    *,
    definition: R2Definition | str = R2Definition.RESIDUAL,
) -> float | None:
    """Return the coefficient of determination, or ``None`` where it is undefined.

    :param measured: the known measurements (the paper's ``x``).
    :param predicted: the model's values for the same rows (the paper's filled
        series, ``y``).
    :param definition: which quantity to report (ambiguity A12, see the module
        docstring). ``residual`` is ``1 - SS_res / SS_tot`` and falls below zero
        for predictions worse than the measured mean; ``squared_correlation`` is
        the squared Pearson correlation and stays within ``[0, 1]``.

    Undefined - and so ``None`` - for an empty subset, for a single pair, and
    wherever every measurement is identical, since a constant series has no
    variance to explain. ``squared_correlation`` is additionally undefined for a
    constant prediction.
    """
    return _r2(*_paired(measured, predicted), R2Definition.coerce(definition))


def _slope(
    measured: np.ndarray[Any, Any],
    predicted: np.ndarray[Any, Any],
    *,
    fit_intercept: bool,
) -> float | None:
    """Compute the least-squares slope over complete pairs."""
    if measured.size == 0:
        return None
    if fit_intercept:
        x = measured - measured.mean()
        y = predicted - predicted.mean()
    else:
        x, y = measured, predicted
    denominator = float(x @ x)
    if denominator == 0.0:
        return None
    return _defined(float(x @ y) / denominator)


def regression_slope(
    measured: MetricInput,
    predicted: MetricInput,
    *,
    fit_intercept: bool = True,
) -> float | None:
    """Return the least-squares slope of predicted on measured.

    The orientation is fixed by the specification and is not an option:
    ``x = measured``, ``y = predicted`` (``docs/method_spec.md`` section 6.1).
    Regressing the other way answers a different question and reports a different
    number for the same data.

    :param measured: the known measurements, ``x``.
    :param predicted: the model's values for the same rows, ``y``.
    :param fit_intercept: whether the fit carries an intercept. The default, and
        the ordinary reading of "linear-regression slope"; ``False`` forces the
        line through the origin, which the paper neither states nor rules out.

    A slope of 1 means the predictions track the measurements across their range;
    the shrinkage typical of a forest shows up as a slope below 1. Undefined - and
    so ``None`` - for an empty subset, and wherever the measurements are constant,
    since a vertical scatter has no slope.
    """
    return _slope(*_paired(measured, predicted), fit_intercept=fit_intercept)


def _rmse(measured: np.ndarray[Any, Any], predicted: np.ndarray[Any, Any]) -> float | None:
    """Compute the root mean squared error over complete pairs."""
    if measured.size == 0:
        return None
    residual = predicted - measured
    return _defined(math.sqrt(float(residual @ residual) / residual.size))


def rmse(measured: MetricInput, predicted: MetricInput) -> float | None:
    """Return the root mean squared error, in the units of the flux.

    :param measured: the known measurements.
    :param predicted: the model's values for the same rows.

    ``None`` for an empty subset. Errors of both signs contribute, so unlike
    :func:`bias` this cannot cancel: a run with an ``RMSE`` of 30 W m-2 and a bias
    of 0 is making large errors that happen to balance.
    """
    return _rmse(*_paired(measured, predicted))


def _bias(measured: np.ndarray[Any, Any], predicted: np.ndarray[Any, Any]) -> float | None:
    """Compute the paper's bias over complete pairs."""
    if measured.size == 0:
        return None
    difference = float(predicted.sum()) - float(measured.sum())
    return _defined(difference / measured.size)


def bias(measured: MetricInput, predicted: MetricInput) -> float | None:
    """Return the paper's bias, ``(sum(predicted) - sum(measured)) / n``.

    :param measured: the known measurements.
    :param predicted: the model's values for the same rows.

    Computed in the published form (``docs/method_spec.md`` section 6.1), which
    for equally weighted observations is the mean prediction error. The sign is
    the direction of the error in the predictions: positive means the fill runs
    high. ``n`` counts the complete pairs, so a row the model left unfilled is
    absent from both sums and from the denominator alike. ``None`` for an empty
    subset.
    """
    return _bias(*_paired(measured, predicted))


@dataclass(frozen=True)
class CoreMetrics(FrozenRecord):
    """The four core metrics of ``docs/method_spec.md`` section 6.1, scored together.

    Produced by :func:`core_metrics` for one subset of one target - all of it, its
    daytime rows, one gap class - and carrying the row accounting that makes the
    numbers readable: :attr:`n` pairs were scored out of :attr:`n_offered` rows,
    and the difference is rows that had no measurement, no prediction, or neither.

    A field is ``None`` where its metric is undefined for this subset, rather than
    zero or ``NaN``; :attr:`is_empty` separates "nothing to score" from "scored,
    but the metric has no value here".
    """

    #: Complete measured/predicted pairs the metrics were computed from.
    n: int
    #: Rows offered, before incomplete pairs were dropped.
    n_offered: int
    #: Coefficient of determination, under :attr:`r2_definition`.
    r2: float | None
    #: Least-squares slope of predicted on measured.
    slope: float | None
    #: Root mean squared error, in the units of the flux.
    rmse: float | None
    #: ``(sum(predicted) - sum(measured)) / n``.
    bias: float | None
    #: Which coefficient of determination :attr:`r2` is (A12).
    r2_definition: R2Definition | str = R2Definition.RESIDUAL

    def __post_init__(self) -> None:
        if self.n < 0 or self.n_offered < 0:
            raise MetricError("row counts cannot be negative")
        if self.n > self.n_offered:
            raise MetricError(
                f"scored {self.n} pair(s) out of {self.n_offered} offered row(s); "
                "dropping incomplete pairs cannot increase the count"
            )
        object.__setattr__(self, "r2_definition", R2Definition.coerce(self.r2_definition))

    @property
    def is_empty(self) -> bool:
        """Whether no complete pair was available to score."""
        return self.n == 0

    @property
    def dropped_incomplete(self) -> int:
        """Offered rows that lacked a measurement, a prediction, or both."""
        return self.n_offered - self.n

    @property
    def r2_kind(self) -> R2Definition:
        """The validated ``R2`` definition (narrowed from the input union)."""
        definition = self.r2_definition
        assert isinstance(definition, R2Definition)
        return definition

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary for the run manifest."""
        return {
            "n": self.n,
            "n_offered": self.n_offered,
            "dropped_incomplete": self.dropped_incomplete,
            "r2": self.r2,
            "r2_definition": self.r2_kind.value,
            "slope": self.slope,
            "rmse": self.rmse,
            "bias": self.bias,
        }


def core_metrics(
    measured: MetricInput,
    predicted: MetricInput,
    *,
    definition: R2Definition | str = R2Definition.RESIDUAL,
    fit_intercept: bool = True,
) -> CoreMetrics:
    """Score one subset on all four core metrics at once.

    A convenience over :func:`r2`, :func:`regression_slope`, :func:`rmse` and
    :func:`bias` that pairs the inputs once and reports how many rows survived.
    The values are identical to calling the four functions separately with the
    same arguments - they share the same kernels.

    :param measured: the known measurements.
    :param predicted: the model's values for the same rows.
    :param definition: which ``R2`` to report (A12).
    :param fit_intercept: whether the slope's fit carries an intercept.
    """
    r2_definition = R2Definition.coerce(definition)
    offered = _aligned(measured=measured, predicted=predicted)
    clean_measured, clean_predicted = _complete(*offered)
    return CoreMetrics(
        n=int(clean_measured.size),
        n_offered=int(offered[0].size),
        r2=_r2(clean_measured, clean_predicted, r2_definition),
        slope=_slope(clean_measured, clean_predicted, fit_intercept=fit_intercept),
        rmse=_rmse(clean_measured, clean_predicted),
        bias=_bias(clean_measured, clean_predicted),
        r2_definition=r2_definition,
    )


# ---------------------------------------------------------------------------
# Energy-balance ratio
# ---------------------------------------------------------------------------


def _ebr(
    sensible_heat: np.ndarray[Any, Any],
    latent_heat: np.ndarray[Any, Any],
    net_radiation: np.ndarray[Any, Any],
    soil_heat_flux: np.ndarray[Any, Any],
) -> float | None:
    """Compute ``sum(H + LE) / sum(NETRAD - G)`` over complete rows."""
    if sensible_heat.size == 0:
        return None
    available_energy = float((net_radiation - soil_heat_flux).sum())
    if available_energy == 0.0:
        # No available energy over this interval: the ratio has no value, and
        # dividing anyway would report an arbitrarily large closure.
        return None
    turbulent_flux = float((sensible_heat + latent_heat).sum())
    return _defined(turbulent_flux / available_energy)


def energy_balance_ratio(
    *,
    sensible_heat: MetricInput,
    latent_heat: MetricInput,
    net_radiation: MetricInput,
    soil_heat_flux: MetricInput,
) -> float | None:
    """Return ``sum(H + LE) / sum(NETRAD - G)`` (``docs/method_spec.md`` section 6.4).

    How well the turbulent fluxes account for the available energy over the rows
    given. A ratio of 1 is closure; eddy-covariance sites typically fall short of
    it, and the quantity of interest is whether *filling* the fluxes changes that
    - see :func:`compare_energy_balance`.

    Every argument is keyword-only. All four are fluxes in W m-2, and swapping
    ``net_radiation`` for ``soil_heat_flux`` positionally would return a plausible
    wrong number rather than fail.

    A row contributes only when all four of its values are finite, so numerator
    and denominator are always summed over exactly the same rows. Summing each
    over whatever it happened to have would compare the turbulent flux of one set
    of half hours against the available energy of another.

    Returns ``None`` when no row is complete, and when the available energy sums
    to zero - which a short interval can do by cancellation between night and day.
    """
    rows = _complete(
        *_aligned(
            sensible_heat=sensible_heat,
            latent_heat=latent_heat,
            net_radiation=net_radiation,
            soil_heat_flux=soil_heat_flux,
        )
    )
    return _ebr(*rows)


@dataclass(frozen=True)
class EnergyBalanceComparison(FrozenRecord):
    """Measured against filled energy-balance ratio, over the same rows.

    The pair required by ``docs/method_spec.md`` section 6.4: the ratio from the
    measured H and LE, the ratio from the filled H and LE, and their difference.
    Both are computed over one shared set of rows, so :attr:`difference` reports
    what the fill did to closure and not what a different row set would have done.
    """

    #: Rows both ratios were computed from.
    n: int
    #: Rows offered, before incomplete ones were dropped.
    n_offered: int
    #: ``EBR`` from the measured H and LE.
    measured: float | None
    #: ``EBR`` from the filled H and LE.
    filled: float | None
    #: ``filled - measured``, or ``None`` unless both are defined.
    difference: float | None

    @property
    def is_empty(self) -> bool:
        """Whether no complete row was available."""
        return self.n == 0

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary for the run manifest."""
        return {
            "n": self.n,
            "n_offered": self.n_offered,
            "measured_ebr": self.measured,
            "filled_ebr": self.filled,
            "difference": self.difference,
        }


def compare_energy_balance(
    *,
    measured_sensible_heat: MetricInput,
    measured_latent_heat: MetricInput,
    filled_sensible_heat: MetricInput,
    filled_latent_heat: MetricInput,
    net_radiation: MetricInput,
    soil_heat_flux: MetricInput,
) -> EnergyBalanceComparison:
    """Return the measured and filled energy-balance ratios and their difference.

    The section 6.4 comparison: the same artificial-gap rows scored twice, once
    with the measured H and LE and once with the values the model put there.

    A row is used only where all six series are finite, so both ratios cover an
    identical set of half hours. Scoring each over its own complete rows would let
    the difference reflect the change in row set as much as the change in flux,
    which is the one thing this comparison exists to rule out.

    Returns ``None`` for either ratio where it is undefined, and for
    :attr:`~EnergyBalanceComparison.difference` unless both are defined.
    """
    offered = _aligned(
        measured_sensible_heat=measured_sensible_heat,
        measured_latent_heat=measured_latent_heat,
        filled_sensible_heat=filled_sensible_heat,
        filled_latent_heat=filled_latent_heat,
        net_radiation=net_radiation,
        soil_heat_flux=soil_heat_flux,
    )
    measured_h, measured_le, filled_h, filled_le, radiation, soil = _complete(*offered)
    measured_ratio = _ebr(measured_h, measured_le, radiation, soil)
    filled_ratio = _ebr(filled_h, filled_le, radiation, soil)
    difference = (
        filled_ratio - measured_ratio
        if measured_ratio is not None and filled_ratio is not None
        else None
    )
    return EnergyBalanceComparison(
        n=int(measured_h.size),
        n_offered=int(offered[0].size),
        measured=measured_ratio,
        filled=filled_ratio,
        difference=difference,
    )
