# Method specification — Zhu et al. (2022) RFR eddy-covariance gap filling

Frozen implementation specification. This document is the contract the code is
checked against. Machine-readable companion: [`method_spec.yaml`](method_spec.yaml).
Numerical reproduction targets: [`supplement_benchmarks.md`](supplement_benchmarks.md).

**Source:** Zhu, S., Clement, R., McCalmont, J., Davies, C. A., & Hill, T. (2022).
*Stable gap-filling for longer eddy covariance data gaps: A globally validated
machine-learning approach for carbon dioxide, water, and energy fluxes.*
Agricultural and Forest Meteorology, 314, 108777.
<https://doi.org/10.1016/j.agrformet.2021.108777>

**Primary rule:** nothing in this document may be labelled "paper exact" unless it
is stated in the article or its supplements. Everything else is a documented
default and appears in [Known ambiguities](#known-ambiguities).

---

## 1. Target methods

| Method | Definition |
|---|---|
| `RFR3` | Random Forest + receptive limiter, using the 3 MDS-equivalent meteorological drivers. |
| `RFR10` | Random Forest + receptive limiter, using all 10 meteorological drivers. |
| `ORF` | Benchmark only. Same estimator family and same driver set as the RFR it is compared against, **without** receptive-limiter features (Supplementary Figure S1). |
| `MDS` | External reference method. Not implemented; used only as a published comparison in benchmark tables. |

Target fluxes: `NEE` (CO2 net ecosystem exchange), `H` (sensible heat), `LE`
(latent heat). The API stays generic over any continuous flux given valid drivers.

Models are fitted **per site and per target**. No pooling of sites in the
paper-faithful workflow.

---

## 2. Canonical driver lists

Canonical names are internal; FLUXNET2015 names are the reference mapping only.
Column names must never be hard-coded — every driver is resolved through a
`ColumnMap`.

### RFR3 (3 drivers)

| # | Canonical name | FLUXNET2015 column | Units |
|---|---|---|---|
| 1 | `shortwave` | `SW_IN_F` | W m-2 |
| 2 | `vpd` | `VPD_F_MDS` | hPa |
| 3 | `air_temperature` | `TA_F_MDS` | degC |

### RFR10 (RFR3 + 7 drivers)

| # | Canonical name | FLUXNET2015 column | Units |
|---|---|---|---|
| 4 | `net_radiation` | `NETRAD` | W m-2 |
| 5 | `wind_speed` | `WS` | m s-1 |
| 6 | `wind_direction` | `WD` | degrees |
| 7 | `soil_heat_flux` | `G_F_MDS` | W m-2 |
| 8 | `soil_temperature` | `TS_F_MDS` | degC |
| 9 | `relative_humidity` | `RH` | % |
| 10 | `soil_water_content` | `SWC_F_MDS` | % |

`net_radiation` and `soil_heat_flux` are additionally required for the EBR metric
(section 6.4) regardless of the selected mode.

---

## 3. Receptive-limiter features

The receptive limiter is the paper's feature-engineering stage and is the
distinguishing component of RFR versus ORF. All four groups below are deterministic
transformers with a stable, documented feature order.

### 3.1 Radiation category

Categorical, derived from `shortwave`:

| Class | Rule (default `boundary_convention="medium_inclusive"`) |
|---|---|
| `weak` | `SW_IN < 10` W m-2 |
| `medium` | `10 <= SW_IN <= 100` W m-2 |
| `strong` | `SW_IN > 100` W m-2 |

Thresholds live in `FeatureConfig.radiation_thresholds = (10.0, 100.0)`. The bins
are inclusive and exhaustive over the reals. Missing `shortwave` yields a missing
category, not a default class.

### 3.2 Elapsed hours (time distance)

Continuous feature `time_distance_hours` = elapsed hours since the first timestamp
of the site time series, computed from actual timestamp differences, never row
position:

```text
time_distance_hours = (timestamp - timestamp.min()) / 1 hour
```

At 30-minute cadence this yields 0.0, 0.5, 1.0, ... Purpose: represent gradual
ecosystem growth, degradation, and other long-term temporal trends.

### 3.3 Hemisphere-aware season

Categorical, derived from calendar month plus hemisphere.

| Month | Northern | Southern |
|---|---|---|
| Dec, Jan, Feb | `winter` | `summer` |
| Mar, Apr, May | `spring` | `autumn` |
| Jun, Jul, Aug | `summer` | `winter` |
| Sep, Oct, Nov | `autumn` | `spring` |

Hemisphere must be supplied explicitly (`hemisphere="north"|"south"`) or inferred
from latitude with the documented rule `latitude >= 0 -> north`. One of the two is
required; there is no silent default.

### 3.4 Daily target-flux statistics

Per calendar day, from quality-controlled observations of the target flux only:

| Feature | Definition |
|---|---|
| `<target>_daily_q1` | 25th percentile |
| `<target>_daily_q2` | 50th percentile (median) |
| `<target>_daily_q3` | 75th percentile |
| `<target>_daily_std` | standard deviation |

Each daily value is joined back to every timestamp of that day. Stated purpose in
the paper: reduce the effect of potential outliers. These features are
target-specific — a separate feature matrix exists per target.

Quantiles use linear interpolation between order statistics (the numpy/pandas
default). The standard deviation uses `daily_std_ddof = 1`, the sample standard
deviation (ambiguity A11); a day with a single visible observation therefore has
defined quartiles and an undefined standard deviation.

A day with fewer than `min_daily_observations` visible target values is handled
by `daily_statistic_strategy` (ambiguity A4). The paper does not say what it did
with such a day, so the choice is explicit, and every option draws exclusively on
observations visible to the model — the strategies differ in how far they reach,
never in what they are allowed to see:

| Strategy | Behaviour for a day below the minimum |
|---|---|
| `missing` (default) | All four statistics stay missing. Nothing is imputed, nothing is borrowed. |
| `within_day_available` | Use whatever that calendar day has, ignoring the minimum. |
| `neighbor_day_fallback` | Take the statistics of the nearest day meeting the minimum, up to `fallback_window_days` away; ties resolve to the earlier day. |
| `rolling_available` | Recompute from the visible observations within ± `fallback_window_days` calendar days; still missing if that pool is below the minimum. |

`fallback_window_days` defaults to 7 for the two reaching strategies and is
rejected for the two that never leave the day, so a recorded window always had an
effect.

**Consequence of the default.** Under `missing`, an artificial gap that covers a
whole calendar day leaves every row of that day without daily statistics, so the
paper's 7-day and 30-day gap classes yield no complete feature rows and would
receive no predictions at all. That is the honest reading of A4 rather than a
defect, and it is reported rather than hidden:
`ValidationFeatureSet.to_dict()` records `holdout_rows_with_complete_features`
alongside `holdout_rows`. A long-gap validation run therefore has to select a
reaching strategy deliberately and record it in the manifest.

### 3.5 Leakage rule (default `feature_mode="paper_safe"`)

Daily target statistics are derived from the target, so they are the main leakage
risk in the artificial-gap validation.

Required default behaviour:

1. build the artificial-gap mask **before** computing target-derived features;
2. compute daily statistics from target observations visible to the model only;
3. never let held-out truth influence a feature used to predict that truth;
4. days with no visible target observations yield missing daily statistics
   (see [Known ambiguities](#known-ambiguities), A4).

`rfrgapfill.leakage` is the workflow that enforces this, and validation features
must be built through it rather than by calling the transformers directly:

- `holdout_mask_from_intervals()` turns gap intervals into the holdout mask
  (step 1), before any target-derived feature exists;
- `observed_target_mask()` / `available_target_mask()` derive what the model may
  see: a genuine QC-passed measurement that is not withheld;
- `hide_target()` removes the held-out values from the frame features are read
  from (step 2);
- `build_validation_features()` computes the daily statistics from the visible
  observations only (step 3), assembles the feature matrix (step 4), and returns
  a `ValidationFeatureSet` carrying the untouched truth as a separate attribute
  used only for scoring (step 5).

The two protections — hiding the values and passing `available_mask` — are
independent, and each is sufficient on its own. `detect_target_leakage()` rebuilds
the features with the hidden truth replaced by absurd values and reports any
feature column that moved; `require_no_target_leakage()` raises `LeakageError`
when one does. Both run with and without the frame-level hiding, so the mask is
shown to suffice by itself. In `paper_safe` mode the probe must return nothing,
for every daily-statistic strategy.

`feature_mode="legacy_fluxlib"` is reserved for a compatibility mode to be added
only if implementation evidence for a different derivation is found. The two modes
must never be mixed silently.

### 3.6 ORF switch

`use_receptive_limiter=False` produces the ORF benchmark: identical estimator
family, identical hyperparameter grid, identical training rows, identical driver
set; sections 3.1–3.4 omitted. ORF is not redefined in any other way.

ORF is defined *relative to* the RFR run it is compared against, so the pairing
is a checked precondition rather than a convention:

- `RFRConfig.as_orf()` derives the benchmark arm, flipping only the limiter flag
  and carrying over mode, drivers, seed, grid, CV policy, column mapping, QC
  rules and hemisphere unchanged;
- `require_orf_pairing(rfr, orf)` rejects a pair that differs in anything else,
  naming the offending settings;
- `receptive_limiter_features(config, target=...)` is exactly what the two arms'
  feature sets differ by, so `feature_names` cannot drift from it.

Both arms must additionally be scored on the **same artificial gap mask**; that
is a property of the validation workflow (section 4) rather than of the
configuration pair. The paired comparison reports R2, slope, RMSE and bias for
each arm and their difference. No individual metric is required to improve:
Supplementary Figure S1 is evidence about the receptive limiter in aggregate,
not a per-site guarantee.

---

## 4. Artificial-gap validation scenario

### 4.1 Gap durations

| Class | Duration |
|---|---|
| `short` | 24 hours |
| `long` | 7 days |
| `very_long` | 30 days |

Durations are elapsed-time quantities, matched against timestamps rather than row
counts.

### 4.2 Artificial missing fraction

`missing_fraction = 0.25` — artificial gaps collectively withhold approximately
25% of the available half-hourly observations. Achieved fraction is reported with a
documented tolerance rather than asserted exactly.

### 4.3 Nominal gap mix

| Class | Share |
|---|---|
| `short` (24 h) | 20% |
| `long` (7 d) | 30% |
| `very_long` (30 d) | 50% |

Interpretation is configurable — `allocation_basis="missing_records"` (default) or
`"gap_events"` — and the choice used is recorded in every result (ambiguity A3).

### 4.4 Generator requirements

- seeded and reproducible;
- intervals sampled from timestamps and requested durations;
- proposed intervals require `min_observed_fraction = 0.50` genuinely observed
  target values, otherwise rejected and resampled;
- no artificial intervals overlap unless explicitly allowed;
- dataset-boundary-safe; bounded retries; explicit failure when the requested
  scenario cannot be constructed;
- joint NEE/H/LE validation uses **identical** gap locations for all three targets;
- returns a manifest with start, end, duration, class, affected rows, and observed
  fraction per gap.

### 4.5 Train/test design

The artificially withheld observations **are** the test set (~25%); remaining
eligible observations are the training set (~75%), matching Figure 2 of the paper.
No naive row-wise random split, which would destroy the intended temporal gap
structure. Hyperparameters are tuned on the training portion only. Blocked or
time-aware CV is an enhancement, never the paper-faithful default.

---

## 5. Model

`sklearn.ensemble.RandomForestRegressor`, with:

- deterministic `random_state`;
- configurable `n_jobs`;
- hyperparameter optimisation via `GridSearchCV`;
- training restricted to rows with a valid target and valid required features;
- explicit reporting of rows dropped for missing predictors;
- one model and one feature transformer per target;
- `joblib`-serializable model and configuration.

The paper states that `GridSearchCV` was used but does not enumerate the grid. The
package therefore ships a documented default grid in configuration
(`hyperparameter_grid`); any archived `fluxlib` grid may be added as a **named
preset**, never as "paper exact" (ambiguity A1).

`rfrgapfill.model.RFRModel` is that estimator plus the search around it, and
nothing else: it is handed a feature matrix and a target vector, so the same class
serves the RFR arm, the ORF arm and an operational fill, and knows nothing about
artificial gaps. `fit(X, y)`, `predict(X)`, `get_feature_names()`,
`get_best_params()`, `save()` and `load()` are the whole surface.

Three points the paper leaves open are settled here explicitly and reported in the
manifest:

| Point | Behaviour |
|---|---|
| Rows with a missing or non-finite predictor | Dropped at fit; left missing at prediction. Recent scikit-learn forests accept `NaN` natively, which would silently impose an undocumented imputation rule instead of section 7's "fail clearly or leave predictions missing". `predict(..., on_incomplete="raise")` is the "fail clearly" half. |
| Row accounting | `FitReport` records rows offered, rows fitted, rows dropped for a missing target, rows dropped for a missing predictor, and a per-feature count. It names the feature that cost the most rows, and points at A4 when that feature is a daily statistic. |
| `n_jobs` | Applied to the forest, not to the grid search, so one configured setting cannot multiply into folds × candidates × trees workers. |

The CV strategy of ambiguity A5 maps to `KFold(n_splits=cv_folds,
shuffle=cv_shuffle)` for the default and `TimeSeriesSplit(n_splits=cv_folds)` for
the labelled enhancement; scoring is scikit-learn's regressor default, R2. The
expected feature order is derived from `feature_names(config, target=...)` whenever
it can be — which needs the target, since the daily statistics are target-specific
— so a matrix built for another target is rejected rather than fitted.

A saved model is a `joblib` payload carrying the fitted search, the feature order,
the fit report, the environment versions and the `RFRConfig` itself. The
configuration is **revalidated as it is unpickled**, so an edited model file fails
on load rather than predicting under settings the constructor would have rejected,
and a scikit-learn version different from the one the model was fitted under is
reported rather than silently accepted.

---

## 6. Required metrics

Computed by comparing predictions against known measurements inside artificial
gaps.

### 6.1 Core

| Metric | Definition |
|---|---|
| `R2` | coefficient of determination |
| `slope` | linear-regression slope, measured on x, filled on y |
| `RMSE` | root mean squared error |
| `bias` | `(sum(filled) - sum(measured)) / n` |

Regression orientation is fixed: `x = measured`, `y = filled`. The bias definition
is the paper's and is equivalent to mean prediction error for equal-weighted
observations.

### 6.2 Day/night subsets

Daytime is defined by `daytime_threshold`: downward shortwave radiation
`> 20 W m-2`. Metrics are reported for **all**, **daytime**, and **nighttime**
subsets. Aggregate-only reporting is not acceptable — nighttime skill is
substantially weaker (see `supplement_benchmarks.md`) and must stay visible.

### 6.3 Reporting by gap class

Core metrics are additionally reported per gap class (24 h / 7 d / 30 d), together
with bias IQR by gap class.

### 6.4 Energy-balance ratio

For H and LE:

```text
EBR = sum(H + LE) / sum(NETRAD - G)
```

Computed over the same artificial-gap intervals using (a) measured H and LE and
(b) RFR-filled H and LE. Both values and their difference are returned.

---

## 7. Data and provenance requirements

- half-hourly FLUXNET2015 is the reference resolution; other regular cadences
  (e.g. hourly) are supported;
- `DatetimeIndex` or explicit timestamp column; monotonic and unique timestamps
  required unless a documented preprocessing option resolves duplicates;
- gap lengths always computed from elapsed time, not row counts;
- observed flux measurements distinguished from pre-ingestion gap-filled values
  using a QC/provenance field; training and validation prefer genuinely observed,
  quality-controlled data;
- meteorological drivers may arrive pre-filled, as in the paper; missing drivers
  are never silently fabricated — either fail clearly or leave predictions missing
  for rows lacking predictors;
- observed target values are never overwritten in place by default; filled output
  carries `<target>_original`, `<target>_filled`, `<target>_is_observed`,
  `<target>_is_filled`, `<target>_fill_method`, `<target>_model_version`.

Every stochastic component (gap sampling, Random Forest, CV) accepts a seed, and a
run manifest records package/Python/scikit-learn versions, column mapping, target,
RFR3/RFR10 mode, feature mode, hyperparameter grid, best parameters, seed, gap
manifest, time resolution, hemisphere/latitude, and QC rules.

---

## Known ambiguities

Unresolved points in Zhu et al. (2022). Each has a defensible default, is exposed
in configuration, and must be reported in run output. None of these may be
described as reproducing the paper exactly.

| ID | Ambiguity | Default chosen | Configuration |
|---|---|---|---|
| A1 | The complete `GridSearchCV` hyperparameter grid is not enumerated in the article. | Documented default grid shipped with the package; any archived `fluxlib` grid only as a named preset. | `hyperparameter_grid` |
| A2 | Radiation-category boundary handling at exactly 10 and 100 W m-2 is not explicit in the prose. | Inclusive, exhaustive bins: `<10` weak, `10–100` medium, `>100` strong. | `radiation_thresholds`, `boundary_convention` |
| A3 | The 20/30/50 gap mix may refer to the number of gap events or the number of withheld half-hours. | `missing_records`. | `allocation_basis` |
| A4 | Handling of daily target statistics when a whole day or longer interval has too few or no visible observations is unspecified. | `missing`: statistics left missing; affected rows excluded from training and flagged at prediction time rather than imputed. Three reaching alternatives (`within_day_available`, `neighbor_day_fallback`, `rolling_available`) are implemented and leakage safe; a long-gap run must choose one deliberately (section 3.4). | `daily_statistic_strategy`, `min_daily_observations`, `fallback_window_days` |
| A5 | Cross-validation details inside `GridSearchCV` (fold count, shuffling, temporal blocking) are unspecified. | Conventional `GridSearchCV` folds on the training portion; blocked/time-aware CV offered as a labelled enhancement. | `cv_strategy` |
| A6 | Whether the historical `fluxlib` implementation computed daily target statistics leakage-safely is unverified. | Leakage-safe `paper_safe` mode is the default; a `legacy_fluxlib` mode is added only on implementation evidence. | `feature_mode` |
| A7 | The achieved artificial-missing fraction cannot always hit exactly 25% given real gaps and series boundaries. | Report achieved fraction and class allocation against a documented tolerance. | `missing_fraction`, tolerance |
| A8 | The denominator of the supplement's normalized joint-uncertainty ratio is not reconstructed. | Bias-IQR by gap class supported now; normalized ratios remain explicitly experimental. | experimental module |
| A9 | Hemisphere inference for sites at or very near the equator. | `latitude >= 0 -> north`; an explicit `hemisphere` always overrides. | `hemisphere`, `latitude` |
| A10 | Units of Table S3 NEE RMSE/bias (`g C m-2 d-1`) differ from the half-hourly model units (`umol m-2 s-1`); the aggregation from half-hourly residuals to daily carbon units is not spelled out. | Report metrics in model units by default; benchmark comparison applies an explicit, documented unit conversion. | reproduction module |
| A11 | The paper names the daily standard deviation but not its degrees-of-freedom convention, nor the quantile interpolation behind Q1/Q2/Q3. | Sample standard deviation (`ddof=1`, the pandas default) and linear quantile interpolation (the numpy/pandas default). | `daily_std_ddof` |
