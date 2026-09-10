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

Both readings are implemented, side by side, in `rfrgapfill.gaps.allocate_gaps`,
which is a pure function of the configuration and the size of the series so the
design can be inspected without sampling anything. Write `S_c` for the configured
share of class `c`, `E_c = duration_c / time_step` for the records one event of
that class covers on a complete grid, `A` for the available observations and
`T = round(missing_fraction * A)` for the records the scenario aims to withhold:

| Basis | Algorithm |
|---|---|
| `missing_records` (default) | The shares apply to withheld records; each class independently gets `events_c = round(S_c * T / E_c)`. The counts sum to no particular total, because what is apportioned is `T`, not a number of gaps. |
| `gap_events` | The shares apply to the gap count. With mean event size `M = sum(S_c * E_c)`, the design needs `N = round(T / M)` events, apportioned by largest remainder so the counts sum to exactly `N`. |

Rounding is half-up, and largest-remainder ties resolve in class order, so an
allocation never depends on floating-point parity. `missing_records` is the
default because the paper frames the scenario as percentages of removed
half-hours — an inference from the prose, not a statement in it, and neither
basis may be called "paper exact".

The two designs are not near each other: at half-hourly cadence a 20/30/50 mix of
gap *events* puts over 80% of the withheld records in the 30-day class. That is
the reason the ambiguity is exposed rather than buried.

Both bases plan from `E_c`, the records a complete grid would hold, because the
plan is made before any interval is placed. What a scenario actually withholds is
smaller wherever a chosen interval overlaps real missing data, so the achieved
fraction and the achieved mix are **measured from the placed intervals** and
reported on *both* bases next to the request — never back-corrected to look
exact. `GapManifest.summary()` and `to_dict()` state which basis was used, what
was asked for and what was achieved, which is what makes a validation report
self-describing.

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

`rfrgapfill.gaps.GapScenarioGenerator` is that generator, and
`rfrgapfill.gaps.GapManifest` is what it returns:

- the withheld fraction is measured against **genuinely observed** target values,
  not against rows — withholding a value that was never measured withholds
  nothing, so `n_available` comes from the QC-aware observed mask;
- `observed_fraction` for a proposed interval divides by `n_expected`, the
  records a complete grid would hold, not by the rows present: an interval laid
  over a real gap holds few rows *and* few observations, which would otherwise
  look like perfect coverage;
- intervals are placed longest class first, because a 30-day interval is far
  harder to fit than a 24-hour one;
- joint NEE/H/LE validation passes several targets at once and a row counts as
  available only where *every* target is observed, so the single set of
  locations is a fair test set for all three;
- `GapManifest.mask()` produces the holdout mask that is step 1 of the
  leakage-safe workflow in section 3.5, built before any target-derived feature
  exists;
- a scenario that cannot be built raises `GapError` naming the classes that came
  up short and what would let them fit; `on_shortfall="warn"` is the deliberate
  opt-out that returns the partial design for inspection;
- everything that did not go as requested — a class the series is too short to
  hold, a class too small a share to earn a gap, an achieved fraction or mix
  outside the tolerances of A7 — is recorded on the manifest and emitted as a
  `GapScenarioWarning` rather than passing quietly.

### 4.5 Train/test design

The artificially withheld observations **are** the test set (~25%); remaining
eligible observations are the training set (~75%), matching Figure 2 of the paper.
No naive row-wise random split, which would destroy the intended temporal gap
structure. Hyperparameters are tuned on the training portion only. Blocked or
time-aware CV is an enhancement, never the paper-faithful default.

`rfrgapfill.validation.validate_rfr` is the workflow that performs the whole
experiment, and it is the only supported entry point to it: the order of its
stages is the leakage protection of section 3.5, so a caller who assembles the
same pieces by hand can get that order wrong and this one cannot. One call places
the intervals, derives the holdout mask from them, hides the truth, builds the
features through `rfrgapfill.leakage`, fits on what remains, predicts the withheld
rows and scores them.

- one `GapManifest` serves every target of a run, which is section 4.4's shared
  locations; `gaps=` reuses it for a second arm or for the ORF benchmark, so a
  paired comparison is scored on identical rows by construction;
- predictions exist **only** inside the artificial gaps. Every other row of the
  returned series is missing, so "no observed value was changed" is a checkable
  property of the output rather than an intention;
- a withheld row whose predictors are incomplete is neither predicted nor scored,
  and `CoreMetrics.n` against `n_offered` says how many that was;
- `ValidationReport.to_frame()` is the tidy table - target, method, mode, gap
  class, subset, the four metrics and the row counts - and
  `ValidationReport.manifest(target)` hands the run to
  `rfrgapfill.provenance.RunManifest`;
- a run that withheld rows and scored none of them raises `ValidationWarning`
  naming ambiguity A4, because that is the default's documented consequence
  (section 3.4) and an empty gap class otherwise reads as a defect.

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
observations. The regression carries an intercept by default (`fit_intercept`);
a line forced through the origin is offered but is not the paper's stated setup.

`rfrgapfill.metrics` is these four quantities and the ratio of section 6.4, and
nothing else: each is an independent function of the values handed to it, so the
same code scores a whole run, one gap class, one daytime subset or a hand-built
fixture. Choosing the subsets belongs to the validation layer, not here.

Two rules apply before any metric is computed, and `CoreMetrics` reports what
they cost (`n` scored out of `n_offered`):

- **alignment is checked, never performed.** Two series carrying different
  indexes raise `MetricError` instead of being aligned into a union of missing
  values, which is how a prediction ends up scored against the wrong half hour;
- **incomplete pairs are dropped on both sides at once**, so a row the model left
  unfilled for want of a predictor (section 5) leaves the score entirely — the
  bias denominator `n` counts the pairs that survived.

An undefined metric is `None`, never `NaN` or zero: an empty subset has no RMSE,
constant measurements have no R2 and no slope, and available energy summing to
zero or less has no EBR (section 6.4). `None` reaches a JSON manifest as `null` and cannot propagate silently
through a later mean the way `NaN` does.

**Which R2 (A12).** The article reports R2 beside a slope without saying which
quantity it is. `R2Definition.RESIDUAL` — `1 - SS_res / SS_tot`, scikit-learn's
`r2_score` — is the default: it is the unqualified meaning of "coefficient of
determination", and the alternative `SQUARED_CORRELATION` is invariant to any
affine rescaling of the predictions, so it scores a systematically offset series
as perfectly as an unbiased one. The two agree exactly when predictions are
unbiased and their spread has shrunk to `r` times the measured spread, which is
ordinary forest behaviour and close to where Table S3 sits — its median R2 and
median slope agree to about 0.01 in every RFR row — so the published numbers do
not settle it and neither reading is paper exact. The choice is
`ValidationConfig.r2_definition` and is recorded in every result.

### 6.2 Day/night subsets

Daytime is defined by `daytime_threshold`: downward shortwave radiation
`> 20 W m-2`. Metrics are reported for **all**, **daytime**, and **nighttime**
subsets. Aggregate-only reporting is not acceptable — nighttime skill is
substantially weaker (see `supplement_benchmarks.md`) and must stay visible.

### 6.3 Reporting by gap class

Core metrics are additionally reported per gap class (24 h / 7 d / 30 d), together
with bias IQR by gap class. A bias IQR is a spread *over* something, and it is
reported over two populations that are different quantities and never mixed:

| Population | One value is | Where |
|---|---|---|
| Gaps within one site | the bias of one placed interval | `TargetValidation.bias_spread`, `bias_spread_frame()` |
| Sites within a study | the bias of one site's cell of `gap_length_table` | `rfrgapfill.uncertainty.bias_iqr()` |

The across-site IQR is the analogue of the Supplementary Table S8 numerator. That
Table S8 spreads bias across sites rather than across gaps is an inference from
the supplement's cross-site framing, not a statement in it (ambiguity A8).

`bias_iqr()` groups by target, method, mode, gap class, day/night subset and
units, and - when site metadata supplies it - by IGBP class. Rows pooling every
ecosystem are labelled `igbp="all"` and are always present; a site without an
IGBP class contributes to the pooled rows only. Quartiles use linear
interpolation (A11), a site with an undefined bias is skipped rather than counted
as zero, and an IQR needs two defined values, so one site or one gap reports
quartiles without a spread.

The **normalized** ratios of Table S8 - bias IQR divided by a flux confidence
interval or by a joint flux-uncertainty confidence interval - are not computed by
this package, because their denominator has not been reconstructed. The published
very-long-gap ranges are carried as data, `rfrgapfill.uncertainty.TABLE_S8_RANGES`,
every record fixed at status `experimental`, and they are kept out of
`rfrgapfill.benchmarks` so `compare_to_benchmarks()` can never difference a run
against them. A ratio implemented later must carry "experimental" in its name
until it has been reproduced against Table S8.

### 6.4 Energy-balance ratio

For H and LE:

```text
EBR = sum(H + LE) / sum(NETRAD - G)
```

Computed over the same artificial-gap intervals using (a) measured H and LE and
(b) RFR-filled H and LE. Both values and their difference are returned, by
`metrics.compare_energy_balance`.

A row enters either ratio only where every component it needs is present, so
numerator and denominator always cover exactly the same half hours — summing each
over whatever it happened to have would divide the turbulent flux of one interval
by the available energy of another. The measured and filled ratios share one row
set for the same reason: otherwise their difference would report the change in
interval as much as the change in flux.

A denominator at or below zero yields `None`. At zero the ratio does not exist;
below zero — a night-dominated interval, where the surface loses energy — it
exists but reads backwards: more turbulent flux gives a *smaller* ratio, so the
closure and the filled-minus-measured difference would both carry the wrong
sign. The rule is on the interval's sum, not on each row, so night rows inside a
positive interval count. The shared denominator is reported beside the ratios
(`available_energy`), so an undefined ratio says why.

In a validation run the check is `validation.EnergyBalanceCheck`, computed
whenever H and LE are validated together: over every withheld row, and again
within each gap class. Every withheld row is offered, and each comparison counts
what the shared row set cost — rows lacking a measured H or LE
(`n_missing_measured`: absent, or not a genuine measurement by its QC flag, since
a pre-filled value is no more closure evidence than it is scoring truth), rows
the model left unpredicted (`n_missing_filled`), and rows lacking NETRAD or G
(`n_missing_available_energy`). A row missing two components counts under both.
The check appears in `ValidationReport.energy_balance_check`, in
`energy_balance_frame()` (one row per gap class plus `all`), in the run summary,
and in the run manifests of the H and LE targets. NETRAD and G are read as given:
like every driver, they may arrive pre-filled.

### 6.5 Reporting across runs and sites

Sections 6.1-6.4 score one arm of one run. The comparison the paper is *about* -
does skill hold up as the gap grows, and how do RFR3, RFR10 and MDS compare -
needs those scores lined up across arms and across sites, and that is
`rfrgapfill.sensitivity`. It computes no metric: every number in its tables came
out of `rfrgapfill.metrics` inside a validation run, so a table can be wrong
about labelling but never about values.

- `gap_length_table()` is the tidy table - site, target, method, mode, gap class,
  day/night subset, units, row counts and the four metrics. It accepts live
  results and equally tidy frames read back from disk, because a 94-site
  reproduction does not hold 94 fitted forests in memory;
- gap classes come back in **duration order**, never alphabetical, and the row
  pooling every class is labelled `all`;
- metric columns are always float, so a subset every run left undefined arrives
  as missing rather than as an object column a later median cannot aggregate;
- `median_across_sites()` is the aggregation Table S3 reports, and it refuses a
  table where one site contributed a cell twice rather than weighting that site
  twice;
- `gap_length_pivot()` is the readable target-by-method view, built on `pivot`
  rather than `pivot_table` so two rows in one cell raise instead of being
  silently averaged.

Units travel with the numbers, because ambiguity A10 makes them load-bearing:
`rmse` and `bias` are in the units of the flux, and Table S3's NEE units are not
this package's. `rfrgapfill.benchmarks` holds the published medians as data -
every value traceable to `supplement_benchmarks.md`, checked by a test that
re-reads that document - and `compare_to_benchmarks()` puts a run beside them
metric by metric, marking a cell `comparable=False` rather than differencing two
numbers in different units. `convert_nee_to_carbon_units()` is the explicit
conversion; it converts a rate, and it is not a claim about how the paper
aggregated half hours into days.

Plotting is `rfrgapfill.plotting`, after the tables and derived only from them.
`matplotlib` is an optional dependency, imported when a figure is drawn and
reported by name when it is missing.

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

`rfrgapfill.fill.RFRGapFiller` is the operational workflow those rules describe:
`fit(data, target=..., qc_column=...)` then `fill(data)`, one site and one target
per instance. Leakage has nothing to prevent here — a genuinely missing value is
already invisible to the daily statistics — so the entry point is this module and
not `rfrgapfill.leakage`. What it does enforce:

- **training rows are observed rows.** Present, finite and QC-accepted. A value
  the QC flag marks as gap-filled before ingestion is neither trained on nor
  admitted to the daily statistics, exactly as in validation.
- **a present value is never replaced.** Only gaps are predicted. A pre-filled
  value is carried through and labelled `pre_filled`; `refill_pre_filled=True`
  is the explicit opt-in, and it is the way to run RFR over a FLUXNET target
  column that already arrived complete.
- **nothing is imputed for a row lacking a predictor.** It stays missing and is
  labelled `unfilled_incomplete_features`, or `on_incomplete="raise"` fails
  instead — the two behaviours this section allows, and no third.
- **the caller's frame is not mutated.** Every stage works on a copy on a
  validated time axis; the result carries the input's columns plus the six above.
- **`time_distance_hours` keeps one origin.** Fixed at `fit` and reused by
  `fill`, so a later slice of the same series does not silently restart the
  clock; rows earlier than that origin are counted in the report.

`<target>_fill_method` partitions the filled series by provenance: `observed`,
`pre_filled`, `unfilled_incomplete_features`, or the arm's own label (`RFR3`,
`RFR10`, `ORF3`, `ORF10`) for a value this package predicted.

`FillReport` is the row accounting — observed, pre-filled and missing rows against
the frame; filled and unfilled against the candidates; and the feature that
blocked the most rows. That last field is where ambiguity A4 becomes visible
operationally: under the default `daily_statistic_strategy="missing"` a gap
covering a whole calendar day has no daily statistics, so **none of its rows can
be filled**, and the fill warns rather than returning a quietly empty gap.

Every stochastic component (gap sampling, Random Forest, CV) accepts a seed, and a
run manifest records package/Python/scikit-learn versions, column mapping, target,
RFR3/RFR10 mode, feature mode, hyperparameter grid, best parameters, seed, gap
manifest, time resolution, hemisphere/latitude, and QC rules.

`rfrgapfill.provenance.RunManifest` is that manifest. Every stage already
describes itself - `RFRConfig.to_dict`, `RFRModel.to_dict`, `GapManifest.to_dict`,
`FillReport.to_dict` - and this is the one place they are assembled, checked, and
exported as JSON:

- `RunManifest.from_fill(filler, result)` and `RunManifest.from_validation(...)`
  take the sections from the objects that own them, so a manifest cannot describe
  a configuration the run did not use; a result paired with another run's filler,
  or a target the model and the features disagree about, is rejected;
- the required list above is a checked construct rather than a hope.
  `REQUIRED_FIELDS` maps each item onto its path in the exported document, and
  `require_complete()` names any the manifest cannot supply. A gap manifest is
  required of an artificial-gap run and not of an operational fill, which places
  no intervals; a hemisphere is required only where the season feature is built.
  `save()` runs the check before writing, because the file is what outlives the
  session that could have explained it;
- **the resolved ambiguities travel with the result.** The preamble to
  [Known ambiguities](#known-ambiguities) requires every choice to be reported in
  run output, not merely exposed in configuration, so `ambiguity_choices()`
  writes this run's answer to A1-A12 into every manifest, under a note stating
  that none of them reproduces the paper exactly;
- `settings_digest` is a SHA-256 over the environment versions, the target and
  the whole validated configuration, and over nothing that varies between two
  runs of the same design - not the timestamp, the row counts, the placed
  intervals or the scores. Two manifests sharing a digest were configured
  identically.

Undefined values reach the document as `null` rather than `NaN`, matching section
6.1, and every section is rendered to plain containers when the manifest is
built, so a manifest that could not be serialised fails there rather than when
someone tries to write the audit file.

---

## Known ambiguities

Unresolved points in Zhu et al. (2022). Each has a defensible default, is exposed
in configuration, and must be reported in run output. None of these may be
described as reproducing the paper exactly.

| ID | Ambiguity | Default chosen | Configuration |
|---|---|---|---|
| A1 | The complete `GridSearchCV` hyperparameter grid is not enumerated in the article. | Documented default grid shipped with the package; any archived `fluxlib` grid only as a named preset. | `hyperparameter_grid` |
| A2 | Radiation-category boundary handling at exactly 10 and 100 W m-2 is not explicit in the prose. | Inclusive, exhaustive bins: `<10` weak, `10–100` medium, `>100` strong. | `radiation_thresholds`, `boundary_convention` |
| A3 | The 20/30/50 gap mix may refer to the number of gap events or the number of withheld half-hours. | `missing_records`. Both readings are implemented in `rfrgapfill.gaps.allocate_gaps` (section 4.3); every manifest reports the achieved mix on both bases alongside the one requested. | `allocation_basis` |
| A4 | Handling of daily target statistics when a whole day or longer interval has too few or no visible observations is unspecified. | `missing`: statistics left missing; affected rows excluded from training and flagged at prediction time rather than imputed. Three reaching alternatives (`within_day_available`, `neighbor_day_fallback`, `rolling_available`) are implemented and leakage safe; a long-gap run must choose one deliberately (section 3.4). | `daily_statistic_strategy`, `min_daily_observations`, `fallback_window_days` |
| A5 | Cross-validation details inside `GridSearchCV` (fold count, shuffling, temporal blocking) are unspecified. | Conventional `GridSearchCV` folds on the training portion; blocked/time-aware CV offered as a labelled enhancement. | `cv_strategy` |
| A6 | Whether the historical `fluxlib` implementation computed daily target statistics leakage-safely is unverified. | Leakage-safe `paper_safe` mode is the default; a `legacy_fluxlib` mode is added only on implementation evidence. | `feature_mode` |
| A7 | The achieved artificial-missing fraction cannot always hit exactly 25% given real gaps and series boundaries. | Report achieved fraction and class allocation against a documented tolerance. | `missing_fraction`, tolerance |
| A8 | The denominator of the supplement's normalized joint-uncertainty ratio is not reconstructed, and whether its bias IQR is taken across sites or across gaps is not stated. | Bias IQR reported over both populations (section 6.3), the across-site one as the Table S8 analogue. The published ratios are carried as data fixed at status `experimental`; no ratio is computed. | `rfrgapfill.uncertainty` |
| A9 | Hemisphere inference for sites at or very near the equator. | `latitude >= 0 -> north`; an explicit `hemisphere` always overrides. | `hemisphere`, `latitude` |
| A10 | Units of Table S3 NEE RMSE/bias (`g C m-2 d-1`) differ from the half-hourly model units (`umol m-2 s-1`); the aggregation from half-hourly residuals to daily carbon units is not spelled out. | Report metrics in model units by default; `compare_to_benchmarks` refuses a cell whose units differ, and `convert_nee_to_carbon_units` applies the rate conversion explicitly (section 6.5). | `rfrgapfill.benchmarks`, `units=` on `gap_length_table` |
| A11 | The paper names the daily standard deviation but not its degrees-of-freedom convention, nor the quantile interpolation behind Q1/Q2/Q3. | Sample standard deviation (`ddof=1`, the pandas default) and linear quantile interpolation (the numpy/pandas default). | `daily_std_ddof` |
| A12 | `R2` is reported beside a regression slope without saying whether it is `1 - SS_res/SS_tot` or the squared Pearson correlation of that regression. | `residual` (`1 - SS_res/SS_tot`), the only one of the two a systematic offset can lower. Both are implemented; the two coincide in the regime Table S3 sits in, so neither is paper exact (section 6.1). | `r2_definition` |
