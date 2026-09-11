# Assumptions and known differences from the paper

Zhu et al. (2022) leave some implementation details open. Wherever they do, this
package picks a default, exposes it as a setting, and writes the choice into
every run manifest (`manifest.to_dict()["ambiguities"]`). **Nothing on this page
may be described as reproducing the paper exactly.**

This page explains each open point for a user: what is unknown, what the package
does, how to change it, and why the default was chosen. The authoritative table
is [Known ambiguities in `method_spec.md`](method_spec.md#known-ambiguities).
After the twelve ambiguities come the package's own conventions, the places where
the paper's supplement disagrees with itself, and what is not implemented.

---

## The twelve ambiguities

### A1: The hyperparameter grid

**Open.** The article says `GridSearchCV` was used and does not list the grid.

**Default.** `DEFAULT_HYPERPARAMETER_GRID`: `max_features` in `(1.0, "sqrt")`,
`min_samples_leaf` in `(1, 5)`, `n_estimators` in `(100, 300)`, eight
candidates, searched with 5 unshuffled folds.

**Change it.**

```python
RFRConfig(mode="RFR10", hemisphere="north", hyperparameter_grid={"n_estimators": (500,)})
RFRConfig(mode="RFR10", hemisphere="north", hyperparameter_preset="legacy_fluxlib", cv_folds=3)
```

`legacy_fluxlib` is the 288-candidate grid found in the archived `fluxlib` code,
which no archived pipeline actually calls; `legacy_fluxlib_fixed` is the single
parameter set those pipelines fitted ([`fluxlib_audit.md`](fluxlib_audit.md),
F10-F11).

**Why.** A small, conventional grid over the settings that matter most for a
random forest. No grid is called paper exact.

### A2: Radiation-category boundaries

**Open.** The categories are weak below 10, medium from 10 to 100 and strong
above 100 W m-2, but the article does not say which category a value of exactly
10 or 100 belongs to.

**Default.** `boundary_convention="medium_inclusive"`: `< 10` weak,
`10 to 100` medium, `> 100` strong. Every value lands in exactly one category,
and a missing radiation value gives a missing category.

**Change it.** `FeatureConfig(boundary_convention="medium_exclusive")` (`<= 10`
weak, `>= 100` strong), or other thresholds with `radiation_thresholds=`.

**Why.** Either convention is defensible; the difference touches only values that
land exactly on a threshold. `fluxlib` puts such values, and missing ones, in a
fourth class (F8); only `feature_mode="legacy_fluxlib"` reproduces that.

### A3: What the 20/30/50 gap mix counts

**Open.** The mix of 24-hour, 7-day and 30-day gaps can mean 20/30/50 percent of
the gap *events* or of the withheld *half-hours*.

**Default.** `allocation_basis="missing_records"`: the shares apply to withheld
records.

**Change it.**
`ValidationConfig(gaps=GapScenarioConfig(allocation_basis="gap_events"))`.

**Why.** The paper frames the scenario as percentages of removed data, and the
`fluxlib` code apportions records (F14). The two readings differ a lot: counted
as events, a 20/30/50 mix puts over 80% of the withheld records in the 30-day
class. Every manifest reports the achieved mix on both bases.

### A4: Days with too few measured values

**Open.** The daily target statistics need measured values on that day. The
article does not say what a day with none, or too few, receives.

**Default.** `daily_statistic_strategy="missing"`: the statistics stay missing,
and rows without them are neither trained on nor predicted. **Consequence:** a
gap covering a whole calendar day cannot be filled, so 7-day and 30-day gaps get
no predictions. The run warns when this happens.

**Change it.**

```python
FeatureConfig(daily_statistic_strategy="rolling_available", fallback_window_days=7)
```

The alternatives are `within_day_available`, `neighbor_day_fallback` and
`rolling_available` ([`validation.md`](validation.md#long-gaps-need-a-daily-statistic-strategy-a4)).
`min_daily_observations` (default 1) sets how many visible values make a day
adequate.

**Why.** The default imputes nothing and borrows nothing, so it cannot be
mistaken for something the paper did. Any run with gaps longer than a day
should choose a reaching strategy deliberately; the synthetic example uses
`rolling_available`. `fluxlib` interpolates the target across every gap first
(F3), which only `legacy_fluxlib` reproduces.

### A5: Cross-validation inside the grid search

**Open.** Fold count, shuffling and any temporal blocking are not stated.

**Default.** `cv_strategy="kfold"`, `cv_folds=5`, `cv_shuffle=False`, on the
training rows only.

**Change it.** `RFRConfig(..., cv_strategy="time_series_split")` for blocked,
time-aware folds, a labelled enhancement that makes the configuration not paper
faithful. `cv_folds=3` matches `fluxlib` (F12).

**Why.** Conventional `GridSearchCV` folds are the plain reading of the article.

### A6: Whether the original code was leakage safe

**Resolved by evidence: it was not.** `fluxlib` 0.0.23 computes the daily
target statistics before the artificial gaps are hidden
([`fluxlib_audit.md`](fluxlib_audit.md), F1).

**Default.** `feature_mode="paper_safe"`: gaps hidden first, statistics from
visible measurements only, and a probe that proves it
([`validation.md`](validation.md#leakage-safe-target-statistics)).

**Change it.** `FeatureConfig(feature_mode="legacy_fluxlib")` reproduces the
historical derivation for comparison. It raises `LegacyFluxlibWarning` on every
build and labels its arms `RFR3-legacy` and `RFR10-legacy`.

**Why.** A validation score must not depend on the truth it is scored against.

### A7: The achieved missing fraction

**Open.** With real gaps in the record and a finite series, the scenario cannot
always withhold exactly 25%.

**Default.** The achieved fraction and mix are measured and reported, and a
departure beyond `fraction_tolerance=0.05` or `mix_tolerance=0.05` raises a
`GapScenarioWarning`. Nothing is adjusted to look exact.

**Change it.** The tolerances are fields of `GapScenarioConfig`.

**Why.** Reporting the achieved scenario is more useful than quietly forcing the
requested one. On the synthetic site, a 25% request typically lands nearer 18%.

### A8: The supplement's normalized uncertainty ratios

**Open.** Table S8 divides a bias IQR by a confidence interval whose construction
is not described, and does not say whether the IQR is taken across sites or
across gaps.

**Default.** Bias IQR is reported over both populations, never mixed: across the
gaps of one site (`report.bias_spread_frame()`) and across sites
(`rfrgapfill.uncertainty.bias_iqr()`, the Table S8 analogue). The ratios
themselves are not computed. The published ranges are carried as
`TABLE_S8_RANGES` with status `experimental`.

**Why.** A ratio whose denominator cannot be reconstructed cannot be reproduced.

### A9: Hemisphere at the equator

**Open.** The season feature depends on hemisphere, which is ambiguous at
latitude 0.

**Default.** `latitude >= 0` is north. An explicit `hemisphere=` always
overrides the latitude.

**Change it.** `RFRConfig(..., hemisphere="south")`.

**Why.** A declared tie-break, not a scientific claim. One of `hemisphere` and
`latitude` is required: there is no silent default.

### A10: Units of NEE error metrics

**Open.** Table S3 gives NEE RMSE and bias in g C m-2 d-1; the model works in
µmol m-2 s-1, and the article does not say how half-hourly errors were
aggregated into daily carbon units.

**Default.** Metrics are reported in model units. `compare_to_benchmarks()`
refuses to difference a cell whose units differ from the published ones, and
`convert_nee_to_carbon_units()` applies the rate conversion
`12.011e-6 x 86400` when asked.

**Complication.** The supplement contradicts itself. Table S9 prints the same
per-site NEE RMSE values as Table S4, whose medians are Table S3, but labels
them µmol m-2 s-1 ([`supplement_benchmarks.md`](supplement_benchmarks.md)
section 8). `read_tables_s4_to_s6(path, nee_units=...)` makes you state which
reading you use.

**Why.** Comparing two numbers in units that may differ would make any agreement
or disagreement meaningless.

### A11: Daily standard deviation and quantiles

**Open.** The degrees-of-freedom convention of the daily standard deviation and
the quantile method of the daily quartiles are not stated.

**Default.** Sample standard deviation (`daily_std_ddof=1`, the pandas default)
and linear quantile interpolation (the numpy and pandas default). A day with one
visible value therefore has quartiles and no standard deviation.

**Change it.** `FeatureConfig(daily_std_ddof=0)`.

**Why.** The library defaults are the most likely thing the original code used.

### A12: Which R2

**Open.** The article reports R2 beside a regression slope without saying
whether it is `1 - SS_res/SS_tot` or the squared correlation of that
regression.

**Default.** `r2_definition="residual"`, `1 - SS_res/SS_tot`.

**Change it.** `ValidationConfig(r2_definition="squared_correlation")`, which is
what `fluxlib` computes (F21) and the setting for a historical comparison.

**Why.** The squared correlation ignores any constant offset or rescaling of the
predictions, so it scores a biased fill as highly as an unbiased one. The two
coincide in the regime Table S3 sits in, so the published numbers do not settle
the question.

---

## The package's own conventions

These are not open points in the paper; they are rules this package applies and
you should know about.

- **What counts as a measurement.** A target value is measured when it is present
  and its QC flag is in `observed_qc_values` (default `(0,)`, FLUXNET's "measured"
  code). A missing or unrecognised flag counts as not measured. Without a QC
  column every present value counts as measured, which cannot reproduce the
  paper's distinction and is discouraged.
- **Drivers are never filled.** A row missing a driver is not trained on and not
  predicted; it stays missing and is counted.
- **Units are never converted.** Supply drivers in the units of
  [`method.md`](method.md#drivers-rfr3-and-rfr10).
- **Time is elapsed time.** Gap lengths and `time_distance_hours` come from
  timestamps, never row counts. Timestamps must be unique; duplicates raise
  unless you choose `on_duplicates="keep_first"` or `"keep_last"`.
- **Undefined metrics are missing, not zero.** They reach JSON as `null`.
- **Energy balance uses one row set.** A row enters the measured and the filled
  ratio only where every component is present, and a total available energy at
  or below zero gives no ratio.
- **"Improved" is our definition.** `method_differences()` counts a site as
  improved when R2 is higher, slope is closer to 1, RMSE is lower or bias is
  closer to 0. The paper does not define per-site improvement.
- **The Welch test treats sites as independent samples**, as Table S10 does,
  although the two methods' values are paired by site. The paired counts of
  improved and worsened sites are reported beside it.
- **One site, one target, one model.** Nothing pools sites.

## Where the supplement disagrees with itself

Found by reading the supplementary files directly
([`supplement_benchmarks.md`](supplement_benchmarks.md) sections 5 and 8). The
package records these and does not correct them.

- **Table S1 and Table S2 count ecosystems differently**: S1 gives 23 deciduous
  broadleaf (DBF) and 12 evergreen broadleaf (EBF) sites, while the rows of S2
  give 22 and 13.
- **Tables S4 and S9 label the same NEE values with different units** (A10).
- **Table S3 is not always the median of Tables S4-S6.** R2 and slope agree to
  rounding; S3's LE MDS nighttime RMSE is 21.93 where the per-site values give
  21.39.
- **One Table S10 cell contradicts its own interval**: the NEE RMSE gain of RFR3
  over MDS is marked significant, but its interval (-0.9, 0.1) includes zero.
- **Table S3's nighttime bias medians for H and LE** are printed in the
  supplement but not yet carried in `PUBLISHED_BENCHMARKS`.
- **`mmc7.xlsx`** has not been inspected and has no role in the package.

## Not implemented

- **MDS.** It appears only as published numbers.
- **The Table S8 normalized uncertainty ratios** (A8).
- **The FLUXNET2015 adapter.** FLUXNET data works today with a few lines of pandas
  or through the command line ([`fluxnet.md`](fluxnet.md)).
