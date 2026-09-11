# Validating RFR with artificial gaps

How to measure how well RFR fills gaps at a site, how the measurement is kept
honest, and how to read and compare the results. The method itself is
[`method.md`](method.md); the contract is sections 4 and 6 of
[`method_spec.md`](method_spec.md).

Two [example notebooks](examples.md) run everything on this page on a synthetic
site:
[`02_artificial_gap_validation.ipynb`](../examples/notebooks/02_artificial_gap_validation.ipynb)
covers the scenario, long gaps, leakage and reading the results, and
[`03_comparing_configurations.ipynb`](../examples/notebooks/03_comparing_configurations.ipynb)
compares the arms and puts a run beside the published medians.

---

## What validation answers

Real gaps have no truth to score against, so validation makes gaps where the
truth is known. Stretches of genuinely measured data are hidden, the model is
fitted on what is left, and its predictions for the hidden stretches are scored
against the measurements.

The result says how RFR would have filled gaps of 24 hours, 7 days and 30 days
**at this site, over this record**. It says nothing about any other site, and one
synthetic site is a software check, not evidence about the method. The paper's
evidence is a distribution of per-site results across 94 and 194 sites.

## The experiment in one call

```python
from rfrgapfill import validate_rfr

report = validate_rfr(
    df,
    config=config,
    targets=["NEE", "H", "LE"],
    qc_columns={"NEE": "NEE_QC", "H": "H_QC", "LE": "LE_QC"},
)
```

`validate_rfr` runs the stages in a fixed order, and the order is part of the
leakage protection. Assembling the same pieces by hand is possible but not
supported, because the order is easy to get wrong:

1. place the artificial gaps and record them in a `GapManifest`;
2. turn the gaps into a holdout mask, before any feature exists;
3. hide the held-out values;
4. build the features, computing the daily target statistics only from what is
   still visible;
5. fit the forest, including its grid search, on the visible measured rows;
6. predict the held-out rows, and only those;
7. score the predictions against the hidden measurements.

The withheld rows **are** the test set, about a quarter of the measured data,
and the remaining measured rows are the training set, as in Figure 2 of the
paper. There is no random row-wise split: it would scatter single test rows among
their training neighbours and destroy the gap structure being tested.

## The gap scenario

| Setting | Paper | Default | Where |
|---|---|---|---|
| Gap durations | 24 h, 7 d, 30 d | same | `GapScenarioConfig.durations` |
| Share of measured data withheld | about 25% | `missing_fraction=0.25` | `GapScenarioConfig` |
| Mix of the three classes | 20 / 30 / 50 | same | `gap_mix` |
| What the mix counts | not stated | withheld records (A3) | `allocation_basis` |
| Minimum measured coverage under a gap | at least 50% | `min_observed_fraction=0.50` | `GapScenarioConfig` |
| Same gaps for NEE, H and LE | yes | yes | `shared_gaps_across_targets` |

Gaps are placed from timestamps and durations, seeded from
`RFRConfig.random_state`, never overlapping and never crossing the ends of the
record. A proposed gap sitting mostly over real missing data is rejected and
redrawn, because withholding a value that was never measured withholds nothing.

What was achieved is measured from the placed gaps and reported next to what was
asked for (`report.gaps.summary()`), never adjusted to look exact. Real outages
under the gaps usually make the achieved fraction smaller than requested (A7). A
scenario that cannot be built at all raises `GapError`, naming the classes that
fell short; `on_shortfall="warn"` returns the partial design for inspection.

To choose the gaps yourself, build them first and pass them in:

```python
from rfrgapfill import GapScenarioGenerator

gaps = GapScenarioGenerator(config).generate(
    df, target=["NEE", "H", "LE"], qc_column={"NEE": "NEE_QC", "H": "H_QC", "LE": "LE_QC"}
)
report = validate_rfr(df, config=config, targets=["NEE", "H", "LE"], qc_columns=qc, gaps=gaps)
```

`SyntheticSite.known_gaps()` is a fixed, hand-checkable plan for the synthetic
site; it is a test fixture, not the paper's scenario.

## Leakage-safe target statistics

The receptive limiter includes daily quartiles and a daily standard deviation of
the flux being predicted. If they are computed before the gaps are hidden, a
hidden value helps build the features used to predict it, and the score improves
for a reason that will never exist when filling a real gap.

The default `feature_mode="paper_safe"` prevents this in two independent ways,
each sufficient on its own:

- the held-out values are **removed** from the frame the features are computed
  from;
- the feature builder is **told** which rows are visible and ignores the rest.

It is also checkable. `validate_rfr(..., check_leakage=True)` runs a probe that
replaces the hidden truth with absurd values, rebuilds every feature with and
without the first protection, and fails if any feature column moved.
`detect_target_leakage()` and `require_no_target_leakage()` run the same probe
outside a validation run.

The paper's own code did not do this: the archived `fluxlib` implementation
computes its daily statistics before the gaps are hidden
([`fluxlib_audit.md`](fluxlib_audit.md), finding F1). To measure how much that
flatters a score, `FeatureConfig(feature_mode="legacy_fluxlib")` reproduces the
historical derivation. It warns on every build, labels its arms `RFR3-legacy`
and `RFR10-legacy`, and is never the default.

## Long gaps need a daily-statistic strategy (A4)

A gap that covers a whole calendar day leaves that day with no visible
measurements, so under the default `daily_statistic_strategy="missing"` its daily
statistics are missing and none of its rows can be predicted. With the default,
7-day and 30-day gaps get no predictions at all. That is the honest reading of
a point the paper does not address, and it is not hidden: the run raises a
`ValidationWarning`, and the feature record shows
`holdout_rows_with_complete_features = 0`.

Any run that includes the 7-day and 30-day classes has to choose a strategy that
reaches further:

| Strategy | For a day without enough visible measurements |
|---|---|
| `missing` (default) | statistics stay missing; nothing is borrowed |
| `within_day_available` | use whatever that day has, ignoring the minimum |
| `neighbor_day_fallback` | copy the statistics of the nearest adequate day, up to `fallback_window_days` away |
| `rolling_available` | recompute from the visible measurements within ± `fallback_window_days` days |

```python
config = config.replace(
    features=config.features.replace(
        daily_statistic_strategy="rolling_available", fallback_window_days=7
    )
)
```

Every strategy uses only visible measurements, so none of them leaks; they differ
in how far they reach. The choice is recorded in every run manifest.

## Reading the results

```python
report.to_frame()          # one row per target x gap class x subset
report["LE"].metric(subset="nighttime", gap_class="very_long")
report.summary()           # a readable account of the scenario and every target
report.gaps.summary()      # what the scenario asked for and achieved
report.bias_spread_frame() # bias quartiles across the gaps of each class
report.energy_balance_frame()
```

The tidy table has the columns `target`, `method`, `mode`, `gap_class`,
`subset`, `n`, `n_offered`, `r2`, `slope`, `rmse` and `bias`.

- **`subset`** is `all`, `daytime` or `nighttime`; daytime is shortwave
  radiation above 20 W m-2, the paper's threshold. A row with missing radiation
  is in neither subset, because it cannot be classified without inventing a
  measurement.
- **`gap_class`** is `all`, `short` (24 h), `long` (7 d) or `very_long` (30 d),
  in that order.
- **`n` against `n_offered`** is how many held-out rows were scored out of how
  many were withheld. The difference is rows the model could not predict for want
  of a predictor or a daily statistic.
- **An undefined metric is missing**, never zero: an empty subset has no RMSE,
  and constant measurements have no R2 or slope.
- **Units are the flux's own.** RMSE and bias for NEE are in µmol m-2 s-1, and
  for H and LE in W m-2.
- **Bias is** `(sum(filled) - sum(measured)) / n`, the paper's definition, and
  the regression slope is filled on measured, with an intercept.

### The energy-balance check

When H and LE are validated together, the run also compares the energy-balance
ratio `sum(H + LE) / sum(NETRAD - G)` computed from the measured and from the
filled values, over exactly the same withheld rows (`report.energy_balance`, and
per gap class in `energy_balance_frame()`). It needs `net_radiation` and
`soil_heat_flux` in the column map, whatever the driver mode. It recognises the
targets named `H` and `LE`; with other names, pass
`energy_balance_targets=("my_H", "my_LE")`.

### Read the metrics together

No single number describes a fill.

- **R2 needs something to explain.** In the README quick start, the synthetic
  site's 30-day gap falls in winter, when NEE barely varies. Its RMSE is the
  smallest of any NEE gap class, yet its R2 is negative, because R2 compares the
  error with a variance that is almost zero. Read R2 beside RMSE and bias, never
  alone.
- **Nighttime skill is weak everywhere.** Across the paper's 94 sites the best
  median nighttime R2 for H is 0.49, and for LE 0.27
  ([`supplement_benchmarks.md`](supplement_benchmarks.md) section 4). Always look
  at the `nighttime` rows; the `all` rows hide them.
- **Which R2** is a setting (A12). The default is `1 - SS_res/SS_tot`; the
  squared correlation is available, and the choice is recorded in every result.

## Comparing RFR3, RFR10 and ORF

Score the arms on the same gaps, which is what makes them comparable:

```python
from rfrgapfill import gap_length_pivot, gap_length_table

rfr3 = validate_rfr(df, config=config.replace(mode="RFR3"), targets=targets, qc_columns=qc)
rfr10 = validate_rfr(
    df, config=config.replace(mode="RFR10"), targets=targets, qc_columns=qc, gaps=rfr3.gaps
)
orf3 = validate_rfr(
    df, config=config.replace(mode="RFR3").as_orf(), targets=targets, qc_columns=qc,
    gaps=rfr3.gaps,
)

table = gap_length_table([rfr3, rfr10, orf3], site="my-site")
gap_length_pivot(table, metric="rmse", gap_class="very_long")
```

The ORF arm is the same forest over the same drivers without the receptive
limiter, so the RFR-minus-ORF difference is the limiter's contribution
(Supplementary Figure S1). Nothing requires every metric to improve.

## Across sites

Every site is validated on its own; a multi-site study compares the resulting
**scores**. No function here fits a model on more than one site's data.

```python
from rfrgapfill import (
    gap_length_table, median_across_sites, method_differences, site_metadata,
    site_report, stratified_summary, welch_comparison,
)

table = gap_length_table({"SITE-A": [rfr3_a, rfr10_a], "SITE-B": [rfr3_b, rfr10_b]})
median_across_sites(table)                       # the aggregation Table S3 reports

sites = site_metadata(my_site_list)              # site_id, latitude, igbp, koppen, ...
report = site_report(table, sites)               # every score beside its site's metadata
stratified_summary(report, by="igbp")            # quartiles across sites, per ecosystem
method_differences(report, method="RFR10", baseline="RFR3")   # per site: improved or not
welch_comparison(report, method="RFR10", baseline="RFR3")     # the Table S10 statistic
```

`gap_length_table` also accepts tidy frames read back from disk, so a many-site
study does not need every fitted forest in memory. Three rules hold throughout:
no site is required to improve, no threshold decides whether a site passes, and
the published evidence is not equally broad (RFR3 NEE has results at 194 sites,
RFR10 only at 94). The details are in section 6.6 of
[`method_spec.md`](method_spec.md).

## Comparing with the paper

The published medians of Supplementary Table S3 are carried as data:

```python
from rfrgapfill import benchmark_table, compare_to_benchmarks, convert_nee_to_carbon_units

benchmark_table()                                   # the published medians
compare_to_benchmarks(median_across_sites(table))   # run | published | difference | comparable
```

They are medians across the paper's 94-site subset of FLUXNET2015, valid as a
reproduction target only for a run over matching sites, inputs and
preprocessing. They are not pass/fail thresholds for any site, and nothing in the
package uses them as one.

NEE RMSE and bias are published in g C m-2 d-1, not the model's µmol m-2 s-1, and
the paper does not say how it converted them (A10). `compare_to_benchmarks`
therefore marks those cells `comparable=False` until you convert explicitly with
`convert_nee_to_carbon_units()`, a rate conversion that is documented as exactly
that.

The Table S10 Welch comparison is carried too: `compare_to_table_s10(welch)` puts
a `welch_comparison()` result beside it. With a local copy of the journal
supplement, `read_tables_s4_to_s6(path)` reads the paper's own per-site results
into the same tables, and the package reproduces Table S10 from them; see
[`supplement_benchmarks.md`](supplement_benchmarks.md) section 5.

## Recording the run

```python
report.manifest("LE").save("LE_validation.json")
```

The manifest records the configuration, column mapping, QC rule, grid and chosen
parameters, the placed gaps, the row accounting, the environment versions and
this run's answer to every ambiguity A1-A12. `save()` refuses to write a manifest
that is missing a required field.
