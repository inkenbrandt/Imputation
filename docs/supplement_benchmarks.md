# Supplement-derived benchmarks and their sources

Companion to [`method_spec.md`](method_spec.md). This document records **which
supplementary table or figure supports each numerical reproduction target**, so no
benchmark number ever appears in code or tests without a traceable source.

## How these numbers may be used

- They are **integration-test and reproduction benchmarks**, valid only with
  matching FLUXNET inputs and matching preprocessing.
- They are **not** universal pass/fail thresholds for arbitrary sites. A new
  dataset is never required to match them.
- Performance varies strongly by ecosystem and by gap length. Never assert a
  blanket claim such as `R2 > 0.8` for every site.
- Every comparison must state the site subset, gap class, and day/night subset it
  was computed on.

---

## 0. Where these numbers live in the code

Sections 2-4 below are also carried as data in
`rfrgapfill.benchmarks.PUBLISHED_BENCHMARKS`, one `PublishedBenchmark` record per
value, each naming its source table and the site population it is a median over.

`tests/test_benchmarks.py` re-reads **this document** and checks every constant
against the table it was transcribed from, so the two cannot drift apart: editing
a number here without editing the code, or the reverse, fails the test suite.
That is what makes the project rule - no benchmark number in code without a
traceable source - enforceable rather than aspirational.

`compare_to_benchmarks()` puts a reproduction run beside these medians metric by
metric. It asserts nothing: a cell whose units differ from the published units is
marked not comparable rather than differenced (ambiguity A10), and no function in
that module turns a benchmark into a pass/fail threshold.

Section 5's Table S10 is carried as `rfrgapfill.sites.TABLE_S10_COMPARISONS`, and
`tests/test_sites.py` re-reads this document to check it. The per-site tables
(S2, S4-S6, S9) are too large to transcribe and are publisher material, so they
are not carried at all: `rfrgapfill.sites.read_table_s2()`,
`read_tables_s4_to_s6()` and `read_table_s9()` parse a local copy into the
package's tidy tables (section 8).

Section 7's Table S8 ranges are carried separately, as
`rfrgapfill.uncertainty.TABLE_S8_RANGES`, and `tests/test_uncertainty.py`
re-reads this document to check them in the same way. They are deliberately
**not** in `PUBLISHED_BENCHMARKS`: the package cannot compute the quantity they
describe (section 7), so there is nothing a run could be compared against.

---

## 1. Source map

Supplementary files as supplied with the article.

| File | Contents | Used for |
|---|---|---|
| `mmc1.docx` | Supplementary Figure S1 — RFR3 vs ORF | receptive-limiter effect (section 6) |
| `mmc2.docx` | Gap-length sensitivity and normalized-uncertainty discussion/figure | section 5 context |
| `mmc3.docx` | Supplementary Tables S1–S2 — site metadata, 94-site membership | section 7 |
| `mmc4.docx` | Supplementary Table S3 — summary performance distributions | sections 2, 3, 4 |
| `mmc5.docx` | Supplementary Tables S4–S6 — site-level diel/day/night results | site-level checks |
| `mmc6.docx` | Supplementary Table S8 — normalized uncertainty ranges | section 5 |
| `mmc8.docx` | Supplementary Table S9 — 194-site NEE, MDS vs RFR3 | section 7 |
| `mmc9.docx` | Supplementary Table S10 — Welch-test confidence intervals | section 4 |
| `mmc7.xlsx` | **Unverified.** Retain as a benchmark data asset. | none until inspected |

All nine files are present locally in `publication/` with the publisher prefix
`1-s2.0-S0168192321004639-`, and are git-ignored rather than committed. Tests
marked `supplement` look for them there (or in `$RFRGAPFILL_SUPPLEMENT_DIR`) and
skip when they are absent.

`mmc7.xlsx` has no assigned role. Its table identity and structure must be
inspected before any programmatic use; do not infer its role from filename
ordering.

> **Verification status.** The supplementary files are **not tracked in the
> repository** (publisher-copyright material, git-ignored). Section 5 is checked
> against `mmc9.docx` itself by the `supplement`-marked tests. Sections 2-4 were
> transcribed from the project context document, and no test parses `mmc4.docx`
> directly; the `supplement`-marked tests in `tests/test_sites.py` recompute Table
> S3's medians from the per-site Tables S4-S6 and find the transcribed R2 and
> slope values to within rounding, with the RMSE discrepancies listed in section 8.
> Tests that need the files skip when they are absent.

---

## 2. Diel median performance — 94-site complete-analysis subset

**Source:** Supplementary Table S3 (`mmc4.docx`). Medians across sites, all gap
classes combined, all observations (diel).

| Flux | Method | Median R2 | Median slope | Median RMSE | Median bias |
|---|---|---:|---:|---:|---:|
| NEE | MDS | 0.72 | 0.79 | 2.98 | 0.00 |
| NEE | RFR3 | 0.78 | 0.79 | 2.63 | -0.01 |
| NEE | RFR10 | 0.84 | 0.84 | 2.42 | 0.02 |
| H | MDS | 0.67 | 0.72 | 50.33 | -1.12 |
| H | RFR3 | 0.78 | 0.78 | 39.37 | 0.26 |
| H | RFR10 | 0.90 | 0.90 | 26.80 | -0.13 |
| LE | MDS | 0.63 | 0.70 | 39.84 | -2.36 |
| LE | RFR3 | 0.75 | 0.76 | 33.50 | -0.11 |
| LE | RFR10 | 0.85 | 0.85 | 25.51 | -0.46 |

**Units in Table S3:** RMSE and bias are `g C m-2 d-1` for NEE and `W m-2` for H
and LE. Model-native NEE units are `umol m-2 s-1`, so a documented unit conversion
is required before comparison (ambiguity A10 in `method_spec.md`).

**Supports:** acceptance test 38 — aggregate RFR3/RFR10 comparison against Table
S3, without requiring exact equality unless preprocessing and source versions are
identical.

---

## 3. Very-long-gap medians (30-day class)

**Source:** Supplementary Table S3 (`mmc4.docx`).

### NEE

| Method | R2 | slope | RMSE | bias |
|---|---:|---:|---:|---:|
| MDS | 0.59 | 0.85 | 2.99 | -0.48 |
| RFR3 | 0.74 | 0.97 | 2.61 | -0.14 |
| RFR10 | 0.74 | 0.97 | 2.48 | -0.04 |

### H

| Method | R2 | slope | RMSE | bias |
|---|---:|---:|---:|---:|
| MDS | 0.62 | 0.88 | 55.00 | -9.41 |
| RFR3 | 0.76 | 0.99 | 40.11 | 0.66 |
| RFR10 | 0.89 | 1.00 | 26.80 | -0.20 |

### LE

| Method | R2 | slope | RMSE | bias |
|---|---:|---:|---:|---:|
| MDS | 0.57 | 0.85 | 45.45 | -20.24 |
| RFR3 | 0.74 | 0.98 | 34.59 | -0.35 |
| RFR10 | 0.83 | 0.99 | 25.49 | -3.66 |

**Supports:** per-gap-class reporting (`method_spec.md` §6.3) and the claim that
the extended RFR10 driver set matters most for H and LE at long gap durations.

---

## 4. Nighttime behaviour

**Source:** Supplementary Table S3 (`mmc4.docx`), nighttime subset medians.

```text
H nighttime:
MDS    R2=0.03, slope=0.16, RMSE=29.88
RFR3   R2=0.16, slope=0.20, RMSE=19.46
RFR10  R2=0.49, slope=0.47, RMSE=14.24

LE nighttime:
MDS    R2=0.02, slope=0.17, RMSE=21.93
RFR3   R2=0.10, slope=0.15, RMSE=12.75
RFR10  R2=0.27, slope=0.32, RMSE=9.79
```

**Supports:** the requirement in `method_spec.md` §6.2 that day/night metrics are
always reported. Nighttime skill is weak even for the best configuration, and
aggregate-only reporting would hide it.

---

## 5. Statistical comparison benchmarks

**Source:** Supplementary Table S10 (`mmc9.docx`) — mean method differences
(`method - baseline`) across the 94-site subset, with Welch-test 95% confidence
intervals. The supplement's asterisk marks `p > 0.05`; the last column carries it
wherever it is printed, on the mean or on the interval. Intervals printed in the
supplement as `8.17×10-02` are written out. Units: R2 and slope are
dimensionless; RMSE and bias are `W m-2` for H and LE and, per the caption,
`g C m-2 d-1` for NEE (ambiguity A10).

| Flux | Metric | Comparison | Mean difference | CI lower | CI upper | p > 0.05 |
|---|---|---|---:|---:|---:|---|
| NEE | R2 | RFR3 vs MDS | 0.07 | 0.0 | 0.1 | |
| NEE | R2 | RFR10 vs RFR3 | 0.05 | 0.02 | 0.09 | |
| NEE | slope | RFR3 vs MDS | 0.01 | -0.0274 | 0.0484 | * |
| NEE | slope | RFR10 vs RFR3 | 0.04 | 0.00 | 0.08 | |
| NEE | bias | RFR3 vs MDS | 0.00 | -0.1 | 0.1 | * |
| NEE | bias | RFR10 vs RFR3 | 0.03 | -0.03 | 0.08 | * |
| NEE | RMSE | RFR3 vs MDS | -0.40 | -0.9 | 0.1 | |
| NEE | RMSE | RFR10 vs RFR3 | -0.24 | -0.67 | 0.18 | * |
| H | R2 | RFR3 vs MDS | 0.11 | 0.0817 | 0.146 | |
| H | R2 | RFR10 vs RFR3 | 0.12 | 0.10 | 0.15 | |
| H | slope | RFR3 vs MDS | 0.06 | 0.0 | 0.1 | |
| H | slope | RFR10 vs RFR3 | 0.11 | 0.09 | 0.13 | |
| H | bias | RFR3 vs MDS | 1.97 | 0.7 | 3.2 | |
| H | bias | RFR10 vs RFR3 | -0.06 | -0.81 | 0.68 | * |
| H | RMSE | RFR3 vs MDS | -10.55 | -15.3 | -5.8 | |
| H | RMSE | RFR10 vs RFR3 | -13.05 | -16.49 | -9.62 | |
| LE | R2 | RFR3 vs MDS | 0.11 | 0.0705 | 0.141 | |
| LE | R2 | RFR10 vs RFR3 | 0.10 | 0.07 | 0.13 | |
| LE | slope | RFR3 vs MDS | 0.05 | 0.0 | 0.1 | |
| LE | slope | RFR10 vs RFR3 | 0.09 | 0.06 | 0.12 | |
| LE | bias | RFR3 vs MDS | 2.99 | 1.6 | 4.4 | |
| LE | bias | RFR10 vs RFR3 | -0.85 | -1.48 | -0.22 | |
| LE | RMSE | RFR3 vs MDS | -7.89 | -12.7 | -3.0 | |
| LE | RMSE | RFR10 vs RFR3 | -8.19 | -11.60 | -4.78 | |

Gains are strongest and most consistent for H and LE. Some NEE differences —
notably bias and RMSE — are **not** statistically significant per Table S10, and
must not be reported as improvements.

**Carried and reproduced.** The table is `rfrgapfill.sites.TABLE_S10_COMPARISONS`,
and `tests/test_sites.py` re-reads this section to check it, and - where a local
copy of the supplement exists - checks it against `mmc9.docx` itself.
`rfrgapfill.sites.welch_comparison()` applied to the per-site values of Table S4
reproduces every mean difference above to within 0.05 (both sides are printed to
two decimals), and `compare_to_table_s10()` lays the two side by side. Table S10
treats the per-site values of the two methods as independent samples, although
they are paired by site; the reproduction does the same, and reports the paired
view - how many sites improved or worsened - beside it rather than instead of it.

One cell disagrees with itself: the NEE RMSE gain of RFR3 over MDS carries no
asterisk, but its printed interval, (-0.9, 0.1), crosses zero. The reproduction
agrees with the interval (p ≈ 0.07). The asterisk is transcribed as printed.

Welch-test reproduction across sites belongs in `rfrgapfill.sites`, separate from
ordinary single-site gap filling.

---

## 6. Receptive-limiter effect (ORF benchmark)

**Source:** Supplementary Figure S1 (`mmc1.docx`) — RFR3 versus ORF, a Random
Forest without receptive-limiter feature engineering.

No tabulated numerical values are available from the figure. The benchmark is
therefore **directional and recorded, not asserted**:

- acceptance test 10 records the RFR-minus-ORF difference in R2, slope, RMSE, and
  bias on a fixed reproducible dataset;
- it does not assert that every individual metric improves;
- ORF must share the estimator family, driver set, grid, and training rows with the
  RFR it is compared against (`method_spec.md` §3.6).

---

## 7. Uncertainty diagnostics

**Source:** Supplementary Table S8 (`mmc6.docx`) and the companion sensitivity
figure (`mmc2.docx`).

Approximate bias-IQR / joint-uncertainty-CI intervals for very-long gaps:

| Flux | MDS | RFR3 | RFR10 |
|---|---|---|---|
| NEE | 5.21–5.34 | 2.90–2.97 | 0.87–0.89 |
| H | 5.37–5.63 | 2.46–2.57 | 1.65–1.72 |
| LE | 7.90–8.93 | 2.25–2.55 | 1.79–2.02 |

Normalized uncertainty rises steeply with gap length for MDS and more gradually for
RFR.

**Implementation position:**

- bias IQR **by gap class** is a supported core output: across sites, stratified
  by IGBP class where site metadata is available, by
  `rfrgapfill.uncertainty.bias_iqr()`; across the gaps of one site by
  `TargetValidation.bias_spread_frame()` (`method_spec.md` §6.3). Treating the
  published IQR as a spread across sites is an inference, not a statement in
  the supplement;
- the normalized ratios above depend on a denominator (flux CI and joint
  flux-uncertainty CI) that has **not** been reconstructed from the supplementary
  methods. Exact reproduction of these ratios must not be claimed. The package
  does not compute them; the ranges are carried in
  `rfrgapfill.uncertainty.TABLE_S8_RANGES` with a fixed `experimental` status and
  a caveat on every row (ambiguity A8). Any later implementation must carry
  "experimental" in its name until it has been reproduced against Table S8.

---

## 8. Site and ecosystem stratification

**Source:** Supplementary Tables S1–S2 (`mmc3.docx`); Supplementary Table S9
(`mmc8.docx`).

- The wider metadata set covers **194 sites** across multiple continents and 11
  IGBP classes.
- The full RFR3/RFR10 × NEE/H/LE comparison used a **94-site subset**, restricted
  by complete QC and RFR10 driver availability.
- Table S9 provides a broader **194-site NEE comparison of MDS versus RFR3**.

Consequences for the implementation:

- do not treat the 94-site subset as the only validation evidence for RFR3;
- do not assume RFR10 can be run at every RFR3 site;
- report site and ecosystem metadata alongside multi-site validation results
  (acceptance test 39).

Metadata to preserve per site:

```text
site_id, latitude, longitude, continent, country, IGBP,
Koppen climate class, start_date, end_date, elevation,
instrument system, instrument-to-canopy height ratio,
included_in_94_site_subset
```

**Implementation.** `rfrgapfill.sites` (`method_spec.md` §6.6): `site_metadata()`
normalises Table S2's headers, check/cross marks and IGBP spellings;
`site_report()` puts every site's own scores beside its metadata;
`stratified_summary()` gives across-site quartiles per IGBP class (or any other
metadata field); `method_differences()` and `welch_comparison()` report per-site
changes and the Table S10 statistic; `site_level_evidence()` records that RFR3 NEE
has site-level results at 194 sites and RFR10 at 94. None of them fits a model or
applies a threshold.

**What reading the tables showed.** The supplement-marked tests in
`tests/test_sites.py` read the publisher's files and pin each of these:

- Table S2 lists 194 sites, 94 of them marked as in the complete-analysis
  subset; Tables S4-S6 score exactly those 94. One wetland is spelled `Wet`.
- Table S1's continent counts match the rows of Table S2; its IGBP counts do
  not quite: S1 gives 23 DBF and 12 EBF sites where the rows of S2 give 22 and 13.
- At the 94 shared sites, Tables S4 and S9 print **identical** NEE R2, slope and
  RMSE for MDS and RFR3, but different biases at nearly every site, so they are
  not the same experiment printed twice. S9's caption gives those RMSE values in
  `umol m-2 s-1`; S3 and S10 give the same numbers in `g C m-2 d-1`. This is new
  evidence on ambiguity A10, recorded there; the package default is unchanged.
- Table S3's medians are the medians of Tables S4-S6 for R2 and slope, to
  rounding. RMSE and bias mostly agree to a few tenths, but not everywhere: S3
  prints LE MDS nighttime RMSE as 21.93 against a median of 21.39 in S6 (the
  digits look transposed), and H RFR3 diel RMSE as 39.37 against 39.68.
- Table S3 prints nighttime bias medians for H and LE (H: MDS -5.82, RFR3 -0.31,
  RFR10 -0.25; LE: -6.03, -0.27, -0.29). Section 4 above, and
  `PUBLISHED_BENCHMARKS`, do not carry them yet.
- In Table S9, RFR3 loses NEE R2 to MDS at some sites. Improvement is a
  tendency across sites, not a property of each one.

---

## 9. Benchmark-to-test map

| Benchmark | Source | Acceptance test |
|---|---|---|
| Diel medians, 94 sites | Table S3 (`mmc4.docx`) | 38 |
| Very-long-gap medians | Table S3 (`mmc4.docx`) | 38, gap-class reporting |
| Nighttime medians | Table S3 (`mmc4.docx`) | 32 (day/night split) |
| Welch mean differences | Table S10 (`mmc9.docx`) | `rfrgapfill.sites`, reproduced from Table S4 |
| RFR3 vs ORF | Figure S1 (`mmc1.docx`) | 7, 8, 9, 10 |
| Bias IQR / normalized uncertainty | Table S8 (`mmc6.docx`) | 40 |
| Site metadata, 94-site membership | Tables S1–S2 (`mmc3.docx`) | 39 |
| 194-site NEE MDS vs RFR3 | Table S9 (`mmc8.docx`) | 39 |
| Unassigned benchmark asset | `mmc7.xlsx` | none — inspect first |
