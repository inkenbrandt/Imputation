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

All nine files are present in the working directory with the publisher prefix
`1-s2.0-S0168192321004639-`, and are git-ignored rather than committed.

`mmc7.xlsx` has no assigned role. Its table identity and structure must be
inspected before any programmatic use; do not infer its role from filename
ordering.

> **Status note:** the supplementary files are present in the working directory as
> `1-s2.0-S0168192321004639-mmc1..9.{docx,xlsx}` but are **not tracked in the
> repository** (publisher-copyright material, git-ignored). The numbers below are
> transcribed from the project context document and have **not yet been verified
> against the supplement files themselves**. Until that verification happens, mark
> tests depending on them with the `supplement` marker and treat the values as
> provisional.

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

**Source:** Supplementary Table S10 (`mmc9.docx`) — mean method differences with
Welch-test 95% confidence intervals.

Selected mean R2 improvements:

```text
RFR3 - MDS:     NEE +0.07    H +0.11    LE +0.11
RFR10 - RFR3:   NEE +0.05    H +0.12    LE +0.10
```

Gains are strongest and most consistent for H and LE. Some NEE differences —
notably bias and RMSE — are **not** statistically significant per Table S10, and
must not be reported as improvements.

Welch-test reproduction across sites is optional and belongs in a separate
statistical module, not in ordinary single-site gap filling.

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

---

## 9. Benchmark-to-test map

| Benchmark | Source | Acceptance test |
|---|---|---|
| Diel medians, 94 sites | Table S3 (`mmc4.docx`) | 38 |
| Very-long-gap medians | Table S3 (`mmc4.docx`) | 38, gap-class reporting |
| Nighttime medians | Table S3 (`mmc4.docx`) | 32 (day/night split) |
| Welch mean differences | Table S10 (`mmc9.docx`) | optional statistical module |
| RFR3 vs ORF | Figure S1 (`mmc1.docx`) | 7, 8, 9, 10 |
| Bias IQR / normalized uncertainty | Table S8 (`mmc6.docx`) | 40 |
| Site metadata, 94-site membership | Tables S1–S2 (`mmc3.docx`) | 39 |
| 194-site NEE MDS vs RFR3 | Table S9 (`mmc8.docx`) | 39 |
| Unassigned benchmark asset | `mmc7.xlsx` | none — inspect first |
