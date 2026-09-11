# FLUXNET2015 adapter

`rfrgapfill.fluxnet` is the only module in the package that knows FLUXNET column
names. Everything else resolves variables through a `ColumnMap`
([`method_spec.md`](method_spec.md) section 2), so the adapter is optional: the
package works on any station naming, and removing this module changes no
science.

Zhu et al. (2022) ran on **FLUXNET2015 FULLSET half-hourly** files, so
reproducing their workflow means reading that product. This document records
what the adapter assumes about it, what the QC flags mean, and the four naming
choices the adapter makes on the user's behalf.

## Scope

The adapter does:

- parse FLUXNET timestamps and the `-9999` missing-value sentinel;
- report which canonical variables, fluxes and QC flags a frame carries;
- build a `ColumnMap` from the columns actually present;
- describe a flux QC column against the flag semantics below.

The adapter does **not**: download anything, store credentials or site lists,
convert units, filter by u\*, interpret driver QC flags, or make any
site-specific assumption. FLUXNET2015 files are obtained by the user under the
data policy they agreed to.

## Column names

The paper's variables, and the canonical names they map to:

| FLUXNET2015 column | Canonical name | Units |
|---|---|---|
| `SW_IN_F` | `shortwave` | W m-2 |
| `VPD_F_MDS` | `vpd` | hPa |
| `TA_F_MDS` | `air_temperature` | degC |
| `NETRAD` | `net_radiation` | W m-2 |
| `WS` | `wind_speed` | m s-1 |
| `WD` | `wind_direction` | degrees |
| `G_F_MDS` | `soil_heat_flux` | W m-2 |
| `TS_F_MDS` | `soil_temperature` | degC |
| `RH` | `relative_humidity` | % |
| `SWC_F_MDS` | `soil_water_content` | % |

Target fluxes and their quality flags:

| Flux | Column | QC column |
|---|---|---|
| NEE | `NEE_VUT_REF` | `NEE_VUT_REF_QC` |
| H | `H_F_MDS` | `H_F_MDS_QC` |
| LE | `LE_F_MDS` | `LE_F_MDS_QC` |

Units are recorded, never converted: a frame in different units is the user's
to convert before it reaches the package.

## QC flag semantics

For the **half-hourly and hourly** FULLSET fluxes, the `_QC` column is an
integer class describing how the value on that row was produced:

| Flag | Meaning |
|---|---|
| 0 | measured — a genuine observation |
| 1 | good-quality gap fill |
| 2 | medium-quality gap fill |
| 3 | poor-quality gap fill |

Only `0` is a measurement. The package's single structural use of the flag is
`flag == 0` (`RFRConfig.observed_qc_values`, default `(0,)`); the 1/2/3 grading
is reported by `qc_summary` and never acted on. Treating a "good" MDS gap fill
as truth would train the Random Forest on another gap-filling model's output,
hide those values inside artificial gaps, and then score RFR predictions against
them — measuring agreement between two models rather than skill against
observations ([`method_spec.md`](method_spec.md) section 7).

Two boundaries the adapter refuses to cross:

- **Aggregated products do not use these codes.** In the daily, weekly, monthly
  and yearly FLUXNET2015 products, `_QC` is the *fraction* of
  measured-or-good-quality records in the aggregation window — a float in
  `[0, 1]`. `qc_summary` raises `FluxnetError` on a non-integer flag rather than
  reading a fraction as a class code. A file whose fractions happen to be all
  0.0 and 1.0 is indistinguishable from flag codes, so use the half-hourly or
  hourly product and do not rely on the check to catch the wrong file for you.
- **Driver QC flags are not interpreted.** The paper used FLUXNET's pre-filled
  meteorological drivers as given, and the `_F` / `_F_MDS` driver flags encode a
  different provenance scheme (including ERA-downscaled values). The adapter
  reports these columns if asked and leaves the decision to the user; consult
  the FLUXNET2015 data product documentation before acting on them.

## Adapter conventions

These are choices about the **data product**, not about the method. They carry
their own `F` identifiers so they can never be confused with the paper's
ambiguities `A1`–`A10` in [`method_spec.md`](method_spec.md), and none of them
may be described as "paper exact".

| ID | Question | Choice | Configuration |
|---|---|---|---|
| F1 | A FULLSET record carries `TIMESTAMP_START` and `TIMESTAMP_END`; the article does not say which labelled its half-hours. | Index by `TIMESTAMP_START` — the record is labelled by the start of its averaging interval. At 30-minute cadence the alternative shifts every stamp by 30 minutes, which moves rows across the daytime threshold and the day boundary of the daily statistics. | `prepare_fluxnet_frame(timestamp=...)` |
| F2 | Real files often carry depth-indexed soil columns (`TS_F_MDS_1`, `SWC_F_MDS_2`, …) rather than the bare names the paper lists. | When the bare name is absent, use the lowest present index (`_1` … `_10`), i.e. the shallowest sensor. Applied only to `soil_temperature` and `soil_water_content`. | `inspect_fluxnet(overrides=...)` |
| F3 | NEE ships in several u\*-threshold variants. | `NEE_VUT_REF`, the variable-u\*-threshold reference series the paper validated. CUT variants are reachable through `fluxes=`. | `inspect_fluxnet(fluxes=...)` |
| F4 | `-9999` is a plausible-looking number, not a missing marker, to any code that does not know the product. | Replace exact `-9999` in numeric columns with `NaN` on ingest, before anything else sees the frame. | `prepare_fluxnet_frame(sentinel=...)` |

## Reporting missing variables

`inspect_fluxnet` reports availability before a model is fitted. This matters
because RFR10 is not available everywhere: the paper's full RFR3/RFR10
comparison used 94 of 194 sites, because complete QC and extended-driver
availability were more restrictive
([`supplement_benchmarks.md`](supplement_benchmarks.md)). A site that cannot run
RFR10 is a normal outcome to be reported, not an error to be worked around by
substituting drivers.

```python
info = inspect_fluxnet(frame)
info.best_mode        # Mode.RFR10, Mode.RFR3, or None
info.missing_drivers  # canonical name -> the FLUXNET column that is absent
info.summary()        # human-readable
info.require("RFR10") # raises FluxnetError listing everything a run would lack
```

`require(..., require_qc=True)` treats a flux without its QC column as a
failure: on a raw FLUXNET file, running without the flag would silently promote
MDS-filled values to measurements.

One more name mismatch is worth knowing about: the energy-balance ratio defaults
to targets literally called `H` and `LE` ([`method_spec.md`](method_spec.md)
section 6.4), which under FLUXNET naming are `H_F_MDS` and `LE_F_MDS`. Pass
`energy_balance_targets=info.heat_targets()` or the ratio is quietly skipped.

## Worked example

```python
from rfrgapfill import RFRConfig, validate_rfr
from rfrgapfill.fluxnet import inspect_fluxnet, read_fluxnet_csv

frame = read_fluxnet_csv("FLX_XX-Site_FLUXNET2015_FULLSET_HH_2004-2014_1-4.csv")
info = inspect_fluxnet(frame)
info.require(info.best_mode)

report = validate_rfr(
    frame,
    targets=list(info.target_columns()),
    mode=info.best_mode,
    scenario="zhu2022",
    column_map=info.column_map(),
    qc_columns=info.qc_columns(),
    energy_balance_targets=info.heat_targets(),
    latitude=51.5,          # or hemisphere="north"; the site's own metadata
    frequency="30min",
)
```

Site latitude, elevation, IGBP class and the rest are site metadata the user
supplies from the FLUXNET site table; the adapter deliberately ships no site
list of its own.
