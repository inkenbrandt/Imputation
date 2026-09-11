# Using FLUXNET2015 data

The paper was validated on FLUXNET2015, and the package's reference column names
are FLUXNET2015's. The core never requires them: any data works through a
`ColumnMap`.

> **Two routes.** This page does every step with pandas and
> `ColumnMap.fluxnet2015()`, so nothing is hidden. The optional adapter,
> `rfrgapfill.fluxnet`, wraps the same steps: `read_fluxnet_csv` reads a file,
> `inspect_fluxnet` reports which drivers, fluxes and QC flags it carries, and
> `qc_summary` describes a flux's flags. Its conventions are recorded in
> [`fluxnet_adapter.md`](fluxnet_adapter.md). Nothing in this package downloads
> data or holds credentials: obtain FLUXNET2015 files under their own licence.

[`04_bring_your_own_data.ipynb`](../examples/notebooks/04_bring_your_own_data.ipynb)
runs every step on this page, including the command line, on a stand-in file
written in the FLUXNET2015 layout.

---

## Which files and columns

Use the half-hourly `FULLSET_HH` files (or `FULLSET_HR` for the sites that
report hourly, with `frequency="1h"`). The daily and coarser aggregations are
not suitable: their QC columns are fractions, not the per-record flags this
package reads.

The paper's targets and drivers:

| Role | FLUXNET2015 column | Canonical name | Units |
|---|---|---|---|
| target | `NEE_VUT_REF`, with `NEE_VUT_REF_QC` | `NEE` | µmol m-2 s-1 |
| target | `H_F_MDS`, with `H_F_MDS_QC` | `H` | W m-2 |
| target | `LE_F_MDS`, with `LE_F_MDS_QC` | `LE` | W m-2 |
| RFR3, RFR10 | `SW_IN_F` | `shortwave` | W m-2 |
| RFR3, RFR10 | `VPD_F_MDS` | `vpd` | hPa |
| RFR3, RFR10 | `TA_F_MDS` | `air_temperature` | degC |
| RFR10 | `NETRAD` | `net_radiation` | W m-2 |
| RFR10 | `WS` | `wind_speed` | m s-1 |
| RFR10 | `WD` | `wind_direction` | degrees |
| RFR10 | `G_F_MDS` | `soil_heat_flux` | W m-2 |
| RFR10 | `TS_F_MDS` | `soil_temperature` | degC |
| RFR10 | `RH` | `relative_humidity` | % |
| RFR10 | `SWC_F_MDS` | `soil_water_content` | % |

`ColumnMap.fluxnet2015()` maps all ten drivers; `ColumnMap.fluxnet2015("RFR3")`
maps only the three RFR3 drivers. Map all ten even for an RFR3 run if you want
the energy-balance check, which needs `NETRAD` and `G_F_MDS` whatever the mode.

**Soil variables may be numbered.** FLUXNET files commonly number soil
measurements by sensor, for example `TS_F_MDS_1` and `SWC_F_MDS_1`, where the
paper names `TS_F_MDS` and `SWC_F_MDS` without saying which sensor. Check your
file's header, choose explicitly, and the choice goes into the run manifest with
the rest of the mapping:

```python
columns = ColumnMap.fluxnet2015(
    overrides={"soil_temperature": "TS_F_MDS_1", "soil_water_content": "SWC_F_MDS_1"}
)
```

This is not one of the twelve recorded ambiguities; it is a property of the
files rather than of the method.

## What the QC flags mean

In the half-hourly files, a `_QC` column beside a gap-filled variable says where
each value came from. The FLUXNET2015 documentation defines `0` as measured, and
`1`, `2` and `3` as gap-filled with good, medium and poor quality. Check the
documentation of the release you are using.

The package's default `observed_qc_values=(0,)` therefore treats only measured
values as measurements. Gap-filled values are neither trained on, nor used in the
daily statistics, nor scored against, exactly as in the paper. A missing flag
counts as not measured. The drivers' own QC columns are not read: the paper used
FLUXNET's filled meteorology, and so does the package.

FLUXNET's target columns arrive already gap-filled. For operational filling that
means there are few genuine holes to fill; to replace FLUXNET's gap-filled values
with RFR predictions, fill with `refill_pre_filled=True`. Those values are then
labelled with the RFR arm in `<target>_fill_method`, and the originals are kept
in `<target>_original`.

## Preparing a file

Four steps, all of them ordinary pandas:

```python
import numpy as np
import pandas as pd

from rfrgapfill import ColumnMap, FeatureConfig, RFRConfig, RFRGapFiller, validate_rfr

# round_trip reads each number exactly as written; pandas' default parser can be
# one unit in the last place off, and the command line reads files this way.
raw = pd.read_csv(
    "FLX_XX-Xxx_FLUXNET2015_FULLSET_HH_2005-2014_1-4.csv", float_precision="round_trip"
)

# 1. FLUXNET writes missing values as -9999. The package would read that as a
#    number, train on it and put it in the daily statistics.
raw = raw.replace(-9999, np.nan)

# 2. TIMESTAMP_START is an integer, YYYYMMDDHHMM. The package refuses numeric
#    timestamps, which pandas would read as nanoseconds since 1970.
raw.index = pd.to_datetime(raw["TIMESTAMP_START"].astype(str), format="%Y%m%d%H%M")

# 3. Rename the targets to the names the reporting layer knows. Default units,
#    the energy-balance check and the published benchmarks are keyed by NEE, H, LE.
raw = raw.rename(
    columns={
        "NEE_VUT_REF": "NEE", "NEE_VUT_REF_QC": "NEE_QC",
        "H_F_MDS": "H", "H_F_MDS_QC": "H_QC",
        "LE_F_MDS": "LE", "LE_F_MDS_QC": "LE_QC",
    }
)

# 4. Map the drivers and check every mapped column is present.
columns = ColumnMap.fluxnet2015()
missing = columns.missing_columns(raw.columns)
if missing:
    raise SystemExit(f"this file lacks {missing}; map them with overrides=")
```

Renaming the targets is optional. With FLUXNET names, pass
`energy_balance_targets=("H_F_MDS", "LE_F_MDS")` to `validate_rfr` and
`units={"NEE_VUT_REF": "umol m-2 s-1", ...}` to `gap_length_table`.

The timestamp convention is also a choice. Indexing by `TIMESTAMP_START` puts
each half hour on the day it starts in, which decides which day's statistics the
record joins. The paper does not say which it used; start is the recommendation
here, and whichever you use, use it throughout.

## Running the paper's experiment on it

```python
config = RFRConfig(
    mode="RFR10",
    frequency="30min",          # "1h" for FULLSET_HR files
    latitude=45.0,              # the site's latitude; it decides the hemisphere
    site_id="XX-Xxx",
    column_map=columns,
    features=FeatureConfig(daily_statistic_strategy="rolling_available"),  # A4
)
qc = {"NEE": "NEE_QC", "H": "H_QC", "LE": "LE_QC"}

report = validate_rfr(raw, config=config, targets=["NEE", "H", "LE"], qc_columns=qc)
report.to_frame()
report.manifest("LE").save("XX-Xxx_LE_validation.json")
```

and to fill:

```python
filler = RFRGapFiller(config).fit(raw, target="LE", qc_column="LE_QC")
result = filler.fill(raw, refill_pre_filled=True)
```

The input frame is never modified, and the validation guide explains the output
([`validation.md`](validation.md)).

## From the command line

The same run needs no Python at all. `--na-value` and `--timestamp-format` do
steps 1 and 2 above; the targets keep their FLUXNET names, so the energy-balance
check is told which columns are H and LE. `XX-Xxx.json` holds the configuration
above, as `config.to_dict()` would write it:

```bash
rfr-gapfill validate FLX_XX-Xxx_FLUXNET2015_FULLSET_HH_2005-2014_1-4.csv \
  --config XX-Xxx.json \
  --target NEE_VUT_REF --target H_F_MDS --target LE_F_MDS \
  --qc-column NEE_VUT_REF=NEE_VUT_REF_QC \
  --qc-column H_F_MDS=H_F_MDS_QC \
  --qc-column LE_F_MDS=LE_F_MDS_QC \
  --energy-balance-targets H_F_MDS LE_F_MDS \
  --timestamp TIMESTAMP_START --timestamp-format %Y%m%d%H%M --na-value -9999 \
  --output XX-Xxx_validation/

rfr-gapfill fill FLX_XX-Xxx_FLUXNET2015_FULLSET_HH_2005-2014_1-4.csv \
  --config XX-Xxx.json --target LE_F_MDS --qc-column LE_F_MDS_QC --refill-pre-filled \
  --timestamp TIMESTAMP_START --timestamp-format %Y%m%d%H%M --na-value -9999 \
  --output XX-Xxx_LE_filled.csv
```

The filled file carries ISO timestamps rather than `YYYYMMDDHHMM` integers. The
README's [command-line section](../README.md#command-line) lists what each
subcommand writes.

## Matching the paper's site set

The paper's 194 sites, their IGBP and Köppen classes, instrument systems and
membership of the 94-site complete-analysis subset are in Supplementary Table S2.
With a local copy of the supplement, `read_table_s2(path)` returns them in the
package's `site_metadata()` form, ready for `site_report()`. The supplement is
publisher material and is not included in this repository.

A comparison against the published medians is meaningful only over matching
sites, inputs and preprocessing, and the medians are never thresholds for one
site ([`supplement_benchmarks.md`](supplement_benchmarks.md)).
