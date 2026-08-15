# Raw NHANES inputs

Place the following unmodified NHANES August 2021-August 2023 SAS transport files in this directory:

- `DEMO_L.xpt` — demographics and sample weights
- `BMX_L.xpt` — body measures
- `BPXO_L.xpt` — oscillometric blood pressure
- `PAQ_L.xpt` — physical activity
- `SMQ_L.xpt` — cigarette use
- `GHB_L.xpt` — glycohemoglobin

The files are intentionally excluded from Git. The processing pipeline will validate their schemas, join them one-to-one by `SEQN`, retain respondents with `RIDAGEYR > 18`, and require a nonmissing `LBXGH` target measurement.

Official source: [CDC/NCHS NHANES August 2021-August 2023](https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx?Cycle=2021-2023)

