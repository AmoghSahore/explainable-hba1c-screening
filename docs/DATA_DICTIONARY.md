# Analytic data dictionary

The unit of analysis is one NHANES August 2021-August 2023 respondent with `RIDAGEYR > 18` and a nonmissing `LBXGH` measurement.

| Analytic field | Source | Construction | Role |
|---|---|---|---|
| `age` | DEMO `RIDAGEYR` | Age at screening; 80 represents 80+ | Numeric predictor |
| `sex` | DEMO `RIAGENDR` | 1 Male, 2 Female | Categorical predictor |
| `education` | DEMO `DMDEDUC2` | Five adult categories; unavailable at age 19 | Categorical predictor |
| `income_ratio` | DEMO `INDFMPIR` | Family income-to-poverty ratio, top-coded at 5 | Numeric predictor |
| `bmi` | BMX `BMXBMI` | kg/m² | Numeric predictor |
| `waist_cm` | BMX `BMXWAIST` | Centimetres | Numeric predictor |
| `mean_systolic_bp` | BPXO `BPXOSY1-3` | Mean of available readings | Numeric predictor |
| `mean_diastolic_bp` | BPXO `BPXODI1-3` | Mean of available readings | Numeric predictor |
| `moderate_activity_min_week` | PAQ `PAD790Q/U`, `PAD800` | Frequency converted to weekly frequency × minutes/session | Numeric predictor |
| `vigorous_activity_min_week` | PAQ `PAD810Q/U`, `PAD820` | Frequency converted to weekly frequency × minutes/session | Numeric predictor |
| `sedentary_min_day` | PAQ `PAD680` | Minutes on a typical day | Numeric predictor |
| `smoking_history` | SMQ `SMQ020` | Ever versus never smoked 100 cigarettes | Categorical predictor |
| `current_smoking` | SMQ `SMQ020`, `SMQ040` | Current if an ever-smoker reports every-day or some-day smoking | Categorical predictor |
| `elevated_hba1c` | GHB `LBXGH` | 1 when HbA1c >= 5.7%, otherwise 0 | Binary target |

NHANES refused/don't-know codes are converted to missing before modelling. `SEQN` is a merge key only and is excluded from predictors. Raw `LBXGH` is used only to construct the target and is then removed to prevent direct target leakage.

