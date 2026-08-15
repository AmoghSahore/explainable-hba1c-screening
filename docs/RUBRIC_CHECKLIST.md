# Faculty rubric traceability

## General instructions

- [x] End-to-end path from raw XPT files to explainable live prediction
- [x] Complete reusable codebase and faculty-facing notebook
- [x] README with environment and execution steps
- [x] Saved fitted pipeline plus reproducible training script
- [x] Open/public NHANES source cited for all six files
- [x] No personally identifiable or confidential data
- [x] External code, libraries, figures, and assets acknowledged
- [x] Exact three-minute pitch-and-demo script prepared

## Question 1 — Real-World Impact Framing (5 marks)

- [x] Mission Health selected
- [x] Problem and intended beneficiaries defined
- [x] Target defined as current elevated HbA1c (`LBXGH >= 5.7%`)
- [x] Measurable sample-level screening impact generated
- [x] Suitability of ML stated without claiming diagnosis or causality
- [x] Dataset source and one-respondent unit of analysis documented
- [x] Responsible-use limitations stated

## Question 2 — Data Wrangling and Feature Engineering (6 marks)

- [x] One-to-one `SEQN` merge and duplicate assertions
- [x] Strict `RIDAGEYR > 18` filter
- [x] Missing-value and special-code audit
- [x] Invalid-range handling and training-fold quantile clipping
- [x] Data-type conversion and categorical encoding
- [x] Numerical scaling inside the pipeline
- [x] Stratification and class-weight tuning for mild imbalance
- [x] Training-only target-aware EDA
- [x] Mean BP, weekly activity, and smoking features derived and justified
- [x] Reproducible 70/15/15 train/validation/test split

## Question 3 — Ensemble Architecture, Tuning, and Comparison (8 marks)

- [x] Logistic Regression baseline
- [x] Random Forest bagging
- [x] XGBoost boosting
- [x] Heterogeneous stacking ensemble
- [x] Stratified cross-validation and hyperparameter tuning
- [x] Out-of-fold stack predictions prevent meta-learning leakage
- [x] Same untouched test set used once for every model
- [x] ROC-AUC, precision, recall, F1, PR-AUC, specificity, calibration, and confusion matrices
- [x] Explicit stack-versus-baseline conclusion generated

## Question 4 — Explainability and Ethics (4 marks)

- [x] Global model-agnostic SHAP output for the selected pipeline
- [x] Individual SHAP output for a fictional record
- [x] Non-causal domain interpretation guidance
- [x] Fairness audit by sex, age, education, and income
- [x] Privacy, uncertainty, false-positive and false-negative costs
- [x] Human oversight and deployment limitations

## Question 5 — Three-Minute Product Pitch and Live Demo (2 marks)

- [x] 0:00–0:35 problem and beneficiaries
- [x] 0:35–1:15 cleaning, EDA, and feature engineering
- [x] 1:15–2:10 baseline, bagging, boosting, and stacking results
- [x] 2:10–3:00 live fictional prediction, local explanation, ethics, and limitations
- [x] Repository, compact result visual, and live model output included

