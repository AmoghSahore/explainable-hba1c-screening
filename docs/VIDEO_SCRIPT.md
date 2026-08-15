# Three-minute pitch and live-demo script

This script follows the faculty's required timing exactly and uses the final full-run results.

## 0:00–0:35 — problem and beneficiaries

“Prediabetes and diabetes can remain unnoticed until complications develop. In clinics with limited laboratory capacity, universal HbA1c testing may be difficult. Our Mission Health project screens adults older than 18 for current elevated HbA1c so healthcare workers can prioritize confirmatory testing. The beneficiaries are adults needing early follow-up and clinics allocating limited tests. This is a screening-support prototype, not a diagnosis.”

Show the repository title and problem statement.

## 0:35–1:15 — cleaning, EDA, and feature engineering

“We use six public NHANES August 2021–August 2023 files, joined one-to-one by the respondent sequence number. We retain adults aged 19 or older with measured HbA1c, producing 5,872 respondents and 2,300 elevated cases. We audit duplicates, missingness, invalid survey codes, ranges, outliers, types, and class balance. We derive mean blood pressure, weekly moderate and vigorous activity, and smoking status. All imputation, clipping, encoding, and scaling occur inside training folds to prevent leakage.”

Show the missingness chart, target balance, and one engineered-feature figure.

## 1:15–2:10 — baseline and ensemble results

“We compare the required Logistic Regression baseline with Random Forest bagging, XGBoost boosting, and a heterogeneous stack. Hyperparameters are tuned with stratified cross-validation. The stack's meta-learner uses out-of-fold predictions only. Thresholds are selected on validation data using F2, which emphasizes recall. Every model is then assessed once on the same untouched test set using ROC-AUC, precision, recall, F1, and confusion matrices. Random Forest was selected, with test ROC-AUC 0.782, recall 0.899, precision 0.534, and F1 0.670. The stack achieved 0.781 ROC-AUC versus 0.771 for the baseline, so it did outperform the baseline descriptively.”

Show `model_curves.png`, the compact comparison table, and `confusion_matrices.png`.

## 2:10–3:00 — live prediction, explanation, ethics, and limitations

Enter the prepared fictional adult record in the Streamlit app.

“For this fictional record, the fitted pipeline estimates an elevated-HbA1c probability of 79.1% against the validation-selected threshold of 36.5%. The app recommends prioritizing confirmatory HbA1c testing. The local SHAP chart shows which inputs raised or lowered this prediction; these are model associations, not causes. False negatives could delay follow-up, while false positives create unnecessary testing and anxiety. A healthcare professional must confirm every result. The model uses one cross-sectional US survey cycle, includes self-reported variables, lacks external clinical validation, and cannot establish racial fairness because race was not among the approved variables.”

End on the disclaimer and repository view before 3:00.
