# ML for Social Good: Explainable Screening for Elevated HbA1c

## 1. Real-world impact framing

This Mission Health project addresses limited access to laboratory screening for abnormal glucose regulation. Its intended beneficiaries are adults who may need early follow-up and healthcare workers allocating limited HbA1c tests. The model estimates whether an adult currently has measured HbA1c of at least 5.7% from non-laboratory demographic, anthropometric, blood-pressure, physical-activity, and smoking variables.

The unit of analysis is one examined NHANES respondent older than 18. The product decision is whether to prioritize a confirmatory HbA1c test. Sample-level impact is expressed per 100 screened adults: referrals, elevated cases captured, elevated cases missed, and false referrals. These figures are simulated from held-out data and are not observed clinical impact.

Machine learning is suitable for combining nonlinear relationships and interactions among routinely understandable variables. It cannot diagnose diabetes, establish causation, or forecast future disease from this cross-sectional dataset. Any flag requires a healthcare professional and confirmatory testing.

## 2. Data source and cohort

Six public NHANES August 2021–August 2023 XPT files are used: DEMO_L, BMX_L, BPXO_L, PAQ_L, SMQ_L, and GHB_L. They are joined one-to-one by `SEQN`, which is then removed. Raw `LBXGH` creates the target and is also removed before modelling.

The strict eligibility rule is `RIDAGEYR > 18`. Starting from 7,969 adults, requiring a nonmissing HbA1c measurement produces 5,872 respondents. There are 2,300 positive cases (39.2%) and 3,572 negative cases (60.8%). No duplicate `SEQN` values occur in any source.

## 3. Wrangling and feature engineering

The pipeline audits and handles missing values, invalid codes, duplicates, types, outliers, category encoding, scaling, and class balance. A pandas XPORT edge case that decoded SAS numeric zero as approximately 5.4×10^-79 was detected and normalized before activity construction.

Important missingness includes family income-to-poverty ratio (752 rows), waist circumference (269), blood pressure (186), education (107), BMI (84), moderate activity (43), vigorous activity (36), and sedentary time (42). Education is unavailable for all 105 included nineteen-year-olds because `DMDEDUC2` begins at age 20; they remain eligible and receive fold-safe missing-category handling.

Domain-informed features are:

- Mean systolic and diastolic pressure from all available oscillometric readings.
- Weekly moderate and vigorous leisure-time activity from reported frequency, unit, and minutes per session.
- Smoking history and current smoking derived jointly to respect the SMQ skip pattern.

Deterministic impossible-range rules are applied before splitting. Learned median imputation, missingness indicators, 1st/99th percentile clipping, standardization, and one-hot encoding are contained inside scikit-learn pipelines and fitted separately in every training fold.

## 4. Reproducible evaluation design

The cohort is stratified into 70% training (4,110), 15% validation (881), and 15% untouched test (881) with random seed 42. Training contains 1,610 positive cases; validation and test each contain 345. Target-aware EDA uses training data only.

Hyperparameter search uses shuffled five-fold stratified cross-validation. ROC-AUC determines refitting during search. The decision threshold for each model is selected on validation data by maximizing F2, emphasizing recall. After choices are frozen, each model is refitted on training plus validation and evaluated once on the same test set.

## 5. Required models and tuning

- Baseline: Logistic Regression
- Bagging: Random Forest
- Boosting: XGBoost
- Heterogeneous ensemble: Logistic Regression, Random Forest, and XGBoost base learners with a Logistic Regression meta-learner

The stacking meta-learner receives five-fold out-of-fold base-model probabilities, never in-sample predictions. Its regularization parameter is selected on validation data.

## 6. Untouched test-set results

| Model | ROC-AUC | PR-AUC | Precision | Recall | F1 | Specificity | Threshold |
|---|---:|---:|---:|---:|---:|---:|---:|
| Random Forest | 0.782 | 0.663 | 0.534 | 0.899 | 0.670 | 0.496 | 0.365 |
| Heterogeneous Stacking | 0.781 | 0.666 | 0.534 | 0.899 | 0.670 | 0.494 | 0.245 |
| XGBoost | 0.779 | 0.663 | 0.517 | 0.904 | 0.658 | 0.455 | 0.245 |
| Logistic Regression | 0.771 | 0.651 | 0.510 | 0.933 | 0.660 | 0.424 | 0.220 |

Random Forest was selected by the predeclared validation ROC-AUC rule. Its test confusion matrix contains 310 true positives, 270 false positives, 35 false negatives, and 266 true negatives. The 95% bootstrap intervals are 0.751–0.814 for ROC-AUC, 0.501–0.572 for precision, 0.866–0.932 for recall, and 0.639–0.703 for F1.

The stack's test ROC-AUC is 0.781 versus 0.771 for Logistic Regression, a descriptive improvement of 0.010. The intervals overlap, so this is not evidence of a statistically conclusive improvement in another sample.

At the selected Random Forest threshold, the test-set simulation per 100 adults produces approximately 65.8 referrals, 35.2 elevated cases captured, 4.0 elevated cases missed, and 30.6 false referrals. This is the expected trade-off of a recall-sensitive screening threshold.

## 7. Explainability

Model-agnostic permutation SHAP explains the complete Random Forest pipeline. The largest global mean absolute contributions are age, waist circumference, BMI, mean systolic pressure, vigorous activity, education, income ratio, and moderate activity.

For the fictional demo record, the model predicts 79.1% against a 36.5% referral threshold. Waist circumference, age, BMI, systolic pressure, and lower income ratio are the strongest upward contributions. Moderate activity and sedentary time lower this particular prediction slightly. These explanations describe the fitted model's associations and must not be interpreted as causal effects.

## 8. Fairness, privacy, and responsible use

The test audit compares recall, false-negative rate, false-positive rate, precision, and ROC-AUC across sex, age, education, and income groups. Recall is similar for females (0.896) and males (0.902). A serious limitation appears by age: recall is 0.379 for ages 19–39, 0.903 for ages 40–59, and 0.964 for ages 60+. The younger estimate is based on only 29 positive cases but is too weak for safe use. The 60+ false-positive rate is 0.910 at the recall-sensitive threshold.

Race/ethnicity is not an approved feature, so racial fairness cannot be assessed. The dataset is public and de-identified; `SEQN` is used only during joining. The live demo uses a fictional record. Human oversight, confirmatory HbA1c testing, and local external validation are mandatory.

## 9. Conclusion

The project fulfills the complete path from raw public data through defensible preprocessing, baseline and ensemble comparison, untouched test evaluation, global and local explanation, ethics analysis, and live prediction. Performance is useful for an educational screening demonstration but not sufficient for clinical deployment, particularly given subgroup variation, modest precision, and the absence of external validation.

