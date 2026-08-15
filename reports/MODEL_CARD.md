# Model card: elevated-HbA1c screening

## Intended use

The model prioritizes adults aged over 18 for confirmatory HbA1c testing when universal laboratory testing is difficult. It is a screening-support prototype, not a diagnosis, prognosis, or treatment recommendation.

## Selected model

- Model: **Random Forest**
- Validation-selected decision threshold: **0.365**
- Target: measured HbA1c >= 5.7%

## Untouched test-set results

| model_label            |   roc_auc |   precision |   recall |    f1 |   specificity |   threshold |
|:-----------------------|----------:|------------:|---------:|------:|--------------:|------------:|
| Random Forest          |     0.782 |       0.534 |    0.899 | 0.670 |         0.496 |       0.365 |
| Heterogeneous Stacking |     0.781 |       0.534 |    0.899 | 0.670 |         0.494 |       0.245 |
| XGBoost                |     0.779 |       0.517 |    0.904 | 0.658 |         0.455 |       0.245 |
| Logistic Regression    |     0.771 |       0.510 |    0.933 | 0.660 |         0.424 |       0.220 |

The heterogeneous stacking model did outperform the Logistic Regression baseline in test ROC-AUC. This conclusion is descriptive for the held-out sample and does not establish superiority in another population.

## Key limitations

- Cross-sectional NHANES data support detection of current elevated HbA1c, not future diabetes prediction.
- Activity and smoking measures are self-reported.
- The model has not received external or prospective clinical validation.
- HbA1c can be affected by anemia, kidney/liver disease, pregnancy, blood disorders, medicines, blood loss, and transfusion; those factors are not fully represented here.
- Only the approved variables are used. Racial fairness cannot be assessed because race/ethnicity is not included.
- Human review and confirmatory testing are mandatory before any health action.
