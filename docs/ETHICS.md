# Ethics and responsible-use statement

## Intended use

This project is an educational screening prototype that estimates whether an adult's measured HbA1c is likely to be at least 5.7%. It may illustrate how limited screening resources could be prioritized. It must not diagnose diabetes, forecast an individual's future disease, prescribe treatment, or replace laboratory testing and clinical review.

## Privacy

The source is the public, de-identified NHANES August 2021-August 2023 release. `SEQN` is used only for one-to-one joins and is removed before modelling. The live demo uses a fictional record and must never display a real patient's data. No names, addresses, contact details, or confidential clinical records are used.

## Bias and fairness

Performance will be evaluated by sex, age group, education, and family income-to-poverty ratio using recall, false-negative rate, false-positive rate, precision, and ROC-AUC where subgroup size permits. Small differences must not be overinterpreted without confidence intervals and external validation.

Race/ethnicity is not among the approved model variables. Consequently, this project cannot claim racial fairness or equal performance across racial and ethnic populations. Removing a protected characteristic also does not remove proxy discrimination through socioeconomic or clinical variables.

## Error costs

- **False negative:** an adult with elevated HbA1c is not prioritized, potentially delaying confirmatory testing or prevention support. This is the more serious screening error, so threshold selection emphasizes recall through the F2 score.
- **False positive:** an adult without elevated HbA1c is prioritized, creating avoidable testing cost, inconvenience, or anxiety.

The decision threshold is selected using validation data only. It can be changed only after the trade-off is reviewed for the intended setting; it must never be tuned on the test set.

## Uncertainty

Outputs are probabilities, not certainties. Test metrics include bootstrap confidence intervals and calibration analysis. HbA1c itself can be influenced by anemia, kidney or liver disease, pregnancy, blood disorders, medicines, blood loss, and transfusion, which are not comprehensively represented in the approved feature set.

## Human oversight

A qualified healthcare worker must review every flag, consider history and symptoms, and confirm status with appropriate laboratory testing. A low score must not be used to deny a test when clinical judgment indicates one.

## Deployment limitations

- Cross-sectional predictors and HbA1c were measured during the same survey cycle; the model detects current elevation rather than future risk.
- NHANES represents the US civilian noninstitutionalized population and does not guarantee transfer to another country, clinic, or time period.
- Activity and smoking variables are self-reported.
- Age is top-coded at 80.
- The release records sex/gender in binary categories and does not represent all identities.
- Known diabetes, treatment, diet, family history, medication, and several biological factors are not included.
- No external, prospective, or clinical workflow validation has been performed.

## Observed subgroup warning

On the untouched test set, the selected Random Forest had similar recall for females (0.896) and males (0.902), but recall was only 0.379 for ages 19–39 compared with 0.903 for ages 40–59 and 0.964 for ages 60+. The younger group had only 29 positive cases, so the estimate is uncertain, but the gap is operationally important. The model must not be used to deny testing to younger adults, and external validation plus threshold review by age group would be required before any deployment.

The 60+ group also had a false-positive rate of 0.910 at the recall-sensitive threshold. This illustrates the deliberate cost of prioritizing recall and means the prototype would refer many older adults who do not have elevated HbA1c. Capacity and harm trade-offs must be reviewed locally.

