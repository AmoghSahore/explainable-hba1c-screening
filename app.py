"""Streamlit demo for the fitted elevated-HbA1c screening pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src.config import ARTIFACT_DIR, DISPLAY_NAMES, FEATURE_COLUMNS
from src.explain import build_explainer, encode_explanation_frame


MODEL_PATH = ARTIFACT_DIR / "model_pipeline.joblib"
METADATA_PATH = ARTIFACT_DIR / "model_metadata.json"
BACKGROUND_PATH = ARTIFACT_DIR / "shap_background.csv"


st.set_page_config(
    page_title="Elevated HbA1c Screening",
    page_icon="🩺",
    layout="wide",
)


@st.cache_resource
def load_assets():
    if not MODEL_PATH.is_file() or not METADATA_PATH.is_file():
        raise FileNotFoundError("Run `python -m src.train` before launching the demo.")
    model = joblib.load(MODEL_PATH)
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    background = pd.read_csv(BACKGROUND_PATH) if BACKGROUND_PATH.is_file() else None
    return model, metadata, background


@st.cache_resource
def cached_explainer(_model, background: pd.DataFrame):
    return build_explainer(_model, background)


st.title("Explainable screening for elevated HbA1c")
st.caption(
    "Mission Health · NHANES August 2021–August 2023 · Adults older than 18"
)
st.warning(
    "Educational screening prototype only. This output is not a diabetes diagnosis, "
    "does not replace an HbA1c test, and must not be used to make treatment decisions."
)

try:
    model, metadata, background = load_assets()
except FileNotFoundError as error:
    st.error(str(error))
    st.stop()

with st.sidebar:
    st.header("Fictional adult record")
    age = st.slider("Age", min_value=19, max_value=80, value=58)
    sex = st.selectbox("Sex recorded by NHANES", ["Female", "Male"], index=1)
    education = st.selectbox(
        "Education",
        [
            "Less than 9th grade",
            "9-11th grade/no diploma",
            "High school/GED",
            "Some college/AA",
            "College graduate or above",
            "Not collected/unknown",
        ],
        index=2,
    )
    income_ratio = st.slider("Family income-to-poverty ratio", 0.0, 5.0, 1.8, 0.1)
    bmi = st.slider("BMI (kg/m²)", 10.0, 75.0, 34.0, 0.1)
    waist_cm = st.slider("Waist circumference (cm)", 50.0, 190.0, 112.0, 0.5)
    systolic = st.slider("Mean systolic BP (mmHg)", 70.0, 230.0, 145.0, 1.0)
    diastolic = st.slider("Mean diastolic BP (mmHg)", 35.0, 140.0, 88.0, 1.0)
    moderate = st.slider("Moderate activity (min/week)", 0.0, 1_500.0, 30.0, 10.0)
    vigorous = st.slider("Vigorous activity (min/week)", 0.0, 1_000.0, 0.0, 10.0)
    sedentary = st.slider("Sedentary time (min/day)", 0.0, 1_380.0, 600.0, 10.0)
    smoking_history = st.selectbox("Smoking history", ["Never", "Ever"], index=1)
    if smoking_history == "Never":
        current_smoking = "Not current"
        st.selectbox("Current smoking", ["Not current"], disabled=True)
    else:
        current_smoking = st.selectbox("Current smoking", ["Not current", "Current"], index=1)

record = pd.DataFrame(
    [
        {
            "age": float(age),
            "income_ratio": float(income_ratio),
            "bmi": float(bmi),
            "waist_cm": float(waist_cm),
            "mean_systolic_bp": float(systolic),
            "mean_diastolic_bp": float(diastolic),
            "moderate_activity_min_week": float(moderate),
            "vigorous_activity_min_week": float(vigorous),
            "sedentary_min_day": float(sedentary),
            "sex": sex,
            "education": education,
            "smoking_history": smoking_history,
            "current_smoking": current_smoking,
        }
    ],
    columns=FEATURE_COLUMNS,
)

probability = float(model.predict_proba(record)[:, 1][0])
threshold = float(metadata["decision_threshold"])
prioritize = probability >= threshold

left, middle, right = st.columns(3)
left.metric("Estimated probability", f"{probability:.1%}")
middle.metric("Screening threshold", f"{threshold:.1%}")
right.metric("Model", metadata["selected_model_label"])

if prioritize:
    st.error("Screening result: prioritize confirmatory HbA1c testing.")
else:
    st.success("Screening result: below the model's referral threshold.")
st.progress(min(max(probability, 0.0), 1.0))

st.subheader("Individual explanation")
if background is None:
    st.info("SHAP background data were not found. Re-run training to enable explanations.")
else:
    with st.spinner("Calculating a model-agnostic SHAP explanation..."):
        explainer = cached_explainer(model, background)
        explanation = explainer(
            encode_explanation_frame(record),
            max_evals=2 * len(FEATURE_COLUMNS) + 1,
        )
        contributions = pd.DataFrame(
            {
                "Feature": [DISPLAY_NAMES[name] for name in FEATURE_COLUMNS],
                "Contribution": np.asarray(explanation.values)[0],
                "Entered value": [str(record.iloc[0][name]) for name in FEATURE_COLUMNS],
            }
        ).sort_values("Contribution", key=np.abs, ascending=False)
        st.bar_chart(contributions.set_index("Feature")["Contribution"], horizontal=True)
        st.dataframe(contributions, hide_index=True, width="stretch")
        st.caption(
            "Positive SHAP contributions raise this prediction; negative contributions lower it. "
            "These are model associations, not causal effects."
        )

with st.expander("Responsible-use limitations"):
    st.markdown(
        """
- Confirm any flag with a clinical HbA1c test and qualified healthcare professional.
- A low model score cannot rule out prediabetes or diabetes.
- The model was developed on one US NHANES cycle and has no external clinical validation.
- Activity and smoking inputs are self-reported and may contain recall error.
- Available fairness checks cover sex, age, education, and income; racial fairness was not assessed.
"""
    )
