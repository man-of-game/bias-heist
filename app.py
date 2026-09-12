"""
Bias Heist - US Visa Approval Predictor (Black Box)
Exposes ONLY inputs and outputs. No internals, no feature importance,
no model type disclosure.
"""

import streamlit as st
import numpy as np
import joblib

st.set_page_config(page_title="Visa Approval Predictor", page_icon="🛂")

@st.cache_resource
def load_model():
    return joblib.load("visa_model.joblib")

bundle = load_model()
model = bundle["model"]
encoder = bundle["encoder"]

st.title("🛂 US Visa Approval Predictor")
st.caption("Enter applicant details and submit to receive a decision.")

with st.form("visa_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age", 18, 80, 30)
        annual_income = st.number_input(
            "Annual Income (INR)", min_value=80000, max_value=15000000,
            value=600000, step=10000
        )
        bank_balance = st.number_input(
            "Bank Balance (INR)", min_value=20000, max_value=20000000,
            value=300000, step=10000
        )
        education_level = st.selectbox(
            "Education Level",
            ["high_school", "bachelors", "masters", "phd"],
            index=1,
        )

    with col2:
        purpose_of_visit = st.selectbox(
            "Purpose of Visit",
            ["travel", "business", "education", "medical"],
        )
        prior_approvals = st.slider("Prior Approvals", 0, 10, 0)
        english_level = st.selectbox(
            "English Communication", ["basic", "good", "fluent"], index=1
        )
        legal_charges = st.selectbox("Any Legal Charges?", ["no", "yes"])

    submitted = st.form_submit_button("Submit Application")

if submitted:
    row_cat = encoder.transform(
        [[education_level, purpose_of_visit, english_level, legal_charges]]
    )
    row = np.hstack([[[age, annual_income, bank_balance, prior_approvals]], row_cat])

    proba = model.predict_proba(row)[0, 1]
    decision = "APPROVED ✅" if proba >= 0.5 else "REJECTED ❌"

    st.divider()
    st.subheader(f"Decision: {decision}")
    st.metric("Confidence", f"{proba*100:.1f}%")
