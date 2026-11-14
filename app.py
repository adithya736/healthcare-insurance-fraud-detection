
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path


# -----------------------------------
# 1. Page Configuration
# -----------------------------------

st.set_page_config(
    page_title="Healthcare Fraud Detection",
    page_icon="🏥",
    layout="wide"
)

st.title("Healthcare Insurance Fraud Detection")

st.write(
    "Enter healthcare claim details to predict "
    "whether a claim is potentially fraudulent."
)


# -----------------------------------
# 2. Load Saved Model and Artifacts
# -----------------------------------

MODEL_PATH = Path(__file__).parent / "healthcare_fraud_detection_model.pkl"


@st.cache_resource
def load_artifacts():
    return joblib.load(MODEL_PATH)


try:
    artifacts = load_artifacts()

    model = artifacts["model"]
    threshold = artifacts["threshold"]
    encoder = artifacts["encoder"]
    scaler = artifacts["scaler"]

    numerical_cols = artifacts["numerical_cols"]
    categorical_cols = artifacts["categorical_cols"]
    feature_columns = artifacts["feature_columns"]

    imputation_values = artifacts["imputation_values"]

except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()


# -----------------------------------
# 3. User Input Form
# -----------------------------------

st.header("Claim Details")

with st.form("claim_form"):

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Numerical Features")

        patient_age = st.number_input(
            "Patient Age",
            min_value=0,
            max_value=120,
            value=35
        )

        claim_amount = st.number_input(
            "Claim Amount",
            min_value=0.0,
            value=500.0,
            step=50.0
        )

        days_between = st.number_input(
            "Days Between Service and Claim",
            min_value=0,
            value=5
        )

        monthly_claims = st.number_input(
            "Number of Claims Per Provider Monthly",
            min_value=0,
            value=10
        )

        length_of_stay = st.number_input(
            "Length of Stay",
            min_value=0,
            value=2
        )

        chronic_condition = st.selectbox(
            "Chronic Condition",
            options=[0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes"
        )

        prior_visits = st.number_input(
            "Prior Visits in Last 12 Months",
            min_value=0.0,
            value=3.0,
            step=1.0
        )

    with col2:

        st.subheader("Categorical Features")

        def get_options(column):
            index = categorical_cols.index(column)
            return encoder.categories_[index].tolist()

        patient_gender = st.selectbox(
            "Patient Gender",
            get_options("Patient_Gender")
        )

        diagnosis_code = st.selectbox(
            "Diagnosis Code",
            get_options("Diagnosis_Code")
        )

        procedure_code = st.selectbox(
            "Procedure Code",
            get_options("Procedure_Code")
        )

        insurance_type = st.selectbox(
            "Insurance Type",
            get_options("Insurance_Type")
        )

        provider_specialty = st.selectbox(
            "Provider Specialty",
            get_options("Provider_Specialty")
        )

        patient_state = st.selectbox(
            "Patient State",
            get_options("Patient_State")
        )

        visit_type = st.selectbox(
            "Visit Type",
            get_options("Visit_Type")
        )

    submit = st.form_submit_button(
        "Predict Fraud"
    )


# -----------------------------------
# 4. Preprocessing and Prediction
# -----------------------------------

if submit:

    input_data = {

        "Patient_Age": patient_age,
        "Patient_Gender": patient_gender,
        "Diagnosis_Code": diagnosis_code,
        "Procedure_Code": procedure_code,
        "Claim_Amount": claim_amount,
        "Insurance_Type": insurance_type,
        "Days_Between_Service_and_Claim": days_between,
        "Number_of_Claims_Per_Provider_Monthly": monthly_claims,
        "Provider_Specialty": provider_specialty,
        "Patient_State": patient_state,
        "Length_of_Stay": length_of_stay,
        "Visit_Type": visit_type,
        "Chronic_Condition_Flag": chronic_condition,
        "Prior_Visits_12m": prior_visits
    }

    input_df = pd.DataFrame([input_data])

    # Handle missing values using saved values
    for column, value in imputation_values.items():
        if column in input_df.columns:
            input_df[column] = input_df[column].fillna(value)

    # One-Hot Encoding
    encoded_data = encoder.transform(
        input_df[categorical_cols]
    )

    encoded_df = pd.DataFrame(
        encoded_data.toarray(),
        columns=encoder.get_feature_names_out()
    )

    # Select numerical features
    numerical_data = input_df[numerical_cols].copy()

    # Combine numerical and encoded features
    final_input = pd.concat(
        [numerical_data, encoded_df],
        axis=1
    )

    # Match the exact training feature order
    missing_features = [
        col for col in feature_columns
        if col not in final_input.columns
    ]

    if missing_features:
        st.error(
            f"Missing model features: {missing_features}"
        )
        st.stop()

    final_input = final_input.reindex(
        columns=feature_columns
    )

    # Standard Scaling
    final_input[numerical_cols] = scaler.transform(
        final_input[numerical_cols]
    )

    # Predict probability
    probabilities = model.predict_proba(final_input)

    fraud_index = list(model.classes_).index(1)

    fraud_probability = probabilities[0][fraud_index]

    # Apply saved threshold
    prediction = int(
        fraud_probability >= threshold
    )


    # -----------------------------------
    # 5. Display Prediction Result
    # -----------------------------------

    st.subheader("Prediction Result")

    st.metric(
        "Estimated Fraud Probability",
        f"{fraud_probability:.2%}"
    )

    st.write(
        f"Decision Threshold: {threshold:.4f}"
    )

    if prediction == 1:

        st.warning(
            "Potentially Fraudulent Claim"
        )

    else:

        st.success(
            "Potentially Legitimate Claim"
        )

    st.caption(
        "This is a machine-learning screening result, "
        "not a final determination of fraud. "
        "Claims should be reviewed by an authorized person."
    )
