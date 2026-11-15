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
    "A Machine Learning application to identify "
    "potentially fraudulent healthcare insurance claims."
)


# -----------------------------------
# 2. Load Model and Artifacts
# -----------------------------------

BASE_DIR = Path(__file__).parent

MODEL_PATH = BASE_DIR / "healthcare_fraud_detection_model.pkl"
DATA_PATH = BASE_DIR / "healthcare_fraud_detection.csv.xlsx"


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

except Exception:
    st.error(
        "Unable to load the saved model. "
        "Please check the model file."
    )
    st.stop()


# -----------------------------------
# 3. Create Application Tabs
# -----------------------------------

tab1, tab2, tab3, tab4 = st.tabs([
    "Predict",
    "Data Insights",
    "Model Performance",
    "About"
])


# -----------------------------------
# 4. Prediction Tab
# -----------------------------------

with tab1:

    st.header("Healthcare Claim Prediction")

    st.write(
        "Enter the healthcare claim details to check "
        "whether the claim is potentially fraudulent."
    )

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
                "Claims Per Provider Monthly",
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

        submit = st.form_submit_button("Predict Fraud")

    # -----------------------------------
    # 5. Preprocessing and Prediction
    # -----------------------------------

    if submit:

        try:

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

            # Handle missing values

            for column, value in imputation_values.items():

                if column in input_df.columns:
                    input_df[column] = input_df[column].fillna(value)

            # One-Hot Encoding

            encoded_data = encoder.transform(
                input_df[categorical_cols]
            )

            if hasattr(encoded_data, "toarray"):
                encoded_data = encoded_data.toarray()

            encoded_df = pd.DataFrame(
                encoded_data,
                columns=encoder.get_feature_names_out()
            )

            # Select numerical features

            numerical_data = input_df[numerical_cols].copy()

            # Combine features

            final_input = pd.concat(
                [numerical_data, encoded_df],
                axis=1
            )

            # Match training feature order

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

            # Display result

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

            # -----------------------------------
            # 6. Why this prediction?
            # -----------------------------------

            st.subheader("Why this prediction?")

            st.write(
                "The chart below shows the features that had the "
                "greatest overall influence on model performance "
                "during permutation importance analysis."
            )

            importance_df = pd.DataFrame({
                "Feature": [
                    "Days_Between_Service_and_Claim",
                    "Claim_Amount",
                    "Number_of_Claims_Per_Provider_Monthly",
                    "Length_of_Stay",
                    "Diagnosis_Code_K21.9",
                    "Procedure_Code_99214",
                    "Procedure_Code_87086",
                    "Insurance_Type_Medicare",
                    "Diagnosis_Code_J18.9",
                    "Diagnosis_Code_F41.9"
                ],
                "Importance": [
                    0.496235,
                    0.122874,
                    0.040171,
                    0.006402,
                    0.003702,
                    0.003023,
                    0.002695,
                    0.002365,
                    0.001690,
                    0.001685
                ]
            })

            st.bar_chart(
                importance_df.set_index("Feature"),
                horizontal=True
            )

            st.caption(
                "Higher importance means that shuffling a feature "
                "caused a greater decrease in the model's F1 score. "
                "These are global model-level results, not explanations "
                "of an individual claim."
            )

        except Exception:
            st.error(
                "Unable to process this claim. "
                "Please check the inputs and try again."
            )


# -----------------------------------
# 7. Data Insights Tab
# -----------------------------------

with tab2:

    st.header("Dataset Insights")

    try:

        df = pd.read_excel(
            DATA_PATH,
            sheet_name="Cleaned Data"
        )

        st.write(
            "Overview of the Healthcare Claims Dataset"
        )

        col1, col2 = st.columns(2)

        col1.metric(
            "Total Claims",
            f"{len(df):,}"
        )

        col2.metric(
            "Total Features",
            len(df.columns)
        )

        st.subheader("Fraud vs Legitimate Claims")

        fraud_counts = df["Is_Fraud"].value_counts().sort_index()

        chart_data = pd.DataFrame({
            "Claim Type": ["Legitimate", "Fraud"],
            "Number of Claims": [
                int(fraud_counts.get(0, 0)),
                int(fraud_counts.get(1, 0))
            ]
        })

        st.bar_chart(
            chart_data.set_index("Claim Type")
        )

        st.subheader("Target Distribution")

        st.dataframe(
            chart_data,
            use_container_width=True
        )

    except Exception:
        st.error(
            "Unable to load dataset insights. "
            "Please check the Excel file and sheet name."
        )


# -----------------------------------
# 8. Model Performance Tab
# -----------------------------------

with tab3:

    st.header("Model Performance")

    st.write(
        "The final selected model is Gradient Boosting Classifier."
    )

    st.subheader("Evaluation Metrics")

    col1, col2, col3 = st.columns(3)

    col1.metric("Accuracy", "93.50%")
    col2.metric("Precision", "57.83%")
    col3.metric("Recall", "80.12%")

    col4, col5 = st.columns(2)

    col4.metric("F1 Score", "67.17%")
    col5.metric("ROC-AUC", "97.05%")

    st.subheader("Confusion Matrix")

    confusion_data = pd.DataFrame(
        [[1737, 97], [33, 133]],
        index=["Actual Legitimate", "Actual Fraud"],
        columns=["Predicted Legitimate", "Predicted Fraud"]
    )

    st.dataframe(
        confusion_data,
        use_container_width=True
    )

    st.caption(
        "Metrics are from the final threshold-adjusted test evaluation."
    )


# -----------------------------------
# 9. About Tab
# -----------------------------------

with tab4:

    st.header("About This Project")

    st.subheader("Project Objective")

    st.write(
        "This project uses Machine Learning to identify "
        "potentially fraudulent healthcare insurance claims."
    )

    st.subheader("Dataset")

    st.write(
        "The dataset contains 10,000 healthcare claim records "
        "with numerical and categorical features."
    )

    st.subheader("Machine Learning Model")

    st.write(
        "Gradient Boosting Classifier was selected as the final model "
        "after model evaluation and hyperparameter tuning."
    )

    st.subheader("How to Use")

    st.markdown("""
    1. Open the Predict tab.
    2. Enter the healthcare claim details.
    3. Click Predict Fraud.
    4. Review the prediction and estimated probability.
    """)

    st.subheader("Disclaimer")

    st.write(
        "This application is an educational machine-learning project. "
        "Its predictions are not proof of fraud and should not be used "
        "as the sole basis for real insurance decisions."
    )