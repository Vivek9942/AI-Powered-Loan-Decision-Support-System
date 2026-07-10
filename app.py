import streamlit as st
import pickle
import pandas as pd
from datetime import datetime

with open("models/best_model.pkl", "rb") as file:
    model = pickle.load(file)

st.set_page_config(
    page_title="AI Loan Decision Support System",
    page_icon="🏦",
    layout="wide"
)

st.title("🏦 AI-Powered Loan Decision Support System")

st.markdown("""
### Machine Learning Workflow

Model Information → Dataset Summary → Model Performance → Feature Importance → Loan Prediction → Download Report
""")

st.markdown("---")

with st.expander("1️⃣ Model Information", expanded=True):

    st.write("Information about the trained model.")

    if st.button("View Model Details"):

        st.success("Model Loaded Successfully")

        st.info("""
**Model Name:** AdaBoost Classifier

**Algorithm:** Ensemble Learning

**Features Used:** 9

**Target Variable:** Loan Approved

**Status:** Ready for Prediction
""")
        # ===========================
# 2️⃣ Dataset Summary
# ===========================

with st.expander("2️⃣ Dataset Summary"):

    st.write("View information about the processed dataset.")

    if st.button("View Dataset Summary"):

        df = pd.read_csv("loanapproval_processed.csv")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Rows", df.shape[0])
            st.metric("Columns", df.shape[1])

        with col2:
            st.metric("Missing Values", df.isnull().sum().sum())
            st.metric("Duplicate Rows", df.duplicated().sum())

        st.markdown("---")

        st.subheader("Dataset Preview")

        st.dataframe(df.head())

        # ===========================
# 3️⃣ Model Performance
# ===========================

with st.expander("3️⃣ Model Performance"):

    st.write("View the performance of the trained AdaBoost model.")

    if st.button("View Performance"):

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Accuracy", "100.00%")
            st.metric("Precision", "100.00%")
            st.metric("Recall", "100.00%")
            st.metric("F1 Score", "100.00%")
            st.metric("Cross Validation", "99.63%")

        with col2:
            st.metric("F1 Score", "100%")
            st.metric("Cross Validation", "99.63%")
            st.metric("Model", "AdaBoost")

        st.success("Model is ready for deployment.")
        
    # ===========================
# 4️⃣ Feature Importance
# ===========================

with st.expander("4️⃣ Feature Importance"):

    st.write("View the importance of each feature used by the trained model.")

    if st.button("Show Feature Importance"):

        feature_df = pd.read_csv("feature_importance.csv")

        feature_df = feature_df.sort_values(
            by="Importance",
            ascending=False
        )

        st.subheader("Feature Importance")

        st.bar_chart(
            feature_df.set_index("Feature")["Importance"]
        )

        st.dataframe(feature_df)

# ===========================
# 5️⃣ Loan Prediction
# ===========================

with st.expander("5️⃣ Loan Prediction", expanded=True):

    st.subheader("Applicant Information")

    col1, col2 = st.columns(2)

    with col1:

        age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=30
    )

        gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

        marital_status = st.selectbox(
        "Marital Status",
        ["Single", "Married", "Divorced"]
    )

        annual_income = st.number_input(
        "Annual Income",
        min_value=0,
        value=50000
    )

        loan_amount = st.number_input(
        "Loan Amount",
        min_value=0,
        value=10000
    )

    with col2:

        credit_score = st.slider(
        "Credit Score",
        300,
        900,
        700
    )

        num_dependents = st.number_input(
        "Number of Dependents",
        min_value=0,
        max_value=10,
        value=0
    )

        existing_loans = st.number_input(
        "Existing Loans",
        min_value=0,
        max_value=10,
        value=0
    )

        employment_status = st.selectbox(
        "Employment Status",
        ["Employed", "Self-Employed", "Unemployed"]
    )

    st.markdown("---")

    predict = st.button(
    "Predict Loan Approval",
    use_container_width=True
)

    if predict:

        gender_value = 1 if gender == "Male" else 0

        marital_mapping = {
        "Single": 0,
        "Married": 1,
        "Divorced": 2
    }

        employment_mapping = {
        "Unemployed": 0,
        "Employed": 1,
        "Self-Employed": 2
    }

        input_df = pd.DataFrame({

        "age":[age],
        "gender":[gender_value],
        "marital_status":[marital_mapping[marital_status]],
        "annual_income":[annual_income],
        "loan_amount":[loan_amount],
        "credit_score":[credit_score],
        "num_dependents":[num_dependents],
        "existing_loans_count":[existing_loans],
        "employment_status":[employment_mapping[employment_status]]

    })

        prediction = model.predict(input_df)[0]

        probability = model.predict_proba(input_df)[0][1]

        if prediction == 1:
            st.success("✅ Loan Approved")
        else:
            st.error("❌ Loan Rejected")

        st.metric(
            "Approval Probability",
            f"{probability*100:.2f}%"
        )
        
        st.markdown("---")

        risk = (
            "Low Risk" if probability >= 0.80
            else "Medium Risk" if probability >= 0.60
            else "High Risk"
        )

        suggestion = (
            "Eligible for loan approval."
            if prediction == 1
            else "Improve credit score or income before applying again."
        )

        report = f"""
            AI LOAN DECISION SUPPORT SYSTEM
        ----------------------------------------

        Prediction Date : {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}

        Applicant Information
        ---------------------
        Age : {age}
        Gender : {gender}
        Marital Status : {marital_status}
        Employment Status : {employment_status}

        Annual Income : {annual_income}
        Loan Amount : {loan_amount}
        Credit Score : {credit_score}

        Dependents : {num_dependents}
        Existing Loans : {existing_loans}

        Prediction Result
        -----------------
        Decision : {"Loan Approved" if prediction==1 else "Loan Rejected"}

        Approval Probability : {probability * 100:.2f}%

        Risk Level : {risk}

        Suggestion
        ----------
        {suggestion}
        """

        st.download_button(
            label="📄 Download Prediction Report",
            data=report,
            file_name="loan_prediction_report.txt",
            mime="text/plain"
)