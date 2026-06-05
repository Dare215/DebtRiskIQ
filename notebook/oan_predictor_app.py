# loan_predictor_app.py
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ---- App settings ----
st.set_page_config(page_title="Loan Default Risk Predictor", page_icon="💳", layout="centered")

# ---- App Logo ----
st.image("loan_logo.png", width=150)  # Make sure 'loan_logo.png' is in the same folder

st.title("💳 Loan Default Risk Predictor")
st.write("Predict default probability for a single applicant or a batch CSV.")

# ---- Load trained model ----
MODEL_PATH = Path("loan_default_model.pkl")
if not MODEL_PATH.exists():
    st.error("Model file 'loan_default_model.pkl' not found. Train & export it first.")
    st.stop()
model = joblib.load(MODEL_PATH)

# ---- Utilities ----
YN = ["Yes", "No"]
def yn_normalize(x: str) -> str:
    x = str(x).strip().lower()
    if x in {"yes","y","true","1"}: return "Yes"
    if x in {"no","n","false","0"}: return "No"
    return "No"

st.sidebar.header("⚙️ Settings")
threshold = st.sidebar.slider("Decision threshold", 0.05, 0.95, 0.50, 0.01)
st.sidebar.caption("Pred ≥ threshold → classify as DEFAULT")

# =========================
# Single applicant section
# =========================
st.header("🧍 Single Applicant")
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=35, step=1)
    income = st.number_input("Annual Income ($)", min_value=0, value=60000, step=1000)
    loan_amount = st.number_input("Loan Amount ($)", min_value=0, value=15000, step=500)
    credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=680, step=1)
    months_employed = st.number_input("Months Employed", min_value=0, value=24, step=1)
    num_credit_lines = st.number_input("Number of Credit Lines", min_value=0, value=5, step=1)

with col2:
    interest_rate = st.number_input("Interest Rate (%)", min_value=0.0, value=7.5, step=0.1, format="%.2f")
    loan_term = st.number_input("Loan Term (months)", min_value=1, value=36, step=1)
    dti_ratio = st.slider("Debt-to-Income Ratio", 0.0, 1.0, 0.30, 0.01)
    education = st.text_input("Education (e.g., high school, undergraduate, graduate)", "undergraduate")
    employment_type = st.text_input("Employment Type (e.g., salaried, self-employed)", "salaried")
    marital_status = st.text_input("Marital Status (e.g., single, married)", "single")

col3, col4, col5 = st.columns(3)
with col3:
    has_mortgage = st.selectbox("Has Mortgage?", YN, index=1)  # default No
with col4:
    has_dependents = st.selectbox("Has Dependents?", YN, index=1)
with col5:
    has_cosigner = st.selectbox("Has Co-Signer?", YN, index=1)

loan_purpose = st.text_input("Loan Purpose (e.g., personal, education, medical, business)", "personal")

if st.button("Predict Default Risk", type="primary"):
    row = pd.DataFrame([{
        "Age": age,
        "Income": income,
        "LoanAmount": loan_amount,
        "CreditScore": credit_score,
        "MonthsEmployed": months_employed,
        "NumCreditLines": num_credit_lines,
        "InterestRate": interest_rate,
        "LoanTerm": loan_term,
        "DTIRatio": dti_ratio,
        "Education": education,
        "EmploymentType": employment_type,
        "MaritalStatus": marital_status,
        "HasMortgage": yn_normalize(has_mortgage),
        "HasDependents": yn_normalize(has_dependents),
        "LoanPurpose": loan_purpose,
        "HasCoSigner": yn_normalize(has_cosigner),
    }])

    try:
        prob = model.predict_proba(row)[:, 1][0]
        pred = int(prob >= threshold)
        st.metric("Default Probability", f"{prob*100:.2f}%")
        st.success("✅ NO DEFAULT (Lower Risk)") if pred == 0 else st.error("⚠️ DEFAULT (Higher Risk)")
    except Exception as e:
        st.error(f"Prediction error: {e}")

st.divider()

# =========================
# Batch scoring section
# =========================
st.header("📦 Batch Predictions (CSV)")
st.caption("Your CSV must include these columns exactly: "
           "`Age, Income, LoanAmount, CreditScore, MonthsEmployed, NumCreditLines, InterestRate, LoanTerm, DTIRatio, "
           "Education, EmploymentType, MaritalStatus, HasMortgage, HasDependents, LoanPurpose, HasCoSigner`")

# Downloadable template
template = pd.DataFrame([{
    "Age": 35, "Income": 60000, "LoanAmount": 15000, "CreditScore": 680, "MonthsEmployed": 24,
    "NumCreditLines": 5, "InterestRate": 7.5, "LoanTerm": 36, "DTIRatio": 0.30,
    "Education": "undergraduate", "EmploymentType": "salaried", "MaritalStatus": "single",
    "HasMortgage": "No", "HasDependents": "No", "LoanPurpose": "personal", "HasCoSigner": "No"
}])
st.download_button("⬇️ Download CSV Template", data=template.to_csv(index=False).encode("utf-8"),
                   file_name="batch_template.csv", mime="text/csv")

uploaded = st.file_uploader("Upload CSV for batch scoring", type=["csv"])
if uploaded is not None:
    try:
        df = pd.read_csv(uploaded)
        req = list(template.columns)
        missing = [c for c in req if c not in df.columns]
        if missing:
            st.error(f"Missing required columns: {missing}")
        else:
            # Normalize Y/N flags
            for c in ["HasMortgage", "HasDependents", "HasCoSigner"]:
                if c in df.columns:
                    df[c] = df[c].apply(yn_normalize)

            probs = model.predict_proba(df[req])[:, 1]
            preds = (probs >= threshold).astype(int)
            out = df.copy()
            out["default_probability"] = probs
            out["prediction"] = preds
            st.write("Preview:")
            st.dataframe(out.head(25))
            st.download_button("⬇️ Download Predictions", data=out.to_csv(index=False).encode("utf-8"),
                               file_name="loan_default_predictions.csv", mime="text/csv")
    except Exception as e:
        st.error(f"Batch scoring error: {e}")

st.info("⚠️ Educational demo. Predictions are probabilistic and must not be used alone for lending decisions.")
