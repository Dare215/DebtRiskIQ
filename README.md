# DebtRiskIQ: Loan Default Prediction

**Author:** Darious Brown  
**GitHub:** [Dare215](https://github.com/Dare215)  
**Email:** dariousbrown3@icloud.com  

## 1) Project Overview
DebtRiskIQ is a machine learning-powered loan default prediction system that analyzes borrower profiles to assess the probability of default.  
The project applies **exploratory data analysis (EDA)**, **feature engineering**, and **predictive modeling** (Logistic Regression & Random Forest) to identify risk factors and classify applicants.

## 2) Dataset
- **Source:** Kaggle Loan Default Dataset  
- **File:** `Loan_default.csv`  
- **Size:** 2,532 rows × 13 columns  
- **Key Variables:** `loan_amount`, `term`, `employment_type`, `annual_income`, `education`, `credit_score`, `age`, `loan_purpose`, `default_status` (target)  

## 3) Key Features
- **EDA:** Histograms, heatmaps, pie charts, scatter plots for distribution & correlation insights.
- **Modeling:** Logistic Regression baseline & Random Forest Classifier for nonlinear relationships.
- **Metrics:** Accuracy, Precision, Recall, F1-score, ROC-AUC.
- **Interpretability:** SHAP values for feature importance at global & individual levels.
- **Deployment:** Streamlit-based app for manual and batch predictions.

## 4) Project Structure
```
DebtRiskIQ/
│── LoanFraudPredictiorEngine-2.ipynb   # Jupyter notebook with EDA, modeling, and SHAP
│── loan_predictor_app.py               # Streamlit app
│── main.py                             # Model serving entry point
│── Loan_default.csv                    # Dataset
│── loan_default_model.pkl              # Trained Random Forest model
│── requirements.txt                    # Python dependencies
│── README.md                           # Project documentation
```

## 5) How to Run

### Option A — Python / Terminal
```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # Mac/Linux
.venv\Scripts\activate      # Windows

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run loan_predictor_app.py
```

### Option B — PyCharm
1. Open the folder in PyCharm  
2. Configure a Python interpreter (point to `.venv`)  
3. Install requirements from `requirements.txt`  
4. Run either the notebook or the Streamlit app

### Option C — GitHub Desktop
1. Add the local repository to GitHub Desktop  
2. Commit all files (including `README.md` and `requirements.txt`)  
3. Push to a public GitHub repository  

## 6) Results Summary
- **Random Forest ROC-AUC:** ~0.85  
- **Top Predictors:** Credit Score, Annual Income, Loan Amount, Employment Type  
- **Key Insight:** Credit score has a strong inverse correlation with default probability.

## 7) Ethical Considerations
- Ensure fairness by monitoring disparate impact across income, education, and job type.
- Avoid using sensitive features as proxies for discriminatory attributes.
- Provide clear explanations for model predictions to build stakeholder trust.

## 8) Future Enhancements
- Integrate **Fairness AI metrics** to monitor bias.
- Add **LIME** alongside SHAP for interpretability comparisons.
- Expand dataset for improved generalization.

## 9) License
MIT License — Free to use with attribution.
