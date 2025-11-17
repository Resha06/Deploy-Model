import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

st.set_page_config(page_title="Loan Eligibility App", layout="centered")

st.title("Loan Eligibility Predictor")
st.write("Upload your loan dataset **or** use a sample CSV from GitHub. Then enter applicant details to predict.")

# --- DATA SOURCE SELECTION ---
use_upload = st.radio("Choose dataset source", ["Use sample from GitHub (default)", "Upload CSV file"])

if use_upload == "Upload CSV file":
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        st.warning("Please upload a CSV file.")
        st.stop()
else:
    # Put your raw GitHub CSV URL here (replace the URL below)
    sample_csv_url = "https://raw.githubusercontent.com/Resha06/your-repo/main/loan_dataset.csv"
    st.write("Using sample CSV. Replace `sample_csv_url` in app.py with your raw GitHub URL.")
    try:
        df = pd.read_csv(sample_csv_url)
    except Exception as e:
        st.error("Couldn't load the sample CSV from GitHub. If you want, upload a local CSV instead.")
        st.stop()

st.subheader("Preview of dataset")
st.dataframe(df.head())

# --- Expectation of columns (adjust mapping for your dataset) ---
st.markdown("**Expected columns (if different, edit the column names in app.py):**")
st.write("`['Gender','Married','Dependents','Education','Self_Employed','Applicant_Income','Coapplicant_Income','Loan_Amount','Loan_Amount_Term','Credit_History','Property_Area','Loan_Status']`")

# Check whether required columns exist; if not show a helpful message and stop
required_cols = [
    'Gender',
    'Married',
    'Dependents',
    'Education',
    'Self_Employed',
    'Applicant_Income',
    'Coapplicant_Income',
    'Loan_Amount',
    'Loan_Amount_Term',
    'Credit_History',
    'Property_Area',
    'Loan_Status'
]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    st.warning(f"Dataset is missing columns: {missing}. You can still proceed, but edit the code to match your column names.")
    # we continue but may fail later

# --- Simple preprocessing + training pipeline ---
# We'll convert Loan_Status to numeric label if it's present and non-numeric
if 'Loan_Status' in df.columns:
    if df['Loan_Status'].dtype == 'object':
        df['Loan_Status'] = df['Loan_Status'].map({'N':0, 'Y':1})
        # if your values are other strings, adjust mapping above

# select features and target (edit these if your dataset differs)
feature_cols = [
    "Gender", "Married", "Dependents", "Education", "Self_Employed",
    "Applicant_Income", "Coapplicant_Income", "Loan_Amount",
    "Loan_Amount_Term", "Credit_History", "Property_Area"]
target_col = 'Loan_Status'

if target_col not in df.columns:
    st.error("Target column Loan_Status not found. Please provide a dataset with a target or adapt app.py.")
    st.stop()

X = df[feature_cols].copy()
y = df[target_col].astype(int).copy()

# Basic cleaning: replace '?' with NaN and coerce numeric
X.replace('?', np.nan, inplace=True)
for col in ['Applicant_Income','Coapplicant_Income','Loan_Amount','Loan_Amount_Term','Credit_History']:
    if col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')

# Define columns for preprocessing
numeric_features = [c for c in X.columns if X[c].dtype in [np.float64, np.int64] or c in ['Applicant_Income','Coapplicant_Income','Loan_Amount','Loan_Amount_Term','Credit_History']]
categorical_features = [c for c in X.columns if c not in numeric_features]

# Pipelines
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='drop'
)

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('clf', LogisticRegression(max_iter=1000))
])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
with st.spinner("Training model..."):
    model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
st.success(f"Model trained. Test accuracy: {acc:.3f}")

st.subheader("Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
ax.matshow(cm)
for (i, j), val in np.ndenumerate(cm):
    ax.text(j, i, f'{val}', va='center', ha='center')
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig)

st.subheader("Classification report (short)")
st.text(classification_report(y_test, y_pred, digits=3))

# --- Prediction UI ---
st.header("Predict loan eligibility for a single applicant")
with st.form("applicant_form"):
    # Create input widgets for each feature
    Gender = st.selectbox("Gender", options=['Male','Female'])
    Married = st.selectbox("Married", options=['Yes','No'])
    Dependents = st.selectbox("Dependents", options=['0','1','2','3+'])
    Education = st.selectbox("Education", options=['Graduate','Not Graduate'])
    Self_emp = st.selectbox("Self Employed", options=['Yes','No'])
    Applicant_income = st.number_input("Applicant Income", min_value=0.0, value=2500.0, step=100.0)
    Coapplicant_income = st.number_input("Coapplicant Income", min_value=0.0, value=0.0, step=100.0)
    Loan_amount = st.number_input("Loan Amount (in thousands)", min_value=0.0, value=100.0, step=10.0)
    Loan_term = st.number_input("Loan Amount Term (in days/months)", min_value=0.0, value=360.0, step=12.0)
    Credit_history = st.selectbox("Credit History (1 = good, 0 = bad)", options=[1.0, 0.0])
    Property_area = st.selectbox("Property Area", options=['Urban','Semiurban','Rural'])
    Submit = st.form_submit_button("Predict")

if submit:
    applicant_df = pd.DataFrame([{
        'Gender': gender,
        'Married': 'Yes' if married == 'Yes' else 'No',
        'Dependents': dependents,
        'Education': education,
        'Self_Employed': 'Yes' if self_emp == 'Yes' else 'No',
        'Applicant_Income': applicant_income,
        'Coapplicant_Income': coapplicant_income,
        'Loan_Amount': loan_amount,
        'Loan_Amount_Term': loan_term,
        'Credit_History': credit_history,
        'Property_Area': property_area
    }])
    proba = model.predict_proba(applicant_df)[:, 1][0]
    pred = model.predict(applicant_df)[0]
    st.metric("Probability of Approval", f"{proba:.2f}")
    st.write("Model prediction:", "✅ Approved" if pred==1 else "❌ Not Approved")
    st.write("Note: This is a simple model trained on your dataset. For production you should validate, tune hyperparameters, and handle imbalanced classes.")



