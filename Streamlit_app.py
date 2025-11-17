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
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Could not read uploaded CSV: {e}")
            st.stop()
    else:
        st.warning("Please upload a CSV file.")
        st.stop()
else:
    # Replace this with your raw GitHub CSV URL if you want to load from GitHub
    sample_csv_url = "https://raw.githubusercontent.com/Resha06/your-repo/main/loan_dataset.csv"
    st.write("Using sample CSV. Replace `sample_csv_url` with your raw GitHub CSV URL if needed.")
    try:
        df = pd.read_csv(sample_csv_url)
    except Exception:
        st.error("Couldn't load the sample CSV from GitHub. Please upload a local CSV instead.")
        st.stop()

st.subheader("Preview of dataset")
st.dataframe(df.head())

# --- EXPECTED / REQUIRED COLUMNS (match your CSV)
required_cols = [
    'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
    'Applicant_Income', 'Coapplicant_Income', 'Loan_Amount', 'Loan_Amount_Term',
    'Credit_History', 'Property_Area', 'Loan_Status'
]

missing = [c for c in required_cols if c not in df.columns]
if missing:
    st.error(f"Dataset is missing required columns: {missing}. Please fix column names in the CSV or update the code.")
    st.stop()

# Convert target to numeric if needed
if df['Loan_Status'].dtype == 'object':
    df['Loan_Status'] = df['Loan_Status'].map({'Y': 1, 'N': 0})
    if df['Loan_Status'].isnull().any():
        st.error("Loan_Status contains values other than 'Y'/'N'. Please standardize the target values.")
        st.stop()

# --- Select features and target
feature_cols = [
    "Gender", "Married", "Dependents", "Education", "Self_Employed",
    "Applicant_Income", "Coapplicant_Income", "Loan_Amount",
    "Loan_Amount_Term", "Credit_History", "Property_Area"
]
target_col = "Loan_Status"

# Safety check
for col in feature_cols + [target_col]:
    if col not in df.columns:
        st.error(f"Column '{col}' not found in dataset.")
        st.stop()

X = df[feature_cols].copy()
y = df[target_col].astype(int).copy()

# Basic cleaning: convert '?' to NaN and coerce numeric columns
X.replace('?', np.nan, inplace=True)

numeric_cols_expected = ['Applicant_Income', 'Coapplicant_Income', 'Loan_Amount', 'Loan_Amount_Term', 'Credit_History']
numeric_features = [c for c in numeric_cols_expected if c in X.columns]
for col in numeric_features:
    X[col] = pd.to_numeric(X[col], errors='coerce')

categorical_features = [c for c in X.columns if c not in numeric_features]

# Preprocessing pipelines
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# For scikit-learn 1.7+, use sparse_output=False (instead of sparse=False)
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='drop'
)

# Model pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('clf', LogisticRegression(max_iter=1000))
])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

with st.spinner("Training model..."):
    model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
st.success(f"Model trained successfully! Test accuracy = {acc:.3f}")

st.subheader("Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
ax.matshow(cm)
for (i, j), val in np.ndenumerate(cm):
    ax.text(j, i, str(val), va='center', ha='center')
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig)

st.subheader("Classification Report")
st.text(classification_report(y_test, y_pred, digits=3))

# --- Prediction UI ---
st.header("Predict loan eligibility for a single applicant")

with st.form("applicant_form"):
    gender = st.selectbox("Gender", options=['Male', 'Female'])
    married = st.selectbox("Married", options=['Yes', 'No'])
    dependents = st.selectbox("Dependents", options=['0', '1', '2', '3+'])
    education = st.selectbox("Education", options=['Graduate', 'Not Graduate'])
    self_emp = st.selectbox("Self Employed", options=['Yes', 'No'])

    applicant_income = st.number_input("Applicant Income", min_value=0.0, value=2500.0, step=100.0)
    coapplicant_income = st.number_input("Coapplicant Income", min_value=0.0, value=0.0, step=100.0)
    loan_amount = st.number_input("Loan Amount", min_value=0.0, value=120.0, step=1.0)
    loan_term = st.number_input("Loan Amount Term", min_value=0.0, value=360.0, step=12.0)
    credit_history = st.selectbox("Credit History (1 = good, 0 = bad)", options=[1.0, 0.0])
    property_area = st.selectbox("Property Area", options=['Urban', 'Semiurban', 'Rural'])

    submit = st.form_submit_button("Predict")

if submit:
    applicant_df = pd.DataFrame([{
        'Gender': gender,
        'Married': married,
        'Dependents': dependents,
        'Education': education,
        'Self_Employed': self_emp,
        'Applicant_Income': applicant_income,
        'Coapplicant_Income': coapplicant_income,
        'Loan_Amount': loan_amount,
        'Loan_Amount_Term': loan_term,
        'Credit_History': credit_history,
        'Property_Area': property_area
    }])

    # Ensure same dtypes as training
    for col in numeric_features:
        applicant_df[col] = pd.to_numeric(applicant_df[col], errors='coerce')

    proba = model.predict_proba(applicant_df)[:, 1][0]
    pred = model.predict(applicant_df)[0]

    st.metric("Probability of Approval", f"{proba:.2f}")
    st.write("Model prediction:", "✅ Approved" if pred == 1 else "❌ Not Approved")

st.info("Note: This is a simple baseline model. For production, do hyperparameter tuning, cross-validation, and more feature engineering.")
