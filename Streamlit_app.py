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
    sample_csv_url = "https://raw.githubusercontent.com/Resha06/your-repo/main/loan_dataset.csv"
    st.write("Using sample CSV. Replace `sample_csv_url` with your raw GitHub CSV URL.")
    try:
        df = pd.read_csv(sample_csv_url)
    except Exception:
        st.error("Couldn't load the sample CSV. Upload a local CSV instead.")
        st.stop()

st.subheader("Preview of dataset")
st.dataframe(df.head())

# REQUIRED COLUMNS
required_cols = [
    'Gender','Married','Dependents','Education','Self_Employed',
    'Applicant_Income','Coapplicant_Income','Loan_Amount','Loan_Amount_Term',
    'Credit_History','Property_Area','Loan_Status'
]

missing = [c for c in required_cols if c not in df.columns]
if missing:
    st.warning(f"Dataset is missing columns: {missing}. Fix names in your CSV or in the code.")

# Convert Loan_Status if needed
if df['Loan_Status'].dtype == 'object':
    df['Loan_Status'] = df['Loan_Status'].map({'Y': 1, 'N': 0})

feature_cols = [
    "Gender","Married","Dependents","Education","Self_Employed",
    "Applicant_Income","Coapplicant_Income","Loan_Amount",
    "Loan_Amount_Term","Credit_History","Property_Area"
]
target_col = "Loan_Status"

X = df[feature_cols]
y = df[target_col].astype(int)

# Clean
X.replace("?", np.nan, inplace=True)
num_cols = ["Applicant_Income","Coapplicant_Income","Loan_Amount","Loan_Amount_Term","Credit_History"]

for col in num_cols:
    X[col] = pd.to_numeric(X[col], errors="coerce")

numeric_features = num_cols
categorical_features = [c for c in X.columns if c not in numeric_features]

# FIXED missing transformer
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("clf", LogisticRegression(max_iter=1000))
])

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
with st.spinner("Training model..."):
    model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

st.success(f"Model trained successfully! Accuracy = {acc:.3f}")

# CONFUSION MATRIX
st.subheader("Confus
