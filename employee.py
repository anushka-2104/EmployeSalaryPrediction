import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor

# =========================
# Load Dataset
# =========================

df = pd.read_csv("1-employee_salary_dataset.csv")

# =========================
# Preprocessing
# =========================

# Convert categorical columns to numeric
for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].astype("category").cat.codes

# Remove missing values
df = df.dropna()

# =========================
# Features and Target
# =========================

target = "Annual_Salary_LPA"

# Remove Employee_ID from features if present
if "Employee_ID" in df.columns:
    X = df.drop(columns=["Employee_ID", target])
else:
    X = df.drop(columns=[target])

y = df[target]

# =========================
# Split Data
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================
# Train Model
# =========================

model = GradientBoostingRegressor(random_state=42)
model.fit(X_train, y_train)

# =========================
# Streamlit UI
# =========================

st.set_page_config(page_title="Employee Salary Prediction")

st.title("💼 Employee Salary Prediction")
st.write("Enter employee details below:")

# =========================
# User Inputs
# =========================

input_values = {}

for col in X.columns:
    input_values[col] = st.number_input(
        label=col,
        value=float(X[col].mean())
    )

input_df = pd.DataFrame([input_values])

# =========================
# Prediction
# =========================

if st.button("Predict Salary"):
    prediction = model.predict(input_df)[0]

    st.success(f"💰 Predicted Salary: ₹ {prediction:.2f} LPA")