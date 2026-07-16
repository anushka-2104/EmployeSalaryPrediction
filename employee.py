import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor

# =========================
# Page Configuration
# =========================

st.set_page_config(page_title="Employee Salary Prediction")
st.title("💼 Employee Salary Prediction")
st.write("Predict an employee's annual salary.")

# =========================
# Load Dataset
# =========================

df = pd.read_csv("1-employee_salary_dataset.csv")

# Remove Employee_ID if present
if "Employee_ID" in df.columns:
    df = df.drop(columns=["Employee_ID"])

target = "Annual_Salary_LPA"

# =========================
# Clean Categorical Columns
# =========================

category_maps = {}

categorical_cols = df.select_dtypes(include="object").columns.tolist()

for col in categorical_cols:
    # Convert FEMALE, Female, female -> Female
    # Convert MALE, Male, male -> Male
    # Remove leading/trailing spaces
    df[col] = (
        df[col]
        .astype(str)
        .str.strip()
        .str.title()
    )

    category_maps[col] = sorted(df[col].unique())

    df[col] = pd.Categorical(
        df[col],
        categories=category_maps[col]
    ).codes

# Remove missing values
df = df.dropna()

# =========================
# Features & Target
# =========================

X = df.drop(columns=[target])
y = df[target]

# =========================
# Train Model
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = GradientBoostingRegressor(random_state=42)
model.fit(X_train, y_train)

# =========================
# User Input
# =========================

st.header("Enter Employee Details")

user_input = {}

for col in X.columns:

    if col in category_maps:

        selected = st.selectbox(
            col,
            category_maps[col]
        )

        user_input[col] = category_maps[col].index(selected)

    else:

        minimum = float(df[col].min())
        maximum = float(df[col].max())
        default = float(df[col].median())

        user_input[col] = st.number_input(
            col,
            min_value=minimum,
            max_value=maximum,
            value=default
        )

# Create DataFrame
input_df = pd.DataFrame([user_input])

# Keep same column order
input_df = input_df[X.columns]

# =========================
# Prediction
# =========================

if st.button("Predict Salary"):

    prediction = model.predict(input_df)[0]

    st.success(f"💰 Predicted Annual Salary: ₹ {prediction:.2f} LPA")