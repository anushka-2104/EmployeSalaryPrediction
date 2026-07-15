import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("xgb_model.pkl")

st.set_page_config(page_title="Employee Salary Prediction")

st.title("💼 Employee Salary Prediction")
st.write("Fill in the employee details below.")

# Age
age = st.number_input("Age", min_value=18, max_value=65, value=25)

# Gender
gender = st.selectbox("Gender", ["Female", "Male"])
gender = 0 if gender == "Female" else 1

# Education
education = st.selectbox(
    "Education",
    ["High School", "Bachelor", "Master", "PhD"]
)

education_map = {
    "High School": 0,
    "Bachelor": 1,
    "Master": 2,
    "PhD": 3
}
education = education_map[education]

# Experience
experience = st.number_input(
    "Years of Experience",
    min_value=0,
    max_value=40,
    value=2
)

# Department
department = st.selectbox(
    "Department",
    ["HR", "IT", "Finance", "Sales", "Marketing", "Operations"]
)

department_map = {
    "HR": 0,
    "IT": 1,
    "Finance": 2,
    "Sales": 3,
    "Marketing": 4,
    "Operations": 5
}
department = department_map[department]

# Job Level
job_level = st.selectbox(
    "Job Level",
    ["Level 1", "Level 2", "Level 3", "Level 4", "Level 5"]
)
job_level = int(job_level.split()[-1])

# Performance Rating
performance = st.slider("Performance Rating", 1, 5, 3)

# Certifications
certifications = st.number_input(
    "Number of Certifications",
    min_value=0,
    max_value=20,
    value=0
)

# Overtime Hours
overtime = st.number_input(
    "Overtime Hours",
    min_value=0,
    max_value=100,
    value=0
)

# Remote Work
remote = st.selectbox(
    "Remote Work",
    ["No", "Yes"]
)
remote = 0 if remote == "No" else 1

# City
city = st.selectbox(
    "City",
    [
        "Delhi",
        "Mumbai",
        "Bangalore",
        "Chennai",
        "Hyderabad",
        "Pune",
        "Kolkata"
    ]
)

city_map = {
    "Delhi": 0,
    "Mumbai": 1,
    "Bangalore": 2,
    "Chennai": 3,
    "Hyderabad": 4,
    "Pune": 5,
    "Kolkata": 6
}
city = city_map[city]

# Company Tenure
tenure = st.number_input(
    "Company Tenure (Years)",
    min_value=0,
    max_value=40,
    value=1
)

# Projects Completed
projects = st.number_input(
    "Projects Completed",
    min_value=0,
    max_value=100,
    value=5
)

# Skill Score
skill = st.slider(
    "Skill Score",
    0,
    100,
    50
)

# DataFrame
input_data = pd.DataFrame({
    "Age": [age],
    "Gender": [gender],
    "Education": [education],
    "Experience_Years": [experience],
    "Department": [department],
    "Job_Level": [job_level],
    "Performance_Rating": [performance],
    "Certifications": [certifications],
    "Overtime_Hours": [overtime],
    "Remote_Work": [remote],
    "City": [city],
    "Company_Tenure": [tenure],
    "Projects_Completed": [projects],
    "Skill_Score": [skill]
})

# Prediction
if st.button("Predict Salary"):
    prediction = model.predict(input_data)
    st.success(f"Predicted Salary: ₹ {prediction[0]:.2f} LPA")