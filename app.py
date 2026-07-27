import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("employee_attrition_model.pkl")
st.sidebar.title("Employee Attrition Prediction")

st.sidebar.markdown("""
### About

This application predicts whether an employee is likely to leave the organization using a Machine Learning model.

### Model

- Logistic Regression
- Scikit-learn Pipeline
- StandardScaler
- OneHotEncoder

### Dataset

IBM HR Analytics Employee Attrition Dataset

### Developer

Y. Manjunath
MBA - Data Analytics
""")

st.set_page_config(
    page_title="Employee Attrition Prediction",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Employee Attrition Prediction System")
st.markdown(
    "Predict the likelihood of employee attrition using a Machine Learning model trained on the IBM HR Analytics dataset."
)
st.write("Predict whether an employee is likely to leave the organization.")

st.subheader("Employee Details")

col1, col2 = st.columns(2)

with col1:
    Age = st.number_input("Age", 18, 60, 30)
    BusinessTravel = st.selectbox(
        "Business Travel",
        ["Travel_Rarely", "Travel_Frequently", "Non-Travel"]
    )
    DailyRate = st.number_input("Daily Rate", 100, 2000, 800)
    Department = st.selectbox(
        "Department",
        ["Sales", "Research & Development", "Human Resources"]
    )
    DistanceFromHome = st.number_input("Distance From Home", 1, 50, 5)
    Education = st.selectbox("Education", [1, 2, 3, 4, 5])
    EducationField = st.selectbox(
        "Education Field",
        [
            "Life Sciences",
            "Medical",
            "Marketing",
            "Technical Degree",
            "Other",
            "Human Resources",
        ],
    )
    EnvironmentSatisfaction = st.selectbox(
        "Environment Satisfaction", [1, 2, 3, 4]
    )
    Gender = st.selectbox("Gender", ["Male", "Female"])
    HourlyRate = st.number_input("Hourly Rate", 30, 100, 60)
    JobInvolvement = st.selectbox("Job Involvement", [1, 2, 3, 4])
    JobLevel = st.selectbox("Job Level", [1, 2, 3, 4, 5])
    JobRole = st.selectbox(
        "Job Role",
        [
            "Sales Executive",
            "Research Scientist",
            "Laboratory Technician",
            "Manufacturing Director",
            "Healthcare Representative",
            "Manager",
            "Sales Representative",
            "Research Director",
            "Human Resources",
        ],
    )

with col2:
    JobSatisfaction = st.selectbox("Job Satisfaction", [1, 2, 3, 4])
    MaritalStatus = st.selectbox(
        "Marital Status",
        ["Single", "Married", "Divorced"]
    )
    MonthlyIncome = st.number_input("Monthly Income", 1000, 25000, 5000)
    MonthlyRate = st.number_input("Monthly Rate", 2000, 30000, 15000)
    NumCompaniesWorked = st.number_input("Companies Worked", 0, 10, 1)
    OverTime = st.selectbox("OverTime", ["Yes", "No"])
    PercentSalaryHike = st.number_input("Salary Hike (%)", 10, 30, 15)
    PerformanceRating = st.selectbox("Performance Rating", [3, 4])
    RelationshipSatisfaction = st.selectbox(
        "Relationship Satisfaction",
        [1, 2, 3, 4]
    )
    StockOptionLevel = st.selectbox("Stock Option Level", [0, 1, 2, 3])
    TotalWorkingYears = st.number_input(
        "Total Working Years", 0, 40, 10
    )
    TrainingTimesLastYear = st.number_input(
        "Training Times Last Year", 0, 10, 2
    )
    WorkLifeBalance = st.selectbox(
        "Work Life Balance",
        [1, 2, 3, 4]
    )
    YearsAtCompany = st.number_input(
        "Years At Company", 0, 40, 5
    )
    YearsInCurrentRole = st.number_input(
        "Years In Current Role", 0, 20, 3
    )
    YearsSinceLastPromotion = st.number_input(
        "Years Since Last Promotion", 0, 15, 1
    )
    YearsWithCurrManager = st.number_input(
        "Years With Current Manager", 0, 20, 3
    )

if st.button("Predict Attrition"):
    Gender = 1 if Gender == "Male" else 0
    OverTime = 1 if OverTime == "Yes" else 0

    input_df = pd.DataFrame({
        "Age": [Age],
        "BusinessTravel": [BusinessTravel],
        "DailyRate": [DailyRate],
        "Department": [Department],
        "DistanceFromHome": [DistanceFromHome],
        "Education": [Education],
        "EducationField": [EducationField],
        "EnvironmentSatisfaction": [EnvironmentSatisfaction],
        "Gender": [Gender],
        "HourlyRate": [HourlyRate],
        "JobInvolvement": [JobInvolvement],
        "JobLevel": [JobLevel],
        "JobRole": [JobRole],
        "JobSatisfaction": [JobSatisfaction],
        "MaritalStatus": [MaritalStatus],
        "MonthlyIncome": [MonthlyIncome],
        "MonthlyRate": [MonthlyRate],
        "NumCompaniesWorked": [NumCompaniesWorked],
        "OverTime": [OverTime],
        "PercentSalaryHike": [PercentSalaryHike],
        "PerformanceRating": [PerformanceRating],
        "RelationshipSatisfaction": [RelationshipSatisfaction],
        "StockOptionLevel": [StockOptionLevel],
        "TotalWorkingYears": [TotalWorkingYears],
        "TrainingTimesLastYear": [TrainingTimesLastYear],
        "WorkLifeBalance": [WorkLifeBalance],
        "YearsAtCompany": [YearsAtCompany],
        "YearsInCurrentRole": [YearsInCurrentRole],
        "YearsSinceLastPromotion": [YearsSinceLastPromotion],
        "YearsWithCurrManager": [YearsWithCurrManager],
    })


    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0]

    leave_prob = probability[1] * 100
    stay_prob = probability[0] * 100

    st.markdown("---")
    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("⚠️ Employee is likely to leave the organization.")
    else:
        st.success("✅ Employee is likely to stay with the organization.")

    st.metric(
        "Prediction Confidence",
        f"{max(stay_prob, leave_prob):.2f}%"
    )

    st.write("### Prediction Probability")
    st.write(f"🟢 Stay: {stay_prob:.2f}%")
    st.progress(stay_prob / 100)

    st.write(f"🔴 Leave: {leave_prob:.2f}%")
    st.progress(leave_prob / 100)

    st.markdown("---")
    st.subheader("Employee Summary")
    st.dataframe(input_df, use_container_width=True)
    st.markdown("---")
st.caption("Developed by Y. Manjunath | Python • Scikit-learn • Streamlit")
if st.button("Reset"):
    st.rerun()

