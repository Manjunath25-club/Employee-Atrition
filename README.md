# Employee Attrition Prediction System

StreamLit Live Link:   
https://employee-atrition-7yu6z9tvwcwln6jtsqq3vq.streamlit.app/

## Overview

The Employee Attrition Prediction System is a Machine Learning web application developed using Python and Streamlit. It predicts whether an employee is likely to leave an organization based on HR-related attributes. The application provides prediction results along with confidence scores and probability distributions to support HR decision-making.

---

## Features

- Predict employee attrition using Machine Learning
- Interactive web interface built with Streamlit
- Displays prediction confidence
- Shows probability of Stay vs Leave
- Employee information summary
- Professional dashboard layout
- Easy-to-use interface

---

## Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Joblib

---

## Machine Learning

### Algorithm

- Logistic Regression

### Data Preprocessing

- Label Encoding
- One-Hot Encoding
- StandardScaler
- Pipeline

---

## Dataset

IBM HR Analytics Employee Attrition Dataset

Dataset includes employee information such as:

- Age
- Gender
- Monthly Income
- Distance From Home
- Total Working Years
- Job Role
- Marital Status
- Overtime
- Education
- Business Travel
- Years at Company
- Work-Life Balance
- Job Satisfaction
- Environment Satisfaction
- and other HR-related attributes.

---

## Project Structure

```
Employee_Attrition_Prediction/
│
├── app.py
├── train_model.py
├── employee_attrition_model.pkl
├── requirements.txt
├── README.md
├── WA_Fn-UseC_-HR-Employee-Attrition.csv
└── images/
    ├── home.png
    ├── prediction1.png
    └── prediction2.png
```

---

## Installation

### Clone Repository

```bash
git clone https://github.com/yourusername/Employee-Attrition-Prediction.git
```

### Navigate to Project

```bash
cd Employee-Attrition-Prediction
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Application

```bash
streamlit run app.py
```

---

## Application Workflow

1. Enter employee details.
2. Click **Predict Attrition**.
3. View prediction result.
4. Check confidence score.
5. Review Stay vs Leave probabilities.
6. Analyze employee summary.

---

## Future Enhancements

- Deploy on Streamlit Community Cloud
- Add feature importance visualization
- Integrate SHAP explainability
- Support multiple ML algorithms
- Export prediction reports as PDF
- Store prediction history in a database

---

## Developer

**Y. Manjunath**

MBA (Data Analytics)

Skills:
- Python
- SQL
- Power BI
- Tableau
- Machine Learning
- Data Analysis
- Streamlit

---

## License

This project is developed for educational and portfolio purposes.



