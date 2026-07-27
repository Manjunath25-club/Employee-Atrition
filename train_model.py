import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ==========================
# Load Dataset
# ==========================
df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")

# ==========================
# Data Cleaning
# ==========================
df.drop(
    columns=[
        "EmployeeCount",
        "Over18",
        "StandardHours",
        "EmployeeNumber"
    ],
    inplace=True
)

# ==========================
# Label Encoding
# ==========================
le = LabelEncoder()

df["Attrition"] = le.fit_transform(df["Attrition"])
df["Gender"] = le.fit_transform(df["Gender"])
df["OverTime"] = le.fit_transform(df["OverTime"])

# ==========================
# Features & Target
# ==========================
X = df.drop("Attrition", axis=1)
y = df["Attrition"]

# ==========================
# Train Test Split
# ==========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# ==========================
# Preprocessing
# ==========================
categorical_features = X.select_dtypes(include="object").columns
numerical_features = X.select_dtypes(exclude="object").columns

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_features),
        (
            "cat",
            OneHotEncoder(
                drop="first",
                handle_unknown="ignore"
            ),
            categorical_features,
        ),
    ]
)

# ==========================
# Pipeline
# ==========================
pipeline = Pipeline(
    [
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(max_iter=1000)),
    ]
)

# ==========================
# Hyperparameter Tuning
# ==========================
param_grid = {
    "model__C": [0.01, 0.1, 1, 10, 100],
    "model__solver": ["liblinear", "lbfgs"],
}

grid = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=5,
    scoring="recall",
)

grid.fit(X_train, y_train)

best_model = grid.best_estimator_

# ==========================
# Evaluation
# ==========================
prediction = best_model.predict(X_test)

print("=" * 50)
print("Best Parameters:", grid.best_params_)
print("=" * 50)

print("Accuracy :", accuracy_score(y_test, prediction))

print("\nClassification Report")
print(classification_report(y_test, prediction))

print("\nConfusion Matrix")
print(confusion_matrix(y_test, prediction))

# ==========================
# Save Model
# ==========================
joblib.dump(best_model, "employee_attrition_model.pkl")

print("\n✅ Model Saved Successfully!")