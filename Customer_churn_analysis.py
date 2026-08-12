# ============================================================
# CUSTOMER CHURN PREDICTION & RISK ANALYSIS
# ============================================================
# Author: Obarayo Oluwaseyi Maryanne
# Project: Customer Churn Analysis
#
# Workflow:
# 1. Data Loading
# 2. Data Audit
# 3. Exploratory Data Analysis
# 4. Feature Engineering
# 5. Machine Learning Preparation
# 6. Baseline Model
# 7. Logistic Regression
# 8. Random Forest
# 9. Model Comparison
# 10. Feature Importance
# 11. Customer Churn Risk Scoring
# 12. High-Risk Customer Profiling
# ============================================================


# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# 2. LOAD DATA
# ============================================================

print("\n" + "=" * 60)
print("CUSTOMER CHURN DATASET")
print("=" * 60)

# Make sure customer_churn.csv is in the same folder
df = pd.read_csv("customer_churn.csv")

print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

print("\nColumn Names:")
print(df.columns.tolist())


# ============================================================
# 3. DATA AUDIT
# ============================================================

print("\n" + "=" * 60)
print("DATA AUDIT")
print("=" * 60)

print("\nFirst 5 Records:")
print(df.head())

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Records:")
print(df.duplicated().sum())


# ============================================================
# 4. CHURN DISTRIBUTION
# ============================================================

print("\n" + "=" * 60)
print("CHURN DISTRIBUTION")
print("=" * 60)

churn_counts = df["Churn"].value_counts()

print(churn_counts)

print("\nChurn Percentage:")
print(
    df["Churn"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)


# ============================================================
# 5. NUMERICAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("NUMERICAL SUMMARY")
print("=" * 60)

print(df.describe())


# ============================================================
# 6. CATEGORICAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("CATEGORICAL SUMMARY")
print("=" * 60)

categorical_columns = df.select_dtypes(
    include=["object"]
).columns

for column in categorical_columns:

    print(f"\n--- {column} ---")
    print(df[column].value_counts())


# ============================================================
# 7. EXPLORATORY CHURN ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("CHURN ANALYSIS")
print("=" * 60)


# ------------------------------------------------------------
# Churn by Contract
# ------------------------------------------------------------

print("\nChurn by Contract Type (%)")

contract_churn = pd.crosstab(
    df["Contract"],
    df["Churn"],
    normalize="index"
) * 100

print(contract_churn.round(2))


# ------------------------------------------------------------
# Churn by Internet Service
# ------------------------------------------------------------

print("\nChurn by Internet Service (%)")

internet_churn = pd.crosstab(
    df["InternetService"],
    df["Churn"],
    normalize="index"
) * 100

print(internet_churn.round(2))


# ------------------------------------------------------------
# Churn by Online Security
# ------------------------------------------------------------

print("\nChurn by Online Security (%)")

security_churn = pd.crosstab(
    df["OnlineSecurity"],
    df["Churn"],
    normalize="index"
) * 100

print(security_churn.round(2))


# ------------------------------------------------------------
# Churn by Tech Support
# ------------------------------------------------------------

print("\nChurn by Tech Support (%)")

support_churn = pd.crosstab(
    df["TechSupport"],
    df["Churn"],
    normalize="index"
) * 100

print(support_churn.round(2))


# ------------------------------------------------------------
# Customer Value Comparison
# ------------------------------------------------------------

print("\nAverage Customer Values")

customer_values = df.groupby("Churn")[
    [
        "TenureMonths",
        "MonthlyCharges",
        "TotalCharges"
    ]
].mean()

print(customer_values.round(2))


# ============================================================
# 8. FEATURE ENGINEERING
# ============================================================

print("\n" + "=" * 60)
print("FEATURE ENGINEERING")
print("=" * 60)


# ------------------------------------------------------------
# Churn Flag
# ------------------------------------------------------------

df["Churn_Flag"] = (
    df["Churn"]
    .map({"No": 0, "Yes": 1})
)


# ------------------------------------------------------------
# Tenure Group
# ------------------------------------------------------------

df["Tenure_Group"] = pd.cut(
    df["TenureMonths"],
    bins=[-1, 12, 24, 48, 72],
    labels=[
        "0-12 Months",
        "13-24 Months",
        "25-48 Months",
        "49-72 Months"
    ]
)


# ------------------------------------------------------------
# Monthly Charge Group
# ------------------------------------------------------------

df["Charge_Group"] = pd.cut(
    df["MonthlyCharges"],
    bins=[-1, 40, 70, np.inf],
    labels=[
        "Low",
        "Medium",
        "High"
    ]
)


# ------------------------------------------------------------
# Total Services
# ------------------------------------------------------------

service_columns = [
    "OnlineSecurity",
    "TechSupport",
    "StreamingService"
]

df["Total_Services"] = (
    df[service_columns]
    .eq("Yes")
    .sum(axis=1)
)


print("\nNew Features:")
print([
    "Churn_Flag",
    "Tenure_Group",
    "Charge_Group",
    "Total_Services"
])

print("\nFeature Engineering Complete.")

print("\nUpdated Dataset:")
print(df.head())


# ============================================================
# 9. MACHINE LEARNING PREPARATION
# ============================================================

print("\n" + "=" * 60)
print("MACHINE LEARNING PREPARATION")
print("=" * 60)


# ------------------------------------------------------------
# Define Features
# ------------------------------------------------------------

categorical_features = [
    "Gender",
    "Partner",
    "Dependents",
    "Contract",
    "InternetService",
    "PaperlessBilling",
    "PaymentMethod",
    "OnlineSecurity",
    "TechSupport",
    "StreamingService",
    "Tenure_Group",
    "Charge_Group"
]

numerical_features = [
    "SeniorCitizen",
    "TenureMonths",
    "MonthlyCharges",
    "TotalCharges",
    "Total_Services"
]


print("\nCategorical Features:")
print(categorical_features)

print("\nNumerical Features:")
print(numerical_features)


# ------------------------------------------------------------
# X and y
# ------------------------------------------------------------

X = df[
    categorical_features +
    numerical_features
]

y = df["Churn_Flag"]


# ------------------------------------------------------------
# Train/Test Split
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\nTrain / Test Split")
print("Training observations:", len(X_train))
print("Testing observations:", len(X_test))

print(
    "\nTraining churn rate:",
    round(y_train.mean() * 100, 2),
    "%"
)

print(
    "Testing churn rate:",
    round(y_test.mean() * 100, 2),
    "%"
)


# ============================================================
# 10. PREPROCESSING PIPELINE
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[

        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_features
        ),

        (
            "numerical",
            StandardScaler(),
            numerical_features
        )

    ]
)


# ============================================================
# 11. BASELINE MODEL
# ============================================================

print("\n" + "=" * 60)
print("BASELINE MODEL")
print("=" * 60)


# Predict the majority class
baseline_prediction = np.zeros(
    len(y_test),
    dtype=int
)


baseline_accuracy = accuracy_score(
    y_test,
    baseline_prediction
)

baseline_precision = precision_score(
    y_test,
    baseline_prediction,
    zero_division=0
)

baseline_recall = recall_score(
    y_test,
    baseline_prediction,
    zero_division=0
)

baseline_f1 = f1_score(
    y_test,
    baseline_prediction,
    zero_division=0
)


print("\nBaseline Performance")

print("Accuracy:", round(baseline_accuracy, 4))
print("Precision:", round(baseline_precision, 4))
print("Recall:", round(baseline_recall, 4))
print("F1 Score:", round(baseline_f1, 4))

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test,
        baseline_prediction
    )
)


# ============================================================
# 12. LOGISTIC REGRESSION
# ============================================================

print("\n" + "=" * 60)
print("LOGISTIC REGRESSION")
print("=" * 60)


logistic_model = Pipeline(
    steps=[

        (
            "preprocessor",
            preprocessor
        ),

        (
            "classifier",
            LogisticRegression(
                max_iter=2000,
                random_state=42
            )
        )

    ]
)


logistic_model.fit(
    X_train,
    y_train
)


logistic_prediction = logistic_model.predict(
    X_test
)


logistic_accuracy = accuracy_score(
    y_test,
    logistic_prediction
)

logistic_precision = precision_score(
    y_test,
    logistic_prediction,
    zero_division=0
)

logistic_recall = recall_score(
    y_test,
    logistic_prediction,
    zero_division=0
)

logistic_f1 = f1_score(
    y_test,
    logistic_prediction,
    zero_division=0
)


print("\n========== MODEL PERFORMANCE ==========")

print(
    "Accuracy:",
    round(logistic_accuracy, 4)
)

print(
    "Precision:",
    round(logistic_precision, 4)
)

print(
    "Recall:",
    round(logistic_recall, 4)
)

print(
    "F1 Score:",
    round(logistic_f1, 4)
)


print("\n========== CONFUSION MATRIX ==========")

print(
    confusion_matrix(
        y_test,
        logistic_prediction
    )
)


print("\n========== CLASSIFICATION REPORT ==========")

print(
    classification_report(
        y_test,
        logistic_prediction,
        zero_division=0
    )
)


# ============================================================
# 13. RANDOM FOREST
# ============================================================

print("\n" + "=" * 60)
print("RANDOM FOREST")
print("=" * 60)


random_forest_model = Pipeline(
    steps=[

        (
            "preprocessor",
            preprocessor
        ),

        (
            "classifier",
            RandomForestClassifier(
                n_estimators=300,
                random_state=42,
                class_weight="balanced",
                n_jobs=-1
            )
        )

    ]
)


random_forest_model.fit(
    X_train,
    y_train
)


rf_prediction = random_forest_model.predict(
    X_test
)


rf_accuracy = accuracy_score(
    y_test,
    rf_prediction
)

rf_precision = precision_score(
    y_test,
    rf_prediction,
    zero_division=0
)

rf_recall = recall_score(
    y_test,
    rf_prediction,
    zero_division=0
)

rf_f1 = f1_score(
    y_test,
    rf_prediction,
    zero_division=0
)


print("\n========== RANDOM FOREST PERFORMANCE ==========")

print(
    "Accuracy:",
    round(rf_accuracy, 4)
)

print(
    "Precision:",
    round(rf_precision, 4)
)

print(
    "Recall:",
    round(rf_recall, 4)
)

print(
    "F1 Score:",
    round(rf_f1, 4)
)


print("\n========== CONFUSION MATRIX ==========")

print(
    confusion_matrix(
        y_test,
        rf_prediction
    )
)


print("\n========== CLASSIFICATION REPORT ==========")

print(
    classification_report(
        y_test,
        rf_prediction,
        zero_division=0
    )
)


# ============================================================
# 14. MODEL COMPARISON
# ============================================================

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)


model_comparison = pd.DataFrame({

    "Model": [
        "Baseline",
        "Logistic Regression",
        "Random Forest"
    ],

    "Accuracy": [
        baseline_accuracy,
        logistic_accuracy,
        rf_accuracy
    ],

    "Precision": [
        baseline_precision,
        logistic_precision,
        rf_precision
    ],

    "Recall": [
        baseline_recall,
        logistic_recall,
        rf_recall
    ],

    "F1 Score": [
        baseline_f1,
        logistic_f1,
        rf_f1
    ]

})


print(
    model_comparison.round(4)
)


# ============================================================
# 15. RANDOM FOREST FEATURE IMPORTANCE
# ============================================================

print("\n" + "=" * 60)
print("FEATURE IMPORTANCE")
print("=" * 60)


# Get trained preprocessing step
rf_preprocessor = (
    random_forest_model
    .named_steps["preprocessor"]
)


# Get trained Random Forest
rf_classifier = (
    random_forest_model
    .named_steps["classifier"]
)


# Get encoded feature names
encoded_features = (
    rf_preprocessor
    .named_transformers_["categorical"]
    .get_feature_names_out(
        categorical_features
    )
)


feature_names = np.concatenate(
    [
        encoded_features,
        numerical_features
    ]
)


# Get importance values
importance_values = (
    rf_classifier
    .feature_importances_
)


feature_importance = pd.DataFrame({

    "Feature": feature_names,

    "Importance": importance_values

})


feature_importance = (
    feature_importance
    .sort_values(
        "Importance",
        ascending=False
    )
    .reset_index(drop=True)
)


print("\n========== FEATURE IMPORTANCE ==========")

print(
    feature_importance.head(15)
)


print("\n========== TOP 10 CHURN DRIVERS ==========")

print(
    feature_importance.head(10)
)


# ============================================================
# 16. CUSTOMER CHURN RISK SCORING
# ============================================================

print("\n" + "=" * 60)
print("CUSTOMER CHURN RISK SCORING")
print("=" * 60)


# Predict probability of churn
churn_probability = (
    logistic_model
    .predict_proba(X)[:, 1]
)


# Create risk dataframe
risk_df = df[
    [
        "CustomerID",
        "Contract",
        "TenureMonths",
        "MonthlyCharges",
        "InternetService"
    ]
].copy()


risk_df["Churn_Probability"] = (
    churn_probability
)


# ------------------------------------------------------------
# Risk Categories
# ------------------------------------------------------------

risk_df["Risk_Category"] = pd.cut(

    risk_df["Churn_Probability"],

    bins=[
        0,
        0.30,
        0.60,
        1.00
    ],

    labels=[
        "Low",
        "Medium",
        "High"
    ],

    include_lowest=True

)


# Sort highest risk first
risk_df = risk_df.sort_values(
    "Churn_Probability",
    ascending=False
)


print("\n========== TOP 20 HIGH-RISK CUSTOMERS ==========")

print(
    risk_df.head(20).to_string(
        index=False
    )
)


# ============================================================
# 17. RISK DISTRIBUTION
# ============================================================

print("\n" + "=" * 60)
print("RISK DISTRIBUTION")
print("=" * 60)


risk_distribution = (
    risk_df["Risk_Category"]
    .value_counts()
)


print(
    risk_distribution
)


print("\nRisk Percentage:")

print(
    risk_df["Risk_Category"]
    .value_counts(
        normalize=True
    )
    .mul(100)
    .round(2)
)


# ============================================================
# 18. HIGH-RISK CUSTOMER PROFILE
# ============================================================

print("\n" + "=" * 60)
print("HIGH-RISK CUSTOMER PROFILE")
print("=" * 60)


high_risk = risk_df[
    risk_df["Risk_Category"] == "High"
]


print(
    "\nNumber of high-risk customers:"
)

print(
    len(high_risk)
)


print(
    "\nAverage tenure:"
)

print(
    round(
        high_risk["TenureMonths"].mean(),
        2
    )
)


print(
    "\nAverage monthly charges:"
)

print(
    round(
        high_risk["MonthlyCharges"].mean(),
        2
    )
)


# ------------------------------------------------------------
# Contract Distribution
# ------------------------------------------------------------

print(
    "\n========== HIGH-RISK CONTRACT DISTRIBUTION =========="
)


print(
    high_risk["Contract"]
    .value_counts(
        normalize=True
    )
    .mul(100)
    .round(2)
)


# ------------------------------------------------------------
# Internet Service Distribution
# ------------------------------------------------------------

print(
    "\n========== HIGH-RISK INTERNET SERVICE =========="
)


print(
    high_risk["InternetService"]
    .value_counts(
        normalize=True
    )
    .mul(100)
    .round(2)
)


# ============================================================
# 19. EXECUTIVE BUSINESS INSIGHTS
# ============================================================

print("\n" + "=" * 60)
print("EXECUTIVE BUSINESS INSIGHTS")
print("=" * 60)


print(
    "\n1. The dataset contains",
    len(df),
    "customers."
)


print(
    "\n2. Overall churn rate:",
    round(
        df["Churn_Flag"].mean() * 100,
        2
    ),
    "%"
)


print(
    "\n3. High-risk customers:",
    len(high_risk),
    "(",
    round(
        len(high_risk) / len(df) * 100,
        2
    ),
    "%)"
)


print(
    "\n4. High-risk customers have an average tenure of",
    round(
        high_risk["TenureMonths"].mean(),
        2
    ),
    "months."
)


print(
    "\n5. High-risk customers have average monthly charges of",
    round(
        high_risk["MonthlyCharges"].mean(),
        2
    )
)


print(
    "\n6. Month-to-month contracts represent",
    round(
        (
            high_risk["Contract"]
            .value_counts(
                normalize=True
            )
            .get(
                "Month-to-month",
                0
            ) * 100
        ),
        2
    ),
    "% of high-risk customers."
)


print(
    "\n7. Fiber optic customers represent",
    round(
        (
            high_risk["InternetService"]
            .value_counts(
                normalize=True
            )
            .get(
                "Fiber optic",
                0
            ) * 100
        ),
        2
    ),
    "% of high-risk customers."
)


# ============================================================
# 20. BUSINESS RECOMMENDATIONS
# ============================================================

print("\n" + "=" * 60)
print("BUSINESS RECOMMENDATIONS")
print("=" * 60)


print("""
1. TARGET MONTH-TO-MONTH CUSTOMERS

Prioritize high-risk customers on month-to-month contracts
for targeted retention campaigns and longer-term contract
conversion offers.


2. STRENGTHEN EARLY CUSTOMER ENGAGEMENT

The high-risk segment has relatively low average tenure.
Customer engagement should therefore be strengthened during
the early stages of the customer relationship.


3. INVESTIGATE FIBER OPTIC CHURN

A large proportion of high-risk customers use Fiber optic
services. Management should investigate pricing, service
quality, customer expectations and support issues.


4. PRIORITIZE RETENTION RESOURCES

Use predicted churn probability to prioritize customers
requiring proactive retention efforts instead of applying
the same strategy to every customer.


5. MONITOR CUSTOMER CHARGES

Monthly charges were among the strongest model features.
The business should assess whether customers perceive
sufficient value relative to the charges they pay.
""")


# ============================================================
# 21. FINAL PROJECT SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("PROJECT COMPLETE")
print("=" * 60)

print("""
Customer Churn Prediction & Risk Analysis completed.

Workflow completed:

✓ Data Audit
✓ Exploratory Data Analysis
✓ Feature Engineering
✓ Machine Learning Preparation
✓ Baseline Model
✓ Logistic Regression
✓ Random Forest
✓ Model Comparison
✓ Feature Importance
✓ Customer Risk Scoring
✓ High-Risk Customer Profiling
✓ Business Recommendations
""")

print("=" * 60)
print("END OF ANALYSIS")
print("=" * 60)
