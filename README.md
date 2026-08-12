 Customer Churn Prediction & Risk Analysis

Project Overview

Customer churn is a major business challenge because losing existing customers can negatively affect revenue and customer lifetime value.

This project uses customer demographic, service, contract and billing data to analyze churn patterns, identify key churn drivers and build machine learning models that predict customers who may be at risk of leaving.

The project follows an end-to-end data analytics and machine learning workflow:

Data Audit → Exploratory Data Analysis → Feature Engineering → Machine Learning → Model Evaluation → Churn Risk Scoring → Business Recommendations

---

 Business Objective

The objectives of this project are to:

- Understand customer churn patterns.
- Identify customer segments with higher churn risk.
- Determine the major factors associated with churn.
- Compare machine learning models for churn prediction.
- Identify individual customers with high predicted churn risk.
- Provide actionable recommendations for customer retention.

---

 Dataset

The dataset contains:

- 1,200 customers
- 16 original variables
- 0 missing values
- 0 duplicate records

Key Variables

- CustomerID
- Gender
- SeniorCitizen
- Partner
- Dependents
- TenureMonths
- Contract
- InternetService
- PaperlessBilling
- PaymentMethod
- OnlineSecurity
- TechSupport
- StreamingService
- MonthlyCharges
- TotalCharges
- Churn

---

 Data Quality Assessment

The dataset was audited before modelling.

 Results

- Rows: 1,200
- Columns: 16
- Missing values: 0
- Duplicate records: 0

 Churn Distribution

| Churn Status | Customers | Percentage |
|---|---:|---:|
| No | 1,059 | 88.25% |
| Yes | 141 | 11.75% |

The dataset is imbalanced, with significantly fewer churned customers than retained customers.

Therefore, accuracy alone was not considered sufficient for evaluating model performance.

---

 Exploratory Data Analysis

 Churn by Contract Type

| Contract | Churn |
|---|---:|
| Month-to-month | 17.08% |
| One year | 7.17% |
| Two year | 2.07% |

Customers on month-to-month contracts showed substantially higher churn than customers on longer-term contracts.

 Churn by Internet Service

| Internet Service | Churn |
|---|---:|
| DSL | 8.12% |
| Fiber optic | 15.82% |
| No internet | 8.79% |

Fiber optic customers showed the highest churn rate among the internet service categories.

 Customer Value Comparison

| Customer Group | Average Tenure | Avg Monthly Charges | Avg Total Charges |
|---|---:|---:|---:|
| Stayed | 37.93 months | 54.02 | 2,062.41 |
| Churned | 25.77 months | 57.92 | 1,532.29 |

Churned customers had lower average tenure and total charges, while their average monthly charges were higher.

---

Feature Engineering

Four new features were created:

- Churn_Flag
- Tenure_Group
- Charge_Group
- Total_Services

These features were created to improve the model's ability to identify customer behaviour and churn patterns.

---

 Machine Learning Preparation

The data was divided into:

- Training set: 960 observations
- Testing set: 240 observations

The churn rate was approximately:

- Training: 11.77%
- Testing: 11.67%

Categorical variables were encoded and numerical variables were prepared for machine learning.

---

Machine Learning Models

Three approaches were evaluated:

1. Baseline Model
2. Logistic Regression
3. Random Forest

Model Comparison

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|
| Baseline | 88.33% | 0.00% | 0.00% | 0.00% |
| Logistic Regression | 70.00% | 22.50% | 64.29% | 33.33% |
| Random Forest | 85.00% | 35.71% | 35.71% | 35.71% |

---

Model Interpretation

The baseline model achieved 88.33% accuracy by predicting all customers as non-churners.

However, it detected none of the actual churners.

Therefore:

- Recall = 0%
- F1 Score = 0%

This demonstrates why accuracy alone can be misleading when dealing with imbalanced classification data.

 Logistic Regression

Logistic Regression achieved:

- Accuracy: 70.00%
- Precision: 22.50%
- Recall: 64.29%
- F1 Score: 33.33%

It identified 18 of the 28 churners in the test dataset.

 Random Forest

Random Forest achieved:

- Accuracy: 85.00%
- Precision: 35.71%
- Recall: 35.71%
- F1 Score: 35.71%

It provided higher accuracy and precision than Logistic Regression but detected fewer actual churners.

Model Selection

For a customer retention use case, Logistic Regression was selected as the operational model because identifying potential churners is a priority.

The model achieved a recall of 64.29%, compared with 35.71% for Random Forest.

---

 Churn Drivers

Random Forest feature importance was used to identify the most influential features in the model.

 Top Churn Drivers

| Feature | Importance |
|---|---:|
| TotalCharges | 13.38% |
| TenureMonths | 13.25% |
| MonthlyCharges | 11.55% |
| Contract - Month-to-month | 8.17% |
| Contract - Two year | 5.55% |
| InternetService - Fiber optic | 3.43% |
| Tenure Group - 49-72 Months | 2.92% |
| PaymentMethod - Electronic check | 2.28% |
| InternetService - DSL | 2.27% |
| Total Services | 2.22% |

The results indicate that customer tenure, charges, contract type and internet service are important factors associated with churn risk.

---

Customer Churn Risk Scoring

The Logistic Regression model was used to estimate churn probability for all 1,200 customers.

Customers were grouped into three risk categories:

- Low Risk
- Medium Risk
- High Risk

Risk Distribution

| Risk Category | Customers | Percentage |
|---|---:|---:|
| Low | 462 | 38.5% |
| Medium | 420 | 35.0% |
| High | 318 | 26.5% |

---

 High-Risk Customer Profile

The model identified 318 customers as high risk.

### Profile

- Average tenure: 22.86 months
- Average monthly charges: 58.54
- Month-to-month contract: 91.51%
- One-year contract: 8.49%
- Fiber optic users: 70.13%
- DSL users: 15.72%
- No internet service: 14.15%

The high-risk segment is strongly concentrated among month-to-month customers and fiber optic users.

---

# Business Recommendations

## 1. Target Month-to-Month Customers

Develop retention campaigns that encourage high-risk month-to-month customers to move to longer-term contracts.

Possible incentives could include discounts, loyalty benefits or improved service packages.

## 2. Strengthen Early Customer Engagement

The high-risk segment has an average tenure of 22.86 months.

Customer engagement should therefore be strengthened during the early stages of the customer relationship.

## 3. Investigate Fiber Optic Churn

Fiber optic customers represent 70.13% of the high-risk segment.

The business should investigate whether pricing, service quality, customer expectations or support issues contribute to this risk.

## 4. Prioritize Retention Resources

Instead of applying the same retention strategy to every customer, the business can prioritize customers with high predicted churn probability.

## 5. Monitor Customer Charges

Monthly and total charges were among the strongest model features.

Management should investigate whether customers with higher charges perceive sufficient value from the services they receive.

---

Business Impact

The project demonstrates how machine learning can move customer retention from a reactive approach to a proactive approach.

Instead of waiting until customers leave, the business can:

1. Identify customers at risk.
2. Prioritize high-risk customers.
3. Understand the characteristics of those customers.
4. Design targeted retention strategies.
5. Monitor churn risk over time.

---

 Tools & Technologies

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Excel
- GitHub

---

 Machine Learning Techniques

- Logistic Regression
- Random Forest
- Feature Engineering
- Classification
- Model Evaluation
- Feature Importance
- Probability-based Risk Scoring

---

 Key Takeaways

The analysis identified several important patterns:

- Month-to-month customers have substantially higher churn.
- Fiber optic customers show higher churn than other internet service groups.
- Churned customers have lower average tenure.
- Customer charges and tenure are among the strongest model features.
- Logistic Regression provides higher churn recall than Random Forest.
- 318 customers were classified as high predicted churn risk.
- High-risk customers are predominantly month-to-month customers and fiber optic users.

---

 Project Outcome

This project demonstrates an end-to-end approach to customer churn analytics, combining exploratory data analysis, feature engineering, machine learning, risk scoring and business recommendations.

The final output can support a hypothetical customer retention team in prioritizing customers who may require proactive engagement.

---

Author

Obarayo Oluwaseyi Maryanne

Data Analyst | Business Intelligence | Data Analytics
