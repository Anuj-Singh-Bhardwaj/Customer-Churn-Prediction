# Customer Churn Prediction

## Problem Statement
Predict telecom customer churn using machine learning. The goal is to identify at-risk customers before they leave, enabling targeted retention strategies.

## Dataset
- **Source:** Kaggle - Telco Customer Churn
- **Records:** 7,043 customers
- **Features:** 21 columns including demographics, account info, and services subscribed
- **Target:** Churn (Yes / No)

## Approach

1. **Exploratory Data Analysis (EDA)**
   - Checked for missing values and duplicates
   - Analyzed correlation between features and churn
   - Identified key churn indicators: contract type, tenure, monthly charges

2. **Data Preprocessing**
   - Dropped irrelevant features (customerID, name)
   - Label Encoding for categorical variables
   - Train-test split (70:30)

3. **Handling Class Imbalance**
   - Detected 73:27 class imbalance (majority class: No Churn)
   - Applied SMOTE **only on training data** to prevent data leakage
   - Synthetic oversampling of minority class

4. **Model Training & Evaluation**
   - Models tested: Logistic Regression, Random Forest, SVM, XGBoost
   - Optimized for **Recall** — minimizing false negatives is critical for churn use cases
   - Hyperparameter tuning via GridSearchCV

## Results

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Random Forest | **84%** | 0.76 | **0.78** | **0.78** |

- **Final Model:** Random Forest
- **Key Metric:** Recall (we prioritize catching actual churners over avoiding false alarms)

## Business Insights

- **Month-to-month contracts** show significantly higher churn risk
- Customers with **tenure < 12 months** are most vulnerable
- **Higher monthly charges** correlate with increased churn probability
- Recommendation: Target month-to-month customers with annual contract incentives

## Tools Used

- Python (Pandas, NumPy, Scikit-learn, Imbalanced-learn, Matplotlib, Seaborn)
- Jupyter Notebook
- SMOTE (Synthetic Minority Oversampling Technique)

## How to Run

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Open `customer_churn.ipynb` in Jupyter Notebook
4. Run all cells

## Author

**Anuj Singh Bhardwaj**
- [LinkedIn](https://www.linkedin.com/in/anuj-singh-a61646234)
- [GitHub](https://github.com/Anuj-Singh-Bhardwaj)
