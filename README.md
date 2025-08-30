# Telecom Customer Churn Prediction

This project focuses on predicting customer churn for a telecom company using machine learning techniques. The dataset contains customer details including service usage, billing information, and demographics, and the goal is to identify customers who are likely to churn.

## Objective

To build a robust classification model that predicts whether a customer will churn based on historical data. The solution also addresses challenges with imbalanced datasets (80% churn rate).

## Problem Statement

Customer churn is a major issue in the telecom industry, leading to significant revenue losses. The aim is to proactively identify high-risk customers and implement retention strategies using machine learning.

## Workflow

1. Data Collection  
   Historical telecom customer data in CSV format.

2. Exploratory Data Analysis (EDA)  
   - Univariate and bivariate analysis  
   - Distribution of churn vs non-churn  
   - Correlation analysis

3. Data Preprocessing  
   - Handling missing values  
   - Encoding categorical variables  
   - Feature scaling  
   - Class imbalance handling (e.g., SMOTE, undersampling)

4. Modeling  
   - Logistic Regression  
   - Random Forest  
   - XGBoost  
   - Evaluation metrics: Accuracy, Precision, Recall, F1-score, ROC-AUC

5. Model Evaluation  
   - Confusion matrix  
   - Classification report  
   - ROC curve  
   - Feature importance

6. Model Export  
   - Save the best-performing model using pickle (`.pkl`)

## Dataset Overview

Typical columns include:
- Customer ID  
- Tenure  
- Monthly Charges  
- Contract Type  
- Internet Service  
- Payment Method  
- Total Charges  
- Churn (target label)

## Tech Stack

- Python (Pandas, NumPy, Scikit-learn, XGBoost)
- Jupyter Notebooks
- Matplotlib and Seaborn
- Imbalanced-learn (for SMOTE)
- Pickle

## Results

- Best-performing model: XGBoost   
- Accuracy: 91%  
- ROC-AUC: 0.86  
- Class imbalance handled using [SMOTE/other technique]

## Installation

```bash
git clone https://github.com/your-username/telecom-churn-prediction.git
cd telecom-churn-prediction
pip install -r requirements.txt
streamlit run app.py
```
After this upload the input.csv file from the Data folder
