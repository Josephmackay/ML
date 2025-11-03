

# 📘 Loan Default Prediction Project Documentation

## 1️⃣ Project Overview

The **Loan Default Prediction** project aims to build a machine learning model that predicts whether a borrower is likely to default on a loan based on their personal, financial, and loan-related attributes.

The goal is to help financial institutions assess risk, reduce losses, and make data-driven lending decisions.

---

## 2️⃣ Data Source and Collection

The dataset used for this project was obtained from Kaggle.

It contains information such as:

* Applicant’s demographic and financial details
* Loan amount and term
* Credit history and income
* Default status (target variable)

---

## 3️⃣ Exploratory Data Analysis (EDA)

Initial exploration and analysis were performed using **Jupyter Notebook**.

### Key Steps:

* Loaded the dataset using `pandas`
* Checked for missing values and data types
* Visualized distributions and relationships using `matplotlib` and `seaborn`
* Identified outliers and inconsistencies
* Examined correlations between features and the target variable


## 4️⃣ Data Preprocessing

Data preprocessing was implemented in `src/feature/preprocess.py` and included:

* **Handling Missing Values:** Using imputation techniques for numeric and categorical columns.
* **Encoding Categorical Variables:** Applied one-hot encoding to categorical features.
* **Feature Scaling:** Standardized numerical features to ensure equal contribution.
* **Outlier Treatment:** Detected and removed extreme values using statistical methods.

---

## 5️⃣ Feature Engineering

Implemented in `src/feature/feature_engineering.py`.

Key transformations:

* Created interaction features such as **income-to-loan ratio**, **credit utilization**, etc.

## 6️⃣ Project Structure Design

To make the project modular and scalable, a well-organized structure was adopted:

```
ML/
│
├── data/
│   └── Loan_default.csv
│
├── src/
│   ├── data/
│   │   └── data_import.py
│   ├── feature/
│   │   ├── preprocess.py
│   │   └── feature_engineering.py
│   ├── model/
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   ├── tune.py
│   │   └── saved_models/
│   ├── logs/
│   │   └── logger_config.py
│   └── main.py
│
├── notebooks/
│   └── Untitled.ipynb
│
└── requirements.txt
```

Each component performs a specific function, allowing easy debugging, updates, and reuse.

---

## 7️⃣ Model Training

The training process is handled in `src/model/train.py`.
Key model used: **Logistic Regression**, chosen for its simplicity and interpretability.

### Steps:

* Split data into training and testing sets using `train_test_split`
* Resampled the data to handle class imbalance (if applicable)
* Trained the logistic regression algorithm
* Saved the trained model using `joblib` in:

  ```
  src/model/saved_models/log_reg_model.pkl
  ```

---

## 8️⃣ Model Evaluation

Model performance was assessed using metrics such as:

* **Accuracy**
* **Precision**
* **Recall**
* **F1-Score**
* **Confusion Matrix**
* **Classification Report**


## 9️⃣ Logging and Monitoring

The logger was configured in:

```
src/logs/logger_config.py
```

Logs are saved in:

```
src/logs/pipeline.log
```
---

## 📈 Results Summary

| Metric    | Score |
| :-------- | :---: |
| Accuracy  | 93.4% |
| Precision |  0.99 |
| Recall    |  0.87 |
| F1-score  |  0.93 |

