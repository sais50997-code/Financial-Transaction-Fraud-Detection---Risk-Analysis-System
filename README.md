# Financial Transaction Fraud Detection & Risk Analysis System

A machine learning project for detecting potentially fraudulent financial transactions and analyzing transaction risk using the PaySim synthetic mobile-money transaction dataset.

## Project Objective

The objective of this project is to:

- Identify potentially fraudulent transactions.
- Estimate fraud probability.
- Convert probability into a Fraud/Normal decision using a configurable threshold.
- Analyze transaction risk.
- Store transaction history.
- Provide a Streamlit dashboard for monitoring.

## Dataset

This project uses the PaySim synthetic mobile-money transaction dataset.

The dataset contains approximately 6.36 million transactions.

The target variable is:

- `isFraud = 0` → Normal transaction
- `isFraud = 1` → Fraudulent transaction

Fraud represents approximately 0.13% of all transactions, making this a highly imbalanced classification problem.

## Dataset Columns

| Column | Description |
|---|---|
| `step` | Simulated time step |
| `type` | Transaction type |
| `amount` | Transaction amount |
| `nameOrig` | Sender account identifier |
| `oldbalanceOrg` | Sender balance before transaction |
| `newbalanceOrig` | Sender balance after transaction |
| `nameDest` | Receiver account identifier |
| `oldbalanceDest` | Receiver balance before transaction |
| `newbalanceDest` | Receiver balance after transaction |
| `isFraud` | Fraud target variable |
| `isFlaggedFraud` | Existing rule-based fraud flag |

## Exploratory Data Analysis

The analysis examined:

- Fraud vs normal transactions
- Transaction type distribution
- Fraud rate by transaction type
- Transaction amount patterns
- Sender and receiver balances
- Fraud behavior over time
- Existing fraud flags

Fraudulent transactions in the dataset were concentrated in `CASH_OUT` and `TRANSFER` transactions.

## Data Preprocessing

The following columns were excluded from the main model:

```text
nameOrig
nameDest
newbalanceOrig
newbalanceDest
isFlaggedFraud

Account identifiers were removed because of their high cardinality.

Post-transaction balance fields were excluded because they may not be available at the fraud-decision point.

isFlaggedFraud was excluded from the main model because it is an existing rule-based fraud indicator.

The type column was converted using one-hot encoding.

Feature Engineering

Three additional features were created:

amount_to_origin_balance
amount_to_destination_balance
log_amount

The final model contains 12 features:

step
amount
oldbalanceOrg
oldbalanceDest
type_CASH_IN
type_CASH_OUT
type_DEBIT
type_PAYMENT
type_TRANSFER
amount_to_origin_balance
amount_to_destination_balance
log_amount
Handling Class Imbalance

The dataset contains approximately 774 normal transactions for every fraudulent transaction in the training data.

Class weighting was used for Logistic Regression and Random Forest to give greater importance to the minority fraud class during training.

Machine Learning Models

The following models were evaluated:

Logistic Regression
Random Forest
XGBoost

The models were evaluated using:

Precision
Recall
F1-score
ROC-AUC
PR-AUC

PR-AUC was included because fraud is a rare positive class.

Model Results
Random Stratified Validation
Model	Precision	Recall	F1-score	ROC-AUC	PR-AUC
Logistic Regression	0.65%	95.44%	1.30%	96.12%	8.37%
Random Forest	99.88%	99.70%	99.79%	99.88%	99.76%
XGBoost	91.93%	86.00%	88.87%	99.56%	92.52%
Time-Based Validation

A time-based validation was also performed using the step column.

Training: step <= 355
Testing : step > 355
Time-Based Results
Model	Precision	Recall	F1-score	ROC-AUC	PR-AUC
Random Forest	99.95%	99.84%	99.89%	~100%	99.99%
XGBoost	97.93%	73.58%	84.03%	98.70%	94.10%
Feature Importance

Random Forest feature importance showed that the most influential features included:

amount_to_origin_balance
oldbalanceOrg
amount_to_destination_balance
amount
type_CASH_OUT
type_PAYMENT
log_amount

The strong influence of amount_to_origin_balance is a dataset-specific pattern observed in PaySim.

Feature importance represents model usage and does not establish causation.

Fraud Detection Threshold

The application converts the model's fraud probability into a Fraud/Normal decision using a configurable threshold.

Probability >= threshold → Fraud
Probability < threshold  → Normal

The threshold can be adjusted in the Streamlit application.

Risk Analysis

The application provides three risk levels:

Low    → probability < 30%
Medium → probability >= 30% and < 70%
High   → probability >= 70%

The application also displays rule-based risk indicators such as:

High transaction amount
Transaction amount close to the origin balance
Transaction type associated with fraud in the PaySim dataset
Zero destination balance before the transaction

These are application-level risk indicators and are separate from the internal reasoning of the Random Forest model.

Streamlit Application

The application allows users to enter:

Transaction type
Step
Transaction amount
Origin balance
Destination balance
Decision threshold

The application then:

Prepares the transaction features.
Loads the trained Random Forest model.
Calculates fraud probability.
Applies the selected threshold.
Determines the risk level.
Displays risk indicators.
Stores the transaction history.
Dashboard

The dashboard displays:

Total transactions
Fraud transactions
Normal transactions
Fraud rate
Fraud vs Normal chart
Risk distribution
Transaction history

Transaction history is stored in:

app/transaction_history.csv
Project Structure
Financial Transaction Fraud Detection & Risk Analysis System/
│
├── app/
│   ├── app.py
│   └── transaction_history.csv
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   ├── random_forest_fraud_model.pkl
│   └── feature_names.pkl
│
├── notebooks/
│   ├── 01_data_inspection.ipynb
│   ├── 02_exploratory_data_analysis.ipynb
│   ├── 03_data_preprocessing.ipynb
│   ├── 04_feature_engineering.ipynb
│   ├── 05_model_training.ipynb
│   ├── 06_model_comparison.ipynb
│   └── 07_model_interpretation.ipynb
│
├── reports/
│   ├── figures/
│   └── model_results/
│
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   └── data_loader.py
│   ├── features/
│   │   ├── __init__.py
│   │   └── feature_engineering.py
│   └── prediction/
│       ├── __init__.py
│       └── predict.py
│
├── tests/
│   └── test_pipeline.py
│
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
Technologies Used
Python
Pandas
NumPy
Matplotlib
Seaborn
Scikit-learn
XGBoost
Joblib
Streamlit
Jupyter Notebook
Visual Studio Code
Git
GitHub
Testing

The project includes pipeline tests covering:

Dataset loading
Feature preparation
Model loading
Prediction generation
Fraud probability generation
Single transaction preparation
Transaction type validation
Input validation

Run the tests from the project root:

python -m tests.test_pipeline
Running the Project
Install Dependencies
pip install -r requirements.txt
Run Tests
python -m tests.test_pipeline
Run Streamlit
streamlit run app/app.py
Limitations
PaySim is a synthetic mobile-money dataset.
Model performance should not be interpreted as real-world banking performance.
Some highly predictive patterns may be specific to the PaySim dataset.
Fraud detection involves severe class imbalance.
Risk-level thresholds are application-specific and are not industry standards.
The model has not been validated for production financial fraud detection.
Future Improvements
SHAP-based model explanations
Hyperparameter tuning
Model calibration
Advanced fraud features
REST API deployment
Database integration
Cloud deployment
Model monitoring
Detection of model drift
Automated model retraining


 ** Disclaimer :

This project is intended for educational and portfolio purposes.

The dataset is synthetic and the model has not been validated for production financial fraud detection.

The predictions and risk indicators should not be used to make real financial decisions.