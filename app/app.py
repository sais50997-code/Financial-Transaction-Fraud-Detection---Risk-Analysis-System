import streamlit as st
import joblib
import os
import pandas as pd
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.feature_engineering import prepare_single_transaction


st.set_page_config(
    page_title="Financial Fraud Detection",
    page_icon="🔍"
)


st.title("Financial Transaction Fraud Detection")


model = joblib.load(
    "models/random_forest_fraud_model.pkl"
)


st.success("Fraud detection model loaded successfully.")
# Transaction inputs
st.subheader("Transaction Details")

transaction_type = st.selectbox(
    "Transaction Type",
    [
        "CASH_IN",
        "CASH_OUT",
        "DEBIT",
        "PAYMENT",
        "TRANSFER"
    ]
)

step = st.number_input(
    "Step",
    min_value=1,
    value=1
)

amount = st.number_input(
    "Transaction Amount",
    min_value=0.0,
    value=1000.0
)

oldbalance_org = st.number_input(
    "Origin Balance Before Transaction",
    min_value=0.0,
    value=1000.0
)

oldbalance_dest = st.number_input(
    "Destination Balance Before Transaction",
    min_value=0.0,
    value=1000.0
)
# Fraud decision threshold
threshold = st.slider(
    "Fraud Detection Threshold",
    min_value=0.1,
    max_value=0.9,
    value=0.4,
    step=0.1
)


if st.button("Check Transaction", key="check_transaction"):

    transaction_features = prepare_single_transaction(
        transaction_type,
        step,
        amount,
        oldbalance_org,
        oldbalance_dest
    )

    probability = model.predict_proba(
        transaction_features
    )[0, 1]

    # Fraud / Normal decision
    if probability >= threshold:
        prediction = 1
    else:
        prediction = 0

    # Risk level
    if probability >= 0.7:
        risk_level = "High"
    elif probability >= 0.3:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    # Risk explanation
    risk_reasons = []

    if amount > 100000:
        risk_reasons.append(
            "High transaction amount"
        )

    amount_origin_ratio = (
        amount / (oldbalance_org + 1)
    )

    if amount_origin_ratio >= 0.9:
        risk_reasons.append(
            "Transaction amount is close to the origin balance"
        )

    if transaction_type in ["TRANSFER", "CASH_OUT"]:
        risk_reasons.append(
            "Transaction type is commonly associated with fraud in the PaySim dataset"
        )

    if oldbalance_dest == 0:
        risk_reasons.append(
            "Destination balance before transaction is zero"
        )

    # Results
    st.write(
        f"Fraud Probability: **{probability:.2%}**"
    )

    st.write(
        f"Decision Threshold: **{threshold:.0%}**"
    )

    st.write(
        f"Risk Level: **{risk_level}**"
    )

    st.subheader("Risk Analysis")

    if risk_reasons:
        for reason in risk_reasons:
            st.write("•", reason)
    else:
        st.write(
            "No major rule-based risk indicators detected."
        )

    if prediction == 1:
        st.error("🚨 Fraud Detected")
    else:
        st.success("✅ Normal Transaction")

    # Save transaction history
    history_file = "app/transaction_history.csv"

    transaction_record = {
        "Transaction Type": transaction_type,
        "Step": step,
        "Amount": amount,
        "Origin Balance": oldbalance_org,
        "Destination Balance": oldbalance_dest,
        "Fraud Probability": probability,
        "Risk Level": risk_level,
        "Decision": "Fraud" if prediction == 1 else "Normal"
    }

    if os.path.exists(history_file):
        history_df = pd.read_csv(history_file)

        history_df = pd.concat(
            [
                history_df,
                pd.DataFrame([transaction_record])
            ],
            ignore_index=True
        )
    else:
        history_df = pd.DataFrame(
            [transaction_record]
        )

    history_df.to_csv(
        history_file,
        index=False
    )
    st.subheader("Risk Analysis Dashboard")

history_file = "app/transaction_history.csv"

if os.path.exists(history_file):

    history_df = pd.read_csv(history_file)

    if not history_df.empty:

        total_transactions = len(history_df)

        fraud_count = (
            history_df["Decision"] == "Fraud"
        ).sum()

        normal_count = (
            history_df["Decision"] == "Normal"
        ).sum()

        high_risk_count = (
            history_df["Risk Level"] == "High"
        ).sum()

        fraud_percentage = (
            fraud_count / total_transactions * 100
        )

        # Dashboard metrics
        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Total Transactions",
            total_transactions
        )

        col2.metric(
            "Fraud Transactions",
            fraud_count
        )

        col3.metric(
            "Normal Transactions",
            normal_count
        )

        col4.metric(
            "Fraud Rate",
            f"{fraud_percentage:.2f}%"
        )

        # Fraud vs Normal
        st.write("### Fraud vs Normal")

        decision_counts = (
            history_df["Decision"]
            .value_counts()
        )

        st.bar_chart(decision_counts)

        # Risk distribution
        st.write("### Risk Distribution")

        risk_counts = (
            history_df["Risk Level"]
            .value_counts()
        )

        st.bar_chart(risk_counts)

        # Transaction history
        st.write("### Transaction History")

        st.dataframe(
            history_df,
            use_container_width=True
        )

    else:

        st.info(
            "No transactions recorded yet."
        )

else:

    st.info(
        "No transaction history available yet."
    )