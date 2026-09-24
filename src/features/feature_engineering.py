import pandas as pd
import numpy as np


def prepare_features(df):
    # Separate features and target
    X = df.drop(columns=["isFraud"])

    # Remove features that we decided not to use
    features_to_drop = [
        "nameOrig",
        "nameDest",
        "newbalanceOrig",
        "newbalanceDest",
        "isFlaggedFraud"
    ]

    X = X.drop(columns=features_to_drop)

    # Convert transaction type into numerical columns
    X = pd.get_dummies(
        X,
        columns=["type"],
        dtype=int
    )

    # Create engineered features
    X["amount_to_origin_balance"] = (
        X["amount"] / (X["oldbalanceOrg"] + 1)
    )

    X["amount_to_destination_balance"] = (
        X["amount"] / (X["oldbalanceDest"] + 1)
    )

    X["log_amount"] = np.log1p(X["amount"])

    return X
def prepare_single_transaction(
    transaction_type,
    step,
    amount,
    oldbalance_org,
    oldbalance_dest
):
    if amount < 0:
        raise ValueError("Transaction amount cannot be negative")

    if step < 1:
        raise ValueError("Step must be at least 1")

    if oldbalance_org < 0:
        raise ValueError(
            "Origin balance cannot be negative"
        )

    if oldbalance_dest < 0:
        raise ValueError(
            "Destination balance cannot be negative"
        )
    valid_transaction_types = [
        "CASH_IN",
        "CASH_OUT",
        "DEBIT",
        "PAYMENT",
        "TRANSFER"
    ]

    if transaction_type not in valid_transaction_types:
        raise ValueError(
            "Invalid transaction type"
        )
    
    # Create one transaction as a DataFrame
    data = pd.DataFrame([{
        "step": step,
        "type": transaction_type,
        "amount": amount,
        "oldbalanceOrg": oldbalance_org,
        "oldbalanceDest": oldbalance_dest
    }])

    # Convert transaction type into dummy variables
    data = pd.get_dummies(
        data,
        columns=["type"],
        dtype=int
    )

    # Make sure all transaction-type columns exist
    for transaction_type_column in [
        "type_CASH_IN",
        "type_CASH_OUT",
        "type_DEBIT",
        "type_PAYMENT",
        "type_TRANSFER"
    ]:
        if transaction_type_column not in data.columns:
            data[transaction_type_column] = 0

    # Create engineered features
    data["amount_to_origin_balance"] = (
        data["amount"] / (data["oldbalanceOrg"] + 1)
    )

    data["amount_to_destination_balance"] = (
        data["amount"] / (data["oldbalanceDest"] + 1)
    )

    data["log_amount"] = np.log1p(data["amount"])

    # Keep exactly the same feature order as model training
    feature_order = [
        "step",
        "amount",
        "oldbalanceOrg",
        "oldbalanceDest",
        "type_CASH_IN",
        "type_CASH_OUT",
        "type_DEBIT",
        "type_PAYMENT",
        "type_TRANSFER",
        "amount_to_origin_balance",
        "amount_to_destination_balance",
        "log_amount"
    ]

    data = data[feature_order]

    return data