from src.data.data_loader import load_data
from src.features.feature_engineering import prepare_features
from src.prediction.predict import load_model, predict_fraud


# 1. Load data
file_path = "data/raw/PS_20174392719_1491204439457_log.csv"

df = load_data(file_path)

print("Data loaded:", df.shape)


# 2. Prepare features
X = prepare_features(df)

print("Features prepared:", X.shape)


# 3. Load saved model
model = load_model(
    "models/random_forest_fraud_model.pkl"
)

print("Model loaded")

# 4. Make predictions on a small sample
X_sample = X.head(5)

prediction, probability = predict_fraud(
    model,
    X_sample
)

print("Predictions:", prediction)
print("Fraud probabilities:", probability)


from src.features.feature_engineering import prepare_single_transaction


# Test single transaction feature preparation
X_single = prepare_single_transaction(
    transaction_type="TRANSFER",
    step=100,
    amount=50000,
    oldbalance_org=60000,
    oldbalance_dest=10000
)

print("\nSingle transaction features:")
print(X_single)

print("\nShape:", X_single.shape)

assert X_single.shape == (1, 12)

print("Single transaction feature test: PASSED")
# Test model prediction
prediction, probability = predict_fraud(
    model,
    X_single
)

print("\nPrediction:", prediction)
print("Fraud probability:", probability)

assert prediction.shape == (1,)
assert probability.shape == (1,)

assert prediction[0] in [0, 1]
assert 0 <= probability[0] <= 1

print("Model prediction test: PASSED")
# Test all transaction types

transaction_types = [
    "CASH_IN",
    "CASH_OUT",
    "DEBIT",
    "PAYMENT",
    "TRANSFER"
]

for transaction_type in transaction_types:

    X_single = prepare_single_transaction(
        transaction_type=transaction_type,
        step=100,
        amount=50000,
        oldbalance_org=60000,
        oldbalance_dest=10000
    )

    assert X_single.shape == (1, 12)

    prediction, probability = predict_fraud(
        model,
        X_single
    )

    assert prediction[0] in [0, 1]
    assert 0 <= probability[0] <= 1

print("All transaction type tests: PASSED")
# Test invalid transaction amount

# Test invalid transaction amount

try:
    prepare_single_transaction(
        transaction_type="TRANSFER",
        step=100,
        amount=-50000,
        oldbalance_org=60000,
        oldbalance_dest=10000
    )

    print("Invalid amount test: FAILED")

except ValueError as e:
    print("Invalid amount test: PASSED")
    print("Error:", e)

# Test invalid step

try:
    prepare_single_transaction(
        transaction_type="TRANSFER",
        step=0,
        amount=50000,
        oldbalance_org=60000,
        oldbalance_dest=10000
    )

    print("Invalid step test: FAILED")

except ValueError as e:
    print("Invalid step test: PASSED")
    print("Error:", e)


# Test invalid origin balance

try:
    prepare_single_transaction(
        transaction_type="TRANSFER",
        step=100,
        amount=50000,
        oldbalance_org=-1000,
        oldbalance_dest=10000
    )

    print("Invalid origin balance test: FAILED")

except ValueError as e:
    print("Invalid origin balance test: PASSED")
    print("Error:", e)


# Test invalid destination balance

try:
    prepare_single_transaction(
        transaction_type="TRANSFER",
        step=100,
        amount=50000,
        oldbalance_org=60000,
        oldbalance_dest=-1000
    )

    print("Invalid destination balance test: FAILED")

except ValueError as e:
    print("Invalid destination balance test: PASSED")
    print("Error:", e)

# Test invalid transaction type

try:
    prepare_single_transaction(
        transaction_type="INVALID",
        step=100,
        amount=50000,
        oldbalance_org=60000,
        oldbalance_dest=10000
    )

    print("Invalid transaction type test: FAILED")

except ValueError as e:
    print("Invalid transaction type test: PASSED")
    print("Error:", e)