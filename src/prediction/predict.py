import joblib


def load_model(model_path):
    model = joblib.load(model_path)

    return model


def predict_fraud(model, X):
    prediction = model.predict(X)
    probability = model.predict_proba(X)[:, 1]

    return prediction, probability