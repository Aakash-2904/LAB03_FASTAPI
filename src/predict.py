import os
import joblib

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "wine_model.pkl")
CLASS_NAMES = ["class_0", "class_1", "class_2"]


def predict_data(X):
    """Predict wine class (0, 1, 2) for input rows X (list of 13-feature lists)."""
    model = joblib.load(MODEL_PATH)
    return model.predict(X), model.predict_proba(X)
