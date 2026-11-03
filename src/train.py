import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from data import load_data, split_data

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "wine_model.pkl")


def fit_model(X_train, y_train):
    """Train a RandomForest classifier and save it to model/wine_model.pkl."""
    clf = RandomForestClassifier(n_estimators=200, random_state=12)
    clf.fit(X_train, y_train)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    return clf


if __name__ == "__main__":
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    model = fit_model(X_train, y_train)
    print(f"Test accuracy: {accuracy_score(y_test, model.predict(X_test)):.4f}")
    print(f"Model saved to {os.path.abspath(MODEL_PATH)}")
