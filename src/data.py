from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split


def load_data():
    """Load the Wine dataset and return features (X) and target (y)."""
    wine = load_wine()
    return wine.data, wine.target


def split_data(X, y):
    """Split data into train and test sets (70/30, stratified)."""
    return train_test_split(X, y, test_size=0.3, random_state=12, stratify=y)
