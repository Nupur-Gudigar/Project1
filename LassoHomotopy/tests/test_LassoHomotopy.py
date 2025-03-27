import csv
import os
import sys
import numpy as np

# Add parent directory to sys.path to import model
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
from model.LassoHomotopy import LassoHomotopyModel

def test_predict():
    model = LassoHomotopyModel()

    # Build path to CSV file relative to current file
    csv_path = os.path.join(os.path.dirname(__file__), "small_test.csv")
    
    data = []
    with open(csv_path, "r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append(row)

    # Extract features and target
    X = np.array([[float(v) for k, v in datum.items() if k.startswith('x')] for datum in data])
    y = np.array([float(datum['y']) for datum in data])

    # Fit model and predict
    results = model.fit(X, y)
    preds = results.predict(X)
    assert not np.allclose(preds, preds[0]), "All predictions are the same — model likely failed to learn"

    # Optional: check prediction error is low
    mse = np.mean((preds - y) ** 2)
    assert mse < 10, f"Model MSE too high: {mse:.4f}"


if __name__ == "__main__":
    test_predict()
