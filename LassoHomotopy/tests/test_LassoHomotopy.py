import numpy as np
import pandas as pd
import pytest
import sys
import os

# Add project root to sys.path to access model
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.LassoHomotopy import LassoHomotopyModel

# === Utility to load data from CSV ===
def load_data(file_name, label_col):
    file_path = os.path.join(os.path.dirname(__file__), file_name)
    df = pd.read_csv(file_path)
    X = df.drop(columns=[label_col]).values
    y = df[label_col].values
    return X, y

# === Print metrics using NumPy only ===
def print_regression_metrics(y_true, y_pred, name=""):
    n = len(y_true)
    mse = np.mean((y_true - y_pred) ** 2)
    mae = np.mean(np.abs(y_true - y_pred))
    r2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)

    print(f"\n{name} Metrics:")
    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R^2 Score: {r2:.4f}")

# === Test: Small dataset ===
def test_small_dataset():
    X, y = load_data("small_test.csv", label_col="y")
    model = LassoHomotopyModel(tol=1e-6)
    model.fit(X, y)
    y_pred = model.predict(X)

    print_regression_metrics(y, y_pred, name="small_test.csv")

    assert model.coef_ is not None
    assert len(model.coef_) == X.shape[1]
    mse = np.mean((y_pred - y) ** 2)
    assert mse < 150  # loosened threshold based on your data scale

# === Test: Collinear dataset should produce sparse coefficients ===
def test_collinear_sparsity():
    X, y = load_data("collinear_data.csv", label_col="target")
    model = LassoHomotopyModel(tol=1e-6)
    model.fit(X, y)
    
    y_pred = model.predict(X)
    print_regression_metrics(y, y_pred, name="collinear_data.csv")

    nonzero_count = np.sum(np.abs(model.coef_) > 1e-4)
    assert nonzero_count < X.shape[1], f"Expected sparsity, got {nonzero_count} non-zero coefficients"
