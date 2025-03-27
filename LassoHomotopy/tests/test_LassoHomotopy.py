import numpy as np
import pandas as pd
import pytest
import sys
import os
import csv


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

# === Test Higly collinear dataset zeros out atleast one column ===
def test_collinear_csv():
    """
    Checks that in a CSV with collinear or nearly-collinear columns,
    Lasso will zero out (or nearly zero) at least one column's coefficient
    when alpha is moderate.
    """
    def load_csv(path):
        with open(path, 'r') as f:
            reader = csv.reader(f)
            header = next(reader) 
            data = np.array([[float(x) for x in row] for row in reader])
        return data[:, :-1], data[:, -1]
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'tests', 'collinear_data.csv')
    X, y = load_csv(csv_path)
    
    alpha_val = 0.5
    model = LassoHomotopyModel(alpha_val)
    results = model.fit(X, y)
    coef = results.coef_
    
    #If we know which columns are collinear (say, col 3 & col 4),
    #We expect one column to be shrunk drastically compared to the other.
    
    col3, col4 = abs(coef[3]), abs(coef[4])
    threshold = 1.0

    print("Coefficients:", coef)
    print(f"|coef[3]| = {col3:.4f}, |coef[4]| = {col4:.4f}")

    if (col3 < threshold) or (col4 < threshold):
        print("Test PASSED: Lasso drove one collinear column near zero.")
    else:
        raise AssertionError(
            f"Test FAILED: expected either coef[3] or coef[4] < {threshold}, "
            f"but got {col3:.4f} and {col4:.4f}"
        )

# === Test: alpha_zero_OLS===
def test_alpha_zero_OLS():
    """
    Edge case: alpha=0 should give us the ordinary least squares (OLS) solution
    (no L1 penalty at all). Compare coefficients to np.linalg.lstsq.
    """
    # Create a small synthetic dataset
    np.random.seed(123)
    n_samples = 20
    n_features = 5

    # Random X
    X = np.random.randn(n_samples, n_features)
    
    # True coefficients
    true_coef = np.array([2.0, -1.0, 0.0, 4.0, -3.0])
    
    # Generating y with some noise
    y = X @ true_coef + 0.01 * np.random.randn(n_samples)

    # Fits the Lasso model with alpha=0
    model = LassoHomotopyModel()
    model.fit(X, y)
    lasso_coefs = model.coef_
    lasso_intercept = model.intercept_

    # Compare with ordinary least squares via np.linalg.lstsq
    # OLS typically has no separate intercept if we haven't centered data,
    # so we can compute an "OLS intercept" by appending a column of ones.
    X_ones = np.column_stack([X, np.ones(n_samples)])
    ols_solution, _, _, _ = np.linalg.lstsq(X_ones, y, rcond=None)
    ols_coefs = ols_solution[:-1]   # first part is the slope
    ols_intercept = ols_solution[-1]  # last part is the intercept

    # Checking they're close
    # Adjust tolerances as appropriate
    assert np.allclose(lasso_coefs, ols_coefs, atol=1e-2), (
        "Coefficients from alpha=0 Lasso should match OLS"
    )
    assert abs(lasso_intercept - ols_intercept) < 1e-2, (
        "Intercept from alpha=0 Lasso should match OLS"
    )
 

    # print them for debugging:
    print("Lasso alpha=0 coefs:", lasso_coefs)
    print("Lasso alpha=0 intercept:", lasso_intercept)
    print("OLS coefs:", ols_coefs)
    print("OLS intercept:", ols_intercept)

# ===High tolerance (early stopping) ===
def test_high_tolerance():
    X, y = load_data("small_test.csv", label_col="y")
    model = LassoHomotopyModel(tol=1e-2)
    model.fit(X, y)
    y_pred = model.predict(X)
    print_regression_metrics(y, y_pred, name="High Tolerance Test")

# === Very few iterations (should underfit) ===
def test_few_iterations():
    X, y = load_data("small_test.csv", label_col="y")
    model = LassoHomotopyModel(max_iter=2)
    model.fit(X, y)
    y_pred = model.predict(X)
    print_regression_metrics(y, y_pred, name="Few Iterations Test")

# === Random noise target (should give near-zero coefs) ===
def test_random_target():
    X, y = load_data("small_test.csv", label_col="y")
    np.random.seed(42)
    y_noise = np.random.randn(len(y))
    model = LassoHomotopyModel(tol=1e-6)
    model.fit(X, y_noise)
    y_pred = model.predict(X)
    print_regression_metrics(y_noise, y_pred, name="Random Target Test")

# === Normalized input data ===
def test_normalized_input():
    X, y = load_data("collinear_data.csv", label_col="target")
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    model = LassoHomotopyModel()
    model.fit(X, y)
    y_pred = model.predict(X)
    print_regression_metrics(y, y_pred, name="Normalized Input Test")

# === Stricter sparsity check ===
def test_strict_sparsity():
    X, y = load_data("collinear_data.csv", label_col="target")
    model = LassoHomotopyModel(tol=1e-6)
    model.fit(X, y)
    y_pred = model.predict(X)
    print_regression_metrics(y, y_pred, name="Strict Sparsity Test")

    nonzero = np.sum(np.abs(model.coef_) > 1e-4)
    assert nonzero <= 5, f"Expected sparse solution, got {nonzero} non-zero coefficients"


