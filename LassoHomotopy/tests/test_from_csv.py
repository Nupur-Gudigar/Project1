import numpy as np
import csv
import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
from model.LassoHomotopy import LassoHomotopyModel

def load_csv(path):
    with open(path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader) 
        data = np.array([[float(x) for x in row] for row in reader])
    return data[:, :-1], data[:, -1]

csv_path = os.path.join(os.path.dirname(__file__), 'small_test.csv')

X, y = load_csv(csv_path)

model = LassoHomotopyModel(alpha=0.01)
results = model.fit(X, y)

y_pred = results.predict(X)

mse = np.mean((y - y_pred) ** 2)

mae = np.mean(np.abs(y - y_pred))

ss_res = np.sum((y - y_pred) ** 2)
ss_tot = np.sum((y - np.mean(y)) ** 2)
r2 = 1 - (ss_res / ss_tot)

print("Mean Squared Error (MSE):", mse)
print("Mean Absolute Error (MAE):", mae)
print("R² Score:", r2)

def test_collinear_csv():
    """
    Checks that in a CSV with collinear or nearly-collinear columns,
    Lasso will zero out (or nearly zero) at least one column's coefficient
    when alpha is moderate.
    """
    
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'tests', 'collinear_data.csv')
    X, y = load_csv(csv_path)
    
    alpha_val = 0.5
    model = LassoHomotopyModel(alpha=alpha_val)
    results = model.fit(X, y)
    coef = results.coef_
    
    #If we know which columns are collinear (say, col 3 & col 4),
    #We expect one column to be shrunk drastically compared to the other.
    
    col3, col4 = abs(coef[3]), abs(coef[4])
    print("Coefficients:", coef)
    assert (col3 < 0.1) or (col4 < 0.1), (
        f"Expected Lasso to zero out one of the collinear columns, "
        f"but got col3={col3:.4f}, col4={col4:.4f}"
    )
    return coef
coef = test_collinear_csv()
tolerance = 1e-12
is_any_zero = np.any(np.abs(coef) < tolerance)

if is_any_zero:
    print("Test Passed: at least one coefficient is zero!")
else:
    print("Test Failed: no coefficients were zero.")
