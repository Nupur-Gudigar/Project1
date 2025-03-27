import os
import numpy as np
import csv
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
from model.LassoHomotopy import LassoHomotopyModel

# Define the path to the collinear_data.csv file
csv_path = os.path.join(os.path.dirname(__file__), 'collinear_data.csv')

# Load the dataset
with open(csv_path, 'r') as f:
    reader = csv.reader(f)
    header = next(reader)  # Skip the header row
    data = np.array([[float(x) if x else np.nan for x in row] for row in reader])

# Separate features (X) and target variable (y)
X = data[:, :-1]
y = data[:, -1]

# Handle missing values by imputing with column mean
col_means = np.nanmean(X, axis=0)
inds = np.where(np.isnan(X))
X[inds] = np.take(col_means, inds[1])

# Standardize features to have zero mean and unit variance
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X = (X - X_mean) / X_std

# Initialize the LassoHomotopyModel with a specified alpha value
alpha = 0.01
model = LassoHomotopyModel(alpha=alpha)

# Fit the model to the data
results = model.fit(X, y)

# Predict target values using the trained model
y_pred = results.predict(X)

# Calculate evaluation metrics
mse = np.mean((y - y_pred) ** 2)  # Mean Squared Error
mae = np.mean(np.abs(y - y_pred))  # Mean Absolute Error
ss_res = np.sum((y - y_pred) ** 2)  # Residual Sum of Squares
ss_tot = np.sum((y - np.mean(y)) ** 2)  # Total Sum of Squares
r2 = 1 - (ss_res / ss_tot)  # R² Score

# Display the evaluation metrics
print("Evaluation Metrics for collinear_data.csv:")
print(f"📉 Mean Squared Error (MSE): {mse:.4f}")
print(f"📉 Mean Absolute Error (MAE): {mae:.4f}")
print(f"📈 R² Score: {r2:.4f}")
