import numpy as np
import csv
import os
from LassoHomotopy import LassoHomotopyModel

def load_csv(path):
    with open(path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader) 
        data = np.array([[float(x) for x in row] for row in reader])
    return data[:, :-1], data[:, -1]

csv_path = os.path.join(os.path.dirname(__file__), '..', 'tests', 'small_test.csv')

X, y = load_csv(csv_path)

model = LassoHomotopyModel(alpha=0.01)
results = model.fit(X, y)

y_pred = results.predict(X)

mse = np.mean((y - y_pred) ** 2)

mae = np.mean(np.abs(y - y_pred))

ss_res = np.sum((y - y_pred) ** 2)
ss_tot = np.sum((y - np.mean(y)) ** 2)
r2 = 1 - (ss_res / ss_tot)

print("📉 Mean Squared Error (MSE):", mse)
print("📉 Mean Absolute Error (MAE):", mae)
print("📈 R² Score:", r2)
