# Project 1 
# Lasso Regression via Homotopy Method (Project 1)

This project implements the **LASSO regularized regression model using the Homotopy Method**, following the approach described in [this paper](https://people.eecs.berkeley.edu/~elghaoui/Pubs/hom_lasso_NIPS08.pdf).

The model is implemented **from scratch** using `numpy`, with no reliance on scikit-learn’s built-in regression models. It supports solving for sparse linear coefficients by incrementally building the solution path as the regularization parameter (`lambda`) decreases — a hallmark of homotopy-style solvers.

---

## Installation

1. Clone the repository and navigate to the root directory.
2. (Optional) Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
3. Install dependencies:
    pip install -r requirements.txt

## Running the model
1. You can run the tests to validate the model’s correctness:
    cd tests
    pytest -v test_LassoHomotopy.py
 or run it directly:
    python tests/test_LassoHomotopy.py

## Usage Example


## Visualization
1. To see how the Lasso coefficients evolve as lambda decreases, run the included notebook:
    jupyter notebook LassoHomotopy_Visualization.ipynb
This notebook shows a plot of the coefficient paths (the “Lasso path”) and helps visualize how sparsity is introduced as regularization decreases.

## Tests Included
1. Small dataset test (basic accuracy)
2. Collinear features test (sparse solution validation)
3. Known collinear pair test (feature elimination)
4. Alpha = 0 → behaves like OLS


* What does the model you have implemented do and when should it be used?
Ans. The Lasso Homotopy model estimates a sparse linear regression model. It is useful when there are many features, and you expect only a few to be important (i.e., the true model is sparse), especially in high-dimensional or collinear data.

* How did you test your model to determine if it is working reasonably correctly?
Ans. Using unit tests and small CSV datasets (small_test.csv, collinear_data.csv). Metrics like MSE, MAE, and R² were printed. Several edge cases (like alpha=0 and highly collinear features) were also verified to ensure correctness and sparsity

* What parameters have you exposed to users of your implementation in order to tune performance? 
Ans. 1. tol : Tolerance for convergance
     2. normalize: Whether to standardize feature before fitting
     3. lambda_min_ratio: The Stopping point for the lambda path in the homotopy method

* Are there specific inputs that your implementation has trouble with? Given more time, could you work around these or is it fundamental?
Ans. 1. Perfectly collinear features may lead to matrix inversion 
        instability — mitigated by using pseudoinverse 
     2. For very large feature counts, the solver may be slower due to
        matrix operations.

## Files Included
1. model/LassoHomotopy.py — core implementation
2. tests/test_LassoHomotopy.py — all unit tests
3. collinear_data.csv and small_test.csv — sample test datasets
4. LassoHomotopy_Visualization.ipynb — optional visual notebook
5. requirements.txt — dependencies


Group Member1: Nupur Gudigar
Group Member2: Nehil Joshi
Group Member3: Riddhi Das
Group Member4: Zaigham Shaikh
