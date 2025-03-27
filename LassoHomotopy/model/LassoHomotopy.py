import numpy as np

class LassoHomotopyModel:
    def __init__(self, alpha=1.0, max_iter=1000, tol=1e-4):
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        n_samples, n_features = X.shape
        X_mean = X.mean(axis=0)
        y_mean = y.mean()
        Xc = X - X_mean
        yc = y - y_mean

        self.coef_ = np.zeros(n_features)
        self.intercept_ = y_mean

        for iteration in range(self.max_iter):
            coef_old = self.coef_.copy()

            for j in range(n_features):
                residual = yc - (Xc @ self.coef_ - Xc[:, j] * self.coef_[j])
                rho = Xc[:, j].T @ residual

                if rho < -self.alpha / 2:
                    self.coef_[j] = (rho + self.alpha / 2) / (Xc[:, j].T @ Xc[:, j])
                elif rho > self.alpha / 2:
                    self.coef_[j] = (rho - self.alpha / 2) / (Xc[:, j].T @ Xc[:, j])
                else:
                    self.coef_[j] = 0.0

            max_coef_change = np.max(np.abs(self.coef_ - coef_old))
            if max_coef_change < self.tol:
                break

        self.intercept_ = y_mean - X_mean @ self.coef_
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return X @ self.coef_ + self.intercept_
