import numpy as np
import logging

class LassoHomotopyModel:
    def __init__(self, alpha=1.0, max_iter=500, tol=1e-4):
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.coef_ = None
        self.intercept_ = None

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initialized LassoHomotopyModel: alpha={alpha}, max_iter={max_iter}, tol={tol}")

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        n_samples, n_features = X.shape
        X_mean = X.mean(axis=0)
        y_mean = y.mean()
        Xc = X - X_mean
        yc = y - y_mean

        if self.alpha == 0.0:
            coef = np.linalg.pinv(Xc.T @ Xc) @ Xc.T @ yc
            self.coef_ = coef
            self.intercept_ = y_mean - X_mean @ coef
            return LassoHomotopyResult(self.coef_, self.intercept_)

        self.coef_ = np.zeros(n_features)
        self.intercept_ = y_mean
        active_set = []
        sign_vector = np.zeros(n_features)
        residual = yc.copy()
        correlation = Xc.T @ residual

        for iteration in range(self.max_iter):
            self.logger.debug(f"Iteration={iteration}, Active Set={active_set}, Coefficients={self.coef_}")

            max_corr = np.max(np.abs(correlation))
            if max_corr < self.alpha:
                break

            max_idx = np.argmax(np.abs(correlation))
            if max_idx not in active_set:
                active_set.append(max_idx)
                sign_vector[max_idx] = np.sign(correlation[max_idx])

            Xa = Xc[:, active_set] * sign_vector[active_set]
            G = Xa.T @ Xa
            G_inv = np.linalg.pinv(G)

            unit_vec = np.ones(len(active_set))
            scaling_factor = 1.0 / np.sqrt(unit_vec @ G_inv @ unit_vec)
            direction = scaling_factor * (G_inv @ unit_vec)

            update_direction = Xa @ direction

            gamma_vals = []
            for i in range(n_features):
                if i not in active_set:
                    ak = Xc[:, i].T @ update_direction
                    ck = correlation[i]
                    denom1 = scaling_factor - ak
                    denom2 = scaling_factor + ak
                    if not np.isclose(denom1, 0):
                        gamma_vals.append((max_corr - ck) / denom1)
                    if not np.isclose(denom2, 0):
                        gamma_vals.append((max_corr + ck) / denom2)

            gamma_vals = [g for g in gamma_vals if g > self.tol]
            if not gamma_vals:
                break

            gamma = min(gamma_vals)

            for i, idx in enumerate(active_set):
                self.coef_[idx] += gamma * sign_vector[idx] * direction[i]

            residual -= gamma * update_direction
            correlation = Xc.T @ residual

            active_set = [idx for idx in active_set if np.sign(self.coef_[idx]) == sign_vector[idx]]

            if np.abs(gamma) < self.tol:
                break

        self.intercept_ = y_mean - X_mean @ self.coef_
        return LassoHomotopyResult(self.coef_, self.intercept_)

class LassoHomotopyResult:
    def __init__(self, coef, intercept):
        self.coef_ = coef
        self.intercept_ = intercept

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return X @ self.coef_ + self.intercept_

if __name__ == "__main__":
    from sklearn.datasets import make_regression

    X, y = make_regression(n_samples=100, n_features=10, noise=0.1, random_state=42)
    model = LassoHomotopyModel(alpha=10.0)
    result = model.fit(X, y)
    predictions = result.predict(X)

    print("Coefficients:", result.coef_)
    print("Intercept:", result.intercept_)
    print("Predictions (first 5):", predictions[:5])
