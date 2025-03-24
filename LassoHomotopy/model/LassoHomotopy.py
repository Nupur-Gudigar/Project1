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
        """
        Fits the model using ordinary least squares (OLS) if alpha is 0,
        otherwise applies the Homotopy/LARS approach for sparse LASSO solutions.
        """
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)

        if self.alpha == 0.0:
            """Applies the LARS-Homotopy method for LASSO regression."""
            X_mean, y_mean = X.mean(axis=0), y.mean()
            Xc, yc = X - X_mean, y - y_mean

            gram_matrix = Xc.T @ Xc
            coef = np.linalg.solve(gram_matrix, Xc.T @ yc)

            self.coef_ = coef
            self.intercept_ = y_mean - X_mean @ coef
        else:
            """Solves the ordinary least squares (OLS) regression problem."""
            n_samples, n_features = X.shape

        X_mean, y_mean = X.mean(axis=0), y.mean()
        Xc, yc = X - X_mean, y - y_mean

        self.coef_ = np.zeros(n_features)
        self.intercept_ = y_mean

        active_set, sign_vector = [], np.zeros(n_features)
        residual, correlation = yc.copy(), Xc.T @ yc

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
                    ck, ak = correlation[i], Xc.T @ update_direction[i]
                    if not np.isclose(scaling_factor - ak, 0):
                        gamma_vals.append((max_corr - ck) / (scaling_factor - ak))
                    if not np.isclose(scaling_factor + ak, 0):
                        gamma_vals.append((max_corr + ck) / (scaling_factor + ak))

            if not gamma_vals:
                break

            gamma = min(filter(lambda x: x > 0, gamma_vals), default=0)

            self.coef_[active_set] += gamma * sign_vector[active_set] * direction
            residual -= gamma * update_direction
            correlation = Xc.T @ residual

            active_set = [idx for idx in active_set if np.sign(self.coef_[idx]) == sign_vector[idx]]

            if np.abs(gamma) < self.tol:
                break

        return LassoHomotopyResult(self.coef_, self.intercept_)

class LassoHomotopyResult:
    def __init__(self, coef, intercept):
        self.coef_ = coef
        self.intercept_ = intercept

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return X @ self.coef_ + self.intercept_
