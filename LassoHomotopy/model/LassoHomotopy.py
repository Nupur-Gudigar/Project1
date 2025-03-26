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

        residual = yc.copy()
        active_set = []
        signs = np.zeros(n_features)

        for _ in range(self.max_iter):
            corr = Xc.T @ residual
            max_corr = np.max(np.abs(corr))
            if max_corr < self.alpha:
                break

            idx = np.argmax(np.abs(corr))
            if idx not in active_set:
                active_set.append(idx)
                signs[idx] = np.sign(corr[idx])

            Xa = Xc[:, active_set] * signs[active_set]
            G = Xa.T @ Xa
            G_inv = np.linalg.pinv(G)

            u = np.ones(len(active_set))
            A = 1 / np.sqrt(u @ G_inv @ u)
            d_theta = A * (G_inv @ u)
            d_residual = Xa @ d_theta

            gammas = []
            for j in range(n_features):
                if j in active_set:
                    continue
                aj = Xc[:, j].T @ d_residual
                cj = corr[j]
                denom1 = A - aj
                denom2 = A + aj
                if not np.isclose(denom1, 0):
                    gammas.append((max_corr - cj) / denom1)
                if not np.isclose(denom2, 0):
                    gammas.append((max_corr + cj) / denom2)

            gammas = [g for g in gammas if g > self.tol]
            if not gammas:
                break
            gamma = min(gammas)

            for i, idx in enumerate(active_set):
                self.coef_[idx] += gamma * signs[idx] * d_theta[i]

            residual -= gamma * d_residual

            active_set = [i for i in active_set if np.sign(self.coef_[i]) == signs[i]]

            if np.abs(gamma) < self.tol:
                break

        self.intercept_ = y_mean - X_mean @ self.coef_
        return LassoHomotopyResults(self.coef_, self.intercept_)

class LassoHomotopyResults:
    def __init__(self, coef, intercept):
        self.coef_ = coef
        self.intercept_ = intercept

    def predict(self, x):
        x = np.asarray(x, dtype=float)
        return x @ self.coef_ + self.intercept_
