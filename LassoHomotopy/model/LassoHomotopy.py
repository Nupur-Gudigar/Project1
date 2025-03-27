# import numpy as np

# class LassoHomotopyModel:
#     def __init__(self, tol=1e-8, max_iter=1000):
#         self.tol = tol
#         self.max_iter = max_iter
#         self.path = []  # (lambda, coef vector at that point)
#         self.coef_ = None
#         self.intercept_ = None

#     def fit(self, X, y):
#         # ===== 1. Center the data =====
#         X = np.asarray(X, dtype=float)
#         y = np.asarray(y, dtype=float)
#         self.X_mean_ = X.mean(axis=0)
#         self.y_mean_ = y.mean()
#         Xc = X - self.X_mean_
#         yc = y - self.y_mean_

#         n, d = Xc.shape
#         theta = np.zeros(d)
#         residual = -yc

#         # ===== 2. Initialize with max correlation =====
#         corr = Xc.T @ yc
#         lambda_val = np.max(np.abs(corr))
#         j0 = np.argmax(np.abs(corr))
#         sign_j0 = np.sign(corr[j0])

#         A = [j0]
#         s = {j0: sign_j0}

#         X_A = Xc[:, A]
#         S_A = np.array([s[j] for j in A])
#         G_A_inv = np.linalg.inv(X_A.T @ X_A)

#         # ===== 3. Homotopy Path =====
#         for _ in range(self.max_iter):
#             d_theta = np.zeros(d)
#             u = X_A @ (G_A_inv @ S_A)
#             a = Xc.T @ u

#             # === Entry Events ===
#             gamma_entries = []
#             for j in range(d):
#                 if j in A:
#                     continue
#                 a_j = a[j]
#                 c_j = corr[j]
#                 if 1 - a_j != 0:
#                     gamma1 = (lambda_val - c_j) / (1 - a_j)
#                     if gamma1 > self.tol:
#                         gamma_entries.append((gamma1, j, np.sign(c_j)))
#                 if 1 + a_j != 0:
#                     gamma2 = (lambda_val + c_j) / (1 + a_j)
#                     if gamma2 > self.tol:
#                         gamma_entries.append((gamma2, j, -np.sign(c_j)))

#             # === Exit Events ===
#             beta_A = theta[A]
#             d_beta_A = G_A_inv @ S_A
#             gamma_exits = []
#             for i, j in enumerate(A):
#                 if d_beta_A[i] != 0:
#                     gamma = -beta_A[i] / d_beta_A[i]
#                     if gamma > self.tol:
#                         gamma_exits.append((gamma, j))

#             # === Select Minimum Gamma ===
#             all_candidates = []
#             all_candidates.extend([(g, 'entry', j, sgn) for g, j, sgn in gamma_entries])
#             all_candidates.extend([(g, 'exit', j, None) for g, j in gamma_exits])

#             if not all_candidates:
#                 break

#             gamma_star, event_type, j_star, sign_star = min(all_candidates, key=lambda x: x[0])

#             # === Update theta ===
#             d_theta[A] = G_A_inv @ S_A
#             theta += gamma_star * d_theta
#             lambda_val -= gamma_star
#             self.path.append((lambda_val, theta.copy()))

#             # === Update active set ===
#             if event_type == 'entry':
#                 A.append(j_star)
#                 s[j_star] = sign_star
#             elif event_type == 'exit':
#                 A.remove(j_star)
#                 del s[j_star]

#             if not A:
#                 break

#             # Update matrices
#             X_A = Xc[:, A]
#             S_A = np.array([s[j] for j in A])
#             G_A_inv = np.linalg.pinv(X_A.T @ X_A)

#             residual = yc - Xc @ theta
#             corr = Xc.T @ residual

#             if lambda_val < self.tol:
#                 break

#         self.coef_ = theta
#         self.intercept_ = self.y_mean_ - self.X_mean_ @ self.coef_
#         return self

#     def predict(self, X):
#         X = np.asarray(X, dtype=float)
#         return X @ self.coef_ + self.intercept_


import numpy as np

class LassoHomotopyModel:
    def __init__(self, tol=1e-8, max_iter=1000, normalize=True, lambda_min_ratio=0.01):
        self.tol = tol
        self.max_iter = max_iter
        self.normalize = normalize
        self.lambda_min_ratio = lambda_min_ratio
        self.path = []
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        # === Center the data ===
        self.X_mean_ = X.mean(axis=0)
        self.y_mean_ = y.mean()
        Xc = X - self.X_mean_
        yc = y - self.y_mean_

        # === Optional feature scaling ===
        if self.normalize:
            self.X_std_ = Xc.std(axis=0) + 1e-8
            Xc = Xc / self.X_std_
        else:
            self.X_std_ = np.ones(Xc.shape[1])

        n, d = Xc.shape
        theta = np.zeros(d)
        residual = -yc

        # === Initialize lambda ===
        corr = Xc.T @ yc
        lambda_max = np.max(np.abs(corr))
        lambda_val = lambda_max
        lambda_min = lambda_max * self.lambda_min_ratio

        j0 = np.argmax(np.abs(corr))
        sign_j0 = np.sign(corr[j0])

        A = [j0]
        s = {j0: sign_j0}

        X_A = Xc[:, A]
        S_A = np.array([s[j] for j in A])
        G_A_inv = np.linalg.pinv(X_A.T @ X_A)

        for _ in range(self.max_iter):
            d_theta = np.zeros(d)
            u = X_A @ (G_A_inv @ S_A)
            a = Xc.T @ u

            # === Entry events ===
            gamma_entries = []
            for j in range(d):
                if j in A:
                    continue
                a_j = a[j]
                c_j = corr[j]
                if 1 - a_j != 0:
                    gamma1 = (lambda_val - c_j) / (1 - a_j)
                    if gamma1 > self.tol:
                        gamma_entries.append((gamma1, j, np.sign(c_j)))
                if 1 + a_j != 0:
                    gamma2 = (lambda_val + c_j) / (1 + a_j)
                    if gamma2 > self.tol:
                        gamma_entries.append((gamma2, j, -np.sign(c_j)))

            # === Exit events ===
            beta_A = theta[A]
            d_beta_A = G_A_inv @ S_A
            gamma_exits = []
            for i, j in enumerate(A):
                if d_beta_A[i] != 0:
                    gamma = -beta_A[i] / d_beta_A[i]
                    if gamma > self.tol:
                        gamma_exits.append((gamma, j))

            # === Select min gamma ===
            all_candidates = []
            all_candidates.extend([(g, 'entry', j, sgn) for g, j, sgn in gamma_entries])
            all_candidates.extend([(g, 'exit', j, None) for g, j in gamma_exits])

            if not all_candidates:
                break

            gamma_star, event_type, j_star, sign_star = min(all_candidates, key=lambda x: x[0])

            # === Update theta ===
            d_theta[A] = G_A_inv @ S_A
            theta += gamma_star * d_theta
            lambda_val -= gamma_star
            self.path.append((lambda_val, theta.copy()))

            # === Update active set ===
            if event_type == 'entry':
                A.append(j_star)
                s[j_star] = sign_star
            elif event_type == 'exit':
                A.remove(j_star)
                del s[j_star]

            if not A:
                break

            # Update matrices
            X_A = Xc[:, A]
            S_A = np.array([s[j] for j in A])
            G_A_inv = np.linalg.pinv(X_A.T @ X_A)
            residual = yc - Xc @ theta
            corr = Xc.T @ residual

            if lambda_val < lambda_min:
                break

        # === Restore unscaled coefficients ===
        coef_scaled = theta / self.X_std_

        # === Debias: OLS refit on non-zero coefficients ===
        active_idx = np.where(np.abs(coef_scaled) > 1e-4)[0]
        if len(active_idx) > 0:
            X_active = X[:, active_idx]
            ols_coef = np.linalg.pinv(X_active.T @ X_active) @ (X_active.T @ y)

            final_coefs = np.zeros(d)
            final_coefs[active_idx] = ols_coef
            self.coef_ = final_coefs
        else:
            self.coef_ = coef_scaled  # fallback

        self.intercept_ = self.y_mean_ - self.X_mean_ @ self.coef_
        return self


    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return X @ self.coef_ + self.intercept_
