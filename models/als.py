import numpy as np

class ALS:
    """
    Alternating Least Squares for Matrix Factorization

    Instead of SGD, we alternate between:
    1. Fix Q, solve for P (least squares)
    2. Fix P, solve for Q (least squares)

    Used by: Spark MLlib, Netflix
    Paper: Zhou et al. (2008) "Large-scale Parallel Collaborative Filtering for the Netflix Prize"
    """

    def __init__(self, n_factors=20, reg=0.1, epochs=15):
        self.n_factors = n_factors
        self.reg = reg
        self.epochs = epochs

    def fit(self, ratings, n_users, n_items, user_to_idx, item_to_idx):
        self.user_to_idx = user_to_idx
        self.item_to_idx = item_to_idx
        self.n_users = n_users
        self.n_items = n_items
        self.global_mean = np.mean([r for _, _, r in ratings])

        # init factors randomly
        self.P = np.random.normal(0, 0.1, (n_users, self.n_factors))
        self.Q = np.random.normal(0, 0.1, (n_items, self.n_factors))

        # build user->items and item->users mappings
        self.user_items = {}  # user -> [(item_idx, rating), ...]
        self.item_users = {}  # item -> [(user_idx, rating), ...]

        for user, item, rating in ratings:
            u = user_to_idx[user]
            i = item_to_idx[item]

            if u not in self.user_items:
                self.user_items[u] = []
            self.user_items[u].append((i, rating))

            if i not in self.item_users:
                self.item_users[i] = []
            self.item_users[i].append((u, rating))

        # ALS iterations
        for epoch in range(self.epochs):
            # step 1: fix Q, update P
            for u in range(n_users):
                if u not in self.user_items:
                    continue

                items = self.user_items[u]
                n_items_u = len(items)

                # build matrices for least squares
                Q_u = np.zeros((n_items_u, self.n_factors))
                r_u = np.zeros(n_items_u)

                for idx, (i, rating) in enumerate(items):
                    Q_u[idx] = self.Q[i]
                    r_u[idx] = rating

                # solve: (Q^T Q + reg*I) P_u = Q^T r_u
                A = Q_u.T @ Q_u + self.reg * np.eye(self.n_factors)
                b = Q_u.T @ r_u
                self.P[u] = np.linalg.solve(A, b)

            # step 2: fix P, update Q
            for i in range(n_items):
                if i not in self.item_users:
                    continue

                users = self.item_users[i]
                n_users_i = len(users)

                # build matrices for least squares
                P_i = np.zeros((n_users_i, self.n_factors))
                r_i = np.zeros(n_users_i)

                for idx, (u, rating) in enumerate(users):
                    P_i[idx] = self.P[u]
                    r_i[idx] = rating

                # solve: (P^T P + reg*I) Q_i = P^T r_i
                A = P_i.T @ P_i + self.reg * np.eye(self.n_factors)
                b = P_i.T @ r_i
                self.Q[i] = np.linalg.solve(A, b)

            # compute RMSE
            if epoch % 3 == 0:
                rmse = self._compute_rmse(ratings)
                print(f"  Epoch {epoch}, RMSE: {rmse:.4f}")

    def _compute_rmse(self, ratings):
        errors = []
        for user, item, rating in ratings:
            pred = self.predict(user, item)
            errors.append((rating - pred) ** 2)
        return np.sqrt(np.mean(errors))

    def predict(self, user, item):
        if user not in self.user_to_idx or item not in self.item_to_idx:
            return self.global_mean

        u = self.user_to_idx[user]
        i = self.item_to_idx[item]

        pred = self.P[u] @ self.Q[i]
        return np.clip(pred, 0.5, 5.0)
