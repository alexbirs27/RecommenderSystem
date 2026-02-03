import numpy as np

class MatrixFactorizationSVD:
    def __init__(self, n_factors=20):
        self.n_factors = n_factors

    def fit(self, ratings, n_users, n_items, user_to_idx, item_to_idx):
        self.user_to_idx = user_to_idx
        self.item_to_idx = item_to_idx
        self.n_users = n_users
        self.n_items = n_items

        # build rating matrix
        self.R = np.zeros((n_users, n_items))
        self.mask = np.zeros((n_users, n_items))

        for user, item, rating in ratings:
            u_idx = user_to_idx[user]
            i_idx = item_to_idx[item]
            self.R[u_idx, i_idx] = rating
            self.mask[u_idx, i_idx] = 1

        # fill missing with global mean
        self.global_mean = np.sum(self.R) / np.sum(self.mask)
        R_filled = self.R.copy()
        R_filled[self.mask == 0] = self.global_mean

        # do SVD
        U, s, Vt = np.linalg.svd(R_filled, full_matrices=False)

        # keep only n_factors
        self.U = U[:, :self.n_factors]
        self.s = s[:self.n_factors]
        self.Vt = Vt[:self.n_factors, :]

        # reconstruct
        self.R_pred = self.U @ np.diag(self.s) @ self.Vt

    def predict(self, user, item):
        if user not in self.user_to_idx or item not in self.item_to_idx:
            return self.global_mean

        u_idx = self.user_to_idx[user]
        i_idx = self.item_to_idx[item]

        pred = self.R_pred[u_idx, i_idx]
        return np.clip(pred, 0.5, 5.0)
