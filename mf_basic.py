import numpy as np

class BasicMF:
    def __init__(self, n_factors=20, lr=0.01, epochs=20, reg=0.02):
        self.n_factors = n_factors
        self.lr = lr
        self.epochs = epochs
        self.reg = reg

    def fit(self, ratings, n_users, n_items, user_to_idx, item_to_idx):
        self.user_to_idx = user_to_idx
        self.item_to_idx = item_to_idx
        self.global_mean = np.mean([r for _, _, r in ratings])

        # init P and Q randomly
        self.P = np.random.normal(0, 0.1, (n_users, self.n_factors))
        self.Q = np.random.normal(0, 0.1, (n_items, self.n_factors))

        # train with gradient descent
        for epoch in range(self.epochs):
            np.random.shuffle(ratings)
            total_loss = 0

            for user, item, rating in ratings:
                u = user_to_idx[user]
                i = item_to_idx[item]

                pred = self.P[u] @ self.Q[i]
                error = rating - pred

                # update P and Q
                self.P[u] += self.lr * (error * self.Q[i] - self.reg * self.P[u])
                self.Q[i] += self.lr * (error * self.P[u] - self.reg * self.Q[i])

                total_loss += error ** 2

            if epoch % 5 == 0:
                print(f"  Epoch {epoch}, Loss: {np.sqrt(total_loss/len(ratings)):.4f}")

    def predict(self, user, item):
        if user not in self.user_to_idx or item not in self.item_to_idx:
            return self.global_mean

        u = self.user_to_idx[user]
        i = self.item_to_idx[item]

        pred = self.P[u] @ self.Q[i]
        return np.clip(pred, 0.5, 5.0)
