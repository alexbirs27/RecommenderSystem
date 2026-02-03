import numpy as np

class TwoTowerModel:
    """
    Two-Tower Neural Network with Genre Features

    Architecture:
    - User Tower: user_id -> embedding -> hidden -> output
    - Item Tower: [item_embedding, genre_vector] -> hidden -> output

    Used by: YouTube, Google, Facebook for recommendations
    """

    def __init__(self, n_genres=20, embed_dim=32, hidden_dim=64, lr=0.001, epochs=30, reg=0.01):
        self.n_genres = n_genres
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.epochs = epochs
        self.reg = reg

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return (x > 0).astype(float)

    def fit(self, ratings, n_users, n_items, user_to_idx, item_to_idx, movie_genres=None):
        self.user_to_idx = user_to_idx
        self.item_to_idx = item_to_idx
        self.global_mean = np.mean([r for _, _, r in ratings])

        # build item_id -> genre vector mapping
        self.item_genres = np.zeros((n_items, self.n_genres))
        if movie_genres:
            for movie_id, idx in item_to_idx.items():
                if movie_id in movie_genres:
                    self.item_genres[idx] = movie_genres[movie_id]

        # user tower weights
        self.user_embed = np.random.normal(0, 0.1, (n_users, self.embed_dim))
        self.user_W1 = np.random.normal(0, 0.1, (self.embed_dim, self.hidden_dim))
        self.user_b1 = np.zeros(self.hidden_dim)
        self.user_W2 = np.random.normal(0, 0.1, (self.hidden_dim, self.embed_dim))
        self.user_b2 = np.zeros(self.embed_dim)

        # item tower weights - input is embed_dim + n_genres
        item_input_dim = self.embed_dim + self.n_genres
        self.item_embed = np.random.normal(0, 0.1, (n_items, self.embed_dim))
        self.item_W1 = np.random.normal(0, 0.1, (item_input_dim, self.hidden_dim))
        self.item_b1 = np.zeros(self.hidden_dim)
        self.item_W2 = np.random.normal(0, 0.1, (self.hidden_dim, self.embed_dim))
        self.item_b2 = np.zeros(self.embed_dim)

        # biases
        self.bu = np.zeros(n_users)
        self.bi = np.zeros(n_items)

        ratings_list = list(ratings)

        for epoch in range(self.epochs):
            np.random.shuffle(ratings_list)
            total_loss = 0

            for user, item, rating in ratings_list:
                u = user_to_idx[user]
                i = item_to_idx[item]

                # forward pass - user tower
                user_e = self.user_embed[u]
                user_h1 = self.relu(user_e @ self.user_W1 + self.user_b1)
                user_out = user_h1 @ self.user_W2 + self.user_b2

                # forward pass - item tower (with genres)
                item_e = self.item_embed[i]
                genre_vec = self.item_genres[i]
                item_input = np.concatenate([item_e, genre_vec])
                item_h1 = self.relu(item_input @ self.item_W1 + self.item_b1)
                item_out = item_h1 @ self.item_W2 + self.item_b2

                # prediction
                pred = self.global_mean + self.bu[u] + self.bi[i] + np.dot(user_out, item_out)
                error = rating - pred
                total_loss += error ** 2

                # backprop - output layer
                d_user_out = -error * item_out
                d_item_out = -error * user_out

                # backprop - user tower
                d_user_W2 = np.outer(user_h1, d_user_out) + self.reg * self.user_W2
                d_user_b2 = d_user_out
                d_user_h1 = d_user_out @ self.user_W2.T * self.relu_derivative(user_e @ self.user_W1 + self.user_b1)
                d_user_W1 = np.outer(user_e, d_user_h1) + self.reg * self.user_W1
                d_user_b1 = d_user_h1
                d_user_e = d_user_h1 @ self.user_W1.T + self.reg * user_e

                # backprop - item tower
                d_item_W2 = np.outer(item_h1, d_item_out) + self.reg * self.item_W2
                d_item_b2 = d_item_out
                d_item_h1 = d_item_out @ self.item_W2.T * self.relu_derivative(item_input @ self.item_W1 + self.item_b1)
                d_item_W1 = np.outer(item_input, d_item_h1) + self.reg * self.item_W1
                d_item_b1 = d_item_h1
                # gradient for item embedding only (genres are fixed input features)
                d_item_input = d_item_h1 @ self.item_W1.T
                d_item_e = d_item_input[:self.embed_dim] + self.reg * item_e

                # update weights - user tower
                self.user_W2 -= self.lr * d_user_W2
                self.user_b2 -= self.lr * d_user_b2
                self.user_W1 -= self.lr * d_user_W1
                self.user_b1 -= self.lr * d_user_b1
                self.user_embed[u] -= self.lr * d_user_e

                # update weights - item tower
                self.item_W2 -= self.lr * d_item_W2
                self.item_b2 -= self.lr * d_item_b2
                self.item_W1 -= self.lr * d_item_W1
                self.item_b1 -= self.lr * d_item_b1
                self.item_embed[i] -= self.lr * d_item_e

                # update biases
                self.bu[u] += self.lr * (error - self.reg * self.bu[u])
                self.bi[i] += self.lr * (error - self.reg * self.bi[i])

            rmse = np.sqrt(total_loss / len(ratings_list))
            if epoch % 5 == 0:
                print(f"  Epoch {epoch}, RMSE: {rmse:.4f}")

    def predict(self, user, item):
        if user not in self.user_to_idx or item not in self.item_to_idx:
            return self.global_mean

        u = self.user_to_idx[user]
        i = self.item_to_idx[item]

        # user tower forward
        user_e = self.user_embed[u]
        user_h1 = self.relu(user_e @ self.user_W1 + self.user_b1)
        user_out = user_h1 @ self.user_W2 + self.user_b2

        # item tower forward (with genres)
        item_e = self.item_embed[i]
        genre_vec = self.item_genres[i]
        item_input = np.concatenate([item_e, genre_vec])
        item_h1 = self.relu(item_input @ self.item_W1 + self.item_b1)
        item_out = item_h1 @ self.item_W2 + self.item_b2

        pred = self.global_mean + self.bu[u] + self.bi[i] + np.dot(user_out, item_out)
        return np.clip(pred, 0.5, 5.0)
