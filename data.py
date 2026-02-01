import numpy as np
import csv

def load_data(path="ml-latest-small/ratings.csv", test_ratio=0.2, seed=42):
    ratings = []
    users = set()
    items = set()

    with open(path, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            user_id = int(row[0])
            item_id = int(row[1])
            rating = float(row[2])
            ratings.append((user_id, item_id, rating))
            users.add(user_id)
            items.add(item_id)

    user_to_idx = {u: i for i, u in enumerate(sorted(users))}
    item_to_idx = {m: i for i, m in enumerate(sorted(items))}

    np.random.seed(seed)
    indices = np.random.permutation(len(ratings))
    split = int(len(ratings) * (1 - test_ratio))

    train_ratings = [ratings[i] for i in indices[:split]]
    test_ratings = [ratings[i] for i in indices[split:]]

    return train_ratings, test_ratings, user_to_idx, item_to_idx, len(users), len(items)


if __name__ == "__main__":
    train, test, u2i, i2i, n_users, n_items = load_data()
    print(f"Train: {len(train)}, Test: {len(test)}")
    print(f"Users: {n_users}, Items: {n_items}")
