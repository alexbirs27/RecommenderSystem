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

    # sparsity analysis
    total_possible = n_users * n_items
    total_ratings = len(train) + len(test)
    sparsity = 1 - (total_ratings / total_possible)
    print(f"\nSparsity: {sparsity*100:.2f}%")
    print(f"Only {total_ratings} ratings out of {total_possible} possible")

    # ratings per user
    from collections import Counter
    user_counts = Counter(u for u, i, r in train)
    avg_per_user = np.mean(list(user_counts.values()))
    min_per_user = min(user_counts.values())
    max_per_user = max(user_counts.values())
    print(f"\nRatings per user: avg={avg_per_user:.1f}, min={min_per_user}, max={max_per_user}")

    # ratings per item
    item_counts = Counter(i for u, i, r in train)
    avg_per_item = np.mean(list(item_counts.values()))
    min_per_item = min(item_counts.values())
    max_per_item = max(item_counts.values())
    print(f"Ratings per item: avg={avg_per_item:.1f}, min={min_per_item}, max={max_per_item}")

    # rating distribution
    rating_counts = Counter(r for u, i, r in train)
    print(f"\nRating distribution:")
    for r in sorted(rating_counts.keys()):
        print(f"  {r}: {rating_counts[r]}")
