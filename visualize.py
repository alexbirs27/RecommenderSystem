import matplotlib.pyplot as plt
from collections import Counter
from data import load_data

# load data
train, test, user_to_idx, item_to_idx, n_users, n_items = load_data()
all_ratings = train + test

# 1. rating distribution
ratings = [r for _, _, r in all_ratings]
rating_counts = Counter(ratings)

plt.figure(figsize=(8, 5))
plt.bar(rating_counts.keys(), rating_counts.values())
plt.xlabel('Rating')
plt.ylabel('Count')
plt.title('Rating Distribution')
plt.savefig('plots/plot_ratings.png')
plt.show()

# 2. ratings per user
user_counts = Counter(u for u, _, _ in all_ratings)
plt.figure(figsize=(8, 5))
plt.hist(user_counts.values(), bins=30)
plt.xlabel('Number of Ratings')
plt.ylabel('Number of Users')
plt.title('Ratings per User')
plt.savefig('plots/plot_users.png')
plt.show()

# 3. ratings per item
item_counts = Counter(i for _, i, _ in all_ratings)
plt.figure(figsize=(8, 5))
plt.hist(item_counts.values(), bins=30)
plt.xlabel('Number of Ratings')
plt.ylabel('Number of Items')
plt.title('Ratings per Item')
plt.savefig('plots/plot_items.png')
plt.show()

# 4. sparsity
total = n_users * n_items
filled = len(all_ratings)
sparsity = (1 - filled/total) * 100
print(f"Sparsity: {sparsity:.2f}%")
print(f"Filled: {filled} / {total}")

# 5. average rating per user (some users rate high, some low)
user_avg = {}
for u, i, r in all_ratings:
    if u not in user_avg:
        user_avg[u] = []
    user_avg[u].append(r)
user_means = [sum(v)/len(v) for v in user_avg.values()]

plt.figure(figsize=(8, 5))
plt.hist(user_means, bins=20)
plt.xlabel('Average Rating')
plt.ylabel('Number of Users')
plt.title('User Rating Tendencies (some users are harsh, some generous)')
plt.savefig('plots/plot_user_bias.png')
plt.show()

# 6. top 10 most rated movies
top_items = item_counts.most_common(10)
plt.figure(figsize=(10, 5))
plt.barh([str(x[0]) for x in top_items], [x[1] for x in top_items])
plt.xlabel('Number of Ratings')
plt.ylabel('Movie ID')
plt.title('Top 10 Most Rated Movies')
plt.savefig('plots/plot_top_movies.png')
plt.show()
