import matplotlib.pyplot as plt
from data import load_data, load_genres, N_GENRES
from utils import rmse, mae
from models import (
    BaselinePredictor,
    UserBasedCF,
    ItemBasedCF,
    BasicMF,
    MatrixFactorizationSVD,
    MatrixFactorizationSGD,
    TwoTowerModel,
    ALS
)

def evaluate(model, test_ratings):
    preds = []
    actuals = []
    for user, item, rating in test_ratings:
        preds.append(model.predict(user, item))
        actuals.append(rating)
    return rmse(preds, actuals), mae(preds, actuals)

def main():
    print("Loading data...")
    train, test, user_to_idx, item_to_idx, n_users, n_items = load_data()
    movie_genres = load_genres()
    print(f"Train: {len(train)}, Test: {len(test)}, Genres: {N_GENRES}")

    results = []

    print("\nTraining Baseline...")
    baseline = BaselinePredictor()
    baseline.fit(train)
    r, m = evaluate(baseline, test)
    results.append(("Baseline", r, m))
    print(f"Baseline - RMSE: {r:.4f}, MAE: {m:.4f}")

    # print("\nTraining User-Based CF...")
    # user_cf = UserBasedCF(k=20)
    # user_cf.fit(train)
    # r, m = evaluate(user_cf, test)
    # results.append(("User-CF", r, m))
    # print(f"User-Based CF - RMSE: {r:.4f}, MAE: {m:.4f}")

    # print("\nTraining Item-Based CF...")
    # item_cf = ItemBasedCF(k=20)
    # item_cf.fit(train)
    # r, m = evaluate(item_cf, test)
    # results.append(("Item-CF", r, m))
    # print(f"Item-Based CF - RMSE: {r:.4f}, MAE: {m:.4f}")

    # print("\nTraining Basic MF (P*Q)...")
    # mf_basic = BasicMF(n_factors=20, lr=0.01, epochs=50)
    # mf_basic.fit(train, n_users, n_items, user_to_idx, item_to_idx)
    # r, m = evaluate(mf_basic, test)
    # results.append(("Basic MF", r, m))
    # print(f"Basic MF - RMSE: {r:.4f}, MAE: {m:.4f}")

    # print("\nTraining SVD...")
    # mf_svd = MatrixFactorizationSVD(n_factors=50)
    # mf_svd.fit(train, n_users, n_items, user_to_idx, item_to_idx)
    # r, m = evaluate(mf_svd, test)
    # results.append(("SVD", r, m))
    # print(f"SVD - RMSE: {r:.4f}, MAE: {m:.4f}")

    # print("\nTraining SGD (with biases)...")
    # mf_sgd = MatrixFactorizationSGD(n_factors=20, lr=0.005, epochs=30, reg=0.02)
    # mf_sgd.fit(train, n_users, n_items, user_to_idx, item_to_idx)
    # r, m = evaluate(mf_sgd, test)
    # results.append(("SGD", r, m))
    # print(f"SGD - RMSE: {r:.4f}, MAE: {m:.4f}")

    print("\nTraining Two-Tower (with genres)...")
    two_tower = TwoTowerModel(n_genres=N_GENRES, embed_dim=32, hidden_dim=64, lr=0.001, epochs=50, reg=0.01)
    two_tower.fit(train, n_users, n_items, user_to_idx, item_to_idx, movie_genres)
    r, m = evaluate(two_tower, test)
    results.append(("Two-Tower", r, m))
    print(f"Two-Tower - RMSE: {r:.4f}, MAE: {m:.4f}")

    print("\nTraining ALS...")
    als = ALS(n_factors=20, reg=0.1, epochs=15)
    als.fit(train, n_users, n_items, user_to_idx, item_to_idx)
    r, m = evaluate(als, test)
    results.append(("ALS", r, m))
    print(f"ALS - RMSE: {r:.4f}, MAE: {m:.4f}")

    # plot results
    names = [r[0] for r in results]
    rmse_vals = [r[1] for r in results]
    mae_vals = [r[2] for r in results]

    plt.figure(figsize=(10, 5))
    plt.bar(names, rmse_vals)
    plt.ylabel('RMSE')
    plt.title('Model Comparison - RMSE (lower is better)')
    plt.xticks(rotation=45)
    for i, v in enumerate(rmse_vals):
        plt.text(i, v + 0.01, f'{v:.3f}', ha='center')
    plt.tight_layout()
    plt.savefig('plots/plot_rmse.png')
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.bar(names, mae_vals)
    plt.ylabel('MAE')
    plt.title('Model Comparison - MAE (lower is better)')
    plt.xticks(rotation=45)
    for i, v in enumerate(mae_vals):
        plt.text(i, v + 0.01, f'{v:.3f}', ha='center')
    plt.tight_layout()
    plt.savefig('plots/plot_mae.png')
    plt.show()

    best = min(results, key=lambda x: x[1])
    print(f"\nBest model: {best[0]} (RMSE: {best[1]:.4f})")

    # generate table for presentation
    generate_results_table(results)

def generate_results_table(results):
    # add complexity info
    complexity = {
        "Baseline": "O(n + m)",
        "User-CF": "O(n² x m)",
        "Item-CF": "O(m² x n)",
        "Basic MF": "O(k x e x r)",
        "SVD": "O(n x m x k)",
        "SGD": "O(k x e x r)",
        "Two-Tower": "O(d x h x e x r)",
        "ALS": "O(k³ x e x (n+m))",
    }

    # sort by RMSE
    results_sorted = sorted(results, key=lambda x: x[1])

    # print table
    print("\n" + "="*60)
    print("RESULTS TABLE FOR PRESENTATION")
    print("="*60)
    print(f"{'Model':<12} {'RMSE':<10} {'MAE':<10} {'Complexity':<15}")
    print("-"*60)
    for name, r, m in results_sorted:
        comp = complexity.get(name, "")
        print(f"{name:<12} {r:<10.4f} {m:<10.4f} {comp:<15}")
    print("-"*60)

    # create simple table image
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('off')

    table_data = [["Model", "RMSE", "MAE", "Complexitate"]]
    for name, r, m in results_sorted:
        comp = complexity.get(name, "")
        table_data.append([name, f"{r:.4f}", f"{m:.4f}", comp])

    table = ax.table(cellText=table_data, loc='center', cellLoc='center')
    table.scale(1.2, 2)

    plt.title('Comparatie Modele - MovieLens Dataset')
    plt.savefig('plots/table_results.png')
    print("\nSaved: plots/table_results.png")

if __name__ == "__main__":
    main()
