import matplotlib.pyplot as plt
import numpy as np
from data import load_data, load_genres, N_GENRES
from utils import rmse, mae
from models import UserBasedCF, ItemBasedCF, MatrixFactorizationSGD, TwoTowerModel, ALS

def evaluate(model, test):
    preds = []
    actuals = []
    for user, item, rating in test:
        preds.append(model.predict(user, item))
        actuals.append(rating)
    return rmse(preds, actuals)

# load data
print("Loading data...")
train, test, user_to_idx, item_to_idx, n_users, n_items = load_data()
movie_genres = load_genres()

# 1. k for User-CF
print("\n" + "="*50)
print("Testing k for User-CF (20 values)...")
print("="*50)
k_values = list(range(1, 101, 5))
k_rmse_user = []
for k in k_values:
    print(f"  k={k}...", end=" ")
    model = UserBasedCF(k=k)
    model.fit(train)
    r = evaluate(model, test)
    k_rmse_user.append(r)
    print(f"RMSE: {r:.4f}")

plt.figure(figsize=(10, 6))
plt.plot(k_values, k_rmse_user, 'o-', color='blue')
plt.xlabel('k (number of neighbors)')
plt.ylabel('RMSE')
plt.title('User-CF: Sensitivity to k')
plt.grid(True)
plt.savefig('plots/tuning_user_cf_k.png')
print("Saved: plots/tuning_user_cf_k.png")

# 2. k for Item-CF
print("\n" + "="*50)
print("Testing k for Item-CF (20 values)...")
print("="*50)
k_rmse_item = []
for k in k_values:
    print(f"  k={k}...", end=" ")
    model = ItemBasedCF(k=k)
    model.fit(train)
    r = evaluate(model, test)
    k_rmse_item.append(r)
    print(f"RMSE: {r:.4f}")

plt.figure(figsize=(10, 6))
plt.plot(k_values, k_rmse_item, 'o-', color='green')
plt.xlabel('k (number of neighbors)')
plt.ylabel('RMSE')
plt.title('Item-CF: Sensitivity to k')
plt.grid(True)
plt.savefig('plots/tuning_item_cf_k.png')
print("Saved: plots/tuning_item_cf_k.png")

# 3. learning rate for SGD
print("\n" + "="*50)
print("Testing learning rate for SGD (20 values)...")
print("="*50)
lr_values = list(np.linspace(0.001, 0.05, 20))
lr_rmse = []
for lr in lr_values:
    print(f"  lr={lr:.4f}...", end=" ")
    model = MatrixFactorizationSGD(n_factors=20, lr=lr, epochs=20, reg=0.02)
    model.fit(train, n_users, n_items, user_to_idx, item_to_idx)
    r = evaluate(model, test)
    lr_rmse.append(r)
    print(f"RMSE: {r:.4f}")

plt.figure(figsize=(10, 6))
plt.plot(lr_values, lr_rmse, 'o-', color='orange')
plt.xlabel('Learning Rate')
plt.ylabel('RMSE')
plt.title('SGD: Sensitivity to Learning Rate')
plt.grid(True)
plt.savefig('plots/tuning_sgd_lr.png')
print("Saved: plots/tuning_sgd_lr.png")

# 4. n_factors for SGD
print("\n" + "="*50)
print("Testing n_factors for SGD (20 values)...")
print("="*50)
nf_values = list(range(5, 105, 5))
nf_rmse = []
for nf in nf_values:
    print(f"  n_factors={nf}...", end=" ")
    model = MatrixFactorizationSGD(n_factors=nf, lr=0.005, epochs=20, reg=0.02)
    model.fit(train, n_users, n_items, user_to_idx, item_to_idx)
    r = evaluate(model, test)
    nf_rmse.append(r)
    print(f"RMSE: {r:.4f}")

plt.figure(figsize=(10, 6))
plt.plot(nf_values, nf_rmse, 'o-', color='purple')
plt.xlabel('Number of Latent Factors')
plt.ylabel('RMSE')
plt.title('SGD: Sensitivity to Number of Factors')
plt.grid(True)
plt.savefig('plots/tuning_sgd_factors.png')
print("Saved: plots/tuning_sgd_factors.png")

# 5. regularization for SGD
print("\n" + "="*50)
print("Testing regularization for SGD (20 values)...")
print("="*50)
reg_values = list(np.linspace(0.001, 0.1, 20))
reg_rmse = []
for reg in reg_values:
    print(f"  reg={reg:.4f}...", end=" ")
    model = MatrixFactorizationSGD(n_factors=20, lr=0.005, epochs=20, reg=reg)
    model.fit(train, n_users, n_items, user_to_idx, item_to_idx)
    r = evaluate(model, test)
    reg_rmse.append(r)
    print(f"RMSE: {r:.4f}")

plt.figure(figsize=(10, 6))
plt.plot(reg_values, reg_rmse, 'o-', color='red')
plt.xlabel('Regularization (lambda)')
plt.ylabel('RMSE')
plt.title('SGD: Sensitivity to Regularization')
plt.grid(True)
plt.savefig('plots/tuning_sgd_reg.png')
print("Saved: plots/tuning_sgd_reg.png")

# 6. epochs for SGD
print("\n" + "="*50)
print("Testing epochs for SGD (20 values)...")
print("="*50)
epoch_values = list(range(5, 105, 5))
epoch_rmse = []
for ep in epoch_values:
    print(f"  epochs={ep}...", end=" ")
    model = MatrixFactorizationSGD(n_factors=20, lr=0.005, epochs=ep, reg=0.02)
    model.fit(train, n_users, n_items, user_to_idx, item_to_idx)
    r = evaluate(model, test)
    epoch_rmse.append(r)
    print(f"RMSE: {r:.4f}")

plt.figure(figsize=(10, 6))
plt.plot(epoch_values, epoch_rmse, 'o-', color='brown')
plt.xlabel('Number of Epochs')
plt.ylabel('RMSE')
plt.title('SGD: Sensitivity to Epochs')
plt.grid(True)
plt.savefig('plots/tuning_sgd_epochs.png')
print("Saved: plots/tuning_sgd_epochs.png")

# 7. embed_dim for Two-Tower
print("\n" + "="*50)
print("Testing embed_dim for Two-Tower (20 values)...")
print("="*50)
embed_values = list(range(8, 128, 6))
embed_rmse = []
for ed in embed_values:
    print(f"  embed_dim={ed}...", end=" ")
    model = TwoTowerModel(n_genres=N_GENRES, embed_dim=ed, hidden_dim=64, lr=0.001, epochs=20, reg=0.01)
    model.fit(train, n_users, n_items, user_to_idx, item_to_idx, movie_genres)
    r = evaluate(model, test)
    embed_rmse.append(r)
    print(f"RMSE: {r:.4f}")

plt.figure(figsize=(10, 6))
plt.plot(embed_values, embed_rmse, 'o-', color='cyan')
plt.xlabel('Embedding Dimension')
plt.ylabel('RMSE')
plt.title('Two-Tower: Sensitivity to Embedding Dimension')
plt.grid(True)
plt.savefig('plots/tuning_twotower_embed.png')
print("Saved: plots/tuning_twotower_embed.png")

# 8. hidden_dim for Two-Tower
print("\n" + "="*50)
print("Testing hidden_dim for Two-Tower (20 values)...")
print("="*50)
hidden_values = list(range(16, 256, 12))
hidden_rmse = []
for hd in hidden_values:
    print(f"  hidden_dim={hd}...", end=" ")
    model = TwoTowerModel(n_genres=N_GENRES, embed_dim=32, hidden_dim=hd, lr=0.001, epochs=20, reg=0.01)
    model.fit(train, n_users, n_items, user_to_idx, item_to_idx, movie_genres)
    r = evaluate(model, test)
    hidden_rmse.append(r)
    print(f"RMSE: {r:.4f}")

plt.figure(figsize=(10, 6))
plt.plot(hidden_values, hidden_rmse, 'o-', color='magenta')
plt.xlabel('Hidden Layer Dimension')
plt.ylabel('RMSE')
plt.title('Two-Tower: Sensitivity to Hidden Dimension')
plt.grid(True)
plt.savefig('plots/tuning_twotower_hidden.png')
print("Saved: plots/tuning_twotower_hidden.png")

# 9. learning rate for Two-Tower
print("\n" + "="*50)
print("Testing learning rate for Two-Tower (20 values)...")
print("="*50)
lr_tt_values = list(np.linspace(0.0005, 0.01, 20))
lr_tt_rmse = []
for lr in lr_tt_values:
    print(f"  lr={lr:.4f}...", end=" ")
    model = TwoTowerModel(n_genres=N_GENRES, embed_dim=32, hidden_dim=64, lr=lr, epochs=20, reg=0.01)
    model.fit(train, n_users, n_items, user_to_idx, item_to_idx, movie_genres)
    r = evaluate(model, test)
    lr_tt_rmse.append(r)
    print(f"RMSE: {r:.4f}")

plt.figure(figsize=(10, 6))
plt.plot(lr_tt_values, lr_tt_rmse, 'o-', color='teal')
plt.xlabel('Learning Rate')
plt.ylabel('RMSE')
plt.title('Two-Tower: Sensitivity to Learning Rate')
plt.grid(True)
plt.savefig('plots/tuning_twotower_lr.png')
print("Saved: plots/tuning_twotower_lr.png")

# 10. n_factors for ALS
print("\n" + "="*50)
print("Testing n_factors for ALS (20 values)...")
print("="*50)
als_nf_values = list(range(5, 105, 5))
als_nf_rmse = []
for nf in als_nf_values:
    print(f"  n_factors={nf}...", end=" ")
    model = ALS(n_factors=nf, reg=0.1, epochs=10)
    model.fit(train, n_users, n_items, user_to_idx, item_to_idx)
    r = evaluate(model, test)
    als_nf_rmse.append(r)
    print(f"RMSE: {r:.4f}")

plt.figure(figsize=(10, 6))
plt.plot(als_nf_values, als_nf_rmse, 'o-', color='navy')
plt.xlabel('Number of Latent Factors')
plt.ylabel('RMSE')
plt.title('ALS: Sensitivity to Number of Factors')
plt.grid(True)
plt.savefig('plots/tuning_als_factors.png')
print("Saved: plots/tuning_als_factors.png")

# 11. regularization for ALS
print("\n" + "="*50)
print("Testing regularization for ALS (20 values)...")
print("="*50)
als_reg_values = list(np.linspace(0.01, 0.5, 20))
als_reg_rmse = []
for reg in als_reg_values:
    print(f"  reg={reg:.3f}...", end=" ")
    model = ALS(n_factors=20, reg=reg, epochs=10)
    model.fit(train, n_users, n_items, user_to_idx, item_to_idx)
    r = evaluate(model, test)
    als_reg_rmse.append(r)
    print(f"RMSE: {r:.4f}")

plt.figure(figsize=(10, 6))
plt.plot(als_reg_values, als_reg_rmse, 'o-', color='darkgreen')
plt.xlabel('Regularization (lambda)')
plt.ylabel('RMSE')
plt.title('ALS: Sensitivity to Regularization')
plt.grid(True)
plt.savefig('plots/tuning_als_reg.png')
print("Saved: plots/tuning_als_reg.png")

print("\n" + "="*50)
print("ALL DONE!")
print("="*50)
print("Generated plots:")
print("  - plots/tuning_user_cf_k.png")
print("  - plots/tuning_item_cf_k.png")
print("  - plots/tuning_sgd_lr.png")
print("  - plots/tuning_sgd_factors.png")
print("  - plots/tuning_sgd_reg.png")
print("  - plots/tuning_sgd_epochs.png")
print("  - plots/tuning_twotower_embed.png")
print("  - plots/tuning_twotower_hidden.png")
print("  - plots/tuning_twotower_lr.png")
print("  - plots/tuning_als_factors.png")
print("  - plots/tuning_als_reg.png")
