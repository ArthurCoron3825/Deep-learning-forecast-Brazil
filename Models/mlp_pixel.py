"""
Precipitation Prediction Using ERA5 Pixel Data and MLP

Description:
This script trains, validates, and tests a Multi-Layer Perceptron (MLP) model 
to predict precipitation at a target pixel location (lat/lon) using ERA5 reanalysis 
variables as input features. The pipeline includes:

1. Data preprocessing:
   - Extraction of features (ERA5 variables excluding total precipitation) and target (tp).
   - Spatial selection of nearest grid point to the target location.
   - Construction of temporal sequences with optional one-hot month encoding.
   - Standardization of inputs and targets.

2. Model training:
   - MLP with dropout and LeakyReLU activations.
   - Loss: Mean Absolute Error (L1Loss).
   - Optimizer: Adam with learning rate scheduler.

3. Evaluation:
   - Computes MAE, RMSE, R², and plots histograms, scatter plots, loss curves, and error distributions.
   - Binary classification evaluation for rainfall detection (F1-score).
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import xarray as xr
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, root_mean_squared_error
import matplotlib.pyplot as plt
import pandas as pd

# Target pixel coordinates
TARGET_LAT = -3
TARGET_LON = -44.25


class PixelPrecipDataset(Dataset):
    """
    Dataset class for ERA5 pixel precipitation prediction.

    Loads multiple NetCDF files, extracts ERA5 variables and total precipitation (tp),
    builds feature sequences, and applies normalization.
    """

    def __init__(self, nc_dir, n_steps, fit_scalers=False, scaler_x=None, scaler_y=None, verbose=True):
        files = sorted([os.path.join(nc_dir, f) for f in os.listdir(nc_dir) if f.endswith(".nc")])
        all_X, all_Y = [], []

        for f in files:
            ds = xr.open_dataset(f, engine="netcdf4")
            lons = ds['longitude'].values
            lats = ds['latitude'].values
            lon_idx = np.argmin(np.abs(lons - TARGET_LON))
            lat_idx = np.argmin(np.abs(lats - TARGET_LAT))

            # Extract variables
            variables = [v for v in ds.data_vars if v != 'tp']
            x = ds[variables].to_array().values  # (nb_var, T, lat, lon)
            y = ds['tp'].values  # (T, lat, lon)
            times = ds['time'].values

            T = x.shape[1]
            for t in range(n_steps - 1, T - 1):
                feat_seq = x[:, t - n_steps + 1:t + 1, lat_idx, lon_idx]
                target = y[t, lat_idx, lon_idx]
                if not np.isnan(feat_seq).any() and not np.isnan(target):
                    # One-hot encoding of month (Nov–Jun)
                    month = pd.to_datetime(times[t]).month
                    month_map = {11: 0, 12: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7}
                    month_idx = month_map.get(month, -1)
                    if month_idx == -1:
                        continue  # skip excluded months

                    month_one_hot = np.eye(8)[month_idx]

                    # Season start flag
                    is_new_season = int(t == n_steps - 1)

                    # Flatten temporal data + concatenate extra features
                    input_vector = feat_seq.flatten()
                    input_vector = np.concatenate([input_vector, month_one_hot, [is_new_season]])

                    all_X.append(input_vector)
                    all_Y.append(target)

        self.X = np.array(all_X)
        self.Y = np.array(all_Y).reshape(-1, 1)

        if verbose:
            print("Raw tp stats (before log+normalization):")
            print("Min:", np.nanmin(self.Y), "Max:", np.nanmax(self.Y), "Mean:", np.nanmean(self.Y))

        if fit_scalers:
            self.scaler_x = StandardScaler().fit(self.X)
            self.scaler_y = StandardScaler().fit(self.Y)
        else:
            self.scaler_x = scaler_x
            self.scaler_y = scaler_y

        self.X = self.scaler_x.transform(self.X)
        self.Y = self.scaler_y.transform(self.Y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.Y[idx], dtype=torch.float32)


class MLP(nn.Module):
    """Simple MLP model for precipitation regression."""

    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 8),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        return self.model(x)


def train(model, loader, optimizer, criterion):
    """Train the model for one epoch."""
    model.train()
    total_loss = 0
    for xb, yb in loader:
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, criterion):
    """Evaluate the model and compute metrics (loss, MAE, R², RMSE)."""
    model.eval()
    total_loss = 0
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in loader:
            out = model(xb)
            loss = criterion(out, yb)
            total_loss += loss.item()
            preds.append(out.cpu().numpy())
            trues.append(yb.cpu().numpy())
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)

    mae = mean_absolute_error(trues, preds)
    r2 = r2_score(trues, preds)
    rmse = root_mean_squared_error(trues, preds)

    return total_loss / len(loader), mae, r2, rmse, preds, trues


def denormalize(scaler_y, arr):
    """Inverse transform predictions to original scale."""
    arr_inv = scaler_y.inverse_transform(arr)
    return np.clip(arr_inv, 0, None)


def plot_histograms(name, y_norm, preds_norm, y_real, preds_real):
    """Plot normalized and real-valued histograms for targets vs predictions."""
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.hist(y_norm, bins=40, alpha=0.7, label='Target (normalized)')
    plt.hist(preds_norm, bins=40, alpha=0.7, label='Prediction (normalized)')
    plt.title(f"Normalized Histograms - {name}")
    plt.xlabel("Normalized Values")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.hist(y_real, bins=40, alpha=0.7, label='Target (real)')
    plt.hist(preds_real, bins=40, alpha=0.7, label='Prediction (real)')
    plt.title(f"Real Histograms - {name}")
    plt.xlabel("Precipitation (mm)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


# =========================
# MAIN TRAINING SCRIPT
# =========================
if __name__ == "__main__":
    n_steps = 1
    train_data = PixelPrecipDataset("data/train", n_steps=n_steps, fit_scalers=True)
    val_data = PixelPrecipDataset("data/val", n_steps=n_steps, fit_scalers=False, scaler_x=train_data.scaler_x, scaler_y=train_data.scaler_y)
    test_data = PixelPrecipDataset("data/test", n_steps=n_steps, fit_scalers=False, scaler_x=train_data.scaler_x, scaler_y=train_data.scaler_y)

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32)
    test_loader = DataLoader(test_data, batch_size=32)

    input_dim = train_data.X.shape[1]
    model = MLP(input_dim=input_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    criterion = nn.L1Loss()
    train_losses, val_losses = [], []
    best_val_loss = float("inf")

    for epoch in range(60):
        train_loss = train(model, train_loader, optimizer, criterion)
        val_loss, _, _, _, _, _ = evaluate(model, val_loader, criterion)
        scheduler.step(val_loss)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f"Epoch {epoch+1}: training loss = {train_loss:.4f} | validation loss = {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()

    model.load_state_dict(best_model_state)

    # Final evaluation
    _, _, _, _, preds_train, trues_train = evaluate(model, train_loader, criterion)
    _, _, _, _, preds_val, trues_val = evaluate(model, val_loader, criterion)
    test_loss, test_mae, test_r2, test_rmse, preds_test, trues_test = evaluate(model, test_loader, criterion)

    # Denormalization
    trues_train_real = denormalize(train_data.scaler_y, trues_train)
    preds_train_real = denormalize(train_data.scaler_y, preds_train)
    trues_val_real = denormalize(train_data.scaler_y, trues_val)
    preds_val_real = denormalize(train_data.scaler_y, preds_val)
    trues_test_real = denormalize(train_data.scaler_y, trues_test)
    preds_test_real = denormalize(train_data.scaler_y, preds_test)

    # Real metric computation
    real_mae = mean_absolute_error(trues_test_real, preds_test_real)
    real_r2 = r2_score(trues_test_real, preds_test_real)
    real_rmse = root_mean_squared_error(trues_test_real, preds_test_real)
    real_mse = mean_squared_error(trues_test_real, preds_test_real)

    metrics_names = ['MSE', 'MAE', 'RMSE', 'R²']
    normalized_values = [test_loss, test_mae, test_rmse, test_r2]
    real_values = [real_mse, real_mae, real_rmse, real_r2]

    # Results table
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.axis('off')
    table_data = [
        [name, f"{norm:.4f}", f"{real:.2f}"]
        for name, norm, real in zip(metrics_names, normalized_values, real_values)
    ]
    table = ax.table(cellText=table_data,
                    colLabels=["Metric", "Normalized", "Real (mm)"],
                    cellLoc='center',
                    loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.4)
    ax.set_title("📊 Test Set Metrics", fontweight='bold')
    plt.tight_layout()
    plt.show()

    # Histograms
    plot_histograms("Train", trues_train.flatten(), preds_train.flatten(), trues_train_real.flatten(), preds_train_real.flatten())
    plot_histograms("Validation", trues_val.flatten(), preds_val.flatten(), trues_val_real.flatten(), preds_val_real.flatten())
    plot_histograms("Test", trues_test.flatten(), preds_test.flatten(), trues_test_real.flatten(), preds_test_real.flatten())

    # Binary rain/no-rain evaluation
    threshold = 0.1
    pred_binary = (preds_test_real > threshold).astype(int)
    true_binary = (trues_test_real > threshold).astype(int)
    tp = np.sum((pred_binary == 1) & (true_binary == 1))
    fp = np.sum((pred_binary == 1) & (true_binary == 0))
    fn = np.sum((pred_binary == 0) & (true_binary == 1))
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1_score = 2 * precision * recall / (precision + recall + 1e-6)
    print(f"F1-score for rainfall (>{threshold}mm): {f1_score:.4f}")

    # Loss curves
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title("Loss Curves")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Time-series comparison
    plt.figure(figsize=(10, 4))
    plt.plot(trues_test_real[:100], label='True Precipitation')
    plt.plot(preds_test_real[:100], label='Predicted', linestyle='--')
    plt.title("True vs Predicted Precipitation (first 100 points)")
    plt.xlabel("Time")
    plt.ylabel("Precipitation (mm)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Error distribution
    plt.figure(figsize=(6, 4))
    plt.hist(preds_test_real - trues_test_real, bins=40, edgecolor='black', alpha=0.7)
    plt.title("Error Distribution (Predicted - True)")
    plt.xlabel("Error (mm)")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Scatter plot real vs predicted
    plt.figure(figsize=(6, 6))
    plt.scatter(trues_test_real, preds_test_real, alpha=0.3, edgecolor='k', linewidth=0.5)
    plt.plot([0, trues_test_real.max()], [0, trues_test_real.max()], 'r--', label='y = x')
    plt.title("Scatter Plot: True vs Predicted")
    plt.xlabel("True Precipitation (mm)")
    plt.ylabel("Predicted Precipitation (mm)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
