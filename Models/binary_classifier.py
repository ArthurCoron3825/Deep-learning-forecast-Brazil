import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import xarray as xr
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple

# =========================
# Global Parameters
# =========================
TARGET_LAT = -3.0
TARGET_LON = -44.25
RAIN_THRESHOLD = 1.0   # mm
EPOCHS = 90
BATCH_SIZE = 32
LEARNING_RATE = 7e-5


# =========================
# PyTorch Dataset
# =========================
class RainBinaryDataset(Dataset):
    def __init__(self, nc_dir: str, n_steps: int = 5,
                 fit_scalers: bool = False,
                 scaler_x: Optional[StandardScaler] = None):
        """
        Custom Dataset that converts NetCDF files into binary classification samples.
        The task is to predict rain (1) vs no-rain (0).

        Args:
            nc_dir (str): Directory containing .nc files.
            n_steps (int): Number of time steps to use as input features.
            fit_scalers (bool): Whether to fit a StandardScaler on features.
            scaler_x (StandardScaler, optional): Pre-fitted scaler for test/val datasets.
        """
        files = sorted([os.path.join(nc_dir, f) for f in os.listdir(nc_dir) if f.endswith(".nc")])
        all_X, all_Y = [], []

        for f in files:
            ds = xr.open_dataset(f, engine="netcdf4")
            lons, lats = ds['longitude'].values, ds['latitude'].values

            # Find nearest grid point to target coordinates
            lon_idx = np.argmin(np.abs(lons - TARGET_LON))
            lat_idx = np.argmin(np.abs(lats - TARGET_LAT))

            # Extract variables except precipitation ("tp")
            variables = [v for v in ds.data_vars if v != 'tp']
            x = ds[variables].to_array().values  # shape: (var, time, lat, lon)
            y = ds['tp'].values  # shape: (time, lat, lon)

            T = x.shape[1]
            for t in range(n_steps - 1, T):
                feat_seq = x[:, t - n_steps + 1:t + 1, lat_idx, lon_idx]
                target = y[t, lat_idx, lon_idx]
                if not np.isnan(feat_seq).any() and not np.isnan(target):
                    input_vector = feat_seq.flatten()
                    label = int(target > RAIN_THRESHOLD)
                    all_X.append(input_vector)
                    all_Y.append(label)

        self.X = np.array(all_X)
        self.Y = np.array(all_Y).reshape(-1, 1)

        # Scale features
        if fit_scalers:
            self.scaler_x = StandardScaler().fit(self.X)
        else:
            self.scaler_x = scaler_x
        self.X = self.scaler_x.transform(self.X)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.tensor(self.X[idx], dtype=torch.float32),
            torch.tensor(self.Y[idx], dtype=torch.float32)
        )


# =========================
# Neural Network Model
# =========================
class BinaryMLP(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# =========================
# Training Function
# =========================
def train(model: nn.Module, loader: DataLoader, optimizer, criterion) -> float:
    model.train()
    total_loss = 0.0
    for xb, yb in loader:
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


# =========================
# Evaluation Function
# =========================
def evaluate(model: nn.Module, loader: DataLoader, criterion):
    model.eval()
    total_loss = 0.0
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
    preds_class = (preds > 0.5).astype(int)

    acc = accuracy_score(trues, preds_class)
    prec = precision_score(trues, preds_class)
    rec = recall_score(trues, preds_class)
    f1 = f1_score(trues, preds_class)

    return total_loss / len(loader), acc, prec, rec, f1, preds, trues


# =========================
# Main Script
# =========================
if __name__ == "__main__":
    n_steps = 1  # Number of time steps for sequence input

    # Load datasets
    train_data = RainBinaryDataset("data/train", n_steps=n_steps, fit_scalers=True)
    val_data = RainBinaryDataset("data/val", n_steps=n_steps, scaler_x=train_data.scaler_x)
    test_data = RainBinaryDataset("data/test", n_steps=n_steps, scaler_x=train_data.scaler_x)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)

    # Initialize model
    input_dim = train_data.X.shape[1]
    model = BinaryMLP(input_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCELoss()

    # Training loop
    best_val_loss = float("inf")
    train_losses, val_losses = [], []

    for epoch in range(EPOCHS):
        train_loss = train(model, train_loader, optimizer, criterion)
        val_loss, *_ = evaluate(model, val_loader, criterion)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()

    # Load best model
    model.load_state_dict(best_model_state)

    # Final evaluation on test set
    print("\n Test set results:")
    test_loss, acc, prec, rec, f1, y_prob, y_true = evaluate(model, test_loader, criterion)
    y_pred = (y_prob > 0.5).astype(int)

    print(f"Test Loss   : {test_loss:.4f}")
    print(f"Accuracy    : {acc:.4f}")
    print(f"Precision   : {prec:.4f}")
    print(f"Recall      : {rec:.4f}")
    print(f"F1 Score    : {f1:.4f}")

    # Confusion matrix - train
    train_loss, acc_train, prec_train, rec_train, f1_train, y_prob_train, y_true_train = evaluate(model, train_loader, criterion)
    y_pred_train = (y_prob_train > 0.5).astype(int)

    print("\n Training set results:")
    print(f"Train Loss   : {train_loss:.4f}")
    print(f"Accuracy     : {acc_train:.4f}")
    print(f"Precision    : {prec_train:.4f}")
    print(f"Recall       : {rec_train:.4f}")
    print(f"F1 Score     : {f1_train:.4f}")

    cm_train = confusion_matrix(y_true_train, y_pred_train)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm_train, annot=True, fmt="d", cmap="Oranges", cbar=False,
                xticklabels=["No Rain", "Rain"], yticklabels=["No Rain", "Rain"])
    plt.title("Confusion Matrix (Training set)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

    # Confusion matrix - test
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["No Rain", "Rain"], yticklabels=["No Rain", "Rain"])
    plt.title("Confusion Matrix (Test set)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

    # Loss curves
    plt.figure(figsize=(6, 4))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("BCELoss")
    plt.title("Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
