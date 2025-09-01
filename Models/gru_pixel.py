import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, root_mean_squared_error

# Target pixel coordinates for precipitation prediction
TARGET_LAT = -3
TARGET_LON = -44.25

# --------------------------- Custom Dataset --------------------------- #
class PixelPrecipDataset(Dataset):
    """
    Custom dataset to create temporal sequences of precipitation data
    for a specific pixel (TARGET_LAT, TARGET_LON), including additional variables.
    """
    def __init__(self, nc_dir, n_steps=10, fit_scalers=False, scaler_x=None, scaler_y=None, verbose=True):
        # List all NetCDF files in the directory
        files = sorted(f for f in os.listdir(nc_dir) if f.endswith(".nc"))
        all_X, all_Y = [], []  # lists to store sequences and targets

        for fname in files:
            ds = xr.open_dataset(os.path.join(nc_dir, fname))  # open dataset
            lons, lats = ds.longitude.values, ds.latitude.values
            i_lon = np.argmin(np.abs(lons - TARGET_LON))  # index of target longitude
            i_lat = np.argmin(np.abs(lats - TARGET_LAT))  # index of target latitude

            # Extract all variables except precipitation
            x = ds[[v for v in ds.data_vars if v != 'tp']].to_array().values  # shape: vars, T, lat, lon
            y = ds.tp.values  # precipitation values
            times = ds.time.values
            T = x.shape[1]

            # Loop to create temporal sequences
            for t in range(n_steps - 1, T - 1):
                seq = x[:, t-n_steps+1:t+1, i_lat, i_lon]  # sequence of length n_steps
                target = y[t, i_lat, i_lon]  # target value

                # Skip sequences with NaNs
                if np.isnan(seq).any() or np.isnan(target):
                    continue

                # One-hot encode the month for additional variables
                month = pd.to_datetime(times[t]).month
                month_map = {11:0,12:1,1:2,2:3,3:4,4:5,5:6,6:7}
                idx = month_map.get(month, -1)
                if idx == -1:
                    continue
                onehot = np.eye(8)[idx]
                is_new = int(t == n_steps - 1)  # flag for first sequence

                # Transpose sequence to shape: n_steps, vars
                seq = seq.transpose(1, 0)
                # Repeat meta info for each timestep
                meta = np.tile(np.concatenate([onehot, [is_new]]), (n_steps, 1))
                # Concatenate sequence and metadata
                full = np.concatenate([seq, meta], axis=1)

                all_X.append(full)
                all_Y.append(target)

        # Convert to numpy arrays
        self.X = np.array(all_X)  # shape: N, n_steps, F
        self.Y = np.array(all_Y).reshape(-1, 1)

        if verbose:
            print("Raw stats:", self.Y.min(), self.Y.max(), self.Y.mean())

        N, _, F = self.X.shape
        flatX = self.X.reshape(-1, F)

        # Scaling/Normalization
        if fit_scalers:
            self.scaler_x = StandardScaler().fit(flatX)
            self.scaler_y = StandardScaler().fit(self.Y)
        else:
            self.scaler_x, self.scaler_y = scaler_x, scaler_y

        self.X = self.scaler_x.transform(flatX).reshape(N, n_steps, F)
        self.Y = self.scaler_y.transform(self.Y)

        if verbose:
            print("→ Final samples:", len(self.X))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Return a single sample as a PyTorch tensor
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.Y[idx], dtype=torch.float32)


# --------------------------- GRU Model --------------------------- #
class GRUModel(nn.Module):
    """
    GRU model with multiple layers followed by fully connected layers.
    """
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super().__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True, dropout=0.15)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.LeakyReLU(),
            nn.Dropout(0.15),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # Forward pass through GRU
        out, _ = self.gru(x)
        # Take only the last timestep output
        out = self.fc(out[:, -1])
        return out


# --------------------------- Training Function --------------------------- #
def train(model, loader, optimizer, criterion):
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


# --------------------------- Evaluation Function --------------------------- #
def evaluate(model, loader, criterion):
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

    # Compute standard metrics
    mae = mean_absolute_error(trues, preds)
    r2 = r2_score(trues, preds)
    rmse = root_mean_squared_error(trues, preds)

    return total_loss / len(loader), mae, r2, rmse, preds, trues


# --------------------------- Denormalization --------------------------- #
def denormalize(scaler_y, arr):
    # Inverse transform and clip negative values
    return np.clip(scaler_y.inverse_transform(arr), 0, None)


# --------------------------- Histogram Plotting --------------------------- #
def plot_histograms(name, y_norm, preds_norm, y_real, preds_test_real):
    # Plot normalized and denormalized histograms for comparison
    plt.figure(figsize=(12, 5))

    # Normalized histograms
    plt.subplot(1, 2, 1)
    plt.hist(y_norm, bins=40, alpha=0.7, label='Normalized Target')
    plt.hist(preds_norm, bins=40, alpha=0.7, label='Normalized Prediction')
    plt.title(f"Normalized Histograms - {name}")
    plt.xlabel("Normalized Values")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)

    # Real (denormalized) histograms
    plt.subplot(1, 2, 2)
    plt.hist(y_real, bins=40, alpha=0.7, label='Real Target')
    plt.hist(preds_test_real, bins=40, alpha=0.7, label='Real Prediction')
    plt.title(f"Denormalized Histograms - {name}")
    plt.xlabel("Precipitation (mm)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


# --------------------------- Main Script --------------------------- #
if __name__ == "__main__":
    n_steps = 10
    # Load datasets
    train_data = PixelPrecipDataset("data/train", n_steps=n_steps, fit_scalers=True)
    val_data = PixelPrecipDataset("data/val", n_steps=n_steps, fit_scalers=False, 
                                  scaler_x=train_data.scaler_x, scaler_y=train_data.scaler_y)
    test_data = PixelPrecipDataset("data/test", n_steps=n_steps, fit_scalers=False, 
                                   scaler_x=train_data.scaler_x, scaler_y=train_data.scaler_y)

    # DataLoaders for batching
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32)
    test_loader = DataLoader(test_data, batch_size=32)

    # Model, optimizer, scheduler, loss
    input_size = train_data.X.shape[2]
    model = GRUModel(input_size=input_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=2.5e-4, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    criterion = nn.L1Loss()

    # Lists to store losses and metrics
    train_losses, val_losses = [], []
    train_maes, val_maes = [], []

    best_val_loss = float("inf")
    best_model_state = None

    # Early stopping parameters
    patience = 6
    min_delta = 4e-3
    early_stop_counter = 0

    # --------------------------- Training Loop --------------------------- #
    for epoch in range(100):  # large number of epochs for early stopping
        train_loss = train(model, train_loader, optimizer, criterion)
        val_loss, val_mae, _, _, _, _ = evaluate(model, val_loader, criterion)
        _, train_mae, _, _, _, _ = evaluate(model, train_loader, criterion)
       
        scheduler.step(val_loss)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_maes.append(train_mae)
        val_maes.append(val_mae)

        print(f"Epoch {epoch+1}: train loss = {train_loss:.4f} | val loss = {val_loss:.4f}")

        # Early stopping
        if best_val_loss - val_loss > min_delta:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            print(f"→ Early stopping patience: {early_stop_counter}/{patience}")
            if early_stop_counter >= patience:
                print("🛑 Early stopping triggered.")
                break

    model.load_state_dict(best_model_state)

    # --------------------------- Plot metrics --------------------------- #
    plt.figure(figsize=(8, 5))
    plt.plot(train_maes, label='Train MAE')
    plt.plot(val_maes, label='Validation MAE')
    plt.xlabel("Epoch")
    plt.ylabel("MAE")
    plt.title("MAE evolution during training")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
