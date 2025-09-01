import os
import torch
import xarray as xr
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import cartopy.crs as ccrs
import cartopy.feature as cfeature

"""
Old Precipitation Forecasting with U-Net

This script trains and evaluates a U-Net model to predict daily precipitation fields
based on ERA5 climate reanalysis data. The workflow includes:
- Loading and normalizing NetCDF climate data
- Building a PyTorch dataset and dataloaders
- Defining and training a U-Net model
- Evaluating model skill with precision, recall, F1, and confusion matrix
- Visualizing true vs. predicted precipitation maps
"""

# ---------- Dataset ----------
class PrecipDataset(Dataset):
    """Custom dataset for precipitation forecasting.
    Loads ERA5 NetCDF files, normalizes features, and creates (x_t, y_{t+1}) pairs.
    """
    def __init__(self, nc_dir, normalizer=None):
        print("Loaded PrecipDataset (concatenated)")
        self.inputs = []
        self.targets = []
        self.normalizer = normalizer

        # Load and concatenate all NetCDF files along the time dimension
        files = sorted([os.path.join(nc_dir, f) for f in os.listdir(nc_dir) if f.endswith(".nc")])
        datasets = [xr.open_dataset(f, engine="netcdf4") for f in files]
        ds = xr.concat(datasets, dim="time")

        # Restrict to the first 6 months of the year
        ds = ds.sel(time=ds['time'].dt.month <= 6)

        self.lons = ds['longitude'].values
        self.lats = ds['latitude'].values

        # Select input variables (all except precipitation "tp")
        variables = [v for v in ds.data_vars if v != 'tp']
        x = ds[variables].to_array().values    # (num_vars, T, H, W)
        y = ds['tp'].values                    # (T, H, W)

        # Check time dimension
        total_timesteps = x.shape[1]
        if total_timesteps < 2:
            raise ValueError("Not enough timesteps in dataset")

        # Normalize if a normalizer is provided
        if self.normalizer is not None:
            x = self.normalizer.normalize(x)

        # Build pairs (x_t, y_{t+1})
        for t in range(total_timesteps - 1):
            self.inputs.append(x[:, t, :, :])
            self.targets.append(y[t + 1][None, :, :])

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        x = torch.tensor(self.inputs[idx], dtype=torch.float32)
        y = torch.tensor(self.targets[idx], dtype=torch.float32)
        return x, y


# ---------- U-Net ----------
class UNet(nn.Module):
    """U-Net model for precipitation prediction."""
    def __init__(self, in_channels=34, out_channels=1):
        super(UNet, self).__init__()

        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Dropout(0.4)
            )

        # Encoder
        self.enc1 = conv_block(in_channels, 32)
        self.enc2 = conv_block(32, 64)
        self.enc3 = conv_block(64, 128)
        self.enc4 = conv_block(128, 256)
        self.pool = nn.MaxPool2d(2)
        self.bottom = conv_block(256, 512)

        # Decoder
        self.up4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec4 = conv_block(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = conv_block(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = conv_block(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = conv_block(64, 32)

        self.out_conv = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Bottleneck
        b = self.bottom(self.pool(e4))

        # Decoder with skip connections
        d4 = self.dec4(torch.cat([self.up4(b), self.crop_to_match(e4, self.up4(b))], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), self.crop_to_match(e3, self.up3(d4))], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), self.crop_to_match(e2, self.up2(d3))], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), self.crop_to_match(e1, self.up1(d2))], dim=1))

        return self.out_conv(d1)

    def crop_to_match(self, enc_feature, dec_feature):
        """Ensure encoder and decoder features have the same spatial size."""
        _, _, h1, w1 = enc_feature.shape
        _, _, h2, w2 = dec_feature.shape
        if h1 != h2 or w1 != w2:
            enc_feature = enc_feature[:, :, :h2, :w2]
        return enc_feature


# ---------- Normalizer ----------
class Normalizer:
    """Compute and apply feature-wise normalization statistics."""
    def __init__(self, stats_file=None):
        self.mean = None
        self.std = None
        self.stats_file = stats_file

        if stats_file and os.path.exists(stats_file):
            data = np.load(stats_file)
            self.mean = data['mean']
            self.std = data['std']

    def fit(self, nc_dir):
        print("Calculating normalization statistics...")
        files = sorted([os.path.join(nc_dir, f) for f in os.listdir(nc_dir) if f.endswith(".nc")])
        ds = xr.open_mfdataset(files, concat_dim="time", combine="by_coords", engine="netcdf4")

        variables = [v for v in ds.data_vars if v != 'precipitation']
        x = ds[variables].to_array().values

        self.mean = x.mean(axis=(1, 2, 3))
        self.std = x.std(axis=(1, 2, 3))

        if self.stats_file:
            np.savez(self.stats_file, mean=self.mean, std=self.std)
        print("Stats saved.")

    def normalize(self, x):
        return (x - self.mean[:, None, None, None]) / (self.std[:, None, None, None] + 1e-8)


# ---------- Training & Evaluation ----------
def train_epoch(model, loader, optimizer, criterion, device):
    """Run one training epoch."""
    model.train()
    total_loss = 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        X += torch.randn_like(X) * 0.01  # Add small Gaussian noise (data augmentation)
        pred = model(X)
        _, _, h_pred, w_pred = pred.shape
        y = y[:, :, :h_pred, :w_pred]
        y_log = torch.log1p(y)
        loss = criterion(pred, y_log)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def eval_epoch(model, loader, criterion, device):
    """Evaluate one epoch without gradient updates."""
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            _, _, h_pred, w_pred = pred.shape
            y = y[:, :, :h_pred, :w_pred]
            y_log = torch.log1p(y)
            loss = criterion(pred, y_log)
            total_loss += loss.item()
    return total_loss / len(loader)


# ---------- Visualization ----------
def plot_example(model, loader, device, lons, lats):
    """Plot true vs predicted precipitation maps for sample batches."""
    model.eval()
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            y_true = y.cpu().numpy()
            y_pred = torch.expm1(pred).cpu().numpy()

            fig, axes = plt.subplots(5, 2, figsize=(14, 25),
                                     subplot_kw={'projection': ccrs.PlateCarree()})

            extent = [-50, -30, -15, 5]  # NE Brazil

            for i in range(min(5, y_true.shape[0])):
                ax_true = axes[i, 0]
                ax_pred = axes[i, 1]

                im0 = ax_true.pcolormesh(lons, lats, y_true[i, 0, :, :], cmap='Blues', shading='auto')
                ax_true.set_title(f"True Precipitation - Sample {i+1}")
                ax_true.coastlines(resolution='10m')
                ax_true.add_feature(cfeature.BORDERS, linestyle=':')
                ax_true.set_extent(extent)

                im1 = ax_pred.pcolormesh(lons, lats, y_pred[i, 0, :, :], cmap='Blues', shading='auto')
                ax_pred.set_title(f"Predicted Precipitation - Sample {i+1}")
                ax_pred.coastlines(resolution='10m')
                ax_pred.add_feature(cfeature.BORDERS, linestyle=':')
                ax_pred.set_extent(extent)

                fig.colorbar(im0, ax=ax_true, orientation='vertical', fraction=0.046, pad=0.04)
                fig.colorbar(im1, ax=ax_pred, orientation='vertical', fraction=0.046, pad=0.04)

            plt.tight_layout()
            plt.show()
            break

def plot_precision_curve(precisions):
    plt.figure(figsize=(8, 4))
    plt.plot(precisions, label="Precision")
    plt.xlabel("Epoch")
    plt.ylabel("Precision")
    plt.title("Precision over epochs")
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_loss_curves(train_losses, val_losses):
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def eval_metrics(model, loader, device, threshold=0.1):
    """Compute precision, recall, F1, and confusion matrix."""
    model.eval()
    y_true_all = []
    y_pred_all = []

    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            _, _, h_pred, w_pred = pred.shape
            y = y[:, :, :h_pred, :w_pred]

            y_np = y.cpu().numpy().flatten()
            pred_np = pred.cpu().numpy().flatten()

            y_bin = (y_np > threshold).astype(int)
            pred_bin = (pred_np > threshold).astype(int)

            y_true_all.extend(y_bin)
            y_pred_all.extend(pred_bin)

    precision = precision_score(y_true_all, y_pred_all, zero_division=0)
    recall = recall_score(y_true_all, y_pred_all, zero_division=0)
    f1 = f1_score(y_true_all, y_pred_all, zero_division=0)
    cm = confusion_matrix(y_true_all, y_pred_all)

    return precision, recall, f1, cm

def plot_confusion_matrix(cm):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=["No Rain", "Rain"], yticklabels=["No Rain", "Rain"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()


# ---------- Main -----------
def main():
    ds = xr.open_dataset("data/test/daily_era5_202201.nc")
    print(ds)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Normalizer
    normalizer = Normalizer("normalization_stats.npz")
    if normalizer.mean is None:
        normalizer.fit("data/train")

    # Datasets & loaders
    train_dataset = PrecipDataset("data/train", normalizer)
    lons, lats = train_dataset.lons, train_dataset.lats

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(PrecipDataset("data/val", normalizer), batch_size=16)
    test_loader = DataLoader(PrecipDataset("data/test", normalizer), batch_size=16)

    # Model & optimizer
    model = UNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.SmoothL1Loss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    # Training loop
    train_losses, val_losses, precision_scores = [], [], []

    for epoch in range(60):
        # Validation precision (binary rain detection)
        y_true_all, y_pred_all = [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                pred_batch = torch.expm1(model(X_batch))
                y_batch = torch.expm1(y_batch)

                _, _, h_pred, w_pred = pred_batch.shape
                y_batch = y_batch[:, :, :h_pred, :w_pred]
                pred_batch = pred_batch[:, :, :h_pred, :w_pred]

                y_pred_bin = (pred_batch > 0.1).cpu().numpy().flatten()
                y_true_bin = (y_batch > 0.1).cpu().numpy().flatten()

                y_pred_all.extend(y_pred_bin)
                y_true_all.extend(y_true_bin)

        precision = precision_score(y_true_all, y_pred_all, zero_division=0)
        precision_scores.append(precision)

        # Train and validate
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = eval_epoch(model, val_loader, criterion, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        scheduler.step(val_loss)

    # Post-training evaluation
    plot_loss_curves(train_losses, val_losses)
    test_loss = eval_epoch(model, test_loader, criterion, device)
    precision, recall, f1, cm = eval_metrics(model, test_loader, device)
    print(f"\nTest Loss: {test_loss:.4f} | Precision: {precision:.3f} | Recall: {recall:.3f} | F1: {f1:.3f}")

    plot_example(model, test_loader, device, lons, lats)
    plot_precision_curve(precision_scores)
    plot_confusion_matrix(cm)


if __name__ == "__main__":
    main()
