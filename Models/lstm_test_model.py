"""
LSTM Model and Pixel Sequence Dataset for Precipitation Prediction

Description:
This module defines:
1. PixelSequenceDataset: A PyTorch Dataset that extracts sequences of pixel-level
   features and corresponding target precipitation values from NetCDF files.
2. LSTMModel: A Long Short-Term Memory network for predicting daily precipitation
   based on the input sequences.

Features:
- Automatic scaling of input features and target values using StandardScaler.
- Log transformation of the target precipitation to stabilize variance.
- Handles missing values by skipping sequences with NaNs.
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import xarray as xr
from sklearn.preprocessing import StandardScaler

# Global constants
SEQ_LEN = 24          # Length of input sequence (hours/days)
TARGET_LAT = -3.0     # Latitude of target pixel
TARGET_LON = -44.25   # Longitude of target pixel

# Dataset class
class PixelSequenceDataset(Dataset):
    """
    PyTorch Dataset that creates sequences of pixel-based variables
    and targets for LSTM training.
    """
    def __init__(self, nc_dir, fit_scalers=False, scaler_x=None, scaler_y=None):
        # Load and concatenate NetCDF datasets
        files = sorted([os.path.join(nc_dir, f) for f in os.listdir(nc_dir) if f.endswith(".nc")])
        datasets = [xr.open_dataset(f, engine="netcdf4") for f in files]
        ds = xr.concat(datasets, dim="time")

        # Determine nearest pixel indices for target location
        self.lons = ds['longitude'].values
        self.lats = ds['latitude'].values
        lon_idx = np.argmin(np.abs(self.lons - TARGET_LON))
        lat_idx = np.argmin(np.abs(self.lats - TARGET_LAT))

        # Extract input features (exclude target variable 'tp') and target variable
        variables = [v for v in ds.data_vars if v != 'tp']
        x = ds[variables].to_array().transpose("time", "variable", "latitude", "longitude").values
        y = ds['tp'].values

        # Generate sequences and targets
        T = x.shape[0]
        self.X, self.Y = [], []
        for t in range(SEQ_LEN, T - 1):
            seq = x[t - SEQ_LEN:t, :, lat_idx, lon_idx]
            target = y[t + 1, lat_idx, lon_idx]
            if not np.isnan(seq).any() and not np.isnan(target):
                self.X.append(seq)
                self.Y.append(target)

        self.X = np.array(self.X)
        self.Y = np.array(self.Y).reshape(-1, 1)

        # Log-transform target to stabilize variance
        self.Y = np.log1p(self.Y)

        # Fit scalers if requested
        if fit_scalers:
            reshaped_X = self.X.reshape(-1, self.X.shape[-1])
            self.scaler_x = StandardScaler().fit(reshaped_X)
            self.scaler_y = StandardScaler().fit(self.Y)
        else:
            self.scaler_x = scaler_x
            self.scaler_y = scaler_y

        # Scale features and target
        reshaped_X = self.X.reshape(-1, self.X.shape[-1])
        reshaped_X = self.scaler_x.transform(reshaped_X)
        self.X = reshaped_X.reshape(self.X.shape)
        self.Y = self.scaler_y.transform(self.Y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.Y[idx], dtype=torch.float32)

# LSTM Model
class LSTMModel(nn.Module):
    """
    LSTM model for predicting precipitation from sequences of pixel features.
    """
    def __init__(self, input_dim, hidden_dim=128, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=0.3
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])
