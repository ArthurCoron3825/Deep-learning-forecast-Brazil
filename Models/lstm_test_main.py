"""
LSTM-based Daily Precipitation Prediction for Pixel Time Series

Description:
This script trains, validates, and tests a Long Short-Term Memory (LSTM) neural network
for predicting daily precipitation at individual pixels.
It supports normalization, denormalization, and visualization of results.

Main steps:
1. Load pixel-based sequences from INMET/ERA5 datasets using PixelSequenceDataset.
2. Train an LSTM model to predict the next-day precipitation.
3. Evaluate performance using Smooth L1 Loss, MAE, and R² metrics.
4. Generate plots for:
   - Training/Validation loss and MAE curves
   - Histograms of predicted vs. true precipitation
   - Scatter plot comparing predicted vs. true
   - Temporal sequence comparison


"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch import nn, optim
from sklearn.metrics import mean_absolute_error, r2_score
from LSTM.lstm_test_model import PixelSequenceDataset, LSTMModel

# Helper functions
def denormalize(scaler, arr):
    """Convert normalized values back to original scale and ensure non-negative precipitation."""
    return np.clip(np.expm1(scaler.inverse_transform(arr)), 0, None)

def train(model, loader, optimizer, criterion):
    """Perform one training epoch."""
    model.train()
    total_loss, total_mae = 0, 0
    for xb, yb in loader:
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        mae = nn.L1Loss()(pred, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_mae += mae.item()
    return total_loss / len(loader), total_mae / len(loader)

def evaluate(model, loader, criterion):
    """Evaluate model performance."""
    model.eval()
    total_loss, total_mae = 0, 0
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in loader:
            pred = model(xb)
            total_loss += criterion(pred, yb).item()
            total_mae += nn.L1Loss()(pred, yb).item()
            preds.append(pred.numpy())
            trues.append(yb.numpy())
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    return total_loss / len(loader), total_mae / len(loader), preds, trues

def plot_all(train_losses, val_losses, train_maes, val_maes, y_true_real, y_pred_real, y_true_norm, y_pred_norm):
    """Visualize training results, predictions, and comparison plots."""
    epochs = range(1, len(train_losses) + 1)

    # Loss and MAE curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.title("Loss Curves")
    plt.legend()
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_maes, label='Training MAE')
    plt.plot(epochs, val_maes, label='Validation MAE')
    plt.title("MAE Curves")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

    # Histogram comparison
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.hist(y_true_norm, bins=40, alpha=0.6, label='Normalized Target')
    plt.hist(y_pred_norm, bins=40, alpha=0.6, label='Normalized Prediction')
    plt.title("Normalized Histogram")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.hist(y_true_real, bins=40, alpha=0.6, label='True Target')
    plt.hist(y_pred_real, bins=40, alpha=0.6, label='Predicted Target')
    plt.title("Real Histogram")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Scatter plot
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true_real, y_pred_real, alpha=0.3)
    plt.plot([0, y_true_real.max()], [0, y_true_real.max()], 'r--')
    plt.xlabel("True (mm)")
    plt.ylabel("Predicted (mm)")
    plt.title("Scatter Plot: Predicted vs True")
    plt.grid()
    plt.tight_layout()
    plt.show()

    # Temporal sequence comparison
    plt.figure(figsize=(12, 4))
    plt.plot(y_true_real[:240], label='True')
    plt.plot(y_pred_real[:240], label='Predicted', linestyle='--')
    plt.title("Temporal Comparison (first 240 points)")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()


# Main execution
if __name__ == "__main__":
    # Load datasets
    train_data = PixelSequenceDataset("data/train", fit_scalers=True)
    val_data = PixelSequenceDataset(
        "data/val", fit_scalers=False,
        scaler_x=train_data.scaler_x, scaler_y=train_data.scaler_y
    )
    test_data = PixelSequenceDataset(
        "data/test", fit_scalers=False,
        scaler_x=train_data.scaler_x, scaler_y=train_data.scaler_y
    )

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32)
    test_loader = DataLoader(test_data, batch_size=32)

    # Initialize model, optimizer, and loss function
    model = LSTMModel(input_dim=train_data.X.shape[-1])
    optimizer = optim.Adam(model.parameters(), lr=4e-3)
    criterion = nn.SmoothL1Loss()

    train_losses, val_losses = [], []
    train_maes, val_maes = [], []

    best_val_loss = float("inf")
    patience, wait = 100, 0

    # Training loop
    for epoch in range(100):
        tr_loss, tr_mae = train(model, train_loader, optimizer, criterion)
        val_loss, val_mae, _, _ = evaluate(model, val_loader, criterion)
        print(f"Epoch {epoch+1:03d} | Train Loss: {tr_loss:.4f} | Val Loss: {val_loss:.4f}")
        train_losses.append(tr_loss)
        val_losses.append(val_loss)
        train_maes.append(tr_mae)
        val_maes.append(val_mae)

        best_val_loss = val_loss
        best_weights = model.state_dict().copy()

    # Load best weights and evaluate on test set
    model.load_state_dict(best_weights)
    test_loss, test_mae, y_pred_norm, y_true_norm = evaluate(model, test_loader, criterion)

    y_true_real = denormalize(train_data.scaler_y, y_true_norm)
    y_pred_real = denormalize(train_data.scaler_y, y_pred_norm)

    print(f"\nTest Results: Loss={test_loss:.4f}, MAE={test_mae:.4f}, R²={r2_score(y_true_real, y_pred_real):.4f}")

    # Plot all results
    plot_all(
        train_losses, val_losses, train_maes, val_maes,
        y_true_real.flatten(), y_pred_real.flatten(),
        y_true_norm.flatten(), y_pred_norm.flatten()
    )
