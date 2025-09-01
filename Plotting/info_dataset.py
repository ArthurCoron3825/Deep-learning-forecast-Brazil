"""
Statistical Analysis of Precipitation at a Target Location

Description:
This script analyzes daily precipitation data from ERA5 (or similar NetCDF datasets) 
for a specific target latitude/longitude point. It computes basic statistics 
(e.g., number of dry days, very light rain days), generates summary tables, 
and produces multiple visualizations to better understand precipitation behavior.
"""

import os
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# === CONFIGURATION ===
DATA_DIR = "data/test"
TARGET_LAT = -8.0       # Target latitude for analysis
TARGET_LON = -34.5      # Target longitude for analysis

# === INITIALIZATION ===
files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".nc"))

zero_rain_days = 0
light_rain_amounts = []      # Days with 0–1 mm rain
very_light_rain_amounts = [] # Days with 0–0.2 mm rain
all_precip = []              # All precipitation values
dates = []                   # Corresponding time values

# === DATA EXTRACTION ===
for fname in files:
    ds = xr.open_dataset(os.path.join(DATA_DIR, fname))
    lats, lons = ds.latitude.values, ds.longitude.values

    # Find nearest grid point to target coordinates
    i_lat = np.argmin(np.abs(lats - TARGET_LAT))
    i_lon = np.argmin(np.abs(lons - TARGET_LON))

    # Extract precipitation and time
    tp = ds.tp.values[:, i_lat, i_lon]
    time = ds.time.values

    # Append to global arrays
    all_precip.extend(tp)
    dates.extend(time)

    # Classify days
    zero_rain_days += np.sum(tp == 0)
    light_mask = (tp > 0) & (tp < 1)
    light_rain_amounts.extend(tp[light_mask])
    very_light_mask = (tp > 0) & (tp <= 0.2)
    very_light_rain_amounts.extend(tp[very_light_mask])

# === SUMMARY STATISTICS ===
light_rain_days = len(light_rain_amounts)
very_light_rain_days = len(very_light_rain_amounts)

# Convert to DataFrame for temporal analysis
df = pd.DataFrame({"date": pd.to_datetime(dates), "precip": all_precip})
df.sort_values("date", inplace=True)
df["rolling_7d"] = df["precip"].rolling(window=7, center=True).mean()

# === VISUALIZATIONS ===

# Plot 1: Summary table
plt.figure(figsize=(6, 4))
plt.axis("off")
table_data = [
    ["Category", "Number of Days"],
    ["Precipitation = 0 mm", f"{zero_rain_days}"],
    ["0–1 mm", f"{light_rain_days}"],
    ["0–0.2 mm", f"{very_light_rain_days}"]
]
table = plt.table(cellText=table_data, loc="center", cellLoc="center")
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.2, 1.5)
plt.title("Summary of Precipitation Days (Target Pixel)", fontweight="bold")
plt.tight_layout()
plt.show()

# Plot 2: Histogram (0–1 mm)
plt.figure(figsize=(8, 5))
bins1 = np.arange(0, 1.01, 0.1)
plt.hist(light_rain_amounts, bins=bins1, edgecolor="black", alpha=0.7)
plt.title("Histogram of Light Rain (0–1 mm)")
plt.xlabel("Precipitation (mm)")
plt.ylabel("Number of Days")
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot 3: Histogram (0–0.2 mm)
plt.figure(figsize=(8, 5))
bins2 = np.arange(0, 0.201, 0.01)
plt.hist(very_light_rain_amounts, bins=bins2, edgecolor="black", alpha=0.7, color="teal")
plt.title("Histogram of Very Light Rain (0–0.2 mm)")
plt.xlabel("Precipitation (mm)")
plt.ylabel("Number of Days")
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot 4: Raw time series
plt.figure(figsize=(12, 5))
plt.plot(df["date"], df["precip"], linestyle="-", color="darkblue")
plt.title("Raw Precipitation Time Series")
plt.xlabel("Date")
plt.ylabel("Precipitation (mm)")
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot 5: Global histogram (1 mm bins)
plt.figure(figsize=(10, 5))
max_precip = np.ceil(np.max(all_precip))
bins_global = np.arange(0, max_precip + 1, 1)
plt.hist(all_precip, bins=bins_global, edgecolor="black", alpha=0.7, color="orange")
plt.title("Global Precipitation Histogram (1 mm bins)")
plt.xlabel("Precipitation (mm)")
plt.ylabel("Number of Days")
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot 6: Full histogram (all precipitation, 1 mm bins)
plt.figure(figsize=(10, 5))
plt.hist(all_precip, bins=bins_global, edgecolor="black", alpha=0.7, color="purple")
plt.title("Complete Precipitation Histogram (1 mm bins)")
plt.xlabel("Precipitation (mm)")
plt.ylabel("Number of Days")
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot 7: Smoothed time series (7-day moving average)
plt.figure(figsize=(12, 5))
plt.plot(df["date"], df["rolling_7d"], color="green", label="7-day Moving Average")
plt.title("Smoothed Precipitation Time Series (7 days)")
plt.xlabel("Date")
plt.ylabel("Precipitation (mm)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
