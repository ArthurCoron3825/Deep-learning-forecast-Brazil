"""
This script compares daily precipitation data from INMET weather stations
(CSV format) with ERA5 reanalysis data (NetCDF format) for a target year.

Steps:
1. Load and preprocess INMET station data (daily totals).
2. Assign random coordinates to stations (replace with actual if available).
3. Extract ERA5 daily precipitation values from the nearest grid point.
4. Generate comparative time-series plots for each selected station,
   including markers for maximum precipitation events.

Purpose:
This analysis helps validate ERA5 reanalysis data against observed
station records, highlighting differences in magnitude and timing.
"""

import os
import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

# Parameters
TARGET_YEAR = 2013
INMET_DIR = "precip/2011-2015/old"
ERA5_DIR = "data/train"
MAX_STATIONS = 10

# Load INMET Stations
print("Loading INMET data...")
stations = {}

for file in os.listdir(INMET_DIR):
    if not file.endswith(".csv"):
        continue

    path = os.path.join(INMET_DIR, file)
    with open(path, encoding="utf-8") as f:
        header_lines = [next(f) for _ in range(10)]
        station_name = header_lines[0].split(":")[1].strip().lower()

        df = pd.read_csv(
            path,
            sep=";",
            skiprows=10,
            encoding="utf-8",
            usecols=[0, 1, 2],
            names=["date", "hour", "precip"],
            dtype=str
        )

        df["precip"] = pd.to_numeric(df["precip"].str.replace(",", "."), errors="coerce")
        df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")
        df = df.dropna(subset=["date", "precip"])
        df = df[df["date"].dt.year == TARGET_YEAR]
        df = df[~df["date"].dt.month.isin([7, 8, 9, 10])]  # exclude dry months
        df_daily = df.groupby("date")["precip"].sum().reset_index()

        if len(df_daily) > 50:
            stations[station_name] = df_daily

    if len(stations) >= MAX_STATIONS:
        break

print(f"{len(stations)} INMET stations selected: {list(stations.keys())}")

# Assign Random Coordinates (replace with actual if available)
coords = {}
for name in stations.keys():
    lat = np.random.uniform(-10, 0)
    lon = np.random.uniform(-45, -35)
    coords[name] = (lat, lon)

# Utility: Find Nearest Grid Index
def find_nearest(array, value):
    return (np.abs(array - value)).argmin()

# Extract ERA5 Data
print("\nLoading ERA5 monthly files...")
data_era5 = {name: pd.Series(dtype="float64") for name in stations.keys()}

for file in sorted(os.listdir(ERA5_DIR)):
    if not file.endswith(".nc") or not file.startswith(f"daily_era5_{TARGET_YEAR}"):
        continue

    path = os.path.join(ERA5_DIR, file)
    ds = xr.open_dataset(path)
    var_name = "precipitation" if "precipitation" in ds.variables else list(ds.data_vars)[0]
    latitudes = ds["latitude"].values
    longitudes = ds["longitude"].values
    times = pd.to_datetime(ds["time"].values)

    for name, (lat, lon) in coords.items():
        lat_idx = find_nearest(latitudes, lat)
        lon_idx = find_nearest(longitudes, lon)

        ts = ds[var_name][:, lat_idx, lon_idx].to_series()
        ts.index = times
        ts = ts[~ts.index.month.isin([7, 8, 9, 10])]

        # Convert cumulative to instantaneous, ensure non-negative
        ts_inst = ts.diff().fillna(0)
        ts_inst[ts_inst < 0] = 0

        ts_daily = ts_inst.groupby(ts_inst.index.date).sum()
        ts_daily.index = pd.to_datetime(ts_daily.index)

        data_era5[name] = pd.concat([data_era5[name], ts_daily])

    ds.close()

# Visual Comparison
print("\nGenerating comparison plots...")

for name in stations:
    df_inmet = stations[name].copy()
    ts_era5 = data_era5[name].sort_index()

    df_era5 = pd.DataFrame({"date": ts_era5.index, "era5": ts_era5.values})
    merged = pd.merge(df_inmet, df_era5, on="date", how="outer").sort_values("date")
    merged.fillna(0, inplace=True)
    merged.set_index("date", inplace=True)

    max_inmet_val = merged["precip"].max()
    max_era5_val = merged["era5"].max()
    max_inmet_date = merged["precip"].idxmax()
    max_era5_date = merged["era5"].idxmax()

    plt.figure(figsize=(14, 5))
    plt.plot(merged.index, merged["precip"], label="INMET", color="steelblue")
    plt.plot(merged.index, merged["era5"], label="ERA5", color="darkorange")

    plt.scatter(max_inmet_date, max_inmet_val, color="red", label="Max INMET", zorder=5)
    plt.scatter(max_era5_date, max_era5_val, color="purple", label="Max ERA5", zorder=5)

    lat, lon = coords[name]
    plt.title(f"{name.title()} - INMET vs ERA5 ({TARGET_YEAR})\nLat={lat:.2f}, Lon={lon:.2f}")
    plt.xlabel("Date")
    plt.ylabel("Precipitation (mm)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
