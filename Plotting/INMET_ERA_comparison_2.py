"""
Comparison of Daily Precipitation between INMET Stations and ERA5 Data

Description:
This script loads daily precipitation data from selected INMET weather stations 
and ERA5 reanalysis data. It filters data for a target year, excludes dry season months,
aggregates daily precipitation, and generates comparison plots including:
- Time series of daily precipitation
- Histograms of daily precipitation distribution
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

# ==========================
# PART 1: INMET STATION DATA
# ==========================
DATA_DIR = "precip/2011-2015/old"
TARGET_STATIONS = ["recife", "chapadinha", "luzilandia"]
TARGET_YEAR = 2012

inmet_data = {}

# Load and process CSV files for the selected stations
for file in os.listdir(DATA_DIR):
    if not file.endswith(".csv"):
        continue

    path = os.path.join(DATA_DIR, file)
    with open(path, encoding="utf-8") as f:
        header_lines = [next(f) for _ in range(10)]
        station_name = header_lines[0].split(":")[1].strip().lower()

        if any(s in station_name for s in TARGET_STATIONS):
            df = pd.read_csv(
                path,
                sep=";",
                skiprows=10,
                encoding="utf-8",
                usecols=[0, 1, 2],
                names=["date", "hour", "precip"],
                dtype=str
            )

            # Convert precipitation to numeric and date column to datetime
            df["precip"] = pd.to_numeric(df["precip"].str.replace(",", "."), errors="coerce")
            df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")
            df = df.dropna(subset=["date", "precip"])

            # Filter for the target year
            df = df[df["date"].dt.year == TARGET_YEAR]

            # Exclude dry-season months: July to October
            df = df[~df["date"].dt.month.isin([7, 8, 9, 10])]

            # Aggregate daily precipitation
            df_daily = df.groupby("date")["precip"].sum().reset_index()
            inmet_data[station_name.title()] = df_daily

# Plot INMET time series and histograms
for station, df in inmet_data.items():
    plt.figure(figsize=(12, 4))
    plt.plot(df["date"], df["precip"], label=station)
    plt.title(f"Daily Precipitation INMET - {station} ({TARGET_YEAR})")
    plt.xlabel("Date")
    plt.ylabel("Precipitation (mm)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    max_precip = int(df["precip"].max()) + 1
    plt.figure(figsize=(8, 4))
    plt.hist(df["precip"], bins=range(0, max_precip), alpha=0.7, edgecolor="black")
    plt.title(f"Daily Precipitation Distribution INMET - {station} ({TARGET_YEAR})")
    plt.xlabel("Precipitation (mm/day)")
    plt.ylabel("Number of Days")
    plt.tight_layout()
    plt.show()

# ==========================
# PART 2: ERA5 REANALYSIS DATA
# ==========================
ERA5_COORDS = {
    "Recife": (-8.0, -34.5),
    "Chapadinha": (-3.5, -43.5),
    "Luzilandia": (-3.5, -42.5)
}
ERA5_DIR = "data/train"

def find_nearest(array, value):
    """Find index of the nearest value in an array."""
    return (np.abs(array - value)).argmin()

era5_data = {}

# Load ERA5 NetCDF files
for file in os.listdir(ERA5_DIR):
    if not (file.endswith(".nc") or file.endswith(".nc4")):
        continue

    path = os.path.join(ERA5_DIR, file)
    ds = xr.open_dataset(path)

    # Determine variable name
    var_name = "precipitation" if "precipitation" in ds.variables else list(ds.data_vars)[0]
    latitudes = ds["latitude"].values
    longitudes = ds["longitude"].values

    # Extract precipitation time series for each station
    for station, (lat, lon) in ERA5_COORDS.items():
        lat_idx = find_nearest(latitudes, lat)
        lon_idx = find_nearest(longitudes, lon)

        ts_cum = ds[var_name][:, lat_idx, lon_idx].to_series()
        ts_cum.index = pd.to_datetime(ts_cum.index)

        # Filter for the target year and exclude dry months
        ts_year = ts_cum[ts_cum.index.year == TARGET_YEAR]
        ts_year = ts_year[~ts_year.index.month.isin([7, 8, 9, 10])]

        # Convert cumulative to daily precipitation
        ts_daily = ts_year.diff().fillna(0)
        ts_daily[ts_daily < 0] = 0  # Correct negative differences
        ts_daily = ts_daily.groupby(ts_daily.index.date).sum()
        ts_daily.index = pd.to_datetime(ts_daily.index)

        # Store in dictionary
        if station in era5_data:
            era5_data[station] = pd.concat([era5_data[station], ts_daily])
        else:
            era5_data[station] = ts_daily

    ds.close()

# Plot ERA5 time series and histograms
for station, ts in era5_data.items():
    plt.figure(figsize=(12, 4))
    plt.plot(ts.index, ts.values, label=station)
    plt.title(f"Daily Precipitation ERA5 - {station} ({TARGET_YEAR})")
    plt.xlabel("Date")
    plt.ylabel("Precipitation (mm)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    max_precip = int(ts.max()) + 1
    plt.figure(figsize=(8, 4))
    plt.hist(ts.values, bins=range(0, max_precip), alpha=0.7, edgecolor="black")
    plt.title(f"Daily Precipitation Distribution ERA5 - {station} ({TARGET_YEAR})")
    plt.xlabel("Precipitation (mm/day)")
    plt.ylabel("Number of Days")
    plt.tight_layout()
    plt.show()
