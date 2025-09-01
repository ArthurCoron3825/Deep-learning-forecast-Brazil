"""
ERA5 Daily Weather Maps Visualization
-------------------------------------
This script loads ERA5 reanalysis data for a given month (NetCDF format) and
creates daily weather maps for a predefined study region in Brazil.

It plots:
- Daily precipitation fields (surface variable `tp`).
- Additional atmospheric variables (humidity, geopotential height) at selected
  pressure levels.
- A context map showing the bounding box of the study area within South America.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import datetime
from matplotlib.patches import Rectangle

# === CONFIGURATION ===
ERA5_DIR = "data/test"       # Directory containing ERA5 monthly NetCDF files
TARGET_MONTH = "202201"      # Target month in format YYYYMM

# Study area (note: ERA5 latitudes are decreasing)
LAT_SLICE = slice(0, -15)
LON_SLICE = slice(-45, -30)
LAT_BOUNDS = (-15, 0)
LON_BOUNDS = (-45, -30)

# Variables of interest
SURFACE_VAR = "tp"  # Total precipitation
OTHER_VARS = {
    "r": ("Relative Humidity @ 850 hPa", "viridis", 850),
    "z": ("Geopotential Height @ 500 hPa", "plasma", 500),
}

def plot_day_maps(date_str, ds):
    """
    Generate maps for precipitation and other weather variables for a given day..
    """
    date = pd.to_datetime(date_str)

    # --- Create subplot figure for precipitation and additional variables ---
    fig, axs = plt.subplots(
        1, len(OTHER_VARS) + 1,
        figsize=(16, 5),
        constrained_layout=True,
        subplot_kw={'projection': ccrs.PlateCarree()}
    )

    # --- Precipitation map ---
    tp = ds[SURFACE_VAR].sel(time=date, latitude=LAT_SLICE, longitude=LON_SLICE)
    tp.plot(ax=axs[0], cmap="Blues", cbar_kwargs={"label": "Precipitation (m)"})
    axs[0].set_title(f"ERA5 Precipitation - {date.date()}")
    axs[0].coastlines()
    axs[0].add_feature(cfeature.BORDERS, linestyle=":")
    axs[0].set_extent([LON_BOUNDS[0], LON_BOUNDS[1], LAT_BOUNDS[0], LAT_BOUNDS[1]])

    # --- Other atmospheric variables ---
    for i, (var, (title, cmap, level)) in enumerate(OTHER_VARS.items(), start=1):
        var_name = f"{var}_{level}hPa"
        da = ds[var_name].sel(time=date, latitude=LAT_SLICE, longitude=LON_SLICE)
        da.plot(ax=axs[i], cmap=cmap, cbar_kwargs={"label": f"{var} ({level} hPa)"})
        axs[i].set_title(f"{title} - {date.date()}")
        axs[i].coastlines()
        axs[i].add_feature(cfeature.BORDERS, linestyle=":")
        axs[i].set_extent([LON_BOUNDS[0], LON_BOUNDS[1], LAT_BOUNDS[0], LAT_BOUNDS[1]])

    plt.show()

    # --- Context map with study area bounding box ---
    fig = plt.figure(figsize=(6, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([-80, -30, -40, 10], crs=ccrs.PlateCarree())
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS)
    ax.add_feature(cfeature.LAND, facecolor="lightgray")
    ax.add_feature(cfeature.OCEAN, facecolor="lightblue")
    ax.set_title(f"Study Area in South America - {date.date()}")

    rect = Rectangle(
        (LON_BOUNDS[0], LAT_BOUNDS[0]),
        LON_BOUNDS[1] - LON_BOUNDS[0],
        LAT_BOUNDS[1] - LAT_BOUNDS[0],
        linewidth=2, edgecolor='red', facecolor='none', zorder=5
    )
    ax.add_patch(rect)
    plt.show()


if __name__ == "__main__":
    # === Load ERA5 file for the target month ===
    filename = os.path.join(ERA5_DIR, f"daily_era5_{TARGET_MONTH}.nc")
    ds = xr.open_dataset(filename)

    # === Generate plots for each day of the month ===
    dates = pd.to_datetime(ds.time.values)
    for date in dates:
        plot_day_maps(date.strftime("%Y-%m-%d"), ds)
