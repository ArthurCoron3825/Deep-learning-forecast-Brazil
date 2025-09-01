"""
Precipitation Data Loading, Processing, and Visualization

Description:
This module provides functions to:
- Load and clean INMET CSV precipitation data.
- Convert ERA5 NetCDF precipitation data from meters to mm.
- Compute daily sums or means from time series.
- Visualize monthly sums, first 12 days, and histograms.
- Analyze maximum precipitation per pixel.
"""

import os
from glob import glob
from io import StringIO
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# --------------------------- FUNCTIONS ---------------------------- #

# --------------------------- Loading data -------------------------- #
def load_clean_precipitation(folder_path):
    """
    Load and clean precipitation CSV files from INMET stations.

    Returns a pandas DataFrame with datetime, precipitation, and station info.
    """
    all_files = glob(os.path.join(folder_path, '*.csv'))
    dfs = []

    print(f"\n🔍 {len(all_files)} CSV files found in {folder_path}")

    for file in all_files:
        try:
            with open(file, 'r', encoding='latin1') as f:
                lines = f.readlines()

            # Extract metadata
            metadata = {}
            for line in lines:
                if line.strip().startswith('Data Medicao'):
                    break
                if ':' in line:
                    key, value = line.strip().split(':', 1)
                    metadata[key.strip()] = value.strip()

            station_name = metadata.get('Nome', 'Unknown')
            station_code = metadata.get('Codigo Estacao', 'Unknown')
            latitude = float(metadata.get('Latitude', 'nan'))
            longitude = float(metadata.get('Longitude', 'nan'))
            altitude = float(metadata.get('Altitude', 'nan'))

            # Locate start of data
            start_idx = next((i for i, line in enumerate(lines) if line.startswith('Data Medicao')), None)
            if start_idx is None:
                print(f"⚠️ 'Data Medicao' line not found. Skipping {file}.")
                continue

            data_str = ''.join(lines[start_idx:])
            df = pd.read_csv(StringIO(data_str), sep=';', usecols=[0, 1, 2],
                             names=['date', 'hour', 'precip'], header=0)

            # Clean precipitation column
            df['precip'] = df['precip'].astype(str).str.strip().str.replace(',', '.').replace({'null': None, '': None})
            df['precip'] = pd.to_numeric(df['precip'], errors='coerce')

            # Parse datetime
            df['hour'] = df['hour'].astype(str).str.zfill(4)
            df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['hour'], format='%Y-%m-%d %H%M', errors='coerce')
            df.dropna(subset=['datetime', 'precip'], inplace=True)

            # Add station info
            df['station_name'] = station_name
            df['station_code'] = station_code
            df['latitude'] = latitude
            df['longitude'] = longitude
            df['altitude'] = altitude

            dfs.append(df)

        except Exception as e:
            print(f"❌ Error in {file}: {e}")

    if dfs:
        full_df = pd.concat(dfs, ignore_index=True)
        full_df['date_day'] = full_df['datetime'].dt.date
        print(f"\n🎉 Total: {len(full_df)} rows loaded from {len(dfs)} files.")
        return full_df
    else:
        print("❌ No valid data loaded.")
        return pd.DataFrame()


# --------------------------- Plotting ---------------------------- #
def plot_monthly_precipitation_sum(ds, var_name="tp", title="Monthly Precipitation (mm)", cmap="Blues"):
    """
    Plot total monthly precipitation from an xarray Dataset.
    """
    tp_sum = ds[var_name].sum(dim="time")

    fig = plt.figure(figsize=(10, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())

    tp_sum.plot(
        ax=ax,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        cbar_kwargs={"label": "Monthly Precipitation (mm)"},
    )

    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=":")
    ax.set_title(title)
    ax.gridlines(draw_labels=True)
    plt.tight_layout()
    plt.show()


def plot_precip_12_days_maps(ds, var_name="tp", cmap="Blues"):
    """
    Plot the first 12 days of precipitation maps.
    """
    da = ds[var_name].isel(time=slice(0, 12))
    fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(20, 10),
                             subplot_kw={"projection": ccrs.PlateCarree()})
    axes = axes.ravel()

    for i in range(12):
        ax = axes[i]
        da_i = da.isel(time=i)
        im = da_i.plot(
            ax=ax,
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            cbar_kwargs={"label": "Precipitation (mm)"}
        )
        ax.set_title(str(np.datetime_as_string(da_i.time.values, unit='D')))
        ax.coastlines()
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        ax.gridlines(draw_labels=False)

    plt.suptitle("Precipitation – First 12 Days of the Month", fontsize=16)
    plt.tight_layout()
    plt.show()


def concat_and_plot_precip_histogram(data_dir, target_lat=-1.75, target_lon=-45,
                                     var_name="tp", bin_size=1, max_bin=40):
    """
    Concatenate NetCDF files, extract precipitation at a pixel, and plot histogram.
    """
    files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".nc")])
    datasets = [xr.open_dataset(f, engine="netcdf4") for f in files]
    ds = xr.concat(datasets, dim="time")

    lons = ds['longitude'].values
    lats = ds['latitude'].values
    lon_idx = np.argmin(np.abs(lons - target_lon))
    lat_idx = np.argmin(np.abs(lats - target_lat))

    precip_series = ds[var_name].isel(latitude=lat_idx, longitude=lon_idx).values
    bins = np.arange(0, max_bin + bin_size, bin_size)

    print("Min precipitation:", np.nanmin(precip_series))
    print("Max precipitation:", np.nanmax(precip_series))
    print("Sample unique values:", np.unique(precip_series)[:20])
    print("Total days:", precip_series.size)

    plt.figure(figsize=(8,5))
    plt.hist(precip_series, bins=bins, color="dodgerblue", edgecolor="black", alpha=0.75)
    plt.title(f"Daily Precipitation Histogram at Pixel ({target_lat}, {target_lon})")
    plt.xlabel("Precipitation (mm)")
    plt.ylabel("Number of days")
    plt.xticks(bins)
    plt.grid(axis="y", alpha=0.7)
    plt.tight_layout()
    plt.show()


# --------------------------- Daily Aggregation ------------------ #
def daily_sum(ds, var_name="tp"):
    """
    Compute daily precipitation from cumulative steps.
    """
    diff = ds[var_name].diff("time", label="upper").fillna(0)
    diff = diff.clip(min=0)
    ds[var_name] = diff
    ds_daily = ds.resample(time="1D").sum()
    return ds_daily


def daily_mean(ds):
    """
    Compute daily mean for all variables in the dataset.
    """
    if "time" not in ds.coords:
        raise ValueError("Dataset has no 'time' coordinate.")

    if not np.issubdtype(ds.time.dtype, np.datetime64):
        ds["time"] = xr.decode_cf(ds).time

    ds_daily = ds.groupby("time.date").mean(dim="time")
    ds_daily = ds_daily.rename({"date": "time"})
    ds_daily = ds_daily.assign_coords(time=pd.to_datetime(ds_daily.time.values))
    return ds_daily


# --------------------------- Conversion ------------------------ #
def convert_precipitation_to_mm(dataset, var_name="tp"):
    """
    Convert precipitation from meters to millimeters.
    """
    if var_name not in dataset:
        raise ValueError(f"Variable '{var_name}' not found in dataset.")
    print(f"Converting {var_name} from meters to mm...")
    dataset[var_name] = dataset[var_name] * 1000
    dataset[var_name].attrs['units'] = 'mm'
    dataset[var_name].attrs['long_name'] = 'Total precipitation (mm)'
    return dataset


# --------------------------- Pixel Analysis -------------------- #
def analyze_max_precip_pixel(nc_dir):
    """
    Identify the pixel with the highest total precipitation.
    """
    files = sorted([os.path.join(nc_dir, f) for f in os.listdir(nc_dir) if f.endswith(".nc")])
    datasets = [xr.open_dataset(f, engine="netcdf4") for f in files]
    ds = xr.concat(datasets, dim="time")

    precip = ds['tp'].sel(longitude=slice(-50, -30), latitude=slice(0, -20))
    agg = precip.sum(dim='time')
    print("Analysis: total precipitation per pixel (mm)")

    max_val = agg.max().item()
    max_idx = np.unravel_index(np.argmax(agg.values), agg.shape)
    max_lat = agg['latitude'].values[max_idx[0]]
    max_lon = agg['longitude'].values[max_idx[1]]

    print(f"\n  Wettest pixel at (lat: {max_lat:.2f}, lon: {max_lon:.2f})")
    print(f"  Maximum value: {max_val:.2f} mm (sum)")

    return {'lat': max_lat, 'lon': max_lon, 'value': max_val}
