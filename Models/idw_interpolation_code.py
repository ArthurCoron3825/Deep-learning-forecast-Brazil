"""
Precipitation Interpolation Pipeline (IDW method)
-------------------------------------------------

This script processes precipitation data from multiple weather stations,
cleans the raw CSV files, aggregates daily precipitation,
and interpolates the values onto a regular grid using Inverse Distance Weighting (IDW).

Main steps:
1. Load and clean precipitation station data from CSV files.
2. Aggregate precipitation values to daily totals for each station.
3. Interpolate precipitation to a 2D grid using IDW for each day of a given month.
4. Save the interpolated data as NetCDF files (one per month).
5. Provide statistical summaries and visualization functions for validation.

Dependencies:
- pandas, numpy, matplotlib, cartopy, xarray
- data must be in INMET station CSV format
"""

import os
import numpy as np
import pandas as pd
from datetime import date
from glob import glob
from io import StringIO
from calendar import monthrange

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr


# ==============================================================================================
# DATA LOADING AND CLEANING
# ==============================================================================================
def load_clean_precipitation(folder_path):
    """
    Load and clean precipitation CSV files from a folder.
    """
    all_files = glob(os.path.join(folder_path, '*.csv'))
    dfs = []
    print(f"\n {len(all_files)} CSV files found in {folder_path}")

    for file in all_files:
        try:
            # Read file with latin1 encoding (Brazilian characters support)
            with open(file, 'r', encoding='latin1') as f:
                lines = f.readlines()

            # Extract metadata (station name, location, etc.)
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

            # Locate the header line where measurements start
            start_idx = next((i for i, line in enumerate(lines) if line.startswith('Data Medicao')), None)
            if start_idx is None:
                print(f" Header line not found in {file}, skipped.")
                continue

            # Convert precipitation data into DataFrame
            data_str = ''.join(lines[start_idx:])
            df = pd.read_csv(
                StringIO(data_str),
                sep=';', usecols=[0, 1, 2],
                names=['date', 'hour', 'precip'],
                header=0, engine='python'
            )

            # Clean precipitation column
            df['precip'] = (
                df['precip'].astype(str)
                .str.strip().str.replace(',', '.')
                .replace({'null': None, '': None})
            )
            df['precip'] = pd.to_numeric(df['precip'], errors='coerce')

            # Create datetime field
            df['hour'] = df['hour'].astype(str).str.zfill(4)
            df['datetime'] = pd.to_datetime(
                df['date'] + ' ' + df['hour'],
                format='%Y-%m-%d %H%M',
                errors='coerce'
            )
            df.dropna(subset=['datetime', 'precip'], inplace=True)

            # Add station metadata
            df['station_name'] = station_name
            df['station_code'] = station_code
            df['latitude'] = latitude
            df['longitude'] = longitude
            df['altitude'] = altitude

            dfs.append(df)

        except Exception as e:
            print(f" Error while processing {file} : {e}")

    if dfs:
        full_df = pd.concat(dfs, ignore_index=True)
        full_df['date_day'] = full_df['datetime'].dt.date
        print(f"\n Total: {len(full_df)} rows loaded from {len(dfs)} files.")
        return full_df
    else:
        print(" No valid data loaded.")
        return pd.DataFrame()


# ==============================================================================================
# STATION DATA PROCESSING
# ==============================================================================================
def generate_station_data(df):
    """
    Aggregate daily precipitation for each station.
    Returns:
        list of dict: Each dict contains lon, lat, daily dates and values.
    """
    station_data = []
    for station_id, group in df.groupby('station_code'):
        if group.empty:
            continue

        lon = group['longitude'].iloc[0]
        lat = group['latitude'].iloc[0]
        group = group.set_index('datetime').sort_index()

        # Aggregate daily totals
        daily = group['precip'].resample('D').sum()
        daily = daily.reindex(
            pd.date_range(daily.index.min(), daily.index.max(), freq='D'),
            fill_value=0
        )
        station_data.append({
            'lon': lon,
            'lat': lat,
            'dates': daily.index,
            'values': daily.values
        })

    print(f"{len(station_data)} stations processed for interpolation.")
    return station_data


# ==============================================================================================
# IDW INTERPOLATION
# ==============================================================================================
def idw_interpolation(points, values, grid_lon, grid_lat, power=6):
    """
    Perform Inverse Distance Weighting (IDW) interpolation.

    Args:
        points (ndarray): Station coordinates (n x 2).
        values (ndarray): Precipitation values at stations.
        grid_lon, grid_lat (ndarray): Meshgrid of longitude/latitude.
        power (int): Power parameter for IDW.

    Returns:
        ndarray: Interpolated values on the grid.
    """
    grid_points = np.vstack((grid_lon.ravel(), grid_lat.ravel())).T
    dists = np.linalg.norm(points[None, :, :] - grid_points[:, None, :], axis=2)
    dists[dists == 0] = 1e-10  # avoid division by zero
    weights = 1 / (dists ** power)
    weighted_values = weights * values[None, :]
    interpolated = np.sum(weighted_values, axis=1) / np.sum(weights, axis=1)

    return interpolated.reshape(grid_lon.shape)


def interpolate_to_grid(station_data, month_str, power=6):
    """
    Interpolate precipitation for a full month onto a regular grid.

    Args:
        station_data (list): Daily aggregated station data.
        month_str (str): Month to interpolate, format "YYYY-MM".
        power (int): IDW power parameter.

    Returns:
        xarray.Dataset: Interpolated precipitation on grid.
    """
    lon = np.linspace(-37, -33, 81)
    lat = np.linspace(-10, -5, 81)
    grid_lon, grid_lat = np.meshgrid(lon, lat)

    year, month = map(int, month_str.split('-'))
    n_days = monthrange(year, month)[1]
    times = pd.date_range(start=f"{year}-{month:02d}-01", periods=n_days, freq='D')

    points = np.array([[s['lon'], s['lat']] for s in station_data])
    daily_values = np.zeros((len(station_data), n_days))

    for i, s in enumerate(station_data):
        mask = (s['dates'] >= times[0]) & (s['dates'] <= times[-1])
        monthly_values = s['values'][mask]

        # Pad with zeros if missing days
        if len(monthly_values) < n_days:
            padded = np.zeros(n_days)
            padded[:len(monthly_values)] = monthly_values
            monthly_values = padded
        daily_values[i, :] = monthly_values

    interpolated_array = np.full((n_days, len(lat), len(lon)), np.nan)

    for day_idx in range(n_days):
        values = daily_values[:, day_idx]
        valid_mask = ~np.isnan(values)
        if np.any(valid_mask):
            interp = idw_interpolation(points[valid_mask], values[valid_mask], grid_lon, grid_lat, power=power)
            interpolated_array[day_idx] = interp

    ds = xr.Dataset(
        {"precipitation": (("time", "lat", "lon"), interpolated_array)},
        coords={
            "time": times,
            "lat": lat,
            "lon": lon
        },
        attrs={"description": f"Interpolated precipitation (IDW) on 0.25° grid for {month_str}"}
    )
    return ds


# ==============================================================================================
# MAIN EXECUTION
# ==============================================================================================
if __name__ == "__main__":
    df = load_clean_precipitation("precip/2011-2015/old")  # Path to raw CSV folder
    station_data = generate_station_data(df)

    year = 2014  # Target year for interpolation

    # Interpolate month by month and save as NetCDF
    for month in range(1, 12 + 1):
        data_time = f"{year}-{month:02d}"
        ds = interpolate_to_grid(station_data, data_time)
        ds.to_netcdf(f"data/raw/new_precipitation_p=6_{data_time}.nc")
        print(f" NetCDF file saved: new_precipitation_p=6_{data_time}.nc")
