"""
ERA5 Data Preprocessing Script
==============================

This script loads monthly ERA5 reanalysis data from NetCDF files,
merges selected surface variables and pressure-level variables, 
and produces a single, harmonized dataset per month.

Steps:
1. Load surface variables (e.g., temperature, pressure, wind, precipitation).
2. Load pressure-level variables at specific pressure levels (e.g., geopotential, humidity, winds).
3. Merge all variables into one dataset with consistent time coordinates.
4. Save the merged dataset as a NetCDF file for later use in machine learning or climate analysis.
"""

import os
import xarray as xr
import numpy as np
from netCDF4 import num2date

# === CONFIGURATION ===
DATA_DIR = "data/raw"  # Path where ERA5 NetCDF files are stored

# Surface variables to extract (mapping: file prefix -> variable name in dataset)
SURFACE_VARIABLES = {
    "2m_temperature": "t2m",
    "mean_sea_level_pressure": "msl",
    "10m_u_component_of_wind": "u10",
    "10m_v_component_of_wind": "v10",
    "total_precipitation": "tp",
}

# Pressure-level variables (mapping: file prefix -> variable name in dataset)
PRESSURE_VARIABLES = {
    "geopotential": "z",
    "relative_humidity": "r",
    "specific_humidity": "q",
    "temperature": "t",
    "u_component_of_wind": "u",
    "v_component_of_wind": "v",
}

# Pressure levels (in hPa) to keep
PRESSURE_LEVELS = [1000, 850, 700, 500, 300]


# === FUNCTIONS ===

def load_monthly_surface_variable(data_dir, file_label, var_name, year, month):
    """
    Load a single surface variable for a given year and month.

    Args:
        data_dir (str): Path to the data directory.
        file_label (str): Prefix of the NetCDF file.
        var_name (str): Variable name inside the NetCDF dataset.
        year (int): Year of data to load.
        month (int): Month of data to load.

    Returns:
        xarray.DataArray: Surface variable for the given month.
    """
    file_name = f"{file_label}_{year}{month:02d}.nc"
    file_path = os.path.join(data_dir, file_name)

    print(f"  Loading surface file: {file_name}")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f" Missing file: {file_name}")

    # Open dataset without automatic time decoding
    ds = xr.open_dataset(file_path, decode_times=False)

    # Ensure consistent time coordinate name
    ds = ds.rename({"valid_time": "time"})

    # Decode time from numeric values to datetime64
    units = ds['time'].attrs.get('units', 'hours since 1900-01-01 00:00:00')
    calendar = ds['time'].attrs.get('calendar', 'standard')
    decoded_time = num2date(ds['time'].values, units=units, calendar=calendar)
    ds['time'] = ('time', np.array(decoded_time, dtype='datetime64[ns]'))

    da = ds[var_name]
    da.name = var_name
    print(f"  Variable {var_name} loaded with shape {da.shape}")
    return da


def load_monthly_pressure_variables(data_dir, var_dict, year, month, levels_to_keep):
    """
    Load multiple pressure-level variables for a given month and subset them to specific levels.

    Args:
        data_dir (str): Path to the data directory.
        var_dict (dict): Dictionary mapping file labels -> variable names.
        year (int): Year of data.
        month (int): Month of data.
        levels_to_keep (list): List of pressure levels to extract (in hPa).

    Returns:
        xarray.Dataset: Merged dataset of all selected pressure variables.
    """
    pressure_data = []

    for file_label, var_name in var_dict.items():
        file_name = f"{file_label}_{year}{month:02d}.nc"
        file_path = os.path.join(data_dir, file_name)

        print(f"\n  Loading pressure file: {file_name}")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f" Missing file: {file_name}")

        ds = xr.open_dataset(file_path, decode_times=False)
        ds = ds.rename({"valid_time": "time"})

        # Decode time
        units = ds['time'].attrs.get('units', 'hours since 1900-01-01 00:00:00')
        calendar = ds['time'].attrs.get('calendar', 'standard')
        decoded_time = num2date(ds['time'].values, units=units, calendar=calendar)
        ds['time'] = ('time', np.array(decoded_time, dtype='datetime64[ns]'))

        # Extract each desired pressure level
        for level in levels_to_keep:
            if level not in ds.coords["pressure_level"]:
                raise ValueError(f" Missing pressure level {level} hPa for {file_label}")

            da = ds[var_name].sel(pressure_level=level)
            da = da.drop_vars("pressure_level")  # Drop redundant dimension
            da.name = f"{var_name}_{level}hPa"  # Rename variable
            pressure_data.append(da)
            print(f"  Variable {da.name} loaded with shape {da.shape}")

    return xr.merge(pressure_data)


def merge_surface_and_pressure(surface_vars, pressure_ds):
    """
    Merge surface variables and pressure-level variables into a single dataset.
    """
    print(f"\n Merging {len(surface_vars)} surface variables and {len(pressure_ds.data_vars)} pressure variables")
    merged = xr.merge(surface_vars + list(pressure_ds.data_vars.values()))

    print(f" Final dataset contains: {list(merged.data_vars)}")
    return merged


# === MAIN ===
YEAR = 2007

def main():
    for month in range(2, 3):  # Example: only February 2007
        print(f"\n Starting ERA5 merge for {month:02d}/{YEAR}\n{'='*40}")

        # --- Load surface data ---
        surface_data = []
        for file_label, var_name in SURFACE_VARIABLES.items():
            try:
                da = load_monthly_surface_variable(DATA_DIR, file_label, var_name, YEAR, month)
                surface_data.append(da)
            except Exception as e:
                print(f" Surface variable error [{file_label}]: {e}")

        # --- Load pressure data ---
        try:
            pressure_ds = load_monthly_pressure_variables(DATA_DIR, PRESSURE_VARIABLES, YEAR, month, PRESSURE_LEVELS)
        except Exception as e:
            print(f" Pressure variable error: {e}")
            continue

        # --- Merge and save ---
        try:
            combined_ds = merge_surface_and_pressure(surface_data, pressure_ds)

            # Ensure consistent time coordinate name
            if "valid_time" in combined_ds.coords:
                combined_ds = combined_ds.rename({"valid_time": "time"})

            print("Sample time values:", combined_ds["time"].values[:5])
            print("Type of time values:", type(combined_ds["time"].values[0]))

            output_path = os.path.join(DATA_DIR, f"era5_merged_{YEAR}{month:02d}.nc")
            combined_ds.to_netcdf(output_path)
            print(f"\n✅ Final merged file saved: {output_path}")
        except Exception as e:
            print(f" Final merge error: {e}")

    print("\n Global process finished.")


if __name__ == "__main__":
    main()
