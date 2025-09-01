import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns


class PrecipAnalysis:
    def __init__(self, nc_dir, lat_range=(-35.5, -34.75), lon_range=(-9.25, -8.0)):
        """
        Load and preprocess NetCDF precipitation data.

        Parameters
        ----------
        nc_dir : str
            Directory containing NetCDF files.
        lat_range : tuple
            Latitude range (min, max) for spatial selection.
        lon_range : tuple
            Longitude range (min, max) for spatial selection.
        """
        print("Loading NetCDF data...")
        self.lat_range = lat_range
        self.lon_range = lon_range

        files = sorted([os.path.join(nc_dir, f) for f in os.listdir(nc_dir) if f.endswith(".nc")])
        datasets = [xr.open_dataset(f) for f in files]
        ds = xr.concat(datasets, dim="time")

        # Keep only January–June
        ds = ds.sel(time=ds['time'].dt.month <= 6)

        # Select spatial subset (precipitation is in meters → convert to mm)
        self.precip = ds['tp'].sel(
            lon=slice(self.lon_range[0], self.lon_range[1]),
            lat=slice(self.lat_range[0], self.lat_range[1])
        ) / 1000

        self.times = self.precip['time'].values
        self.lons = self.precip['longitude'].values
        self.lats = self.precip['latitude'].values

        print("Data successfully loaded.")

    def plot_daily_maps(self, days=5):
        """
        Plot precipitation maps for the first N days.
        """
        print(f"Displaying {days} daily precipitation maps...")
        selected_times = self.times[:days]
        for date in selected_times:
            date_str = str(np.datetime_as_string(date, unit='D'))
            self.plot_daily_map(date_str)

    def plot_all_pixel_histograms(self):
        """
        Plot precipitation histograms for every pixel in the selected area.
        """
        print("Displaying histograms for each pixel...")
        for lat in self.lats:
            for lon in self.lons:
                self.plot_pixel_histogram(lon, lat)

    def plot_daily_map(self, date_str):
        """
        Plot the precipitation map for a given day.
        """
        day_precip = self.precip.sel(time=date_str)
        plt.figure(figsize=(6, 5))
        plt.contourf(day_precip['lon'], day_precip['lat'], day_precip.values.squeeze(), cmap="Blues")
        plt.title(f"Precipitation Map – {date_str}")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.colorbar(label="Precipitation (m)")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()

    def plot_pixel_histogram(self, lon, lat):
        """
        Plot precipitation histogram for a given pixel at the coordinates (lon-lat).
        """
        pixel = self.precip.sel(lon=lon, lat=lat, method="nearest")
        data = pixel.values.flatten()

        plt.figure(figsize=(8, 5))
        sns.histplot(data[~np.isnan(data)], bins=30, kde=True)
        plt.title(f"Histogram – Pixel ({lon:.2f}, {lat:.2f})")
        plt.xlabel("Precipitation (mm)")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    nc_dir = "data/test"

    analysis = PrecipAnalysis(nc_dir)

    # Display precipitation maps for the first 5 days
    analysis.plot_daily_maps(days=5)

    # Display histograms for all pixels
    analysis.plot_all_pixel_histograms()

    print("Analysis complete.")
