"""
Brazil Study Area Map
---------------------
This script generates a map of Brazil with geographic features
(coastlines, borders, rivers, lakes, etc.) using Cartopy. A red rectangle
marks a predefined study area specified by latitude/longitude boundaries.

The purpose of this visualization is to show the selected study region
within Brazil for further analysis (e.g., precipitation prediction).
"""

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Define study area as [lat_max, lon_min, lat_min, lon_max]
LAT_MAX, LON_MIN, LAT_MIN, LON_MAX = 0, -45, -15, -30

# Create Figure
fig = plt.figure(figsize=(10, 10))
ax = plt.axes(projection=ccrs.PlateCarree())

# Add geographic features
ax.add_feature(cfeature.BORDERS, linestyle=":")
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.LAND, facecolor="lightgray")
ax.add_feature(cfeature.OCEAN, facecolor="lightblue")
ax.add_feature(cfeature.LAKES, facecolor="lightblue")
ax.add_feature(cfeature.RIVERS)

# Focus on Brazil
ax.set_extent([-60, -15, -40, 5], crs=ccrs.PlateCarree())

# Draw rectangle for study area
rect_lon = [LON_MIN, LON_MAX, LON_MAX, LON_MIN, LON_MIN]
rect_lat = [LAT_MAX, LAT_MAX, LAT_MIN, LAT_MIN, LAT_MAX]
ax.plot(rect_lon, rect_lat, color="red", linewidth=2, transform=ccrs.PlateCarree())

# Add title
plt.title("Study Area over Brazil", fontsize=15)
plt.show()
