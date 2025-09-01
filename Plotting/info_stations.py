"""
Visualization of Study Area, ERA5 Pixels, and Station Location

Description:
This script creates geospatial visualizations of the study region in Brazil. 
It highlights:
1. The defined study area bounding box.
2. Selected ERA5 grid pixels used for precipitation analysis.
3. The INMET station 82287 (Parnaíba) location.
"""

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# === STUDY AREA COORDINATES ===
PIXELS = [(-8, -34.8), (-3, -44.25)]   # Example ERA5 pixel locations (lat, lon)
STATION_PARNAIBA = (-2.91, -41.77)     # INMET Station 82287 coordinates (lat, lon)

# Study area bounding box
LAT_MIN, LAT_MAX = -15, 0
LON_MIN, LON_MAX = -45, -30


# === BASE MAP FUNCTION ===
def add_base_features(ax, title: str):
    """Add common geographic features to the map."""
    ax.coastlines(resolution="10m")
    ax.add_feature(cfeature.BORDERS, linestyle=":")
    ax.add_feature(cfeature.LAND, facecolor="lightgray")
    ax.add_feature(cfeature.OCEAN, facecolor="lightblue")
    ax.add_feature(cfeature.LAKES, facecolor="lightblue")
    ax.add_feature(cfeature.RIVERS)
    ax.set_extent([LON_MIN - 10, LON_MAX + 5, LAT_MIN - 5, LAT_MAX + 5], crs=ccrs.PlateCarree())
    ax.set_title(title, fontsize=14, fontweight="bold")


# === FIGURE 1: Study area with ERA5 pixels ===
fig1 = plt.figure(figsize=(10, 7))
ax1 = plt.axes(projection=ccrs.PlateCarree())
add_base_features(ax1, "Study Area in Brazil with ERA5 Pixel Locations")

# Plot ERA5 pixel points
ax1.plot(
    [lon for lat, lon in PIXELS],
    [lat for lat, lon in PIXELS],
    "ro", markersize=8, label="ERA5 Pixels"
)

# Plot study area bounding box
ax1.plot(
    [LON_MIN, LON_MAX, LON_MAX, LON_MIN, LON_MIN],
    [LAT_MIN, LAT_MIN, LAT_MAX, LAT_MAX, LAT_MIN],
    color="darkgreen", linewidth=2, linestyle="--", label="Study Area"
)

ax1.legend(loc="lower left")
plt.tight_layout()
plt.show()


# === FIGURE 2: INMET Parnaíba station ===
fig2 = plt.figure(figsize=(8, 6))
ax2 = plt.axes(projection=ccrs.PlateCarree())
add_base_features(ax2, "INMET Station 82287 - Parnaíba")

# Plot station location
ax2.plot(
    STATION_PARNAIBA[1], STATION_PARNAIBA[0],
    "ro", markersize=7, label="Station 82287 (Parnaíba)"
)

# Plot study area bounding box
ax2.plot(
    [LON_MIN, LON_MAX, LON_MAX, LON_MIN, LON_MIN],
    [LAT_MIN, LAT_MIN, LAT_MAX, LAT_MAX, LAT_MIN],
    color="darkgreen", linewidth=2, linestyle="--", label="Study Area"
)

ax2.legend(loc="lower left")
plt.tight_layout()
plt.show()
