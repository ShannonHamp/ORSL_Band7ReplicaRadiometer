# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 17:34:17 2025

@author: ORSL
"""


import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.cm as cm
from matplotlib.ticker import MaxNLocator
import matplotlib.colors as colors
from matplotlib_scalebar.scalebar import ScaleBar
from matplotlib.ticker import MaxNLocator, FormatStrFormatter



# File paths
geopackage_path = r"D:\NASA_EPSCoR_Snow\Band_7_Sensor\Data\Processing_2025_0425\Results\pixelGrid_cropped_to_UAV_Area_3604.gpkg"  # Pixel grid in GPKG format
radiometer_path = r"D:\NASA_EPSCoR_Snow\Band_7_Sensor\Data\Processing_2025_0425\Data\Radiometer\processed_reflectance_2025_0407_flight1.txt"



# Read the pixel grid from the GeoPackage
pixel_grid = gpd.read_file(geopackage_path)


# --- Config ---
radius_m = 7.19235  # Radius for 15m diameter circle

# --- Read CSV ---
df = pd.read_csv(radiometer_path)

# Ensure value column is numeric (if needed)
df['value'] = pd.to_numeric(df['reflectance'], errors='coerce')

# Create GeoDataFrame with WGS84 coordinates and value column
gdf_points = gpd.GeoDataFrame(
    df[['value']],  # Include the value column
    geometry=[Point(lon, lat) for lon, lat in zip(df.longitude, df.latitude)],
    crs="EPSG:4326"  # WGS84
)

# Project to a metric CRS (EPSG:2256)
gdf_projected = gdf_points.to_crs(epsg=3604)


# Create 15m diameter buffer circles (2256 is in meters, so no conversion needed)
gdf_circles = gdf_projected.copy()
gdf_circles['geometry'] = gdf_projected.buffer(radius_m)




#-------------------------------------- PLOT --------------------------------------------
# --- Plot the circles colored by value with colorbar and overlay cropped grid ---
cmap = cm.viridis
norm = colors.Normalize(vmin=8, vmax=13)

fig, ax = plt.subplots(figsize=(10, 8))

# Plot circles, colored by 'value'
circles_plot = gdf_circles.plot(
    column='value',
    cmap=cmap,
    norm=norm,
    legend=False,
    edgecolor='black',
    linewidth=0.5,
    alpha=0.8,
    ax=ax
)

# COLORBAR
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm._A = []  # required workaround for older versions of matplotlib
cbar = fig.colorbar(sm, ax=ax, fraction=0.05, pad=0.04, aspect=20)
cbar.set_label("Reflectance (%)", fontsize=30)
cbar.ax.tick_params(labelsize=20)
cbar.locator = MaxNLocator(nbins=6)  # Set to 5 tick marks
cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%2.0f'))


# Plot cropped pixel grid overlay
pixel_grid.plot(ax=circles_plot, facecolor='none', edgecolor='red', linewidth=3, label='≥50% Overlap Grid')



scalebar = ScaleBar(
    dx=1,       # Each unit is 1 foot, so dx = feet per meter
    units="m",               # Display scale bar in meters
    location='lower left',
    scale_loc='bottom',
    box_alpha=0.7,
    color='black',
    fixed_value=100,         # Show a label for 100 m
    fixed_units='m',
    font_properties={'size': 20}
)

ax.add_artist(scalebar)
 

# --- Add compass rose (north arrow) ---
# Customize location and size here
x_arrow, y_arrow = 0.95, 0.1  # Relative location in axes coordinates

ax.annotate(
    'N', xy=(x_arrow, y_arrow + 0.05), xytext=(x_arrow, y_arrow),
    arrowprops=dict(facecolor='black', width=5, headwidth=15),
    ha='center', va='center', fontsize=20,
    xycoords='axes fraction'
)  
    
# Formatting
ax.set_axis_off()
plt.tight_layout()
plt.show()

