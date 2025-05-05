# -*- coding: utf-8 -*-
"""
Created on Fri Apr 25 13:34:04 2025

@author: ORSL
"""

import geopandas as gpd
import shapely
from shapely.geometry import Point
from pyproj import CRS, Transformer
import matplotlib.pyplot as plt
import numpy as np
from shapely.ops import transform as shapely_transform
import pandas as pd
import rasterio
from rasterio.plot import show

from rasterio.transform import from_origin
from rasterio.features import rasterize
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import MaxNLocator
import matplotlib.colors as colors
from matplotlib.ticker import MaxNLocator, FormatStrFormatter

from matplotlib_scalebar.scalebar import ScaleBar  # <-- import scale bar



# ------------------- FILE PATHS -------------------
geopackage_path = r"D:\NASA_EPSCoR_Snow\Band_7_Sensor\Data\Processing_2025_0425\Results\pixelGrid_cropped_to_UAV_Area_3604.gpkg"
radiometer_path = r"D:\NASA_EPSCoR_Snow\Band_7_Sensor\Data\Processing_2025_0425\Data\Radiometer\processed_reflectance_2025_0407_flight1.txt"



# Read the pixel grid from the GeoPackage
pixel_grid = gpd.read_file(geopackage_path)


# --- Config ---
radius_m = 7.19235  # Radius for 15m diameter circle

# --- Step 1: Read CSV ---
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


#----------------------------------------------------------------------------------------
#--------------------------CALCUATE STANDARD DEVIATION-----------------------------------
# Calculate the standard deviation of circles within each pixel
# --- Spatial join: assign each circle to overlapping pixel ---
joined = gpd.sjoin(gdf_circles, pixel_grid, how='inner', predicate='intersects')

# --- Group by pixel index and calculate standard deviation ---
# First assign a unique ID to each pixel so we can join later
pixel_grid = pixel_grid.reset_index().rename(columns={"index": "pixel_id"})
joined = gpd.sjoin(gdf_circles, pixel_grid[['pixel_id', 'geometry']], how='inner', predicate='intersects')

# Group and calculate standard deviation of circle values
std_by_pixel = joined.groupby('pixel_id')['value'].std().reset_index().rename(columns={'value': 'std_reflectance'})

# Merge the std values back to pixel grid
pixel_grid = pixel_grid.merge(std_by_pixel, on='pixel_id', how='left')


#----------------------------------------------------------------------------------------
#------------------------------------STATISTICS------------------------------------------------
mean_std = np.mean(pixel_grid['std_reflectance']) 
max_std = np.max(pixel_grid['std_reflectance']) 

#----------------------------------------------------------------------------------------
#------------------------------------PLOT------------------------------------------------
# --- Plot the standard deviation with colorbar and overlay cropped grid ---
cmap = cm.RdYlBu
norm = colors.Normalize(vmin=pixel_grid['std_reflectance'].min(), vmax=pixel_grid['std_reflectance'].max())

fig, ax = plt.subplots(figsize=(10, 8))

# Plot circles, colored by 'value'
std_plot = pixel_grid.plot(
    column='std_reflectance',
    cmap=cmap,
    norm=norm,
    legend=False,
    edgecolor='red',
    linewidth=3,
    alpha=0.8,
    ax=ax
)

# COLORBAR
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm._A = []  # required workaround for older versions of matplotlib
cbar = fig.colorbar(sm, ax=ax, fraction=0.05, pad=0.04, aspect=20)
cbar.set_label("Standard Deviation (%)", fontsize=30)
cbar.ax.tick_params(labelsize=20)
cbar.locator = MaxNLocator(nbins=6)  # Set to 5 tick marks
cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%2.2f'))


# # Plot cropped pixel grid overlay
# pixel_grid.plot(ax=circles_plot, facecolor='none', edgecolor='red', linewidth=3, label='≥50% Overlap Grid')



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
