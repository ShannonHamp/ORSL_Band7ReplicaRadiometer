# -*- coding: utf-8 -*-
"""
Created on Fri Apr 25 13:36:16 2025

@author: ORSL
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 11:32:00 2025

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

import rasterio
from rasterio.features import shapes
import geopandas as gpd
from shapely.geometry import shape
import numpy as np




# ------------------- FILE PATHS -------------------
geopackage_path = r"D:\NASA_EPSCoR_Snow\Band_7_Sensor\Data\Processing_2025_0425\Results\pixelGrid_cropped_to_UAV_Area_3604.gpkg"
radiometer_path = r"D:\NASA_EPSCoR_Snow\Band_7_Sensor\Data\Processing_2025_0425\Data\Radiometer\processed_reflectance_2025_0407_flight1.txt"
landsat_path = r"D:\NASA_EPSCoR_Snow\Band_7_Sensor\Data\Processing_2025_0425\Results\croppedLandsat3604.tif"



# Read the Landsat pixel grid
gdf_pixels = gpd.read_file(geopackage_path)

# --- Read the raster ---
with rasterio.open(landsat_path) as src:
    image = src.read(1)  # Read the first band
    mask = image != src.nodata  # Mask out nodata values
    crs = src.crs

    # --- Step 2: Extract shapes (polygonize) ---
    results = (
        {"geometry": shape(geom), "value": value/1000}
        for geom, value in shapes(image, mask=mask, transform=src.transform)
    )

    # --- Step 3: Convert to GeoDataFrame ---
    gdf_landsat = gpd.GeoDataFrame.from_records(results)
    gdf_landsat.set_crs(crs, inplace=True)
    



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

# Reproject
target_crs = "EPSG:3604"  # Use EPSG:3604 which is in meters
gdf_pixels = gdf_pixels.to_crs(target_crs)
gdf_landsat = gdf_landsat.to_crs(target_crs)
gdf_projected = gdf_points.to_crs(target_crs)


# Create 15m diameter buffer circles (2256 is in meters, so no conversion needed)
gdf_circles = gdf_projected.copy()
gdf_circles['geometry'] = gdf_projected.buffer(radius_m)


#----------------------------------------------------------------------------------------
#--------------------------CALCUATE PERCENT DIFFERENCE-----------------------------------
# --- Step 1: Add index to circles ---
gdf_circles = gdf_circles.reset_index(drop=True)
gdf_circles['circle_index'] = gdf_circles.index

# --- Step 2: Spatial join to get overlapping Landsat pixel values for each circle ---
joined = gpd.sjoin(
    gdf_landsat[['geometry', 'value']],  # Landsat pixels with value
    gdf_circles[['circle_index', 'geometry']],  # Circles with index
    how='inner',
    predicate='intersects'
)

# --- Step 3: Group by circle_index and average the Landsat values ---
avg_landsat = joined.groupby('circle_index')['value'].mean().reset_index()
avg_landsat.rename(columns={'value': 'avg_landsat_value'}, inplace=True)

# --- Step 4: Merge back into original gdf_circles ---
gdf_circles = gdf_circles.merge(avg_landsat, on='circle_index', how='left')

# --- Step 5: Compute percent difference ---
gdf_circles['percent_difference'] = (
    (gdf_circles['value'] - gdf_circles['avg_landsat_value']) /
    ((gdf_circles['value'] + gdf_circles['avg_landsat_value']) / 2)
) * 100


#----------------------------------------------------------------------------------------
#------------------------------------STATISTICS------------------------------------------------
mean_diff = np.mean(gdf_circles['percent_difference']) 
max_diff = np.max(gdf_circles['percent_difference']) 
min_diff = np.min(gdf_circles['percent_difference']) 

#----------------------------------------------------------------------------------------
#------------------------------------PLOT------------------------------------------------
# --- Plot the standard deviation with colorbar and overlay cropped grid ---
cmap = cm.RdYlBu
norm = colors.Normalize(vmin=gdf_circles['percent_difference'].min(), vmax=gdf_circles['percent_difference'].max())

fig, ax = plt.subplots(figsize=(10, 8))

# Plot circles, colored by 'value'
percent_diff_plot = gdf_circles.plot(
    column='percent_difference',
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
cbar.set_label("Percent Difference (%)", fontsize=30)
cbar.ax.tick_params(labelsize=20)
cbar.locator = MaxNLocator(nbins=6)  # Set to 5 tick marks
cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%2.0f'))


# Plot cropped pixel grid overlay
gdf_pixels.plot(ax=percent_diff_plot, facecolor='none', edgecolor='black', linewidth=3, label='≥50% Overlap Grid')



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
