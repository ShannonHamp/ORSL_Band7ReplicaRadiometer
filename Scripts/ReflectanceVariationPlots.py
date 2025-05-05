# -*- coding: utf-8 -*-
"""
Created on Fri Apr 25 13:31:48 2025

@author: ORSL
"""

import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
from matplotlib_scalebar.scalebar import ScaleBar
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
from shapely.geometry import Point, shape
import rasterio
from rasterio.features import shapes
import numpy as np


# ------------------- FILE PATHS -------------------
geopackage_path = r"D:\NASA_EPSCoR_Snow\Band_7_Sensor\Data\Processing_2025_0425\Results\pixelGrid_cropped_to_UAV_Area_3604.gpkg"
radiometer_path = r"D:\NASA_EPSCoR_Snow\Band_7_Sensor\Data\Processing_2025_0425\Data\Radiometer\processed_reflectance_2025_0407_flight1.txt"
landsat_path = r"D:\NASA_EPSCoR_Snow\Band_7_Sensor\Data\Processing_2025_0425\Results\croppedLandsat3604.tif"

target_crs = "EPSG:3604"
radius_m = 7.19235  # 15m diameter / 2


# ------------------- READ PIXEL GRID -------------------
pixel_grid = gpd.read_file(geopackage_path).to_crs(target_crs)

# ------------------- PROCESS RADIOMETER DATA -------------------
df = pd.read_csv(radiometer_path)
df['value'] = pd.to_numeric(df['reflectance'], errors='coerce')
gdf_points = gpd.GeoDataFrame(
    df[['value']],
    geometry=[Point(lon, lat) for lon, lat in zip(df.longitude, df.latitude)],
    crs="EPSG:4326"
).to_crs(target_crs)

gdf_circles = gdf_points.copy()
gdf_circles['geometry'] = gdf_points.buffer(radius_m)

# ------------------- PROCESS LANDSAT RASTER -------------------
with rasterio.open(landsat_path) as src:
    image = src.read(1)
    mask = image != src.nodata
    crs = src.crs
    transform = src.transform
    results = (
        {"geometry": shape(geom), "value": value / 1000}
        for geom, value in shapes(image, mask=mask, transform=transform)
    )
    gdf_landsat = gpd.GeoDataFrame.from_records(results)
    gdf_landsat.set_crs(crs, inplace=True)
    gdf_landsat = gdf_landsat.to_crs(target_crs)

# Crop Landsat to pixel grid
gdf_landsat_cropped = gpd.overlay(gdf_landsat, pixel_grid, how='intersection')

# ------------------- SHARED EXTENT -------------------
combined_bounds = gdf_circles.total_bounds  # [minx, miny, maxx, maxy]

# ------------------- CALCULATE STANDARD DEVIATION -------------------
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


# ------------------- PLOT 1: STANDARD DEVIATION -------------------
# ------------------- COLOR MAP SETTINGS -------------------
cmap = cm.Purples
norm = colors.Normalize(vmin=pixel_grid['std_reflectance'].min(), vmax=pixel_grid['std_reflectance'].max())

fig1, ax1 = plt.subplots(figsize=(10, 8))
pixel_grid.plot(column='std_reflectance', cmap=cmap, norm=norm, edgecolor='black', linewidth=0.5, alpha=0.8, ax=ax1)
pixel_grid.plot(ax=ax1, facecolor='none', edgecolor='red', linewidth=3)

# Apply same extent
ax1.set_xlim(combined_bounds[0], combined_bounds[2])
ax1.set_ylim(combined_bounds[1], combined_bounds[3])

# Add scalebar and compass
scalebar1 = ScaleBar(dx=1, units="m", location='lower left', scale_loc='bottom',
                     box_alpha=0.7, color='black', fixed_value=100, fixed_units='m',
                     font_properties={'size': 20})
ax1.add_artist(scalebar1)
ax1.annotate('N', xy=(0.95, 0.15), xytext=(0.95, 0.1),
             arrowprops=dict(facecolor='black', width=5, headwidth=15),
             ha='center', va='center', fontsize=20, xycoords='axes fraction')

# Add colorbar
sm1 = cm.ScalarMappable(cmap=cmap, norm=norm)
sm1._A = []
cbar1 = fig1.colorbar(sm1, ax=ax1, fraction=0.05, pad=0.04, aspect=20)
cbar1.set_label("Standard Deviation (%)", fontsize=30)
cbar1.ax.tick_params(labelsize=20)
cbar1.locator = MaxNLocator(nbins=6)
cbar1.ax.yaxis.set_major_formatter(FormatStrFormatter('%2.2f'))

ax1.set_axis_off()
plt.tight_layout()

# ------------------- PLOT 2: PERCENT DIFFERENCE -------------------
cmap = cm.RdYlBu
norm = colors.Normalize(vmin=gdf_circles['percent_difference'].min(), vmax=gdf_circles['percent_difference'].max())

fig2, ax2 = plt.subplots(figsize=(10, 8))
gdf_circles.plot(column='percent_difference', cmap=cmap, norm=norm, edgecolor='black', linewidth=0.5, alpha=0.8, ax=ax2)
pixel_grid.plot(ax=ax2, facecolor='none', edgecolor='black', linewidth=3)

# Apply same extent
ax2.set_xlim(combined_bounds[0], combined_bounds[2])
ax2.set_ylim(combined_bounds[1], combined_bounds[3])

# Add scalebar and compass
scalebar2 = ScaleBar(dx=1, units="m", location='lower left', scale_loc='bottom',
                     box_alpha=0.7, color='black', fixed_value=100, fixed_units='m',
                     font_properties={'size': 20})
ax2.add_artist(scalebar2)
ax2.annotate('N', xy=(0.95, 0.15), xytext=(0.95, 0.1),
             arrowprops=dict(facecolor='black', width=5, headwidth=15),
             ha='center', va='center', fontsize=20, xycoords='axes fraction')

# Add colorbar
sm2 = cm.ScalarMappable(cmap=cmap, norm=norm)
sm2._A = []
cbar2 = fig2.colorbar(sm2, ax=ax2, fraction=0.05, pad=0.04, aspect=20)
cbar2.set_label("Percent Difference (%)", fontsize=30)
cbar2.ax.tick_params(labelsize=20)
cbar2.locator = MaxNLocator(nbins=6)
cbar2.ax.yaxis.set_major_formatter(FormatStrFormatter('%2.0f'))

ax2.set_axis_off()
plt.tight_layout()
plt.show()
