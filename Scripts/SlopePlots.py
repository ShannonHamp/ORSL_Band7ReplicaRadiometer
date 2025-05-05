# -*- coding: utf-8 -*-
"""
Created on Fri Apr 25 15:09:34 2025

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
lidar_path = r"D:\NASA_EPSCoR_Snow\Band_7_Sensor\Data\Processing_2025_0425\Results\slopeAngle_1.tif"  # EPSG:3604

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

# ------------------- PROCESS LIDAR RASTER -------------------
with rasterio.open(lidar_path) as src:
    image = src.read(1)
    mask = image != src.nodata
    crs = src.crs
    transform = src.transform
    results = (
        {"geometry": shape(geom), "value": value}
        for geom, value in shapes(image, mask=mask, transform=transform)
    )
    gdf_lidar = gpd.GeoDataFrame.from_records(results)
    gdf_lidar.set_crs(crs, inplace=True)
    gdf_lidar = gdf_lidar.to_crs(target_crs)


# --- Reproject both to EPSG:3604 ---
target_crs = "EPSG:3604"
pixel_grid = pixel_grid.to_crs(target_crs)
pixel_grid1 = pixel_grid
gdf_lidar = gdf_lidar.to_crs(target_crs)

# --- Step 3: Intersect slope with pixel grid to crop it ---
gdf_lidar_cropped = gpd.overlay(gdf_lidar, pixel_grid, how='intersection')



# ------------------- SHARED EXTENT -------------------
combined_bounds = gdf_circles.total_bounds  # [minx, miny, maxx, maxy]


#--------------------CALCULATE STANDARD DEVIATION OF SLOPE ANGLE---------------------
# Set values greater than 30 to NaN
gdf_lidar_cropped['value_1'] = gdf_lidar_cropped['value_1'].where(gdf_lidar_cropped['value_1'] <= 30, np.nan)

joined = gpd.sjoin(gdf_lidar_cropped, pixel_grid, how='inner', predicate='intersects')

# --- Group by pixel index and calculate standard deviation ---
# First assign a unique ID to each pixel so we can join later
pixel_grid = pixel_grid.reset_index().rename(columns={"index": "pixel_id"})
joined = gpd.sjoin(gdf_lidar_cropped, pixel_grid[['pixel_id', 'geometry']], how='inner', predicate='intersects')

# Group and calculate standard deviation of circle values
std_by_pixel = joined.groupby('pixel_id')['value_1'].std().reset_index().rename(columns={'value_1': 'std_slope'})

# Merge the std values back to pixel grid
pixel_grid = pixel_grid.merge(std_by_pixel, on='pixel_id', how='left')


#----------------------------------------------------------------------------------------
#------------------------------------STATISTICS------------------------------------------------
mean_std = np.mean(pixel_grid['std_slope']) 
max_std = np.max(pixel_grid['std_slope']) 

#----------------------------CALCULATE AVERAGE SLOPE ANGLE---------------------------
joined = gpd.sjoin(gdf_lidar_cropped, pixel_grid1, how='inner', predicate='intersects')

# --- Group by pixel index and calculate standard deviation ---
# First assign a unique ID to each pixel so we can join later
pixel_grid1 = pixel_grid1.reset_index().rename(columns={"index": "pixel_id"})
joined = gpd.sjoin(gdf_lidar_cropped, pixel_grid1[['pixel_id', 'geometry']], how='inner', predicate='intersects')

# Group and calculate standard deviation of circle values
avg_by_pixel = joined.groupby('pixel_id')['value_1'].mean().reset_index().rename(columns={'value_1': 'avg_slope'})

# Merge the std values back to pixel grid
pixel_grid1 = pixel_grid1.merge(avg_by_pixel, on='pixel_id', how='left')


#----------------------------------------------------------------------------------------
#------------------------------------STATISTICS------------------------------------------------
mean_avg = np.mean(pixel_grid1['avg_slope']) 
max_avg = np.max(pixel_grid1['avg_slope']) 
min_avg = np.min(pixel_grid1['avg_slope']) 


# ------------------- PLOT 1: SLOPE ANGLE -------------------
# ------------------- COLOR MAP SETTINGS -------------------
cmap = cm.Greens
norm = colors.Normalize(vmin=gdf_lidar_cropped['value_1'].min(), vmax=gdf_lidar_cropped['value_2'].max())
norm = colors.Normalize(vmin=0, vmax=20)

fig1, ax1 = plt.subplots(figsize=(10, 8))
gdf_lidar_cropped.plot(column='value_1', cmap=cmap, norm=norm, edgecolor='none', linewidth=0.5, alpha=0.8, ax=ax1)
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
cbar1.set_label("Slope Angle (degrees)", fontsize=30)
cbar1.ax.tick_params(labelsize=20)
cbar1.locator = MaxNLocator(nbins=6)
cbar1.ax.yaxis.set_major_formatter(FormatStrFormatter('%2.2f'))

ax1.set_axis_off()
plt.tight_layout()

# # ------------------- PLOT 2: STANDARD DEVIATION OF SLOPE ANGLE -------------------
cmap = cm.Purples
norm = colors.Normalize(vmin=pixel_grid['std_slope'].min(), vmax=pixel_grid['std_slope'].max())

fig2, ax2 = plt.subplots(figsize=(10, 8))
pixel_grid.plot(column='std_slope', cmap=cmap, norm=norm, edgecolor='black', linewidth=0.5, alpha=0.8, ax=ax2)
pixel_grid.plot(ax=ax2, facecolor='none', edgecolor='red', linewidth=3)

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
cbar2.set_label("Standard Deviation (degrees)", fontsize=30)
cbar2.ax.tick_params(labelsize=20)
cbar2.locator = MaxNLocator(nbins=6)
cbar2.ax.yaxis.set_major_formatter(FormatStrFormatter('%2.2f'))

ax2.set_axis_off()
plt.tight_layout()

# # ------------------- PLOT 3: AVERAGE SLOPE ANGLE -------------------cmap = cm.Purples
cmap = cm.Greys
norm = colors.Normalize(vmin=pixel_grid1['avg_slope'].min(), vmax=pixel_grid1['avg_slope'].max())

fig3, ax3 = plt.subplots(figsize=(10, 8))
pixel_grid1.plot(column='avg_slope', cmap=cmap, norm=norm, edgecolor='black', linewidth=0.5, alpha=0.8, ax=ax3)
pixel_grid1.plot(ax=ax3, facecolor='none', edgecolor='red', linewidth=3)

# Apply same extent
ax3.set_xlim(combined_bounds[0], combined_bounds[2])
ax3.set_ylim(combined_bounds[1], combined_bounds[3])

# Add scalebar and compass
scalebar3 = ScaleBar(dx=1, units="m", location='lower left', scale_loc='bottom',
                     box_alpha=0.7, color='black', fixed_value=100, fixed_units='m',
                     font_properties={'size': 20})
ax3.add_artist(scalebar3)
ax3.annotate('N', xy=(0.95, 0.15), xytext=(0.95, 0.1),
             arrowprops=dict(facecolor='black', width=5, headwidth=15),
             ha='center', va='center', fontsize=20, xycoords='axes fraction')

# Add colorbar
sm3 = cm.ScalarMappable(cmap=cmap, norm=norm)
sm3._A = []
cbar3 = fig3.colorbar(sm3, ax=ax3, fraction=0.05, pad=0.04, aspect=20)
cbar3.set_label("Average Slope Angle (degrees)", fontsize=30)
cbar3.ax.tick_params(labelsize=20)
cbar3.locator = MaxNLocator(nbins=6)
cbar3.ax.yaxis.set_major_formatter(FormatStrFormatter('%2.2f'))

ax3.set_axis_off()
plt.tight_layout()