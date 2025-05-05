# -*- coding: utf-8 -*-
"""
Created on Fri Apr 25 17:07:53 2025

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
from scipy.stats import pearsonr, shapiro, normaltest
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

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
pixel_grid2 = pixel_grid
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
joined = gpd.sjoin(gdf_circles, pixel_grid1[['pixel_id', 'geometry']], how='inner', predicate='intersects')

# Group and calculate standard deviation of circle values
avg_by_pixel = joined.groupby('pixel_id')['value'].mean().reset_index().rename(columns={'value': 'avg_slope'})

# Merge the std values back to pixel grid
pixel_grid1 = pixel_grid1.merge(avg_by_pixel, on='pixel_id', how='left')


#----------------------------------------------------------------------------------------
#------------------------------------STATISTICS------------------------------------------------
mean_avg = np.mean(pixel_grid1['avg_slope']) 
max_avg = np.max(pixel_grid1['avg_slope']) 
min_avg = np.min(pixel_grid1['avg_slope']) 



# ------------------- CALCULATE STANDARD DEVIATION IN REFLECTANCE -------------------
# Calculate the standard deviation of circles within each pixel
# --- Spatial join: assign each circle to overlapping pixel ---
joined = gpd.sjoin(gdf_circles, pixel_grid2, how='inner', predicate='intersects')

# --- Group by pixel index and calculate standard deviation ---
# First assign a unique ID to each pixel so we can join later
pixel_grid2 = pixel_grid2.reset_index().rename(columns={"index": "pixel_id"})
joined = gpd.sjoin(gdf_circles, pixel_grid2[['pixel_id', 'geometry']], how='inner', predicate='intersects')

# Group and calculate standard deviation of circle values
std_by_pixel = joined.groupby('pixel_id')['value'].std().reset_index().rename(columns={'value': 'std_reflectance'})

# Merge the std values back to pixel grid
pixel_grid2 = pixel_grid2.merge(std_by_pixel, on='pixel_id', how='left')

#----------------------------------------------------------------------------------------
#------------------------------------STATISTICS------------------------------------------------
mean_std = np.mean(pixel_grid2['std_reflectance']) 
max_std = np.max(pixel_grid2['std_reflectance']) 

#----------------------------------------------------------------------------------------
#-----------------------------------ASSESS NORMAL DISTRIBUTION ASSUMPTION------------------------------------------------
# Shapiro-Wilk test
shapiro_stat, shapiro_p = shapiro(pixel_grid1['avg_slope'])
print(f"Shapiro-Wilk test: W={shapiro_stat:.4f}, p={shapiro_p:.4f}")

# D'Agostino and Pearson's test
dagostino_stat, dagostino_p = normaltest(pixel_grid1['avg_slope'])
print(f"D'Agostino and Pearson test: stat={dagostino_stat:.4f}, p={dagostino_p:.4f}")

# Interpretation
if shapiro_p > 0.05 and dagostino_p > 0.05:
    print("Data appears to be normally distributed.")
else:
    print("Data does not appear to be normally distributed.")

#----------------------------------------------------------------------------------------
#-----------------------------------CALCULATE THE PEARSON CORRELATION------------------------------------------------
corr_coefficient_std, p_value_std = pearsonr(pixel_grid['std_slope'], pixel_grid2['std_reflectance'])
print(f"Pearson Correlation Coefficient for std of slope: {corr_coefficient_std}")
print(f"P-value for std of slope: {p_value_std}")


corr_coefficient_avg, p_value_avg = pearsonr(pixel_grid1['avg_slope'], pixel_grid2['std_reflectance'])
print(f"Pearson Correlation Coefficient for avg slope: {corr_coefficient_avg}")
print(f"P-value for avg slope: {p_value_avg}")


#----------------------------------------------------------------------------------------
#-----------------------------------CALCULATE LINEAR REGRESSION------------------------------------------------
# Linear regression model
model_std = LinearRegression()
model_std.fit(pixel_grid['std_slope'].to_numpy().reshape(-1, 1), pixel_grid2['std_reflectance'])

# Predictions
y_pred_std = model_std.predict(pixel_grid['std_slope'].to_numpy().reshape(-1, 1))

# RMSE
rmse_std = np.sqrt(mean_squared_error(pixel_grid2['std_reflectance'], y_pred_std))

# Output
print(f"Intercept std: {model_std.intercept_}")
print(f"Slope std: {model_std.coef_[0]}")
print(f"RMSE std: {rmse_std}")



# Linear regression model
model_avg = LinearRegression()
model_avg.fit(pixel_grid1['avg_slope'].to_numpy().reshape(-1, 1), pixel_grid2['std_reflectance'])

# Predictions
y_pred_avg = model_avg.predict(pixel_grid1['avg_slope'].to_numpy().reshape(-1, 1))

# RMSE
rmse_avg = np.sqrt(mean_squared_error(pixel_grid2['std_reflectance'], y_pred_avg))

# Output
print(f"Intercept avg: {model_avg.intercept_}")
print(f"Slope avg: {model_avg.coef_[0]}")
print(f"RMSE avg: {rmse_avg}")

#----------------------------------------------------------------------------------------
#------------------------------------PLOT------------------------------------------------
plt.figure(figsize=(10, 8))

# Scatter plot of the data
plt.scatter(pixel_grid1['avg_slope'], pixel_grid2['std_reflectance'], color='blue', alpha=0.8, s=100)

# # Regression line
# x_vals = np.linspace(pixel_grid1['avg_slope'].min(), pixel_grid1['avg_slope'].max(), 100).reshape(-1, 1)
# y_vals = model_avg.predict(x_vals)
# plt.plot(x_vals, y_vals, color='gray', linewidth=3, label='Linear Regression')

# # Annotate RMSE and regression equation
# slope = model_avg.coef_[0]
# intercept = model_avg.intercept_
# plt.text(0.70, 0.10,
#          f' RMSE = {rmse_avg:.3f}%',
#          transform=plt.gca().transAxes,
#          fontsize=16,
#          verticalalignment='top',
#          bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

# Labeling the plot
plt.ylabel("Standard Deviation of Reflectance (%)", fontsize=20)
plt.xlabel("Average Slope Angle (degrees)", fontsize=20)
plt.tick_params(axis='both', which='major', labelsize=20)
# Show the plot
plt.grid(False)
plt.show()


#------------------------------------PLOT------------------------------------------------
plt.figure(figsize=(10, 8))

# Scatter plot of the data
plt.scatter(pixel_grid['std_slope'], pixel_grid2['std_reflectance'], color='blue', alpha=0.8, s=100)

# # Regression line
# x_vals = np.linspace(pixel_grid['std_slope'].min(), pixel_grid['std_slope'].max(), 100).reshape(-1, 1)
# y_vals = model_std.predict(x_vals)
# plt.plot(x_vals, y_vals, color='gray', linewidth=3, label='Linear Regression')

# # Annotate RMSE and regression equation
# slope = model_std.coef_[0]
# intercept = model_std.intercept_
# plt.text(0.70, 0.10,
#          f' RMSE = {rmse_std:.3f}%',
#          transform=plt.gca().transAxes,
#          fontsize=16,
#          verticalalignment='top',
#          bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

# Labeling the plot
plt.ylabel("Standard Deviation of Reflectance (%)", fontsize=20)
plt.xlabel("Standard Deviation of Slope Angle (degrees)", fontsize=20)
plt.tick_params(axis='both', which='major', labelsize=20)

# Show the plot
plt.grid(False)
plt.show()


