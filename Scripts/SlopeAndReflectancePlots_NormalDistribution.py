# -*- coding: utf-8 -*-
"""
Created on Sun Apr 27 17:30:45 2025

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
from sklearn.preprocessing import PowerTransformer
from scipy.stats import boxcox

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
#-----------------------ASSESS NORMAL DISTRIBUTION ASSUMPTION-----------------------------
# Shapiro-Wilk test
shapiro_stat, shapiro_p = shapiro(pixel_grid['std_slope'])
print(f"Shapiro-Wilk test: W={shapiro_stat:.4f}, p={shapiro_p:.4f}")

# D'Agostino and Pearson's test
dagostino_stat, dagostino_p = normaltest(pixel_grid['std_slope'])
print(f"D'Agostino and Pearson test: stat={dagostino_stat:.4f}, p={dagostino_p:.4f}")

# Interpretation
if shapiro_p > 0.05 and dagostino_p > 0.05:
    print("Data appears to be normally distributed.")
else:
    print("Data does not appear to be normally distributed.")


# Plot each distribution
# Create a figure with 3 subplots for histograms
fig, axs = plt.subplots(3, 1, figsize=(12, 16), constrained_layout=True)

# 1. Histogram: Standard Deviation of Reflectance
axs[0].hist(pixel_grid2['std_reflectance'].dropna(), bins=30, color='skyblue', edgecolor='black')
axs[0].set_title('Distribution of Standard Deviation of Reflectance (%)', fontsize=16)
axs[0].set_xlabel('Std Reflectance (%)')
axs[0].set_ylabel('Frequency')

# 2. Histogram: Average Slope Angle
axs[1].hist(pixel_grid1['avg_slope'].dropna(), bins=30, color='orange', edgecolor='black')
axs[1].set_title('Distribution of Average Slope Angle (degrees)', fontsize=16)
axs[1].set_xlabel('Average Slope Angle (°)')
axs[1].set_ylabel('Frequency')

# 3. Histogram: Standard Deviation of Slope Angle
axs[2].hist(pixel_grid['std_slope'].dropna(), bins=30, color='green', edgecolor='black')
axs[2].set_title('Distribution of Standard Deviation of Slope Angle (degrees)', fontsize=16)
axs[2].set_xlabel('Std Slope Angle (°)')
axs[2].set_ylabel('Frequency')

plt.show()


#----------------------------------------------------------------------------------------
#-----------------------APPLY LOG TRANSFORM TO SLOPE DATA (AVG and STD)-----------------------------
#-------- Apply box-cox transform to avg slope
# pixel_grid1['log_avg_slope'] = np.log1p(pixel_grid1['avg_slope'])
valid_avg_slope = pixel_grid1['avg_slope'].dropna()
valid_avg_slope = valid_avg_slope[valid_avg_slope > 0]

# Apply Box-Cox transformation
boxcox_transformed, lambda_ = boxcox(valid_avg_slope)

# Create a new column and assign transformed values back by index
pixel_grid1['boxcox_avg_slope'] = np.nan  # Initialize with NaNs
pixel_grid1.loc[valid_avg_slope.index, 'boxcox_avg_slope'] = boxcox_transformed


#------- Apply box-cox transform to std of slope
valid_std_slope = pixel_grid['std_slope'].dropna()
valid_std_slope = valid_std_slope[valid_std_slope > 0]

# Apply Box-Cox transformation
boxcox_transformed, lambda_ = boxcox(valid_std_slope)

# Create a new column and assign transformed values back by index
pixel_grid['boxcox_std_slope'] = np.nan  # Initialize with NaNs
pixel_grid.loc[valid_std_slope.index, 'boxcox_std_slope'] = boxcox_transformed

#----------------------------------------------------------------------------------------
#-----------------------REASSESS NORMAL DISTRIBUTION ASSUMPTION-----------------------------
# Shapiro-Wilk test
shapiro_stat, shapiro_p = shapiro(pixel_grid['boxcox_std_slope'])
print(f"Shapiro-Wilk test: W={shapiro_stat:.4f}, p={shapiro_p:.4f}")

# D'Agostino and Pearson's test
dagostino_stat, dagostino_p = normaltest(pixel_grid['boxcox_std_slope'])
print(f"D'Agostino and Pearson test: stat={dagostino_stat:.4f}, p={dagostino_p:.4f}")

# Interpretation
if shapiro_p > 0.05 and dagostino_p > 0.05:
    print("Data appears to be normally distributed.")
else:
    print("Data does not appear to be normally distributed.")


# Plot each distribution
# Create a figure with 3 subplots for histograms
fig, axs = plt.subplots(3, 1, figsize=(12, 16), constrained_layout=True)

# 1. Histogram: Standard Deviation of Reflectance
axs[0].hist(pixel_grid2['std_reflectance'].dropna(), bins=30, color='skyblue', edgecolor='black')
axs[0].set_title('Distribution of Standard Deviation of Reflectance (%)', fontsize=16)
axs[0].set_xlabel('Std Reflectance (%)')
axs[0].set_ylabel('Frequency')

# 2. Histogram: Average Slope Angle
axs[1].hist(pixel_grid1['boxcox_avg_slope'].dropna(), bins=30, color='orange', edgecolor='black')
axs[1].set_title('Distribution of Average Slope Angle (degrees)', fontsize=16)
axs[1].set_xlabel('Average Slope Angle (°)')
axs[1].set_ylabel('Frequency')

# 3. Histogram: Standard Deviation of Slope Angle
axs[2].hist(pixel_grid['boxcox_std_slope'].dropna(), bins=30, color='green', edgecolor='black')
axs[2].set_title('Distribution of Standard Deviation of Slope Angle (degrees)', fontsize=16)
axs[2].set_xlabel('Std Slope Angle (°)')
axs[2].set_ylabel('Frequency')

plt.show()

#----------------------------------------------------------------------------------------
#-----------------------------------CALCULATE THE PEARSON CORRELATION------------------------------------------------
corr_coefficient_std, p_value_std = pearsonr(pixel_grid['boxcox_std_slope'], pixel_grid2['std_reflectance'])
print(f"Pearson Correlation Coefficient for std of slope: {corr_coefficient_std}")
print(f"P-value for std of slope: {p_value_std}")


corr_coefficient_avg, p_value_avg = pearsonr(pixel_grid1['boxcox_avg_slope'], pixel_grid2['std_reflectance'])
print(f"Pearson Correlation Coefficient for avg slope: {corr_coefficient_avg}")
print(f"P-value for avg slope: {p_value_avg}")



#----------------------------------------------------------------------------------------
#------------------------------------PLOT------------------------------------------------
plt.figure(figsize=(10, 8))

# Scatter plot of the data
plt.scatter(pixel_grid1['boxcox_avg_slope'], pixel_grid2['std_reflectance'], color='blue', alpha=0.8, s=100)


# Labeling the plot
plt.ylabel("Standard Deviation of Reflectance (%)", fontsize=20)
plt.xlabel("Boxcox Transform of Slope Angle", fontsize=20)
plt.tick_params(axis='both', which='major', labelsize=20)
# Show the plot
plt.grid(False)
plt.show()


#------------------------------------PLOT------------------------------------------------
plt.figure(figsize=(10, 8))

# Scatter plot of the data
plt.scatter(pixel_grid['boxcox_std_slope'], pixel_grid2['std_reflectance'], color='blue', alpha=0.8, s=100)



# Labeling the plot
plt.ylabel("Standard Deviation of Reflectance (%)", fontsize=20)
plt.xlabel("Boxcox Transform of Standard Deviation of Slope Angle", fontsize=20)
plt.tick_params(axis='both', which='major', labelsize=20)

# Show the plot
plt.grid(False)
plt.show()


