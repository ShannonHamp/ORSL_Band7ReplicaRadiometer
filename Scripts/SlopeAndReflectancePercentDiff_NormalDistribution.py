# -*- coding: utf-8 -*-
"""
Created on Sun Apr 27 18:24:03 2025

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
import seaborn as sns


# ------------------- FILE PATHS -------------------
geopackage_path = r"D:\NASA_EPSCoR_Snow\Band_7_Sensor\Data\Processing_2025_0425\Results\pixelGrid_cropped_to_UAV_Area_3604.gpkg"
radiometer_path = r"D:\NASA_EPSCoR_Snow\Band_7_Sensor\Data\Processing_2025_0425\Data\Radiometer\processed_reflectance_2025_0407_flight1.txt"
lidar_path = r"D:\NASA_EPSCoR_Snow\Band_7_Sensor\Data\Processing_2025_0425\Results\slopeAngle_1.tif"  # EPSG:3604
landsat_path = r"D:\NASA_EPSCoR_Snow\Band_7_Sensor\Data\Processing_2025_0425\Results\croppedLandsat3604.tif"

target_crs = "EPSG:3604"
radius_m = 7.19235  # 15m diameter / 2


# ------------------- READ PIXEL GRID -------------------
pixel_grid = gpd.read_file(geopackage_path).to_crs(target_crs)


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





#----------------------------------------------------------------------------------------
#-----------------------ASSESS NORMAL DISTRIBUTION ASSUMPTION-----------------------------
# Shapiro-Wilk test
shapiro_stat, shapiro_p = shapiro(gdf_circles['percent_difference'])
print(f"Shapiro-Wilk test: W={shapiro_stat:.4f}, p={shapiro_p:.4f}")

# D'Agostino and Pearson's test
dagostino_stat, dagostino_p = normaltest(gdf_circles['percent_difference'])
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
axs[0].hist(gdf_circles['percent_difference'].dropna(), bins=30, color='skyblue', edgecolor='black')
axs[0].set_title('Distribution of Percent Difference (%)', fontsize=16)
axs[0].set_xlabel('Percent Difference (%)')
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
#-------- Apply yeo-johnson transform (generalization of box-cox) to percent difference
data = gdf_circles['percent_difference'].dropna().values.reshape(-1, 1)

# Initialize the transformer
pt = PowerTransformer(method='yeo-johnson', standardize=True)

# Fit and transform the data
transformed = pt.fit_transform(data)

# Create a new column in the original GeoDataFrame (preserving original indices)
gdf_circles.loc[gdf_circles['percent_difference'].notna(), 'yeo_percent_difference'] = transformed.flatten()


#-------- Apply box-cox transform to avg slope
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
shapiro_stat, shapiro_p = shapiro(gdf_circles['yeo_percent_difference'])
print(f"Shapiro-Wilk test: W={shapiro_stat:.4f}, p={shapiro_p:.4f}")

# D'Agostino and Pearson's test
dagostino_stat, dagostino_p = normaltest(gdf_circles['yeo_percent_difference'])
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
axs[0].hist(gdf_circles['yeo_percent_difference'].dropna(), bins=30, color='skyblue', edgecolor='black')
axs[0].set_title('Distribution of Percent Difference (%)', fontsize=16)
axs[0].set_xlabel('Percent Difference (%)')
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
#-----------------------------------CALCULATE THE PEARSON CORRELATION W/R/T AVG SLOPE------------------------------------------------
# Ensure gdf_circles has no unexpected index issues
gdf_circles = gdf_circles.reset_index(drop=True)

# Spatial join: assign each circle to a pixel
circle_with_pixel = gpd.sjoin(
    gdf_circles[['yeo_percent_difference', 'geometry']],
    pixel_grid1[['pixel_id', 'geometry']],
    how='inner',
    predicate='intersects'
)

# Ensure pixel_id is just a Series, not part of index
circle_with_pixel = circle_with_pixel.reset_index(drop=True)


# Now safely group
avg_yeo_by_pixel = circle_with_pixel[['pixel_id', 'yeo_percent_difference']].groupby('pixel_id').mean().reset_index()
# Merge back with pixel_grid to align with 'boxcox_std_slope'
merged_df = pd.merge(
    pixel_grid1[['pixel_id', 'boxcox_avg_slope']],
    avg_yeo_by_pixel,
    on='pixel_id',
    how='inner'
)

# Drop rows with NaN values in either column
valid = merged_df[['yeo_percent_difference', 'boxcox_avg_slope']].dropna()

# Compute Pearson correlation
corr_coef_avg, p_value_avg = pearsonr(valid['yeo_percent_difference'], valid['boxcox_avg_slope'])

# Print results
print("Pearson correlation between Yeo-Johnson percent difference and Box-Cox avg slope:")
print(f"  r = {corr_coef_avg:.4f}, p = {p_value_avg:.20f}")

#----------------------------------------------------------------------------------------
#-----------------------------------CALCULATE THE PEARSON CORRELATION W/R/T STD SLOPE------------------------------------------------
# Ensure gdf_circles has no unexpected index issues
gdf_circles = gdf_circles.reset_index(drop=True)

# Spatial join: assign each circle to a pixel
circle_with_pixel = gpd.sjoin(
    gdf_circles[['yeo_percent_difference', 'geometry']],
    pixel_grid[['pixel_id', 'geometry']],
    how='inner',
    predicate='intersects'
)

# Ensure pixel_id is just a Series, not part of index
circle_with_pixel = circle_with_pixel.reset_index(drop=True)



# Now safely group
avg_yeo_by_pixel = circle_with_pixel[['pixel_id', 'yeo_percent_difference']].groupby('pixel_id').mean().reset_index()
# Merge back with pixel_grid to align with 'boxcox_std_slope'
merged_df = pd.merge(
    pixel_grid[['pixel_id', 'boxcox_std_slope']],
    avg_yeo_by_pixel,
    on='pixel_id',
    how='inner'
)

# Drop rows with NaN values in either column
valid = merged_df[['yeo_percent_difference', 'boxcox_std_slope']].dropna()

# Compute Pearson correlation
corr_coef_std, p_value_std = pearsonr(valid['yeo_percent_difference'], valid['boxcox_std_slope'])

# Print results
print("Pearson correlation between Yeo-Johnson percent difference and Box-Cox std slope:")
print(f"  r = {corr_coef_std:.4f}, p = {p_value_std:.20f}")


#----------------------------------------------------------------------------------------
#------------------------------------PLOT------------------------------------------------
# ------------------- Scatterplot: Percent Difference vs. Avg Slope -------------------
# Recreate merged_df for avg slope correlation if needed
merged_df_avg = pd.merge(
    pixel_grid1[['pixel_id', 'boxcox_avg_slope']],
    circle_with_pixel[['pixel_id', 'yeo_percent_difference']],
    on='pixel_id',
    how='inner'
)

fig, ax = plt.subplots(figsize=(8, 6))
sns.regplot(
    x='boxcox_avg_slope',
    y='yeo_percent_difference',
    data=merged_df_avg.dropna(subset=['boxcox_avg_slope', 'yeo_percent_difference']),
    scatter_kws={'s': 50, 'alpha': 0.7},
    line_kws={'color': 'red'},
    ax=ax
)
#ax.set_title("Yeo-Johnson Percent Difference vs. Box-Cox Avg Slope", fontsize=14)
ax.set_xlabel("Transformed Average Slope", fontsize=20)
ax.set_ylabel("Transformed Percent Difference", fontsize=20)
plt.tick_params(axis='both', which='major', labelsize=20)
plt.tight_layout()
plt.show()

# ------------------- Scatterplot: Percent Difference vs. Std Slope -------------------
# Recreate merged_df for avg slope correlation if needed
merged_df_std = pd.merge(
    pixel_grid[['pixel_id', 'boxcox_std_slope']],
    circle_with_pixel[['pixel_id', 'yeo_percent_difference']],
    on='pixel_id',
    how='inner'
)

fig, ax = plt.subplots(figsize=(8, 6))
sns.regplot(
    x='boxcox_std_slope',
    y='yeo_percent_difference',
    data=merged_df_std.dropna(subset=['boxcox_std_slope', 'yeo_percent_difference']),
    scatter_kws={'s': 50, 'alpha': 0.7},
    line_kws={'color': 'red'},
    ax=ax
)
#ax.set_title("Yeo-Johnson Percent Difference vs. Box-Cox Std Slope", fontsize=14)
ax.set_xlabel("Transformed Standard Deviation of Slope", fontsize=20)
ax.set_ylabel("Transformed Percent Difference", fontsize=20)
plt.tick_params(axis='both', which='major', labelsize=20)
plt.tight_layout()
plt.show()




