# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 17:29:31 2025

@author: ORSL
"""

import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt


# File paths
geopackage_path = r"D:\NASA_EPSCoR_Snow\Band_7_Sensor\Data\Processing_2025_0425\Results\pixel_grid_3604.gpkg"  # Pixel grid in GPKG format
radiometer_path = r"D:\NASA_EPSCoR_Snow\Band_7_Sensor\Data\Processing_2025_0425\Data\Radiometer\processed_reflectance_2025_0407_flight1.txt"

# Read the pixel grid from the GeoPackage
gdf_pixels = gpd.read_file(geopackage_path)


# --- Config ---
radius_m = 7.19235  # Radius for 15m diameter circle

# --- Read CSV containing radiometer data ---
df = pd.read_csv(radiometer_path)
latitudes = df.latitude
longitudes = df.longitude

# Create GeoDataFrame with WGS84 coordinates
gdf_points = gpd.GeoDataFrame(
    df,
    geometry=gpd.points_from_xy(df.longitude, df.latitude),
    crs="EPSG:4326"  # WGS84
)

# Project to a metric CRS (EPSG:3604 - NAD83 / California zone 4 (meters))
# EPSG:3604 uses meters, so no need to convert the radius.
gdf_projected = gdf_points.to_crs(epsg=3604)

# Buffer each point by 7.5 meters directly (since EPSG:3604 uses meters)
gdf_circles = gdf_projected.copy()
gdf_circles['geometry'] = gdf_projected.buffer(radius_m)

# --- Reproject both to EPSG:3604 (if not already in that CRS) ---
target_crs = "EPSG:3604"  # Use EPSG:3604 which is in meters
gdf_pixels = gdf_pixels.to_crs(target_crs)
gdf_circles = gdf_circles.to_crs(target_crs)

# ---  Dissolve all circles into one multipolygon (the "total area" covered) ---
gdf_union = gdf_circles.unary_union  # this is a shapely geometry, not a GeoDataFrame

# --- Intersect each pixel with the unioned area ---
gdf_pixels["pixel_area"] = gdf_pixels.geometry.area
gdf_pixels["intersection_area"] = gdf_pixels.geometry.intersection(gdf_union).area

# --- Filter pixels with ≥50% area covered by the union of circles ---
gdf_cropped = gdf_pixels[gdf_pixels["intersection_area"] >= 0.5 * gdf_pixels["pixel_area"]].copy()

# Drop temp columns
gdf_cropped = gdf_cropped.drop(columns=["pixel_area", "intersection_area"])


print(f"Pixels retained after cropping: {len(gdf_cropped)}")


gdf_cropped = gdf_cropped.drop(index=108)

# --- Save cropped pixel grid to file ---
gdf_cropped.to_file(r"D:\NASA_EPSCoR_Snow\Band_7_Sensor\Data\Processing_2025_0425\Results\pixelGrid_cropped_to_UAV_Area_3604.gpkg", driver="GPKG")


# PLOT: Check the resulting pixel grid
fig, ax = plt.subplots(figsize=(10, 10))

# Plot the 15m diameter circles
gdf_circles.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=2)

# Plot only the grid pixels with ≥50% overlap
gdf_cropped.plot(ax=ax, color='none', edgecolor='red')

# Formatting
ax.set_title("15m Circles and Pixel Grid (≥50% Overlap)")
ax.legend()
plt.axis('equal')
plt.show()
