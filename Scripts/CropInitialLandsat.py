# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 17:20:21 2025

@author: ORSL
"""


import rasterio
from rasterio.mask import mask
from shapely.geometry import box, mapping
import geopandas as gpd
import matplotlib.pyplot as plt

# File paths
raster_path = r"D:\NASA_EPSCoR_Snow\Band_7_Sensor\Data\Processing_2025_0425\Data\LC09_CU_009004_20250406_20250413_02\LC09_CU_009004_20250406_20250413_02_SR_B7.tif"  # EPSG:4326
cropped_raster_output = r"D:\NASA_EPSCoR_Snow\Band_7_Sensor\Data\Processing_2025_0425\Results\croppedLandsat4326.tif"  # EPSG:4326


# --- Define bounding box in lat/lon (EPSG:4326) ---
#Center of field site: 45.231907, -111.476894    
min_lon = -111.476894 - 0.0025
min_lat = 45.231907 - 0.001
max_lon = -111.476894 + 0.0025
max_lat = 45.231907 + 0.0025
    
bbox_geom = box(min_lon, min_lat, max_lon, max_lat)
gdf_bbox = gpd.GeoDataFrame({'geometry': [bbox_geom]}, crs="EPSG:4326")

# --- Open the raster and reproject bbox to match raster's CRS ---
with rasterio.open(raster_path) as src:
    raster_crs = src.crs
    gdf_bbox_proj = gdf_bbox.to_crs(raster_crs)
    bbox_projected_geom = [mapping(gdf_bbox_proj.iloc[0].geometry)]

    # --- Crop the raster using the projected bounding box ---
    out_image, out_transform = mask(src, bbox_projected_geom, crop=True)
    out_meta = src.meta.copy()

    out_meta.update({
        "height": out_image.shape[1],
        "width": out_image.shape[2],
        "transform": out_transform,
        "crs": raster_crs
    })
    
    # --- Save the cropped raster ---
    with rasterio.open(cropped_raster_output, "w", **out_meta) as dest:
        dest.write(out_image)

# --- Plot the cropped raster ---
plt.figure(figsize=(10, 8))
plt.imshow(out_image[0], cmap='gray')
plt.title("Cropped Raster")
plt.axis('off')
plt.show()