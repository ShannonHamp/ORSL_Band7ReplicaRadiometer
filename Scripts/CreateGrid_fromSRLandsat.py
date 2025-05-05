# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 16:38:11 2025

@author: ORSL
"""

import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import geopandas as gpd
import matplotlib.pyplot as plt
from rasterio.plot import show
from shapely.geometry import box

# File paths
input_raster_path = r"D:\NASA_EPSCoR_Snow\Band_7_Sensor\Data\Processing_2025_0425\Results\croppedLandsat4326.tif"  # EPSG:4326
reprojected_raster_path = r"D:\NASA_EPSCoR_Snow\Band_7_Sensor\Data\Processing_2025_0425\Results\croppedLandsat3604.tif"
output_geopackage_path = r"D:\NASA_EPSCoR_Snow\Band_7_Sensor\Data\Processing_2025_0425\Results\pixel_grid_3604.gpkg"

# --- Reproject the raster to EPSG:3604 ---
dst_crs = 'EPSG:3604'  # NAD83 (meters)

with rasterio.open(input_raster_path) as src:
    transform, width, height = calculate_default_transform(
        src.crs, dst_crs, src.width, src.height, *src.bounds
    )
    
    kwargs = src.meta.copy()
    kwargs.update({
        'crs': dst_crs,
        'transform': transform,
        'width': width,
        'height': height
    })

    with rasterio.open(reprojected_raster_path, 'w', **kwargs) as dst:
        for i in range(1, src.count + 1):
            reproject(
                source=rasterio.band(src, i),
                destination=rasterio.band(dst, i),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=dst_crs,
                resampling=Resampling.nearest
            )

# --- Polygonize raster into pixel grid ---
with rasterio.open(reprojected_raster_path) as src:
    transform = src.transform
    width = src.width
    height = src.height
    crs = src.crs
    data = src.read(1)

    polygons = []
    values = []

    for row in range(height):
        for col in range(width):
            value = data[row, col]
            if value == src.nodata:
                continue
            x, y = transform * (col, row)
            x_max, y_max = transform * (col + 1, row + 1)
            pixel_geom = box(x, y_max, x_max, y)  # Note y-axis is flipped
            polygons.append(pixel_geom)
            values.append(value)

    gdf_pixels = gpd.GeoDataFrame({'value': values, 'geometry': polygons}, crs=crs)

# --- Save pixel grid to GeoPackage ---
gdf_pixels.to_file(output_geopackage_path, driver="GPKG")

print(f"GeoPackage saved to: {output_geopackage_path}")


# --- Plot raster and overlay pixel grid ---
fig, ax = plt.subplots(figsize=(12, 10))

with rasterio.open(reprojected_raster_path) as src:
    show(src, ax=ax, title="Reprojected Landsat Raster with Pixel Grid", cmap='gray')

gdf_pixels.boundary.plot(ax=ax, edgecolor='red', linewidth=0.5)

plt.show()