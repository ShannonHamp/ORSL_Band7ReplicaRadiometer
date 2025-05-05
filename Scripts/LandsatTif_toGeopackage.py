# -*- coding: utf-8 -*-
"""
Created on Fri Apr 25 10:18:29 2025

@author: ORSL
"""

import geopandas as gpd
import rasterio
from rasterio.features import shapes
from rasterio.warp import calculate_default_transform, reproject, Resampling
from shapely.geometry import shape
import numpy as np



# File paths
input_raster_path = r"D:\NASA_EPSCoR_Snow\Band_7_Sensor\Data\Processing_2025_0425\Results\croppedLandsat4326.tif"
output_gpkg_path = r"D:\NASA_EPSCoR_Snow\Band_7_Sensor\Data\Processing_2025_0425\Results\LandsatGeopackage3604.gpkg"

output_layer_name = "raster_polygons"

# Target CRS
target_crs = "EPSG:3604"

# Step 1: Open and reproject the raster
with rasterio.open(input_raster_path) as src:
    transform, width, height = calculate_default_transform(
        src.crs, target_crs, src.width, src.height, *src.bounds
    )
    kwargs = src.meta.copy()
    kwargs.update({
        'crs': target_crs,
        'transform': transform,
        'width': width,
        'height': height
    })

    reprojected = np.empty((height, width), dtype=src.meta['dtype'])

    reproject(
        source=rasterio.band(src, 1),
        destination=reprojected,
        src_transform=src.transform,
        src_crs=src.crs,
        dst_transform=transform,
        dst_crs=target_crs,
        resampling=Resampling.nearest
    )

# Step 2: Convert raster to polygons
mask = reprojected != src.nodata if src.nodata is not None else np.ones_like(reprojected, dtype=bool)
shapes_gen = shapes(reprojected, mask=mask, transform=transform)

records = []
for geom, val in shapes_gen:
    records.append({'geometry': shape(geom), 'value': val})

# Step 3: Create GeoDataFrame and save to GeoPackage
gdf = gpd.GeoDataFrame(records, crs=target_crs)
gdf.to_file(output_gpkg_path, layer=output_layer_name, driver="GPKG")

print(f"Saved {len(gdf)} polygons to {output_gpkg_path}")