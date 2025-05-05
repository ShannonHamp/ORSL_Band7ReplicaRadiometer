from rasterio.warp import calculate_default_transform, reproject, Resampling
import rasterio
import numpy as np
from scipy import ndimage

# File paths
elevation_raster_path = r"D:\NASA_EPSCoR_Snow\Band_7_Sensor\Data\Processing_2025_0425\Data\Lidar\20250305_Namaste_YS_dsm.tif"  # EPSG:4326
slope_raster_path = r"D:\NASA_EPSCoR_Snow\Band_7_Sensor\Data\Processing_2025_0425\Results\slopeAngle_1.tif"  # EPSG:3604
target_crs = "EPSG:3604"

# with rasterio.open(elevation_raster_path) as src:
#     src_data = src.read(1)
#     src_transform = src.transform
#     src_crs = src.crs
#     nodata_value = src.nodata

#     # Calculate transform for the new CRS
#     transform, width, height = calculate_default_transform(
#         src_crs, target_crs, src.width, src.height, *src.bounds
#     )

#     # Allocate destination array
#     elevation_data = np.empty((height, width), dtype=np.float32)

#     # Perform reprojection
#     reproject(
#         source=src_data,
#         destination=elevation_data,
#         src_transform=src_transform,
#         src_crs=src_crs,
#         dst_transform=transform,
#         dst_crs=target_crs,
#         resampling=Resampling.bilinear
#     )

# # Mask nodata
# mask = elevation_data != nodata_value if nodata_value is not None else ~np.isnan(elevation_data)

# # Compute slope
# gradient_x, gradient_y = np.gradient(elevation_data, axis=(1, 0))
# slope_radians = np.arctan(np.sqrt(gradient_x**2 + gradient_y**2))
# slope_degrees = np.degrees(slope_radians)

with rasterio.open(elevation_raster_path) as src:
    # Read the elevation data
    elevation_data = src.read(1)  # assuming single-band raster
    # Get the spatial resolution of the raster
    x_res, y_res = src.res[0], src.res[1]

# Calculate the gradient (slope) in the x and y directions
dx, dy = np.gradient(elevation_data, x_res, y_res)

# Calculate the slope angle in radians
slope_angle = np.arctan(np.sqrt(dx**2 + dy**2))

# Convert to degrees
slope_angle_degrees = np.degrees(slope_angle)



# # Apply mask
# slope_angle_degrees[~mask] = nodata_value if nodata_value is not None else -9999

# # Save to new GeoTIFF
# with rasterio.open(
#     slope_raster_path,
#     'w',
#     driver='GTiff',
#     height=slope_degrees.shape[0],
#     width=slope_degrees.shape[1],
#     count=1,
#     dtype=slope_degrees.dtype,
#     crs=target_crs,
#     transform=transform,
#     nodata=nodata_value if nodata_value is not None else -9999
# ) as dst:
#     dst.write(slope_degrees, 1)
# Create a new raster file to save the slope angle data
with rasterio.open(slope_raster_path, 'w', driver='GTiff', 
                   count=1, dtype='float32', crs=src.crs, 
                   transform=src.transform, width=src.width, height=src.height) as dst:
    dst.write(slope_angle_degrees, 1)  # Write the slope angle to the first band
    
print(f"Slope raster has been saved to {slope_raster_path}")
