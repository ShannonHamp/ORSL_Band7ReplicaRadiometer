import os
import rasterio
from rasterio.mask import mask
from shapely.geometry import box, mapping
import geopandas as gpd
import matplotlib.pyplot as plt
from rasterio.warp import calculate_default_transform, reproject, Resampling
import numpy as np
import pandas as pd


#-------------------------------------------------------------------
#----------------------------- FUNCTIONS ---------------------------
#-------------------------------------------------------------------
def crop_raster_to_bbox(
    raster_path, 
    cropped_raster_output_path,
    center_lat=45.231907, 
    center_lon=-111.476894, 
    lon_offset=0.0025, 
    lat_offset_min=0.001, 
    lat_offset_max=0.0025
):
    """
    Crops a raster to a bounding box centered at a given lat/lon coordinate.

    Inputs:
        raster_path (str): Path to the input raster file.
        cropped_raster_output_path (str): Path to save cropped raster file.
        center_lat (float): Latitude of the bounding box center.
        center_lon (float): Longitude of the bounding box center.
        lon_offset (float): Longitude offset for min/max bounding box extent.
        lat_offset_min (float): Latitude offset for min bound.
        lat_offset_max (float): Latitude offset for max bound.

    Returns:
        out_image (tuple): array of image pixel data.
        out_meta (tuple): metadata corrsponsing to out_image.
    """

    # --- Define bounding box in lat/lon ---
    min_lon = center_lon - lon_offset
    min_lat = center_lat - lat_offset_min
    max_lon = center_lon + lon_offset
    max_lat = center_lat + lat_offset_max

    bbox_geom = box(min_lon, min_lat, max_lon, max_lat)
    gdf_bbox = gpd.GeoDataFrame({'geometry': [bbox_geom]}, crs="EPSG:4326")

    # --- Open raster and reproject bbox to raster CRS ---
    with rasterio.open(raster_path) as src:
        raster_crs = src.crs
        gdf_bbox_proj = gdf_bbox.to_crs(raster_crs)
        bbox_projected_geom = [mapping(gdf_bbox_proj.iloc[0].geometry)]

        # --- Crop the raster using mask ---
        out_image, out_transform = mask(src, bbox_projected_geom, crop=True)
        out_meta = src.meta.copy()

        out_meta.update({
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform,
            "crs": raster_crs
        })

    # --- Save cropped raster if output path is provided ---
    with rasterio.open(cropped_raster_output_path, "w", **out_meta) as dest:
        dest.write(out_image)
    print(f"Cropped raster saved to: {cropped_raster_output_path}")

    # --- Plot the cropped raster ---
    plt.figure(figsize=(10, 8))
    plt.imshow(out_image[0], cmap='gray')
    plt.title("Cropped Raster")
    plt.axis('off')
    plt.show()

    return out_image, out_meta



def raster_to_pixel_grid(input_raster_path, output_gpkg_path, target_crs="EPSG:3604", plot=False):
    """
    Reads a raster file, reprojects it to EPSG:3604, and polygonizes it into a pixel grid.

    Inputs:
        input_raster_path (str): Path to the input raster tif file.
        output_gpkg_path (str): Path to save pixel grid greopackage.
        target_crs (str): CRS to reproject to (default EPSG:3604 for Montana).
        plot (bool): Set to plot resulting pixel grid or not.

    Returns:
        gdf_pixels (GeoDataFrame): Pixel grid reprojected to target_crs.
    """

    # --- Read in Raster and Reproject ---
    with rasterio.open(input_raster_path) as src:
        src_data = src.read(1)  # Single band
        src_transform = src.transform
        src_crs = src.crs
        src_nodata = src.nodata
        src_dtype = src.dtypes[0]

        # Compute transform and shape for target CRS
        transform, width, height = calculate_default_transform(
            src_crs, target_crs, src.width, src.height, *src.bounds
        )

        # Create empty array for reprojected data
        reprojected = np.empty((height, width), dtype=src_dtype)

        # Perform reprojection
        reproject(
            source=src_data,
            destination=reprojected,
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=transform,
            dst_crs=target_crs,
            resampling=Resampling.nearest
        )

    # --- Polygonize into pixel grid ---
    polygons = []
    values = []

    for row in range(height):
        for col in range(width):
            value = reprojected[row, col]
            if src_nodata is not None and value == src_nodata:
                continue

            x_min, y_max = transform * (col, row)
            x_max, y_min = transform * (col + 1, row + 1)
            pixel_geom = box(x_min, y_min, x_max, y_max)

            polygons.append(pixel_geom)
            values.append(value)

    gdf_pixels = gpd.GeoDataFrame({'value': values, 'geometry': polygons}, crs=target_crs)

    # --- Plot ---
    if plot:
        fig, ax = plt.subplots(figsize=(10, 10))
        gdf_pixels.boundary.plot(ax=ax, linewidth=0.5, edgecolor='red')
        ax.set_title("Pixel Grid (Reprojected to EPSG:3604)")
        plt.axis('equal')
        plt.tight_layout()
        plt.show()
        
    gdf_pixels.to_file(output_gpkg_path, driver="GPKG")    

    return gdf_pixels




def crop_pixel_grid_to_UAV_area(geopackage_path, radiometerReflectance_path, cropped_output_gpkg_path, radius_m=7.19235):
    """
    Crops a satellite pixel grid to retain only pixels that intersect ≥50% with radiometer footprints.

    Inputs:
        geopackage_path (str): Path to the original pixel grid GeoPackage.
        radiometerReflectance_path (str): Path to the radiometer txt file with 'latitude' and 'longitude' columns.
        cropped_output_gpkg_path (str): Path to save the cropped pixel grid as geopackage.
        radius_m (float): Buffer radius in meters (default 7.19235 for ~15m diameter footprint).


    Returns:
        gdf_cropped (GeoDataFrame): Pixel grid cropped to area of UAV flight.
    """
    # --- Read pixel grid ---
    gdf_pixels = gpd.read_file(geopackage_path)

    # --- Read and convert radiometer points to GeoDataFrame ---
    df = pd.read_csv(radiometerReflectance_path)
    gdf_points = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df.longitude, df.latitude),
        crs="EPSG:4326"
    )

    # --- Reproject to EPSG:3604 (meters) ---
    target_crs = "EPSG:3604"
    gdf_projected = gdf_points.to_crs(target_crs)
    gdf_circles = gdf_projected.copy()
    gdf_circles['geometry'] = gdf_projected.buffer(radius_m)

    gdf_pixels = gdf_pixels.to_crs(target_crs)
    gdf_circles = gdf_circles.to_crs(target_crs)

    # --- Union all buffered circles ---
    gdf_union = gdf_circles.unary_union  # shapely geometry

    # --- Compute overlap ---
    gdf_pixels["pixel_area"] = gdf_pixels.geometry.area
    gdf_pixels["intersection_area"] = gdf_pixels.geometry.intersection(gdf_union).area

    gdf_cropped = gdf_pixels[gdf_pixels["intersection_area"] >= 0.5 * gdf_pixels["pixel_area"]].copy()

    # Drop temp columns
    gdf_cropped = gdf_cropped.drop(columns=["pixel_area", "intersection_area"], errors="ignore")
    gdf_cropped = gdf_cropped.drop(index=108)

    print(f"Pixels retained after cropping: {len(gdf_cropped)}")
    

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(10, 10))
    gdf_circles.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=2, label="15m Circles")
    gdf_cropped.plot(ax=ax, color='none', edgecolor='red', label="Pixels ≥50% Overlap")
    ax.set_title("Radiometer footprints (black) and satellite pixel grid (red)")
    ax.legend()
    plt.axis('equal')
    plt.show()
    
    gdf_cropped.to_file(cropped_output_gpkg_path, driver="GPKG")    


    return gdf_cropped
#-------------------------------------------------------------------


#-------------------------------------------------------------------
#-------------------------------- MAIN -----------------------------
#-------------------------------------------------------------------
# File paths
# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
# Navigate up one directory level
base_dir = os.path.abspath(os.path.join(script_dir, "..", ".."))
# Construct full path to data file
raster_path = os.path.join(base_dir, "Data\Landsat9", "LC09_CU_009004_20250406_20250413_02_SR_B7.tif")

#-----CROP SATELLITE RASTER FOR FASTER PROCESSING
cropped_raster_path = os.path.join(base_dir, "Data\Processed", "croppedLandsat4326.tif")
croppedLandsat_image, cropped_meta = crop_raster_to_bbox(raster_path, cropped_raster_path)
croppedLandsat_raster = {
    "array": croppedLandsat_image,
    "metadata": cropped_meta
}


#----- CREATE PIXEL GRID (ESPG 3604) FROM CROPPSED SATELLITE RASTER
gdf = raster_to_pixel_grid(
    input_raster_path=cropped_raster_path,
    output_gpkg_path = os.path.join(base_dir, "Data\Processed", "pixel_grid_3604.gpkg"),
    target_crs="EPSG:3604",
    plot=True
)

#----- CROP PIXEL GRID TO REGION OF UAV FLIGHT
cropped_pixel_grid = crop_pixel_grid_to_UAV_area(
    geopackage_path = os.path.join(base_dir, "Data\Processed", "pixel_grid_3604.gpkg"),
    radiometerReflectance_path = os.path.join(base_dir, "Data\Processed", "processed_reflectance_2025_0407.txt"),
    cropped_output_gpkg_path = os.path.join(base_dir, "Data\Processed", "pixelGrid_cropped_to_UAV_Area_3604.gpkg"))