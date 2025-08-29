import os
import geopandas as gpd
import matplotlib.pyplot as plt
import rasterio
import matplotlib.cm as cm
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
import matplotlib.colors as colors
from matplotlib_scalebar.scalebar import ScaleBar  # <-- import scale bar
from rasterio.features import shapes
from shapely.geometry import shape
from rasterio.warp import calculate_default_transform, reproject, Resampling
import numpy as np


#-------------------------------------------------------------------
#----------------------------- FUNCTIONS ---------------------------
#-------------------------------------------------------------------
def reproject_satellite_raster(satellite_path):
    """
    Crops a raster to a bounding box centered at a given lat/lon coordinate.

    Inputs:
        satellite_path (str): Path to the input satellite raster file.

    Returns:
        gdf_landsat (Dataframe): Reprojected (espg 3604) satellite reflectance.
    """
    with rasterio.open(satellite_path) as src:
        src_crs = src.crs
        dst_crs = "EPSG:3604"
        src_transform = src.transform
        src_nodata = src.nodata
        src_dtype = src.dtypes[0]

        # Calculate destination transform and shape
        dst_transform, dst_width, dst_height = calculate_default_transform(
            src_crs, dst_crs, src.width, src.height, *src.bounds
        )

        # Prepare array for reprojected raster
        reprojected_image = np.empty((dst_height, dst_width), dtype=src_dtype)

        # Reproject the first band
        reproject(
            source=src.read(1),
            destination=reprojected_image,
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.nearest
        )

    # --- Create mask to ignore nodata values ---
    mask = reprojected_image != src_nodata if src_nodata is not None else np.ones_like(reprojected_image, dtype=bool)

    # --- Polygonize the reprojected raster ---
    results = (
        {"geometry": shape(geom), "value": value / 1000}
        for geom, value in shapes(reprojected_image, mask=mask, transform=dst_transform)
    )

    # --- Convert to GeoDataFrame ---
    gdf_landsat = gpd.GeoDataFrame.from_records(results)
    gdf_landsat.set_geometry('geometry', inplace=True)
    gdf_landsat.set_crs(dst_crs, inplace=True)
    
    return gdf_landsat

def plot_satellite_raster(gdf_landsat_cropped, gdf_pixels):
    """
    Crops a raster to a bounding box centered at a given lat/lon coordinate.

    Inputs:
        gdf_landsat_cropped (Dataframe): Landsat dataframe cropped to UAV flight area
        gdf_pixels (Dataframe): Pixel grid dataframe cropped to UAV flight area

    """
    cmap = cm.viridis
    norm = colors.Normalize(vmin=8, vmax=13)

    fig, ax = plt.subplots(figsize=(10, 8))

    landsat_plot = gdf_landsat_cropped.plot(
        column='value_1',
        cmap=cmap,
        norm=norm,
        legend=False,
        edgecolor='black',
        linewidth=0.5,
        alpha=0.8,
        ax=ax
    )

    # COLORBAR
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm._A = []  # required workaround for older versions of matplotlib
    cbar = fig.colorbar(sm, ax=ax, fraction=0.05, pad=0.04, aspect=20)
    cbar.set_label("Reflectance (%)", fontsize=30)
    cbar.ax.tick_params(labelsize=20)
    cbar.locator = MaxNLocator(nbins=6)  # Set to 5 tick marks
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%2.0f'))


    # Plot cropped pixel grid overlay
    gdf_pixels.plot(ax=landsat_plot, facecolor='none', edgecolor='red', linewidth=3, label='≥50% Overlap Grid')



    scalebar = ScaleBar(
        dx=1,       # Each unit is 1 foot, so dx = feet per meter
        units="m",               # Display scale bar in meters
        location='lower left',
        scale_loc='bottom',
        box_alpha=0.7,
        color='black',
        fixed_value=100,         # Show a label for 100 m
        fixed_units='m',
        font_properties={'size': 20}
    )

    ax.add_artist(scalebar)
     

    # --- Add compass rose (north arrow) ---
    # Customize location and size here
    x_arrow, y_arrow = 0.95, 0.1  # Relative location in axes coordinates

    ax.annotate(
        'N', xy=(x_arrow, y_arrow + 0.05), xytext=(x_arrow, y_arrow),
        arrowprops=dict(facecolor='black', width=5, headwidth=15),
        ha='center', va='center', fontsize=20,
        xycoords='axes fraction'
    )  
        
    # Formatting
    ax.set_axis_off()
    plt.tight_layout()
    plt.show()
    
 

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

satellite_path = os.path.join(base_dir, "Data\Processed", "croppedLandsat4326.tif")
pixel_grid_path = os.path.join(base_dir, "Data\Processed", "pixelGrid_cropped_to_UAV_Area_3604.gpkg")

# Read in geopackage pixel grid 
gdf_pixels = gpd.read_file(pixel_grid_path)

# #----- READ AND REPROJECT SATELLITE RASTER
gdf_landsat = reproject_satellite_raster(satellite_path)

#----- CROP THE SATELLITE GEODATAFRAME TO THE UAV FLIGHT AREA
gdf_landsat_cropped = gpd.overlay(gdf_landsat, gdf_pixels, how='intersection')

#----- SAVE THE SATELLITE GEODATAFRAME
gdf_landsat_cropped.to_file(os.path.join(base_dir, "Data\Processed", "satelliteReflectance.gpkg"), driver="GPKG")    


#----- PLOT THE SATELLITE SURFACE REFLECTANCE AND PIXEL GRID FOR UAV FLIGHT AREA
plot_satellite_raster(gdf_landsat_cropped, gdf_pixels)
