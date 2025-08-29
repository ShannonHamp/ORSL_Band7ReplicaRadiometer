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
import rasterio
import numpy as np
from scipy import ndimage



#-------------------------------------------------------------------
#----------------------------- FUNCTIONS ---------------------------
#-------------------------------------------------------------------
def calculate_slope_angle(elevation_path, output_slope_path):
    """
    Calculates slope angle from elevation raster from UAV-based lidar

    Inputs:
        elevation_path (str): File path for elevation raster
        output_slope_path (str): File path to save resulting slope angles

    """
    with rasterio.open(elevation_path) as src:
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

    with rasterio.open(output_slope_path, 'w', driver='GTiff', 
                       count=1, dtype='float32', crs=src.crs, 
                       transform=src.transform, width=src.width, height=src.height) as dst:
        dst.write(slope_angle_degrees, 1)  # Write the slope angle to the first band
    

    
def plot_elevation_slope(gdf_elevation_cropped, gdf_slope_cropped, gdf_circles, gdf_pixels):
    """
    Plots the elevation and slope measurements as well as the satellite pixel grid for the UAV flight area

    Inputs:
        gdf_elevation_cropped (Dataframe): Elevation dataframe cropped to UAV flight area
        gdf_slope_cropped (Dataframe): Slope dataframe cropped to UAV flight area
        gdf_circles (Dataframe): UAV-based radiometer reflectance dataframe
        gdf_pixels (Dataframe): Pixel grid dataframe cropped to UAV flight area

    """
    # ------------------- SHARED EXTENT -------------------
    combined_bounds = gdf_circles.total_bounds  # [minx, miny, maxx, maxy]

    gdf_slope_cropped['value_1'] = gdf_slope_cropped['value_1'].where(gdf_slope_cropped['value_1'] <= 30, np.nan)


    # ------------------- PLOT 1: SLOPE ANGLE -------------------
    # ------------------- COLOR MAP SETTINGS -------------------
    cmap = cm.Purples
    norm = colors.Normalize(vmin=gdf_slope_cropped['value_1'].min(), vmax=gdf_slope_cropped['value_2'].max())
    norm = colors.Normalize(vmin=0, vmax=20)

    fig1, ax1 = plt.subplots(figsize=(10, 8))
    gdf_slope_cropped.plot(column='value_1', cmap=cmap, norm=norm, edgecolor='none', linewidth=0.5, alpha=0.8, ax=ax1)
    gdf_pixels.plot(ax=ax1, facecolor='none', edgecolor='red', linewidth=3)

    # Apply same extent
    ax1.set_xlim(combined_bounds[0], combined_bounds[2])
    ax1.set_ylim(combined_bounds[1], combined_bounds[3])

    # Add scalebar and compass
    scalebar1 = ScaleBar(dx=1, units="m", location='lower left', scale_loc='bottom',
                         box_alpha=0.7, color='black', fixed_value=100, fixed_units='m',
                         font_properties={'size': 20})
    ax1.add_artist(scalebar1)
    ax1.annotate('N', xy=(0.95, 0.15), xytext=(0.95, 0.1),
                 arrowprops=dict(facecolor='black', width=5, headwidth=15),
                 ha='center', va='center', fontsize=20, xycoords='axes fraction')

    # Add colorbar
    sm1 = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm1._A = []
    cbar1 = fig1.colorbar(sm1, ax=ax1, fraction=0.05, pad=0.04, aspect=20)
    cbar1.set_label("Slope Angle (degrees)", fontsize=30)
    cbar1.ax.tick_params(labelsize=20)
    cbar1.locator = MaxNLocator(nbins=6)
    cbar1.ax.yaxis.set_major_formatter(FormatStrFormatter('%2.0f'))

    ax1.set_axis_off()
    plt.tight_layout()

    # # ------------------- PLOT 2: ELEVATION -------------------
    cmap = cm.terrain
    norm = colors.Normalize(vmin=2640, vmax=2680)

    fig2, ax2 = plt.subplots(figsize=(10, 8))
    gdf_elevation_cropped.plot(column='value_1', cmap=cmap, norm=norm, edgecolor='none', linewidth=0.5, alpha=0.8, ax=ax2)
    gdf_pixels.plot(ax=ax2, facecolor='none', edgecolor='red', linewidth=3)

    # Apply same extent
    ax2.set_xlim(combined_bounds[0], combined_bounds[2])
    ax2.set_ylim(combined_bounds[1], combined_bounds[3])

    # Add scalebar and compass
    scalebar2 = ScaleBar(dx=1, units="m", location='lower left', scale_loc='bottom',
                         box_alpha=0.7, color='black', fixed_value=100, fixed_units='m',
                         font_properties={'size': 20})
    ax2.add_artist(scalebar2)
    ax2.annotate('N', xy=(0.95, 0.15), xytext=(0.95, 0.1),
                 arrowprops=dict(facecolor='black', width=5, headwidth=15),
                 ha='center', va='center', fontsize=20, xycoords='axes fraction')

    # Add colorbar
    sm2 = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm2._A = []
    cbar2 = fig2.colorbar(sm2, ax=ax2, fraction=0.05, pad=0.04, aspect=20)
    cbar2.set_label("Elevation (m)", fontsize=30)
    cbar2.ax.tick_params(labelsize=20)
    cbar2.locator = MaxNLocator(nbins=6)
    cbar2.ax.yaxis.set_major_formatter(FormatStrFormatter('%4.0f'))

    ax2.set_axis_off()
    plt.tight_layout()

   

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
elevation_path = os.path.join(base_dir, "Data\Lidar", "YS-20250407-164619_50cm_dsm.tif")
pixelGrid_path = os.path.join(base_dir, "Data\Processed", "pixelGrid_cropped_to_UAV_Area_3604.gpkg")
radiometer_path = os.path.join(base_dir, "Data\Processed", "radiometerReflectance.gpkg")


# Read in geopackage pixel grid 
gdf_pixels = gpd.read_file(pixelGrid_path)


#----- CALCULATE SLOPE ANGLE
output_slope_path = os.path.join(base_dir, "Data\Processed", "slopeAngle.tif")
calculate_slope_angle(elevation_path, output_slope_path)


# --- READ IN THE ELEVATION RASTER
with rasterio.open(elevation_path) as src:
    image = src.read(1)  # Read the first band
    mask = image != src.nodata  # Mask out nodata values
    crs = src.crs

    # --- Extract shapes (polygonize) ---
    results = (
        {"geometry": shape(geom), "value": value}
        for geom, value in shapes(image, mask=mask, transform=src.transform)
    )

    # --- Convert to GeoDataFrame ---
    gdf_elevation = gpd.GeoDataFrame.from_records(results)
    gdf_elevation = gdf_elevation.set_geometry('geometry')
    gdf_elevation.set_crs(crs, inplace=True)
    


# --- READ IN THE SLOPE RASTER
target_crs = "EPSG:3604"
with rasterio.open(output_slope_path) as src:
    image = src.read(1)
    mask = image != src.nodata
    crs = src.crs
    transform = src.transform
    results = (
        {"geometry": shape(geom), "value": value}
        for geom, value in shapes(image, mask=mask, transform=transform)
    )
    gdf_slope = gpd.GeoDataFrame.from_records(results)
    gdf_slope = gdf_slope.set_geometry('geometry')
    gdf_slope.set_crs(crs, inplace=True)
    gdf_slope = gdf_slope.to_crs(target_crs)


# Reproject
target_crs = "EPSG:3604"  # Use EPSG:3604 which is in meters
gdf_pixels = gdf_pixels.to_crs(target_crs)
gdf_elevation = gdf_elevation.to_crs(target_crs)
gdf_slope = gdf_slope.to_crs(target_crs)


#----- CROP THE ELEVATION AND SLOPE TO THE UAV FLIGHT AREA
gdf_elevation_cropped = gpd.overlay(gdf_elevation, gdf_pixels, how='intersection')
gdf_slope_cropped = gpd.overlay(gdf_slope, gdf_pixels, how='intersection')

#----- PLOT THE ELEVATION AND PIXEL GRID, AND SLOPE AND PIXEL GRID FOR UAV FLIGHT AREA
gdf_circles = gpd.read_file(pixelGrid_path)

plot_elevation_slope(gdf_elevation_cropped, gdf_slope_cropped, gdf_circles, gdf_pixels)

#----- SAVE THE ELEVATION AND SLOPE GEODATAFRAMES
gdf_elevation_cropped.to_file(os.path.join(base_dir, "Data\Processed", "lidarElevation.gpkg"), driver="GPKG")  
gdf_slope_cropped.to_file(os.path.join(base_dir, "Data\Processed", "lidarSlope.gpkg"), driver="GPKG")  


