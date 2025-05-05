# -*- coding: utf-8 -*-
"""
Created on Fri Apr 25 10:13:18 2025

@author: ORSL
"""

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



landsat_path = r"D:\NASA_EPSCoR_Snow\Band_7_Sensor\Data\Processing_2025_0425\Results\croppedLandsat3604.tif"  # EPSG:3604
geopackage_path = r"D:\NASA_EPSCoR_Snow\Band_7_Sensor\Data\Processing_2025_0425\Results\pixelGrid_cropped_to_UAV_Area_3604.gpkg"  # Pixel grid in GPKG format


# Read in geopackage pixelg grid 
gdf_pixels = gpd.read_file(geopackage_path)

# --- Read the raster ---
with rasterio.open(landsat_path) as src:
    image = src.read(1)  # Read the first band
    mask = image != src.nodata  # Mask out nodata values
    crs = src.crs

    # --- Extract shapes (polygonize) ---
    results = (
        {"geometry": shape(geom), "value": value/1000}
        for geom, value in shapes(image, mask=mask, transform=src.transform)
    )

    # --- Convert to GeoDataFrame ---
    gdf_landsat = gpd.GeoDataFrame.from_records(results)
    gdf_landsat.set_crs(crs, inplace=True)
    

# Reproject
target_crs = "EPSG:3604"  # Use EPSG:3604 which is in meters
gdf_pixels = gdf_pixels.to_crs(target_crs)
gdf_landsat = gdf_landsat.to_crs(target_crs)

# Plot both to see if they are aligned
cmap = cm.viridis
norm = colors.Normalize(vmin=8, vmax=13)

fig, ax = plt.subplots(figsize=(10, 8))

landsat_plot = gdf_landsat.plot(
    column='value',
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
    dx=1, 
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



# Crop the Landsat geopackage
gdf_landsat_cropped = gpd.overlay(gdf_landsat, gdf_pixels, how='intersection')



# Plot both to see if they are aligned
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