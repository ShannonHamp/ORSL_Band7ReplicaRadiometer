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

# ------------------- FILE PATHS -------------------
geopackage_path = r"D:\NASA_EPSCoR_Snow\Band_7_Sensor\Data\Processing_2025_0425\Results\pixelGrid_cropped_to_UAV_Area_3604.gpkg"
radiometer_path = r"D:\NASA_EPSCoR_Snow\Band_7_Sensor\Data\Processing_2025_0425\Data\Radiometer\processed_reflectance_2025_0407_flight1.txt"
landsat_path = r"D:\NASA_EPSCoR_Snow\Band_7_Sensor\Data\Processing_2025_0425\Results\croppedLandsat3604.tif"

target_crs = "EPSG:3604"
radius_m = 7.19235  # 15m diameter / 2

# ------------------- COLOR MAP SETTINGS -------------------
cmap = cm.viridis
norm = colors.Normalize(vmin=8, vmax=13)

# ------------------- READ PIXEL GRID -------------------
pixel_grid = gpd.read_file(geopackage_path).to_crs(target_crs)

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

# ------------------- SHARED EXTENT -------------------
combined_bounds = gdf_circles.total_bounds  # [minx, miny, maxx, maxy]

# ------------------- PLOT 1: Radiometer Circles -------------------
fig1, ax1 = plt.subplots(figsize=(10, 8))
gdf_circles.plot(column='value', cmap=cmap, norm=norm, edgecolor='black', linewidth=0.5, alpha=0.8, ax=ax1)
#pixel_grid.plot(ax=ax1, facecolor='none', edgecolor='red', linewidth=3)

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
cbar1.set_label("Reflectance (%)", fontsize=30)
cbar1.ax.tick_params(labelsize=20)
cbar1.locator = MaxNLocator(nbins=6)
cbar1.ax.yaxis.set_major_formatter(FormatStrFormatter('%2.0f'))

ax1.set_axis_off()
plt.tight_layout()

# ------------------- PLOT 2: Landsat Pixels -------------------
fig2, ax2 = plt.subplots(figsize=(10, 8))
gdf_landsat_cropped.plot(column='value_1', cmap=cmap, norm=norm, edgecolor='black', linewidth=0.5, alpha=0.8, ax=ax2)
pixel_grid.plot(ax=ax2, facecolor='none', edgecolor='red', linewidth=3)

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
cbar2.set_label("Reflectance (%)", fontsize=30)
cbar2.ax.tick_params(labelsize=20)
cbar2.locator = MaxNLocator(nbins=6)
cbar2.ax.yaxis.set_major_formatter(FormatStrFormatter('%2.0f'))

ax2.set_axis_off()
plt.tight_layout()
plt.show()
