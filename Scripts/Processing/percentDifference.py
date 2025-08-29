import os
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
import matplotlib.colors as colors
from matplotlib_scalebar.scalebar import ScaleBar  # <-- import scale bar





#-------------------------------------------------------------------
#-------------------------------- MAIN -----------------------------
#-------------------------------------------------------------------
# File paths
# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
# Navigate up one directory level
base_dir = os.path.abspath(os.path.join(script_dir, "..", ".."))
# Construct full path to data file
satellite_path = os.path.join(base_dir, "Data\Processed", "satelliteReflectance.gpkg")
pixelGrid_path = os.path.join(base_dir, "Data\Processed", "pixelGrid_cropped_to_UAV_Area_3604.gpkg")
radiometer_path = os.path.join(base_dir, "Data\Processed", "radiometerReflectance.gpkg")

# Read in dataframes
gdf_pixels = gpd.read_file(pixelGrid_path)
gdf_satellite = gpd.read_file(satellite_path)
gdf_radiometer = gpd.read_file(radiometer_path)



#----------------------------------------------------------------------------------------
#--------------------------CALCUATE PERCENT DIFFERENCE-----------------------------------
# --- Step 1: Add index to circles ---
gdf_radiometer = gdf_radiometer.reset_index(drop=True)
gdf_radiometer['circle_index'] = gdf_radiometer.index

# --- Step 2: Spatial join to get overlapping Landsat pixel values for each circle ---
joined = gpd.sjoin(
    gdf_satellite[['geometry', 'value_1']],  # Landsat pixels with value
    gdf_radiometer[['circle_index', 'geometry']],  # Circles with index
    how='inner',
    predicate='intersects'
)

# --- Step 3: Group by circle_index and average the Landsat values ---
avg_landsat = joined.groupby('circle_index')['value_1'].mean().reset_index()
avg_landsat.rename(columns={'value_1': 'avg_landsat_value'}, inplace=True)

# --- Step 4: Merge back into original gdf_circles ---
gdf_radiometer = gdf_radiometer.merge(avg_landsat, on='circle_index', how='left')

# --- Step 5: Compute percent difference ---
gdf_radiometer['percent_difference'] = (
    (gdf_radiometer['value'] - gdf_radiometer['avg_landsat_value']) /
    ((gdf_radiometer['value'] + gdf_radiometer['avg_landsat_value']) / 2)
) * 100


#----------------------------------------------------------------------------------------
#------------------------------------STATISTICS------------------------------------------------
mean_diff = np.mean(gdf_radiometer['percent_difference']) 
max_diff = np.max(gdf_radiometer['percent_difference']) 
min_diff = np.min(gdf_radiometer['percent_difference']) 

#----------------------------------------------------------------------------------------
#------------------------------------PLOT------------------------------------------------
# --- Plot the standard deviation with colorbar and overlay cropped grid ---
cmap = cm.RdYlBu
norm = colors.Normalize(vmin=gdf_radiometer['percent_difference'].min(), vmax=gdf_radiometer['percent_difference'].max())

fig, ax = plt.subplots(figsize=(10, 8))

# Plot circles, colored by 'value'
percent_diff_plot = gdf_radiometer.plot(
    column='percent_difference',
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
cbar.set_label("Percent Difference (%)", fontsize=30)
cbar.ax.tick_params(labelsize=20)
cbar.locator = MaxNLocator(nbins=6)  # Set to 5 tick marks
cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%2.0f'))


# Plot cropped pixel grid overlay
gdf_pixels.plot(ax=percent_diff_plot, facecolor='none', edgecolor='black', linewidth=3, label='≥50% Overlap Grid')



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
