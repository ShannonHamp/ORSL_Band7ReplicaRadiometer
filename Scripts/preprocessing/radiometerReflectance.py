# -*- coding: utf-8 -*-
"""
Created on Tue Aug 12 10:57:09 2025

@author: ORSL
"""

import os
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.cm as cm
from matplotlib.ticker import MaxNLocator
import matplotlib.colors as colors
from matplotlib_scalebar.scalebar import ScaleBar
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
import numpy as np
import re

#-------------------------------------------------------------------
#----------------------------- FUNCTIONS ---------------------------
#-------------------------------------------------------------------
def apply_temperature_calibration(rawRadiometer):
    """
    Applies temperature calibration to raw upwelling and downwelling sensor measurements
    using the temperature measurements (in °C) to calculate the difference w/r/t 
    room temperature (21°C, the temperature at which the rad cal was completed)

    Inputs:
        rawRadiometer (pd.DataFrame): DataFrame containing raw signal measurements
        (DN) from upwelling and downwelling sensor, as well as temp measurements:
            ' Upwelling Temp (°C)', ' Downwelling Temp (°C)',
            ' Upwelling Signal (DN)', ' Downwelling Signal (DN)'

    Returns:
        tuple: (upSig_TempAdjusted, downSig_TempAdjusted)
               Temperature-corrected radiometer measurements (DN) for upwelling and downwelling
    """

    room_temp = 21
    
    # Calculate temperature adjustments
    tempAdjust_upwelling = -(room_temp - rawRadiometer[' Upwelling Temp (°C)'])
    tempAdjust_downwelling = -(room_temp - rawRadiometer[' Downwelling Temp (°C)'])

    #Percent change in upwelling signal due to deviation from room temp (from second-order fit)
    upPercentChange = -0.0121 * (tempAdjust_upwelling ** 2) - 0.7995 * tempAdjust_upwelling + 0.3124
    
    #Percent change in downwelling signal due to deviation from room temp (from second-order fit)
    downPercentChange = -0.0143 * (tempAdjust_downwelling ** 2) - 3.1585 * tempAdjust_downwelling + 1.4461

    # Raw radiometer signals temperature adjusted for application of room temp radiometric calibration
    upSig_TempAdjusted = rawRadiometer[' Upwelling Signal (DN)'] / (1 + (upPercentChange / 100))
    downSig_TempAdjusted = rawRadiometer[' Downwelling Signal (DN)'] / (1 + (downPercentChange / 100))

    return upSig_TempAdjusted, downSig_TempAdjusted


def apply_upwelling_radiometric_calibration(upSig_TempAdjusted, gain):
    """
    Applies radiometric calibration to temperature-corrected upwelling measurements

    Inputs:
        upSig_TempAdjusted: Series containing temperature-corrected upwelling measurements (DN)
        gain: Radiometer amplifier gain setting, set by user to be "LOW", "MED", or "HIGH"

    Returns:
        Series: radiance
               Radiance values (W/m^2 sr) 
    """
    match gain:
        case "LOW":
            radiance = (2.5581e-4)*(upSig_TempAdjusted)+(-0.7496) # LOW GAIN (R-squared = 0.9658)
        case "MED":
            radiance = (1.3342e-5)*(upSig_TempAdjusted)+0.0569    # MEDIUM GAIN (R-squared = 0.9989)
        case "HIGH":
            radiance = (1.1056e-6)*(upSig_TempAdjusted)+0.0686      # HIGH GAIN (R-squared = 0.9989)


    return radiance

def apply_downwelling_radiometric_calibration(downSig_TempAdjusted, gain):
    """
    Applies radiometric calibration to temperature-corrected downwelling measurements

    Inputs:
        downSig_TempAdjusted: Series containing temperature-corrected downwelling measurements (DN)
        gain: Radiometer amplifier gain setting, set by user to be "LOW", "MED", or "HIGH"

    Returns:
        Series: irradiance
               Irradiance values (W/m^2) 
    """
    match gain:
        case "LOW":
            irradiance = (7.7076e-4)*(downSig_TempAdjusted)+(-1.8868)    # LOW GAIN (R-squared = 0.9775)
        case "MED":
            irradiance = (4.2280e-5)*(downSig_TempAdjusted)+(0.1038)   # MEDIUM GAIN (R-squared = 0.9984)
        case "HIGH":
            irradiance = (3.6731e-6)*(downSig_TempAdjusted)+(0.1372)   # HIGH GAIN (R-squared = 0.9997)

    return irradiance

def georeference_radiometer_reflectance(rawRadiometer, groundElevation, reflectance, min_height=100, remove_start=214, remove_end=226, 
                        buffer_radius_m=7.19235):
    """
    Georeferences reflectance measurements using latitude and longitude coordinates

    Inputs:
        rawRadiometer (pd.DataFrame): Must contain columns 
            ' Altitude (m)', ' Latitude', ' Longitude (degrees * 10^-7)'.
        groundElevation (float or pd.Series): Ground elevation(s) in meters.
        reflectance (list or pd.Series): Reflectance values matching rawRadiometer rows.
        min_height (float): Minimum AGL (in meters) to select points (default 100 m).
        remove_start (int): Start index in elevationIndices to remove.
        remove_end (int): End index in elevationIndices to remove.
        buffer_radius_m (float): Radius in meters for buffer circles (default 7.19235 m, i.e., 15 m diameter).

    Returns:
        tuple: (gdf_points, gdf_circles)
               gdf_points: GeoDataFrame of selected points.
               gdf_circles: GeoDataFrame of buffer circles around points.
    """

    # Calculate Above Ground Level (AGL)
    AGL = rawRadiometer[' Altitude (m)'] - groundElevation
    elevationIndices = rawRadiometer.index[AGL > min_height].tolist()

    # Remove specified range of indices
    del elevationIndices[remove_start:remove_end]

    # Extract filtered data
    reflectance_atElevation = [reflectance[i] for i in elevationIndices]
    latitude_atElevation = (rawRadiometer.loc[elevationIndices, ' Latitude']) / 1e7
    longitude_atElevation = (rawRadiometer.loc[elevationIndices, ' Longitude (degrees * 10^-7)']) / 1e7

    # Create DataFrame
    processedReflectance = pd.DataFrame({
        'reflectance': reflectance_atElevation,
        'latitude': latitude_atElevation,
        'longitude': longitude_atElevation
    })

    # Ensure numeric
    processedReflectance['value'] = pd.to_numeric(processedReflectance['reflectance'], errors='coerce')

    # Create GeoDataFrame with WGS84 coordinates
    gdf_points = gpd.GeoDataFrame(
        processedReflectance[['value']],
        geometry=[Point(lon, lat) for lon, lat in zip(processedReflectance.longitude, processedReflectance.latitude)],
        crs="EPSG:4326"  # WGS84
    )

    # Project to metric CRS
    gdf_projected = gdf_points.to_crs(epsg=3604)

    # Create buffer circles
    gdf_circles = gdf_projected.copy()
    gdf_circles['geometry'] = gdf_projected.buffer(buffer_radius_m)

    gdf_circles.to_file(os.path.join(base_dir, "Data\Processed", "radiometerReflectance.gpkg"), driver="GPKG")  

    return gdf_circles, reflectance_atElevation, latitude_atElevation, longitude_atElevation


def plot_radiometer_reflectance(gdf_circles):
    cmap = cm.viridis
    norm = colors.Normalize(vmin=8, vmax=13)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot circles, colored by 'value'
    circles_plot = gdf_circles.plot(
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

#-------------------- EXTRACT RAW DATA FROM FILE -------------------
# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
# Navigate up one directory level
base_dir = os.path.abspath(os.path.join(script_dir, "..", ".."))
# Construct full path to data file
filepath_rawRadiometer = os.path.join(base_dir, "Data\Radiometer_UAV", "rawFlightData_2025_0407.txt")


# Read the ground elevation from the file header
with open(filepath_rawRadiometer, 'r') as f:
    first_line = f.readline()
groundElevation = re.findall(r'-?\d+\.?\d*', first_line)
groundElevation = [float(num) for num in groundElevation]

# Read all measurements from the raw radiometer file
rawRadiometer = pd.read_csv(filepath_rawRadiometer, skiprows=1)  #The first row serves as a header and includes the ground elevation

# Convert the dataframe (except for the times) to numbers (ints and floats)
rawRadiometer.iloc[:, 1:] = rawRadiometer.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')


#-------------------- APPLY TEMPERATURE CALIBRATION -------------------
upSig_TempAdjusted, downSig_TempAdjusted = apply_temperature_calibration(rawRadiometer)

#-------------------- APPLY RADIOMETRIC CALIBRATION -------------------
# Upwelling Sensor
radiance = apply_upwelling_radiometric_calibration(upSig_TempAdjusted, "HIGH")

# Downwelling Sensor 
irradiance = apply_downwelling_radiometric_calibration(downSig_TempAdjusted, "LOW")

#-------------------- CALCULATE REFLECTANCE -------------------
reflectance = (radiance/irradiance)*np.pi*100

#-------------------- VALIDATE -------------------
#---- GEOREFERENCE REFLECTANCE MEASUREMENTS
gdf_circles, reflectance_atElevation, latitude_atElevation, longitude_atElevation = georeference_radiometer_reflectance(rawRadiometer, groundElevation, reflectance)


#---- PLOT GEOREFERENCED REFLECTANCE MEASUREMENTS
plot_radiometer_reflectance(gdf_circles)

#---- EMPIRICAL VALIDATION WITH REFLECTANCE TARGET
#Calculate average reflectance of calibration target using ground-based measurements of 11% reflectance tarp
#Indexing must be changed manually based on time stamps corresponding to period when reflectance tarp was measured
tarp_avg_reflectance = np.mean([reflectance[332:341]]) 

#-------------------- SAVE REFLECTANCE DATAFRAME TO FILE -------------------
output_filepath = os.path.join(base_dir, "Data\Processed", "processed_reflectance_2025_0407.txt")

reflectance_toSave = pd.DataFrame({
    'reflectance': reflectance_atElevation,
    'latitude': latitude_atElevation,
    'longitude': longitude_atElevation
})

reflectance_toSave.to_csv(output_filepath, index=False)