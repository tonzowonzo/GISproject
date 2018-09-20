# Combine bands of landsat 8.
import arcpy
import os

# Define workspace.
rootdir = r"C:/Users/Tim/Desktop/GIS/GISproject/"
output_dir = r"C:/Users/Tim/Desktop/GIS/GISproject/test_data/"
# Define band names to be combined.
band_names = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9", "B10", "B11"]
band_locations = []
complete_locations = []

# Iterate through the files.
for i, file in enumerate(os.listdir(rootdir + "landsat//Bulk Order 945105//Landsat 8 OLI_TIRS C1 Level-1")):
    if file.endswith(".TIF"):
        split_file = file.split("_")
        split_file = split_file[:7]
        joint_file = "_".join(split_file)
        band_locations.append(joint_file)

# Remove duplications
band_locations = set(band_locations)
band_locations = list(band_locations)

# Create location for each band.
for location in band_locations:
    for i, band in enumerate(band_names):
        split_file = location.split("_")
        date = split_file[3]
        complete_locations.append(rootdir + "landsat//Bulk Order 945105//Landsat 8 OLI_TIRS C1 Level-1//" + location + "_" + band + ".TIF")
        
    # Combine the bands.
    complete_locations = ";".join(complete_locations)
    arcpy.CompositeBands_management(complete_locations, output_dir + date + ".TIF")
    
    # Refresh complete locations list for the next one.
    complete_locations = []