import arcpy
import os
from arcpy import env

path = r"C:/Users/Tim/Desktop/BOKU/GIS/GISproject/landsat/"

files = os.listdir(path + "landsat8/")
files = [file for file in files if file.endswith(".tif")]
sr_in = arcpy.SpatialReference(32632)
sr_out = arcpy.SpatialReference(31463)

for file in files:
    try:     
        arcpy.ProjectRaster_management(path + "landsat8/" + file, 
                                       path + "landsat_8new/" + file,
                                       sr_out, "BILINEAR", "30 30", 
                                       "DHDN_To_WGS_1984_3x", "", sr_in)
    except:
        print("problem")