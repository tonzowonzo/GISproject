# Import the libraries
import arcpy
from arcpy import env
import os

# Root directory.
rootdir = r"C:/Users/Tim/Desktop/GIS/GISproject/"

# Landsat row and path.
landsat_path_row = "194026"

# Clip features.
clip_features = rootdir + "EC6.shp"

# xy tolerance
xy_tolerance = ""

# Features to be clipped.
for filename in os.listdir(rootdir + "landsat//2010_to_2018_landsat5-8"):
  split_file = filename.split("_")
  if filename.endswith("T1.tif") and landsat_path_row == split_file[2]:
    in_features = rootdir + "landsat//2010_to_2018_landsat5-8//" + filename
    out_feature_class = r"C:/Users/Tim/Desktop/GIS/GISproject/EC6_CLIP_PRACTICE/" + split_file[3] + ".tif"
    arcpy.Clip_management(in_features, "#", out_feature_class, clip_features, "0", "ClippingGeometry",
    maintain_clipping_extent = True)

      
