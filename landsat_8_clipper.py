# Clips all 11 bands of each landsat image for each field area.
# Import libraries.
import os
import arcpy

# Define rootdir.
rootdir = r"C:/Users/Tim/Desktop/GIS/GISproject/"
# Define variables to iterate over.
clip_shape = ["EC1", "EC2", "EC3", "EC4", "EC5", "EC6"]
band_locations = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11"]
bands = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9", "B10", "B11"]

# Define function for clipping the image.
def clip_image(clip_features, in_features, output_location):
    arcpy.Clip_management(in_features, "#", output_location, clip_features,
                          "0", "ClippingGeometry",
                          maintain_clipping_extent=True)
    
for shape in clip_shape:
    if clip_shape in ["EC1", "EC2", "EC3"]:
        # Define the path/row that is used.
        path_row = "195026"
    else:
        path_row = "194026"
        
    # Loop over landsat image files.
    for file in os.listdir(rootdir + "landsat//landsat_8"):
        if file.endswith(".TIF"):
            split_file = file.split(".")
            split_file = split_file[0]
            split_file = split_file.split("_")
            if path_row == split_file[2]:
                for i, band in enumerate(bands):
                    i = str(i + 1)
                    if split_file[3] + ".TIF" in os.listdir(rootdir + "landsat_8//" + shape + "//" + i):
                        continue
                    elif split_file[-1] == band:
                        clip_image(rootdir + shape + ".shp",
                                   rootdir + "landsat//landsat_8//" + file,
                                   rootdir + "landsat_8//" + shape + "//" + i + "//" +
                                   split_file[3] + ".TIF")
            