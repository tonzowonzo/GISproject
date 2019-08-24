# Clips all 11 bands of each landsat image for each field area.
# Import libraries.
import os
import arcpy

# Define rootdir.
rootdir = r"C:/Users/Tim/Desktop/BOKU/GIS/GISproject/"
# Define variables to iterate over.
clip_shape = [str(i+1) for i in range(20)]
#clip_shape = ["EC1", "EC2", "EC3", "1_1", "2", "3", "4", "5", "6", 
 #             "8", "9", "11", "13", "15", "17"]
band_locations = ["1", "2", "3", "4", "5", "6", "7", "8"]
bands = ["band1", "band2", "band3", "band4", "band5", "band6", "band7", "qa"]

# xy tolerance
xy_tolerance = ""

# Define function for clipping the image.
def clip_image(clip_features, in_features, output_location):
    arcpy.Clip_management(in_features, "#", output_location, clip_features,
                          "0", "ClippingGeometry",
                          maintain_clipping_extent=False
                          )
    
for shape in clip_shape:
    if shape in clip_shape:
        # Define the path/row that is used.
        path_row = "194026"
    else:
        path_row = "195026"

#    for file in os.listdir(rootdir + "landsat//landsat5_new"):
#            for i, band in enumerate(bands):
#                if file.endswith(".tif"):
#                    split_file = file.split(".")
#                    split_file = split_file[0]
#                    split_file = split_file.split("_")
#                    if path_row == split_file[2]:
#                        for i, band in enumerate(bands):
#                            num = str(i + 1)
#                            print(band)
#                            print(rootdir + shape + ".shp",
#                                       rootdir + "landsat//landsat5_new//" + file,
#                                       rootdir + "landsat_5new//" + shape + "//" + num + "//" +
#                                       split_file[3] + ".tif")
    file_list = [file for file in os.listdir(rootdir + "landsat//2019") for band in bands if band in file]                 
    # Loop over landsat image files.
    for file in file_list:
        if file.endswith(".tif"):
            split_file = file.split(".")
            split_file = split_file[0]
            split_file = split_file.split("_")
            if path_row == split_file[2]:
                for i, band in enumerate(bands):
                    if band == "qa" and split_file[-2] == "pixel":
                        band = "pixel_qa"
                        
                    if band == "pixel_qa":
                        split_file[-1] = "pixel_qa"
                    num = str(i + 1)
                    if split_file[3] + ".tif" in os.listdir(rootdir + "2019//" + shape + "//" + num):
                        continue
                    elif split_file[-1] == band:
                        if os.path.exists(rootdir + "/2019_shapes/" + shape + ".shp"):
                            clip_image(rootdir + "/2019_shapes/" + shape + ".shp",
                                       rootdir + "landsat//2019//" + file,
                                       rootdir + "2019//" + shape + "//" + num + "//" +
                                       split_file[3] + ".tif")
                        else:
                            continue
            