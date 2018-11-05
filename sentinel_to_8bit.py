# Changes sentinel images into 8 bit images.
# Import libraries.
import os
import arcpy

# Path to begin.
path = r"C:/Users/Tim/Desktop/GIS/GISproject/"

# Bands we will save.
bands = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B09",
         "B10", "B11", "B12"]

# Copy the raster to 8 bit function.
def copy_raster(in_raster, out_raster):
    '''
    in_raster: the path of the input raster to be changed to 8 bit.
    out_raster: the path of the output raster that has been changed.
    8_BIT_UNSIGNED: Range of pixel values from 0 -> 255.
    ScalePixelValue: Scale the pixels values down from 16 bit to 8 bit.
    '''
    arcpy.CopyRaster_management(in_raster, out_raster, "", "", "0", "", "",
                                "8_BIT_UNSIGNED", "ScalePixelValue", "")
    
# Loop over folders.
for file in os.listdir(path + "landsat/Bulk Order 956415/Sentinel-2/"):
    # Get the file name within the sentinel path that contains the imagery.
    image_location_name = os.listdir(path + "landsat/Bulk Order 956415/Sentinel-2/" + file + "/GRANULE/")[0]
    # Loop over the image file locations.
    for image in os.listdir(path + "landsat/Bulk Order 956415/Sentinel-2/" + file + "/GRANULE/"
                            + image_location_name + "/IMG_DATA"):
        # Check to see if image is in bands.
        for band in bands:
            if image.endswith(band + ".jp2"):
                copy_raster(path + "landsat/Bulk Order 956415/Sentinel-2/" + file + "/GRANULE/"
                            + image_location_name + "/IMG_DATA/" + image,
                            path + "landsat/sentinel_8bit/" + image)
                
        
        
        