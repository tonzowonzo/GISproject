# Tool for converting landsat 8 bands to 8 bit.
import os
import arcpy

# Path
path = r"C:/Users/Tim/Desktop/GIS/GISproject/"

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
    
    
# Bands we will use.
bands = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9", "B10", "B11"]
# Look through the 16 bit landsat images.
for image in os.listdir(path + "landsat/landsat_8"):
    # Check if the layer to convert is in our wanted bands.
    for band in bands:
        if image.endswith(band + ".TIF"):
            # Copy the raster.
            copy_raster(path + "landsat/landsat_8/" + image, path + "landsat/landsat_8_8bit/" + image)
            