# Field areas to NDVI.
# Import libraries.
import cv2
import numpy as np
import os

# Field areas.
field_areas = ["1_1", "2", "3", "4", "5", "6", "8", "9", "11", "13", "15", "17",
               "18_1", "18_2", "18_4", "19", "20", "21", "23", "25", "26", "27",
               "28", "29", "30", "33", "EC1", "EC2", "EC3", "EC4", "EC5", "EC6"]

# Path of files.
path = r"C:/Users/Tim/Desktop/GIS/GISproject/"

def create_ndvi(landsat):
    for field_area in field_areas:
        # Band locations for each landsat.
        if landsat == 8:
            r_band = "4"
            nir_band = "5"
        elif landsat == 7 or landsat == 5:
            r_band = "3"
            nir_band = "4"
            
        for image in os.listdir(path + "landsat_" + str(landsat) + "/" + field_area + "/" + r_band):
            if image.endswith(".tif"):
                # Read in the images.
                r = cv2.imread(path + "landsat_" + str(landsat) + "/" + field_area + "/" + r_band + "/" + image, 0)
                nir = cv2.imread(path + "landsat_" + str(landsat) + "/" + field_area + "/" + nir_band + "/" + image, 0)
                
                # Change to float.
                r = r.astype(float)
                nir = nir.astype(float)
                
                # Initialise the ndvi array.
                ndvi = np.empty(r.shape)
                
                # Check if number > 0.
                check = np.logical_or(r > 0, nir > 0)
                
                # Calculate ndvi.
                ndvi = np.where(check, (nir - r) / (nir + r), -1)
                
                # Change nan's to 0.
                ndvi = np.nan_to_num(ndvi)
                
                # Interpolate the image so it's 0 -> 255 rgb.
                ndvi = np.interp(ndvi, [-1, 1], [0, 255])
                # Save the image.
                cv2.imwrite(path + "landsat_" + str(landsat) + "/" + field_area + 
                            "/NDVI/" + image, ndvi)

            
create_ndvi(5)