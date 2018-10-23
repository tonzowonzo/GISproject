# Import libraries.
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

# Allow 0 division in numpy.
np.seterr(divide="ignore", invalid="ignore")
# Path to work with.
path = r"C:/Users/Tim/Desktop/GIS/GISproject/"

def create_ndvi_image(landsat=8):
    for i, image in enumerate(os.listdir(path + "landsat/landsat_" + str(landsat) + "_8bit")):
        if image.endswith(".TIF"):
            # Split up the image file name.
            split_img = image.split(".")
            split_img = split_img[0]
            split_img = split_img.split("_")
            date = split_img[3]
            band = split_img[-1]
            if band in ["B4"]:
                image_name = split_img[0:7]
                image_name = "_".join(image_name)
                print(image)
                # Load in red data.
                if landsat == 8:
                    r = cv2.imread(path + "landsat/landsat_8_8bit/" + image_name + "_B4.TIF", 0)
                    # Load in the nir data.
                    nir = cv2.imread(path + "landsat/landsat_8_8bit/" + image_name + "_B5.TIF", 0)
                elif landsat == 7:
                    r = cv2.imread(path + "landsat/landsat_7_8bit/" + image_name + "_B3.TIF", 0)
                    # Load in the nir data.
                    nir = cv2.imread(path + "landsat/landsat_7_8bit/" + image_name + "_B4.TIF", 0)
                elif landsat == 5:
                    r = cv2.imread(path + "landsat/landsat_5_8bit/" + image_name + "_B3.TIF", 0)
                    # Load in the nir data.
                    nir = cv2.imread(path + "landsat/landsat_5_8bit/" + image_name + "_B4.TIF", 0)
                # Turn to floats.
                r = r.astype(float)
                nir = nir.astype(float)
                
                # Initialise ndvi array.
                ndvi = np.empty(r.shape)
                
                # Check if number > 0.
                check = np.logical_or(r > 0, nir > 0)
                
                # Calculate ndvi.
                ndvi = np.where(check, (nir - r) / (nir + r), -1)
                
                # Change nan's to 0.
                ndvi = np.nan_to_num(ndvi)
                
                # Plot the image.
            #    plt.imshow(ndvi, cmap="brg")
                
                # Interpolate the image so it's 0 -> 255 rgb.
                ndvi = np.interp(ndvi, [-1, 1], [0, 255])
                # Save the image.
                cv2.imwrite(path + "landsat/ndvi/" + image_name + ".tif", ndvi)
                print("Iteration {} has now finished and been saved".format(str(i)))
    
create_ndvi_image(landsat=7)
