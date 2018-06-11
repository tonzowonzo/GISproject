# Import libraries.
import os
import cv2
import PIL
from PIL import Image, ImageStat, ImageFilter
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy
from scipy import ndimage

# Matplotlib defaults
#plt.axis("off")

# Import example images.
image1 = "C:/Users/Tim/Desktop/GIS/GISproject/landsat/Bulk Order 914722/Landsat 8 OLI_TIRS C1 Level-1/LC08_L1TP_194026_20170828_20170914_01_T1.tif"
image2 = "C:/Users/Tim/Desktop/GIS/GISproject/landsat/Bulk Order 914722/Landsat 8 OLI_TIRS C1 Level-1/LC08_L1TP_194026_20171015_20171024_01_T1.tif"
# Load in tif image.
def preprocess_image(filename, prepare_for="time_series", sharpen=True):
    '''
    Preprocess the image in preparation for use in classification algorithms.
    
    prepare_for: the type of analysis the data will be prepared for, values that
    can be taken include: time_series, stationary, multiple_zones
    '''
    # Load the image in rgb format.
    img = Image.open(filename).convert("RGB")
    # Turn the image into a numpy array for manipulation.
    img_array = np.array(img)
    print(img_array.shape)
    # Display Color bands.
    # Red.
    plt.imshow(img_array[:, :, 0], cmap="binary")
    plt.title("Red")
    plt.colorbar()
    plt.show()
    # Green.
    plt.imshow(img_array[:, :, 1], cmap="binary")
    plt.title("Green")
    plt.colorbar()
    plt.show()
    # Blue.
    plt.imshow(img_array[:, :, 2], cmap="binary")
    plt.title("Blue")
    plt.colorbar()
    plt.show()
    
    # Compare green matrixes of two images.
#    if filename2 is not None:
#        img2 = Image.open(filename2).convert("RGB")
#        img_array2 = np.array(img2)
#        difference = abs(img_array[:, :, 1] - img_array2[:, :, 1])
#        plt.imshow(difference, cmap="binary")
#        plt.title("Difference in greeness between two images")
#        plt.colorbar()
#        plt.show()
        
    # Display rgb composite.
    plt.imshow(img_array)
    plt.show()
    
    # Sharpen Image.
    if sharpen:
        blurred_img = ndimage.gaussian_filter(img, 3)
        
        filter_blurred_img = ndimage.gaussian_filter(blurred_img, 1)
        
        alpha = 30
        sharpened = blurred_img + alpha * (blurred_img - filter_blurred_img)
        print(sharpened.shape)
        plt.imshow(sharpened)
        plt.title("Sharpened")
        plt.show()
    

    # Atmospheric correction (dark object subtraction).
    # Only required for time series, or multiple row data.
    # Find the brightest pixels
    # Convert to greyscale to calculate brightness
    if prepare_for in ["time_series", "multiple_zones"]:
        luminosity = img.convert("L")
        luminosity = np.array(luminosity)
        # Find the top brightest value and indice in the array.
        max_indices = np.unravel_index(luminosity.argmax(), luminosity.shape)
        print(max_indices)
    







# Sun angle correction.





# Greyscale conversion.

img_array = preprocess_image(image1)


def load_images(path):
    '''
    Loads all of the images into an array.
    '''
    images = pd.DataFrame(index=[x for x in range(100)], columns=["dates", "image_arrays"])
    i = 0

    for file in os.listdir(path):
        if file.endswith("T1.tif"):
            # Load in image array.
            img = Image.open(os.path.join(path, file)).convert("RGB")
            img_array = np.array(img)
            # Get the dates of each image.
            split_file = file.split("_")
            # Add the data to a dataframe.
            images.dates[i] = split_file[3]
            images.image_arrays[i] = img_array
            i+=1
    return images
df = load_images("C:/Users/Tim/Desktop/GIS/GISproject/landsat/Bulk Order 914722/Landsat 8 OLI_TIRS C1 Level-1")
