'''

'''

# Import libraries.
import cv2
import PIL
from PIL import Image, ImageStat
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy

# Matplotlib defaults
#plt.axis("off")

# Import example images.
image1 = "C:/Users/Tim/Desktop/GIS/GISproject/landsat/Bulk Order 914722/Landsat 8 OLI_TIRS C1 Level-1/LC08_L1TP_194026_20170828_20170914_01_T1.tif"
image2 = "C:/Users/Tim/Desktop/GIS/GISproject/landsat/Bulk Order 914722/Landsat 8 OLI_TIRS C1 Level-1/LC08_L1TP_194026_20171015_20171024_01_T1.tif"
# Load in tif image.
def preprocess_image(filename, filename2=None):
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
    if filename2 is not None:
        img2 = Image.open(filename2).convert("RGB")
        img_array2 = np.array(img2)
        difference = abs(img_array[:, :, 1] - img_array2[:, :, 1])
        plt.imshow(difference, cmap="binary")
        plt.title("Difference in greeness between two images")
        plt.colorbar()
        plt.show()
        
    # Display rgb composite.
    plt.imshow(img_array)
    plt.show()


    
    # Atmospheric correction (dark object subtraction).
    # Find the brightest pixels
    # Convert to greyscale to calculate brightness

    luminosity = img.convert("L")
    luminosity = np.array(luminosity)
#    plt.imshow(luminosity)
    # Find the top brightest value and indice in the array.
    max_indices = np.unravel_index(luminosity.argmax(), luminosity.shape)
    print(max_indices)
    









# Sun angle correction.





# Greyscale conversion.

img_array = preprocess_image(image1, image2)
