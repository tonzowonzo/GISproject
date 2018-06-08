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

# Load in tif image.
def preprocess_image(filename):
    # Load the image in rgb format.
    img = Image.open(filename).convert("RGB")
    # Turn the image into a numpy array for manipulation.
    img_array = np.array(img)
    plt.imshow(img_array)
    
    # Atmospheric correction (dark object subtraction).
    # Find the brightest pixels
    # Convert to greyscale to calculate brightness
    luminosity = img.convert("L")
    luminosity = np.array(luminosity)
    plt.imshow(luminosity)
    # Find the top brightest value and indice in the array.
    max_indices = np.unravel_index(luminosity.argmax(), luminosity.shape)
    print(max_indices)
    









# Sun angle correction.





# Greyscale conversion.

img_array = preprocess_image("C:/Users/Tim/Desktop/GIS/Murica/LE07_L1TP_014034_20180607_20180607_01_RT.tif")
