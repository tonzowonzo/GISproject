# Function for calculating land surface temperature of a system.
# Import libraries.
import os
import math
import cv2
from skimage.measure import block_reduce
import numpy as np
import matplotlib.pyplot as plt

# Path of rgb images
path = r"C:/Users/Tim/Desktop/GIS/GISproject/landsat/Bulk Order 943029/Landsat 8 OLI_TIRS C1 Level-1/"
band_10 = "LC08_L1TP_195025_20180806_20180815_01_T1_B10.tif"
band_11 = "LC08_L1TP_195025_20180806_20180815_01_T1_B11.tif"

def calculate_LST(band_10, band_11, mult_10, add_10, mult_11, add_11, k1_10, k2_10, k1_11, k2_11):
    '''
    Takes in band 10 and band 11 from landsat 8 images and approximates land
    surface temperature. Also requires the values of RADIANCE_MULT_BAND_10, 11 and
    RADIANCE_ADD_BAND_10, 11 from the metadata. Lastly, the K1 and K2_CONSTANT_BAND_10
    and 11 are required
    '''
    # Pool image down to a smaller size.
    def pool_image(arr, kernel_size=(24, 24, 1)):
        return block_reduce(arr, kernel_size, np.max)
        
    # Read in the images.
    TIR_10 = cv2.imread(path + band_10)
    TIR_11 = cv2.imread(path + band_11)
    
    TIR_10 = pool_image(TIR_10)
    TIR_11 = pool_image(TIR_11)
    
    # Display TIR images.
    plt.imshow(TIR_10)
    plt.show()
    
    plt.imshow(TIR_11)
    plt.show()
    
    # Calculate top of atmosphere reflectance.
    TOA_10 = np.zeros((TIR_10.shape[0], TIR_10.shape[1]))
    TOA_11 = np.zeros((TIR_11.shape[0], TIR_11.shape[1]))
    for i, col in enumerate(TIR_10[:, :, 0]):
        for j, TIR_val in enumerate(col):
            TOA_10[i][j] = mult_10 * TIR_val + add_10
     
    for i, col in enumerate(TIR_11[:, :, 0]):
        for j, TIR_val in enumerate(col):
            TOA_11[i][j] = mult_11 * TIR_val + add_11
    
    # Display TOA images.
    plt.imshow(TOA_10)
    plt.show()
    
    plt.imshow(TOA_11)
    plt.show()
    
    # Calculate land surface temperature in celcius.
    temp10 = np.zeros((TIR_10.shape[0], TIR_10.shape[1]))
    temp11 = np.zeros((TIR_11.shape[0], TIR_11.shape[1]))    
  
    for i, col in enumerate(TOA_10):
        for j, TOA_val in enumerate(col):
            if TOA_10[i][j] == 0:
                temp10[i][j] = 100
            else:
                temp10[i][j] = k2_10/(math.log10(k1_10/TOA_val + 1))
    
    for i, col in enumerate(TOA_11):
        for j, TOA_val in enumerate(col):
            if TOA_11[i][j] == 0:
                temp11[i][j] = 100
            else:
                temp11[i][j] = k2_11/(math.log(k1_11/TOA_val + 1))

    # Convert from kelvin to celcius.
    temp10 = temp10 - 273.15
    temp11 = temp11 - 273.15
    return temp10, temp11
    
    
temp10, temp11 = calculate_LST(band_10, band_11, 3.3420E-04, 0.10000, 3.3420E-04, 0.10000, 774.8853,
              1321.0789, 480.8883, 1201.1442)

plt.imshow(temp10)
plt.colorbar()
plt.show()

plt.imshow(temp11)
plt.colorbar()
plt.show()

