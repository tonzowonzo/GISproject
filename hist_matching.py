# Import libraries.
import matplotlib.pyplot as plt
from skimage import exposure
from  match_histograms import match_histograms
import cv2
import rasterio 
import numpy as np

with rasterio.Env():
    with rasterio.open(r"C:/Users/Tim/Desktop/Wien work/36_N_UM__Orbit_'078'_Period_1.tif") as src:
        meta_data_image = src.profile
        image = src.read(1)
        
    with rasterio.open(r"C:/Users/Tim/Desktop/Wien work/36_N_UN__Orbit_'078'_Period_1.tif") as src2:
        meta_data_reference = src2.profile
        reference = src2.read(1)
    
    with rasterio.open(r"C:/Users/Tim/Desktop/Wien work/36_N_UM__Orbit_'078'_Period_1.tif") as src3:
        meta_data_matched = src3.profile
        matched_ = match_histograms(image, reference, multichannel=False)
        matched = src3.read(1)
        matched[:, :] = matched_
    
    
    
    def display_image(img):
        plt.imshow(img)
        plt.show()
        
    display_image(image)
    display_image(reference)
    display_image(matched)
    
    # Update metadata
    meta_data_image.update(count=1)
    meta_data_reference.update(count=1)
    meta_data_matched.update(count=1)
    
    
    #cv2.imwrite(r"C:/Users/Tim/Desktop/Wien work/matching_hist.tif", matched.astype(np.uint16))
    #cv2.imwrite(r"C:/Users/Tim/Desktop/Wien work/image.tif", image.astype(np.uint16))
    #cv2.imwrite(r"C:/Users/Tim/Desktop/Wien work/reference.tif", reference.astype(np.uint16))
    with rasterio.open(r"C:/Users/Tim/Desktop/Wien work/image.tif", "w", **meta_data_image) as dst:
        dst.write_band(1, image)
        
    with rasterio.open(r"C:/Users/Tim/Desktop/Wien work/reference.tif", "w", **meta_data_reference) as dst:
        dst.write_band(1, reference)
        
    with rasterio.open(r"C:/Users/Tim/Desktop/Wien work/matched.tif", "w", **meta_data_matched) as dst:
        dst.write_band(1, matched)
    