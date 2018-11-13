# Import libraries
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.externals import joblib
import pandas as pd

# Set current working directory.
os.chdir(r"C:/Users/Tim/Desktop/GIS/GISproject/")

# Load binary classifier.
binary_classifier = joblib.load("binary_classifier.pkl")

# Function for full classification of an image's crop types.
def crop_classification(ndvi_march, nir_may, nir_july, ndvi_april):
    '''
    Aims to classify a landsat image based on the method from Harfenmeister et
    al.
    
    ndvi_march: input for summer vs winter crop classification.
    nir_may: input for rape from barley/rye separation.
    nir_july: input for separation of sugarbeet vs corn.
    ndvi_april: separation of wheat from barley and rye.
    '''
    image_date = ndvi_march.split("_")
    image_date = image_date[3]
    image_date = pd.to_datetime(image_date, format="%Y%m%d")
    month = image_date.month
    
    # Step one - classify summer vs winter crop.
    ndvi_march = cv2.imread(ndvi_march, 0)
    ndvi_march = cv2.resize(ndvi_march, (300, 300))
    
    shape_1, shape_2 = ndvi_march.shape
    print(shape_1, shape_2)
    
    # Create dataframe for making predictions.
    df_summer_winter = pd.DataFrame(columns=["month", "ndvi", "ndvi_ratio"])
    df_summer_winter["ndvi"] = ndvi_march.ravel()
    df_summer_winter["month"] = month
    df_summer_winter["ndvi_ratio"] = df_summer_winter["ndvi"] * month
    df_summer_winter["predictions"] = binary_classifier.predict(df_summer_winter.iloc[:, :3])
    return df_summer_winter

    # Step two - classify rape vs barley where the crop is winter.
    
    #
    
    
    
ndvi_march = r"C:/Users/Tim/Desktop/GIS/GISproject/landsat/ndvi/LC08_L1TP_194026_20140313_20170425_01_T1.tif"
df = crop_classification(ndvi_march, 1, 2, 3)
