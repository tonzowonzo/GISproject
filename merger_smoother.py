# Import libraries.
import cv2
from osgeo import gdal
import numpy as np

try:
    ds = gdal.Open(r"C:/Users/Tim/Desktop/Wien work/S1A_20180207T033416_020495_123_6274_merged (5).tif")
    myarray = np.array(ds.GetRasterBand(1).ReadAsArray())
except:
    print("Hello")