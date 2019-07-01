# Import libraries.
import gdal
import os.path
import cv2
import os
import numpy as np
from numpy import inf
import rasterio
import matplotlib.pyplot as plt

path = r"C:/Users/Tim/Desktop/BOKU/GIS/GISproject/landsat/"
bands = ["band1", "band2", "band3", "band4", "band5", "band6", "band7"]

# File from which projection is copied.
def copy_projection(input_georeferencing, file_to_reproject):
    '''
    Copies projection of one image to another of the same dimensions.
    Where:
        input_georeferencing: the path to the image that has the projection information.
        file_to_reproject: the path to the file that you want to reproject.
    '''
    dataset = gdal.Open(input_georeferencing)
    if dataset is None:
        print('Unable to open', input_georeferencing, 'for reading')
    else:
        projection   = dataset.GetProjection()
        geotransform = dataset.GetGeoTransform()
        print(projection, geotransform)
        if projection is None and geotransform is None:
            print('No projection or geotransform found on file' + input_georeferencing)
        else:
            # File to which projection is copied to.
            dataset2 = gdal.Open( file_to_reproject, gdal.GA_Update )

        if dataset2 is None:
            print('Unable to open', file_to_reproject, 'for writing')

        if geotransform is not None:
            dataset2.SetGeoTransform( geotransform )

        if projection is not None:
            dataset2.SetProjection( projection )
      
for band in bands:
    for file in os.listdir(path + "landsat8/"):
        if file.endswith(band + ".tif"):
            try:
                copy_projection(path + "landsat8/" + file, path + "landsat_8test/" + file)
            except:
                pass
            