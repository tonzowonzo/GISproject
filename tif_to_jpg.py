# Convert all files in a folder to jpg.
import os
from PIL import Image, ImageFile
import cv2

file_ending = "QB.tif"
ImageFile.LOAD_TRUNCATED_IMAGES = True

def tif_to_jpg(path, file_ending):
    for file in os.listdir(path):
        if file.endswith(file_ending):
            outfile = file[:-3] + "jpg"
            img = cv2.imread(path + "/" + file, 0)
            cv2.imwrite(path + "/" + outfile, img)
#            img.save(path + "/" + outfile, "JPG", quality=90)
            
tif_to_jpg("C:/Users/Tim/Desktop/GIS/GISproject/landsat/Bulk Order 917016/Landsat 8 OLI_TIRS C1 Level-1", file_ending)