# Adds files to field areas for landsat 8 - Fields for each band.
# Import libraries.
import os

# Path
path = r"C:/Users/Tim/Desktop/GIS/GISproject/landsat_5/"

# Band types.
bands = ["1", "2", "3", "4", "5", "6", "7", "8"]

for file in os.listdir(path):
    for band in bands:
        os.makedirs(path + file + "//" + band)