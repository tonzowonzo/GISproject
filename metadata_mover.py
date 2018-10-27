# Move the metadata files to another file.
import os

# Path.
path = r"C:/Users/Tim/Desktop/GIS/GISproject/landsat/Bulk Order 948434/Landsat 7 ETM_ C1 Level-1/"
move_to = r"C:/Users/Tim/Desktop/GIS/GISproject/landsat/metadata/"
def file_mover(path, move_to):
    for file in os.listdir(path):
        if file.endswith("MTL.txt"):
            os.rename(path + file, move_to + file)
            
file_mover(path, move_to)