import os

# Landsat path and row
landsat_path_row = "195026"

# Path to the rgb images.
path_rgb = r"C:/Users/Tim/Desktop/GIS/GISproject/EC6"
path_IR = r"C:/Users/Tim/Desktop/GIS/GISproject/EC6_IR"
# Delete IR photos that don't match the currect colour files present.
def delete_IR(path_rgb, path_IR):
    list_rgb =  os.listdir(path_rgb)
    list_ir = os.listdir(path_IR)
    matching = set(list_rgb) & set(list_ir)
    for file in os.listdir(path_IR):
        if file not in matching:
            os.remove(path_IR + "/" + file)
        
delete_list = delete_IR(path_rgb, path_IR)
        
      
    
print(len(os.listdir(path_rgb)))
print(len(os.listdir(path_IR)))

