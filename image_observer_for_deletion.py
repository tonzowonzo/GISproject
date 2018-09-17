# Import libraries.
import os
import cv2
import matplotlib.pyplot as plt

# Path of rgb images
path = r"C:/Users/Tim/Desktop/GIS/GISproject/17"

# Lists for images with each type.
good_list = []
delete_list = []

for img in os.listdir(path):
    '''
    Press 1 to keep the image.
    Press 2 to delete the image.
    '''
    if img.endswith("tif"):
        print(img)
        rgb_img = cv2.imread(path + "/" + img)
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        plt.imshow(rgb_img)
        plt.show()
        
        # Add image to the list that we want.
        is_image_good = input()
        if is_image_good == "1":
            good_list.append(img)
        elif is_image_good == "2":
            delete_list.append(img)
            
            
# Delete bad list imagery.
for file in os.listdir(path):
    for bad_img in delete_list:
        split_file = bad_img.split(".")
        split_file = split_file[0]
        if split_file in file:
            os.remove(path + "/" + file)
            
        
