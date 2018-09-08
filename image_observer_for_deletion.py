# Import libraries.
import os
import cv2
import matplotlib.pyplot as plt

# Path of rgb images
path = r"C:/Users/Tim/Desktop/GIS/GISproject/EC6"

# Lists for images with each type.
good_list = []
delete_list = []

for img in os.listdir(path):
    if img.endswith("tif"):
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
            
            
        
