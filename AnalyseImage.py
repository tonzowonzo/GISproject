# Analyse image based on a given sklearn model.
# Import libraries.
import os
import datetime

import cv2
import PIL
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import sklearn
import matplotlib.patches as mpatches
from skimage.measure import block_reduce
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier

# Set cwd.
os.chdir("C:/Users/Tim/Desktop/GIS/GISproject")

# Load the model.
model = joblib.load("random_forest_2.pkl")

# Analyse an image.
def analyse_image(image_file):
    # Load in image.
    img = Image.open(image_file).convert("RGB")
    img = img.resize((2000, 2000))
    img = np.array(img)
    
    # Get the image date required for the prediction.
    image_date = image_file.split("/")
    image_date = image_date[-1]
    image_date = image_date[:-4]
    image_date = pd.to_datetime(image_date, format="%Y%m%d")
    
    # Get the day of the year required for the prediction.
    month = datetime.datetime.date(image_date).month
    year = datetime.datetime.date(image_date).year
    day_of_year = datetime.datetime.timetuple(image_date).tm_yday
    

    
    # Pool image down to a smaller size.
    def pool_image(arr, kernel_size=(4, 4, 1)):
        return block_reduce(arr, kernel_size, np.max)
        
    
    # Pool.
    img = pool_image(img)
    print(img.shape)
    
        # Image shape.
    img_shape = img.shape
    # Create prediction array.
    pred_array = np.zeros((img_shape[0], img_shape[1]))
    
    # Loop counter.
    counter = 0
    
    # Loop over all of the image array and make predictions, then add to prediction array.
    for i, img_array in enumerate(img):
        for j, channel_values in enumerate(img_array):
            # Get the RGB value from the image.
            r = channel_values[0]
            g = channel_values[1]
            b = channel_values[2]
            
            # Prediction count.
            if counter % 1000 == 0:
                print(counter)
            
            # Ignore analysis if there's a mask.
            if (r == 0 and g == 0 and b == 0) or (r == 255 and g == 255 and b == 255):
                pred_array[i][j] = 9
                counter +=1
            
                
            # Predict.
            else:
                X = np.array([month, year, r, g, b])
                X = X.reshape(1, -1)
                prediction = model.predict(X)
                # Add prediction to prediction array.
                pred_array[i][j] = prediction[0]
                counter +=1
            
            
                
    return img, pred_array
    
#true_img = analyse_image(r"C:/Users/Tim/Desktop/GIS/GISproject/Kraichgau/20141014.tif")
true_img, pred_array = analyse_image(r"C:/Users/Tim/Desktop/GIS/GISproject/Kraichgau/20100731.tif")

def transform_prediction_array(pred_array):
    '''
    Turns the prediction array into an rgb coloured array for display in GIS etc.
    
    Classes are mapped to numbers:
        
        0: CC-GM
            
        1: CC-SB
            
        2: CC-SM
            
        3: Cloud
            
        4: Cloud Shadow
            
        5: SP
                
        6: WB
        
        7: WR
            
        8: WW
            
    '''
    display_img = pred_array.copy()
    return display_img
    
display_img = transform_prediction_array(pred_array)


# Plot the data.
labels = ["CC-GM", "CC-SB", "CC-SM", "Cloud", "Cloud Shadow", "SP", "WB", "WW", "Background"]
# Get unique values.
plt.figure(figsize=(20, 20))
colours = ["yellow", "brown", "white", "white", "pink", "green", "black"]
im = plt.imshow(display_img, cmap=matplotlib.colors.ListedColormap(colours))
values = np.unique(display_img.ravel())

# Create a colourmap.
plt.figure(figsize=(20, 20))
im = plt.imshow(display_img)
colours = [ im.cmap(im.norm(value)) for value in values]
# create a patch (proxy artist) for every color 
patches = [ mpatches.Patch(color=colours[i], label="{l}".format(l=labels[i]) ) for i in range(len(labels)) ]# put those patched as legend-handles into the legend
plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )
plt.show()

# Plot display img.
plt.figure(figsize=(20, 20))
plt.imshow(display_img, cmap=matplotlib.colors.ListedColormap(colours))
plt.legend()
plt.show()

plt.figure(figsize=(20, 20))
plt.imshow(true_img)
plt.show()

# Save the numpy array from analysis.
import scipy.misc
scipy.misc.imsave("classified_example1.tif", display_img)



# Get image statistics.
from scipy import stats

def get_summary_image_statistics(img):
    # Remove the background pixels from array.
    stats_array = img[img != 9]

    # Mean of image.
    print("mean: ", np.mean(stats_array))
    
    # Mode of image.
    mode = int(stats.mode(stats_array)[0][0])
    print("mode: ", labels[mode])
    
    #  CC-GM.
    CC_GM = stats_array[stats_array == 0]
    print("Amount of CC-GM in image: ", len(CC_GM)/len(stats_array))
    
    # CC-SB
    CC_SB = stats_array[stats_array == 1]
    print("Amount of CC-SB in image: ", len(CC_SB)/len(stats_array))
    
    #  CC-SM.
    CC_SM = stats_array[stats_array == 2]
    print("Amount of CC-SM in image: ", len(CC_SM)/len(stats_array))
    
    #  Cloud.
    cloud = stats_array[stats_array == 3]
    print("Amount of cloud in image: ", len(cloud)/len(stats_array))
    
    #  Cloud Shadow.
    cloudShadow = stats_array[stats_array == 4]
    print("Amount of cloud shadow in image: ", len(cloudShadow)/len(stats_array))
    
    # SP
    SP = stats_array[stats_array == 5]
    print("Amount of SP in image: ", len(SP)/len(stats_array))

    #  WB.
    WB = stats_array[stats_array == 6]
    print("Amount of WB in image: ", len(WB)/len(stats_array))
    
    # WR.
    WR = stats_array[stats_array == 7]
    print("Amount of WR in image: ", len(WR)/len(stats_array))
    
    #  WW.
    WW = stats_array[stats_array == 8]
    print("Amount of WW in image: ", len(WW)/len(stats_array))
    

    return CC_GM
    
stats_array = get_summary_image_statistics(display_img)
