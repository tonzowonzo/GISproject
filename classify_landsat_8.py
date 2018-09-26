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

# Variables for testing.
path = r"C:/Users/Tim/Desktop/GIS/GISproject/landsat/landsat_8_test/"
band_numbers = [str(x) for x in range(1, 12)]
image_name = "20130716.tif"

# Load the models, first is for the initial pred, 2nd is for predictions with time.
model_first = joblib.load("no_last_crop_forest.pkl")
model_second = joblib.load("no_last_crop_forest.pkl")

# Analyse an image.
# For the initial image in a set, use this function.
def analyse_image(path, band_numbers, image_name):
    # Load half of the imagery first.
    
    # Pool image down to a smaller size.
    def pool_image(arr, kernel_size=(12, 12)):
        return block_reduce(arr, kernel_size, np.max)
    
    # Load in coastal aerosol.
    ca = cv2.imread(path + band_numbers[0] + "/" + image_name, 0)
    # Load in blue.
    b = cv2.imread(path + band_numbers[1] + "/" + image_name, 0)
    # Load in green.
    g = cv2.imread(path + band_numbers[2] + "/" + image_name, 0)
    # Load in red.
    r = cv2.imread(path + band_numbers[3] + "/" + image_name, 0)
    # Load in near infrared.
    nir = cv2.imread(path + band_numbers[4] + "/" + image_name, 0)
    
    # Pool the above bands.
    ca = pool_image(ca)
    b = pool_image(b)
    g = pool_image(g)
    r = pool_image(r)
    nir = pool_image(nir)
    
    # Load the rest of the images.
    # Load in short wave infrared 1.
    swir1 = cv2.imread(path + band_numbers[5] + "/" + image_name, 0)
    # Load in short wave infrared 2.
    swir2 = cv2.imread(path + band_numbers[6] + "/" + image_name, 0)
    # Load in panchromatic.
    pan = cv2.imread(path + band_numbers[7] + "/" + image_name, 0)
    # Load in cirrus.
    cir = cv2.imread(path + band_numbers[8] + "/" + image_name, 0)
    # Load in thermal infrared 1.
    tirs1 = cv2.imread(path + band_numbers[9] + "/" + image_name, 0)
    # Load in thermal infrared 2.
    tirs2 = cv2.imread(path + band_numbers[10] + "/" + image_name, 0)
    
    # Pool the rest of the images.
    swir1 = pool_image(swir1)
    swir2 = pool_image(swir2)
    pan = pool_image(pan)
    cir = pool_image(cir)
    tirs1 = pool_image(tirs1)
    tirs2 = pool_image(tirs2)
    # Reshape pan.
    # Image shape.
    img_shape = r.shape
    pan = cv2.resize(pan, dsize=(img_shape[1], img_shape[0]), 
                                 interpolation=cv2.INTER_NEAREST)

    # Get the image date required for the prediction.
    image_date = image_name.split(".")
    image_date = image_date[0]
    print(image_date)
    image_date = pd.to_datetime(image_date, format="%Y%m%d")
    
    # Get the day of the year required for the prediction.
    month = datetime.datetime.date(image_date).month
    year = datetime.datetime.date(image_date).year
    day_of_year = datetime.datetime.timetuple(image_date).tm_yday
    
    # What shape are the images.
    print(r.shape)
    print(pan.shape)
    
    # Create prediction array.
    pred_array = np.zeros((img_shape[0], img_shape[1]))
    
    # Probability array.
    proba_array = np.zeros((img_shape[0], img_shape[1], 8))
    
    # Loop counter.
    counter = 0
    
    # Array to store X values
    test_array = []
    
    # Loop over all of the image array and make predictions, then add to prediction array.
    for i, array_value in enumerate(ca):
        for j, ca_value in enumerate(array_value):
            
            # Get the RGB value from the image.
            ca_value = ca_value
            b_value = b[i][j]
            g_value = g[i][j]
            r_value = r[i][j]
            nir_value = nir[i][j]
            swir1_value = swir1[i][j]
            swir2_value = swir2[i][j]
            pan_value = pan[i][j]
            cir_value = cir[i][j]
            tirs1_value = tirs1[i][j]
            tirs2_value = tirs2[i][j]

            
            # Prediction count.
            if counter % 1000 == 0:
                print(counter)
                print(r_value, g_value, b_value)
                
            # Ignore analysis if there's a mask.
            if (r_value == 0 and g_value == 0 and b_value == 0):
                pred_array[i][j] = 10
                counter +=1
                proba_array[i][j] = 0
                
            # Predict.
            else:
                X = np.array([month, year, ca_value, b_value, g_value, r_value, 
                              nir_value, swir1_value, swir2_value, pan_value,
                              cir_value, tirs1_value, tirs2_value])
                X = X.reshape(1, -1)
                prediction = model_first.predict(X)
                test_array.append(X)
                # Add prediction to prediction array.
                proba_array[i][j] = model_first.predict_proba(X)
                pred_array[i][j] = prediction[0]
                counter +=1
            
            
                
    return b, g, r, pred_array, proba_array
    
b, g, r, pred_array, proba_array = analyse_image(path, band_numbers, image_name)


def transform_prediction_array(pred_array):
    '''
    Turns the prediction array into an rgb coloured array for display in GIS etc.
    
    Classes are mapped to numbers:
        
        0: Cloud
            
        1: Cloud Shadow
            
        2: GM
                        
        3: SB 
        
        4: SM
            
        5: SP
                
        6: WB
        
        7: WR
            
        8: WW
            
    '''
    display_img = pred_array.copy()
    return display_img
    
display_img = transform_prediction_array(pred_array)

# Plot the data.
labels = ["CC-GM", "CC-SB", "CC-SM", "Cloud", "WB", "WW", "Background"]
# Get unique values.
plt.figure(figsize=(20, 20))
colours = ["yellow", "brown", "red", "white", "white", "green", "blue", "orange", "purple", "black"]
im = plt.imshow(display_img, cmap=matplotlib.colors.ListedColormap(colours))
values = np.unique(display_img.ravel())
patches = [ mpatches.Patch(color=colours[i], label="{}".format(l=labels[i]) ) for i in range(len(labels)) ]# put those patched as legend-handles into the legend
plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )
plt.show()

# Plot initial prediction.
plt.figure(figsize=(12, 12))
im = plt.imshow(display_img)
#colours = [ im.cmap(im.norm(value)) for value in values]
# create a patch (proxy artist) for every color 
patches = [ mpatches.Patch(color=colours[i], label="{}".format(l=labels[i]) ) for i in range(len(labels)) ]# put those patched as legend-handles into the legend
plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )
plt.show()

# Plot display img.
plt.figure(figsize=(20, 20))
plt.imshow(display_img, cmap=matplotlib.colors.ListedColormap(colours))
plt.legend()
plt.show()

# Plot the true image.
plt.figure(figsize=(12, 12))
plt.imshow(true_img)
plt.show()

# Plot the infrared image.
plt.figure(figsize=(20, 20))
plt.imshow(ir_array)
plt.show()

# Show the probability maps.
def plot_probability_maps(probability_array):
    # Define the plot params
    def plot_image(image_array, title):
        plt.figure(figsize=(12, 12))
        plt.title(title)
        plt.imshow(image_array)
        plt.show()
        
    # Cloud
    plot_image(probability_array[:, :, 0], "Cloud")
    # Cloud Shadow
    plot_image(probability_array[:, :, 1], "Cloud Shadow")
    # GM
    plot_image(probability_array[:, :, 2], "GM")
    # SB
    plot_image(probability_array[:, :, 3], "SB")
    # SM
    plot_image(probability_array[:, :, 4], "SM")
    # SP
    plot_image(probability_array[:, :, 5], "SP")
    # WB
    plot_image(probability_array[:, :, 6], "WB")
    # WR
    plot_image(probability_array[:, :, 7], "WR")
    # WW
    plot_image(probability_array[:, :, 8], "WW")
    
plot_probability_maps(proba_array)

        
# Save the numpy array from analysis.
import scipy.misc
scipy.misc.imsave("classified_example1.tif", display_img)



# Get image statistics.
from scipy import stats

def get_summary_image_statistics(img):
    # Remove the background pixels from array.
    stats_array = img[img != 10]
    
    #  Cloud
    Cloud = stats_array[stats_array == 0]
    print("Amount of Cloud in image: ", len(Cloud)/len(stats_array))
    
    # Cloud Shadow
    CloudShadow = stats_array[stats_array == 1]
    print("Amount of Cloud Shadow in image: ", len(CloudShadow)/len(stats_array))
    
    # GM
    GM = stats_array[stats_array == 2]
    print("Amount of GM in image: ", len(GM)/len(stats_array))
    
    #  SB.
    SB = stats_array[stats_array == 3]
    print("Amount of SB in image: ", len(SB)/len(stats_array))
    
    #  SM.
    SM = stats_array[stats_array == 4]
    print("Amount of SM in image: ", len(SM)/len(stats_array))
    
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
    
    # Total area.
    max_pool_amount = 4
    spatial_resolution = 30
    pixel_size = max_pool_amount * spatial_resolution * spatial_resolution
    area = len(stats_array) * pixel_size
    print("The area of the image is: ", area, " metres squared")
    print("Or ", area/10000, " hectares")
    return WW
    
stats_array = get_summary_image_statistics(display_img)