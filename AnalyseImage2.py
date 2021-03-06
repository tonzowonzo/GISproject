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

# Load the models, first is for the initial pred, 2nd is for predictions with time.
model_first = joblib.load("no_last_crop_forest.pkl")
model_second = joblib.load("random_forest_2.pkl")

# Analyse an image.
# For the initial image in a set, use this function.
def analyse_image(image_file, image_file_IR):
    # Load in image.
    img = Image.open(image_file).convert("RGB")
    img = img.resize((2000, 2000))
    img = np.array(img)
    
    # Load in infrared image.
    img_IR = Image.open(image_file_IR)
    img_IR = img_IR.resize((2000, 2000))
    img_IR = np.array(img_IR)
    

    
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
        
    
    # Pool the normal one.
    img = pool_image(img)

    # Pool image IR.
    img_IR = pool_image(img_IR, kernel_size=(4, 4))
    
    # What shape are the images.
    print(img_IR.shape)
    print(img.shape)
    
    # r g and b arrays.
    r_array = img[:, :, 0].ravel()
    g_array = img[:, :, 1].ravel()
    b_array = img[:, :, 2].ravel()
    ir_array = img_IR.ravel()
    
    # Image shape.
    img_shape = img.shape
    # Create prediction array.
    pred_array = np.zeros((img_shape[0], img_shape[1]))
    
    # Probability array.
    proba_array = np.zeros((img_shape[0], img_shape[1], 9))
    # Loop counter.
    counter = 0
    
    # Array to store X values
    test_array = []
    
    # Loop over all of the image array and make predictions, then add to prediction array.
    for i, img_array in enumerate(img):
        for j, channel_values in enumerate(img_array):
            # Get the RGB value from the image.
            r = channel_values[0]
            g = channel_values[1]
            b = channel_values[2]
            ir = img_IR[i][j]
            
            # Normalise.
            r = r / 255
            g = g / 255
            b = b / 255
            ir = ir / 255
            red_factor = r * month
            green_factor = g * month
            blue_factor = b * month
            ir_factor = ir * month
            # Prediction count.
            if counter % 1000 == 0:
                print(counter)
                print(r, g, b, ir)
                print(channel_values)
                
            # Ignore analysis if there's a mask.
            if (r == 0 and g == 0 and b == 0) or (r == 255 and g == 255 and b == 255):
                pred_array[i][j] = 10
                counter +=1
                proba_array[i][j] = 0
                
            # Predict.
            else:
                X = np.array([month, r, g, b, ir, red_factor, green_factor,
                              blue_factor, ir_factor])
                X = X.reshape(1, -1)
                prediction = model_first.predict(X)
                test_array.append(X)
                # Add prediction to prediction array.
                proba_array[i][j] = model_first.predict_proba(X)
                pred_array[i][j] = prediction[0]
                counter +=1
            
            
                
    return img, pred_array, ir_array, img_array, r_array, proba_array
    
#true_img = analyse_image(r"C:/Users/Tim/Desktop/GIS/GISproject/Kraichgau/20141014.tif")
true_img, pred_array, ir_array, look_at_img_array, red_array, proba_array = analyse_image(r"C:/Users/Tim/Desktop/GIS/GISproject/Kraichgau/20170429.tif",
                                     r"C:/Users/Tim/Desktop/GIS/GISproject/Kraichgau_IR/20170429.tif")


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

## Create prediction using initial prediction of subsequent imagery.
#def predict_subsequent_image(initial_prediction, next_image):
#    '''
#    Passed a numpy array of predictions of an image classified by model_first
#    and uses its values as the last_crop feature of the algorithm.
#    '''
#    # Load in the image.
#    img = Image.open(next_image).convert("RGB")
#    img = img.resize((2000, 2000))
#    img = np.array(img)
#    
#    # Get the image date required for the prediction.
#    image_date = next_image.split("/")
#    image_date = image_date[-1]
#    image_date = image_date[:-4]
#    image_date = pd.to_datetime(image_date, format="%Y%m%d")
#    
#    # Get the day of the year required for the prediction.
#    month = datetime.datetime.date(image_date).month
#    year = datetime.datetime.date(image_date).year
#    day_of_year = datetime.datetime.timetuple(image_date).tm_yday
#
#    # Pool image down to a smaller size.
#    def pool_image(arr, kernel_size=(2, 2, 1)):
#        return block_reduce(arr, kernel_size, np.max)
#        
#    
#    # Pool.
#    img = pool_image(img)
#    print(img.shape)
#    
#    # Image shape.
#    img_shape = img.shape
#    # Create prediction array.
#    pred_array = np.zeros((img_shape[0], img_shape[1]))
#    
#    # Loop counter.
#    counter = 0
#    
#    # Loop over all of the image array and make predictions, then add to prediction array.
#    for i, img_array in enumerate(img):
#        for j, channel_values in enumerate(img_array):
#            # Get the RGB value from the image.
#            r = channel_values[0]
#            g = channel_values[1]
#            b = channel_values[2]
#            
#            # Prediction count.
#            if counter % 1000 == 0:
#                print(counter)
#            
#            # Ignore analysis if there's a mask.
#            if (r == 0 and g == 0 and b == 0) or (r == 255 and g == 255 and b == 255):
#                pred_array[i][j] = 9
#                counter +=1
#            
#                
#            # Predict.
#            else:
#                last_crop = initial_prediction[i][j]
#                X = np.array([month, year, last_crop, r, g, b])
#                X = X.reshape(1, -1)
#                prediction = model_second.predict(X)
#                # Add prediction to prediction array.
#                pred_array[i][j] = prediction[0]
#                counter +=1
#            
#            
#                
#    return img, pred_array
#
#input_img, predicted_subsequent = predict_subsequent_image(display_img, "C:/Users/Tim/Desktop/GIS/GISproject/Kraichgau/20111022.tif")


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

## Plot next in series prediction.
#plt.figure(figsize=(20, 20))
#im = plt.imshow(predicted_subsequent)
#colours = [ im.cmap(im.norm(value)) for value in values]
## create a patch (proxy artist) for every color 
#patches = [ mpatches.Patch(color=colours[i], label="{l}".format(l=labels[i]) ) for i in range(len(labels)) ]# put those patched as legend-handles into the legend
#plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )
#plt.show()

## Show the next in series initial image.
#plt.figure(figsize=(20, 20))
#plt.imshow(input_img)
#plt.show()

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