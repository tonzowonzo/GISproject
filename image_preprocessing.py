# Import libraries.
import os
import cv2
import PIL
from PIL import Image, ImageStat, ImageFilter, ImageFile
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy
from scipy import ndimage
ImageFile.LOAD_TRUNCATED_IMAGES = True
# Matplotlib defaults
#plt.axis("off")

# Import example images.
image1 = "C:/Users/Tim/Desktop/GIS/GISproject/landsat/Bulk Order 917016/Landsat 8 OLI_TIRS C1 Level-1/LC08_L1TP_193027_20150917_20180522_01_T1.tif"
image2 = "C:/Users/Tim/Desktop/GIS/GISproject/landsat/Bulk Order 914722/Landsat 8 OLI_TIRS C1 Level-1/LC08_L1TP_194026_20171031_20171109_01_T1.tif"
test_mask = "C:/Users/Tim/Desktop/GIS/GISproject/landsat/Bulk Order 914722/Landsat 8 OLI_TIRS C1 Level-1/LC08_L1TP_194026_20171031_20171109_01_T1_QB.jpg"
landsat_path = "194"
landsat_row = "026"

# Latitude and Longitude of field area.
latitude = 48.5
longitude = 9.0
# Load in tif image.
def preprocess_image(filename, maskname, prepare_for="stationary", show_images=False):
    '''
    Preprocess the image in preparation for use in classification algorithms.
    
    prepare_for: the type of analysis the data will be prepared for, values that
    can be taken include: time_series, stationary, multiple_zones
    '''
    # Load the image in rgb format.
    img = Image.open(filename).convert("RGB")
    img = img.resize((8000, 8000))

    # Turn the image into a numpy array for manipulation.
    img_array = np.array(img)
    print(img_array.shape)
    
    # Load in mask.
    mask = Image.open(maskname)
    # Turn the image into a numpy array for manipulation.
    mask = mask.resize((8000, 8000))
    mask_array = np.array(mask)
    mask_array = mask_array == 255
    
    # Turn the mask into an actual mask.
    final_mask = np.ma.masked_array(mask_array, mask_array != 0)
    
    # Return masked img.
    masked_color = img_array.copy()
    masked_color[final_mask] = 0
    
#    plt.figure(figsize=(12, 12))
#    plt.imshow(masked_color)
#    plt.show()
    
    # Atmospheric correction (dark object subtraction).
    # Only required for time series, or multiple row data.
    # Find the brightest pixels
    # Get the brightness of the image.
    luminosity = img.convert("L")
    luminosity = np.array(luminosity)
    
    # Convert to greyscale to calculate brightness
    if prepare_for in ["time_series", "multiple_zones"]:
        # Find the top brightest value and indice in the array.
        max_indices = np.unravel_index(luminosity.argmax(), luminosity.shape)
        print(max_indices)
    
    # Display all of the images created.
    if show_images:
        # Display Color bands.
        # Red.
        plt.imshow(img_array[:, :, 0], cmap="binary")
        plt.title("Red")
        plt.colorbar()
        plt.show()
        
        # Green.
        plt.imshow(img_array[:, :, 1], cmap="binary")
        plt.title("Green")
        plt.colorbar()
        plt.show()
        
        # Blue.
        plt.imshow(img_array[:, :, 2], cmap="binary")
        plt.title("Blue")
        plt.colorbar()
        plt.show()
        

        
        # Show luminosity.
        plt.imshow(luminosity)
        plt.title("Luminosity")
        plt.colorbar()
        plt.show()
        
        # Display masked cloud brightness image.
        plt.figure(figsize=(12, 12))
        plt.imshow(mask_array, cmap="binary")
        plt.colorbar()
        plt.title("Masked Clouds")
        plt.show()
    
        # Display rgb composite.
        plt.figure(figsize=(12, 12))
        plt.imshow(img_array)
        plt.title("RGB composite")
        plt.show()

    return masked_color, luminosity, final_mask




# Sun angle correction.





# Greyscale conversion.

#img_array, green_array, mask_array = preprocess_image(image2, maskname=test_mask)


def load_images(path, landsat_row, landsat_path):
    '''
    Loads all of the images into an array.
    '''
    images = pd.DataFrame(index=[x for x in range(100)], columns=["dates", "image_arrays", "indiv_pixel", "green_array"])
    i = 0
    images["indiv_pixel"] = 0
    images["green_level"] = 0
    landsat_path_row = landsat_path + landsat_row
    for file in os.listdir(path):
        # Get the dates of each image.
        split_file = file.split("_")
        
        
        if file.endswith("T1.tif") and split_file[2]==landsat_path_row:
            # Load in image array.
            print(file)
            maskname = file[:-4] + "_QB.jpg"
            img, _, _ = preprocess_image(os.path.join(path, file), os.path.join(path, maskname))
            # Add the data to a dataframe.
            # Hours and minutes.
            hours_minutes = split_file[4]
            hours_minutes = hours_minutes[4:]
            print(hours_minutes)
            time = split_file[3] + hours_minutes
            images.dates[i] = pd.to_datetime(time, format="%Y%m%d%H%M", errors="ignore")
            images.image_arrays[i] = img
            images.green_array[i] = img[:, :, 1]
            i+=1

            
    images = images.dropna()
    return images
#df = load_images("C:/Users/Tim/Desktop/GIS/GISproject/landsat/Bulk Order 914722/Landsat 8 OLI_TIRS C1 Level-1")

def create_time_series(path, landsat_row, landsat_path):
    '''
    Create a dataframe of timeseries values and plot the data.
    '''
    df = load_images(path, landsat_row, landsat_path)
    # Do time series on just one pixel.
#    for i, image in enumerate(df.image_arrays):
#        width, height, _ = image.shape
#        print(image[int(5001)][int(5001)])
#        df["indiv_pixel"][i] = image[int(5001)][int(5001)]
#
    # Time series for green.
    for i, image in enumerate(df.green_array):
        width, height = image.shape
        print(image[int(4452)][int(936)])
        df["green_level"][i] = image[int(4452)][int(936)]
    df = df.sort_values(by="dates")
#    df = df[~(df["indiv_pixel"] > 0)]
    df = df[df["green_level"] > 0]
    # Plot X values.
    X = df["dates"]
    # Data for plotting.
    y_brightness = df["indiv_pixel"]
    y_green = df["green_level"]


    # Plot the results for one pixel.
    plt.plot(X, y_brightness)
    plt.plot(X, y_green)
    return df
        
df2 = create_time_series("C:/Users/Tim/Desktop/GIS/GISproject/landsat/Bulk Order 917016/Landsat 8 OLI_TIRS C1 Level-1", landsat_row, landsat_path)

def plot_series(X, y, title, xlabel, ylabel, plot_type="timeseries"):
    plt.figure(figsize=(12, 12))
    plt.plot(X, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    if plot_type == "map":
        plt.colorbar()
        
    plt.show()
    
    
    
plot_series(df2["dates"], df2["green_level"], "Pixel green value over time",
            "Time (year)", "Green pixel value")

plt.figure(figsize=(12, 12))
plt.imshow(df2["image_arrays"][4])
plt.show()
print(df2["dates"][4])

    