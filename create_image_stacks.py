# Create image stacks with April, June and August imagery.
# Randomly match all months values to create a large dataset (in order).
# Import libraries.
import os
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from get_label import get_label

# Landsats.
landsats = [5, 7, 8]

# Years.
years = ["2010", "2011", "2012", "2013", "2014", "2015", "2016", "2017"]

# Field areas.
field_areas = ["1_1", "2", "3", "4", "5", "6", "8", "9", "11", "13", "15", "17",
               "18_1", "18_2", "18_4", "19", "20", "21", "23", "25", "26", "27",
               "28", "29", "30", "33", "EC1", "EC2", "EC3", "EC4", "EC5", "EC6"]

# Create dicts to hold images of each month as well as their classification.
april_may_date = {}
june_july_date = {}
august_september_date = {}

def get_ndvi_by_month(landsat, field_area):
    # Path we will use.
    # Create dicts to hold images of each month as well as their classification.
    path = r"C:/Users/Tim/Desktop/GIS/GISproject/" + "landsat_" + str(landsat)
    for image in os.listdir(path + "/" + field_area + "/NDVI/"):
        date = image.split(".")[0]
        pandas_date = pd.to_datetime(date, format="%Y%m%d")
        label, last_crop = get_label(field_area, pandas_date)
        # Put the image names into their respective lists.
        if pandas_date.month == 3:
            april_may_date[date + ".tif"] = [label, landsat]
            
        elif pandas_date.month == 5 or pandas_date.month == 6:
            june_july_date[date + ".tif"] = [label, landsat]
            
        elif pandas_date.month == 8:
            august_september_date[date + ".tif"] = [label, landsat]
                
def create_all_combinations(year, april_may, june_july, august_september):
    yearly_list_april_may = []
    yearly_list_june_july = []
    yearly_list_august_september = []
    # Create all band combinations by year in order to create an image stack.
    for image in april_may.keys():
        if image[:4] == year:
            yearly_list_april_may.append(image)
    for image in june_july.keys():
        if image[:4] == year:
            yearly_list_june_july.append(image)
    for image in august_september.keys():
        if image[:4] == year:
            yearly_list_august_september.append(image)
    
    # Create every possible data combination.
    import itertools
    combinations_list = list(itertools.product(*[yearly_list_april_may, 
                                                 yearly_list_june_july,
                                                 yearly_list_august_september]))

    print(combinations_list)
    return combinations_list
            
    
def create_image_stacks(combinations_list, april_may, june_july, august_september, field_area):
    path = r"C:/Users/Tim/Desktop/GIS/GISproject/landsat_"
    for combination in combinations_list:
        # Def
        april_may_file = combination[0]
        june_july_file = combination[1]
        august_september_file = combination[2]
        
        # Load in the ndvi files as numpy arrays.
        ndvi_april_may = cv2.imread(path + str(april_may_date[april_may_file][1]) + "/" +
                                    field_area + "/NDVI/" + april_may_file, 0)
        print(path + str(april_may_date[april_may_file][1]) + "/" +
                                    field_area + "/NDVI/" + april_may_file)
        # Return to start of loop is there is no april or may array.
        if ndvi_april_may.size == 0:
            continue

        ndvi_june_july = cv2.imread(path + str(june_july_date[june_july_file][1]) + "/" +
                            field_area + "/NDVI/" + june_july_file, 0)
        
        # Return to start of loop if there is no june or july array.
        if ndvi_june_july.size == 0:
            continue
        # Make sure the file is the same shape.
        ndvi_june_july = np.resize(ndvi_june_july, ndvi_april_may.shape)
        ndvi_august_september = cv2.imread(path + str(august_september_date[august_september_file][1]) + "/" +
                            field_area + "/NDVI/" + august_september_file, 0)
        
        # Return to top of loop if there is no august or september array.
        if ndvi_august_september.size == 0:
            continue
        
        # Make sure the file is the same shape as april mays file.
        ndvi_august_september = np.resize(ndvi_august_september, ndvi_april_may.shape)

        
        
        # Create a stack of images.
        stacked_array = np.dstack((ndvi_april_may, ndvi_june_july, ndvi_august_september))
        
        plt.imshow(stacked_array)
        plt.show()
        
        # Save the image to its file.
        print(r"C:/Users/Tim/Desktop/GIS/GISproject/temporal_stacks/" +
                    april_may_date[april_may_file][0] + "/" )
        cv2.imwrite(r"C:/Users/Tim/Desktop/GIS/GISproject/temporal_stacks/" +
                    april_may_date[april_may_file][0] + "/" + 
                    april_may_file + june_july_file + august_september_file +
                    field_area + ".tif", stacked_array)
    
# Loop over all datasets.
'''
import random
for field_area in field_areas:
    for landsat in landsats:
        i = random.randint(1, 10000000)
        # Get the monthly ndvi dictionaries.
        april_may_date, june_july_date, august_september_date = get_ndvi_by_month(landsat, field_area=field_area)
        # For each year -> run the experiment.
        combinations_list = create_all_combinations("2017", april_may_date, 
                                                    june_july_date, 
                                                    august_september_date)
            
        print("combo list {}".format(combinations_list))
        create_image_stacks(combinations_list, april_may_date, june_july_date,
                            august_september_date, field_area, i)
'''


april_may_date = {}
june_july_date = {}
august_september_date = {}
field_area = "EC6"
for landsat in landsats: 
    get_ndvi_by_month(landsat, field_area)      

for year in years:
    combinations_list = create_all_combinations(year, april_may_date, june_july_date, august_september_date)
    create_image_stacks(combinations_list, april_may_date, june_july_date, 
                        august_september_date, field_area)
#
#create_image_stacks(combinations_list, april_may_date, june_july_date, august_september_date, "EC2")