# Import libraries.
import cv2
import numpy as np
import os
import pandas as pd
import datetime
import matplotlib.pyplot as plt

# Variables.
path = r"C:/Users/Tim/Desktop/BOKU/GIS/GISproject/"
tile = r"/EC1/"
landsats = ["landsat_5new/", "landsat_7new/", "landsat_8new/"]
years = [#"2008", "2009", "2010", "2011", "2012", 
         "2013", "2014", "2015",
         "2016", "2017", "2018"]
bands = ["/1/", "/2/", "/3/", "/4/", "/5/", "/7/", "/8/"]
tiles = ["/EC1", "/EC2", "/EC3", "/EC4", "/EC5", "/EC6", "/1_1", "/3", "/5",
               "/6", "/8", "/9", "/11", "/13", "/15", "/17", "/18_1", "/18_4", "/19",
               "/20", "/21", "/23", "/25", "/26", "/27", "/28", "/29", "/30", "/33",
               "/final_farmland_polygon"]
img_list = []
starting_date = "0301"
STACK_HEIGHT = 14
for tile in tiles:
    print(tile)
    for band in bands:
#        list_of_imgs = os.listdir(path + landsats[0] + tile + band)
#        list_of_imgs = [img for img in list_of_imgs if img.endswith(".tif")]
#        

        for landsat in landsats:
            list_dir = os.listdir(path + landsat + tile + band)
            list_dir = [path + landsat + tile + band + img for img in list_dir if img.endswith(".tif")]
            img_list.extend(list_dir)
            
            initial_img = cv2.imread(img_list[0], -1)
            x_shape, y_shape = initial_img.shape
            
        img_list = [img for img in img_list if img.endswith(".tif")]

            
            
        for year in years:
            # Array for holding the input information.
            if os.path.exists(path + "stacks/" + "stack-" + tile[1:] + "-" + band[1:-1] + "-" + year + ".npy"):
                continue
            else:
                arr = np.zeros((x_shape, y_shape, STACK_HEIGHT))
                lower_date = year + starting_date
                lower_date = pd.to_datetime(lower_date, format="%Y%m%d")
                df = pd.DataFrame()
                for i in range(STACK_HEIGHT):
                    upper_date = lower_date + datetime.timedelta(weeks=2, days=3, hours=12)
                    for image_date in img_list:
                        date = image_date.split("/")[-1]
                        date = date.split(".")[0]
                        date = pd.to_datetime(date, format="%Y%m%d")
                        if date >= lower_date and date <= upper_date:
                            temp_img = cv2.imread(image_date, -1)
    #                        print(temp_img)
                            temp_img = cv2.resize(temp_img, (y_shape, x_shape))
                            arr[:, :, i] = temp_img
                            print(date)
    #                        print(temp_img)
    #                        plt.imshow(arr[:, :, i])
    #                        plt.show()
                            continue
                    arr[arr == 0] = np.nan
                    df[i] = arr[:, :, i].ravel()
                    df = df.interpolate(axis=1, method='linear')
                    df = df.interpolate(axis=1, limit_direction='backward',
                                        method='linear', order=2)
                    lower_date = upper_date
                    print(lower_date)
                arr[arr == 0] = np.nan
            
                final_arr = np.zeros((x_shape, y_shape, STACK_HEIGHT))
                for stack in range(STACK_HEIGHT):
                    img = df[stack].values.reshape(x_shape, y_shape)
    #                print(img)
                    final_arr[:, :, stack] = img
                np.save(path + "stacks/" + "stack-" + tile[1:] + "-" + band[1:-1] + "-" + year, final_arr)
        img_list = []