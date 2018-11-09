# Binary classifier.
# Creates a training and test dataframe from full landsat 5 - 8 information.
# Classifies whether crop is summer or winter.
# Current accuracy - 93%
# 2010 Accuracy - NA
# 2011 Accuracy - 44%
# 2012 Accuracy - 46%
# 2013 Accuracy - NA
# 2014 Accuracy - 72%
# 2015 Accuracy - NA
# 2016 Accuracy - 82%
# 2017 Accuracy - 100%
# Import libraries.
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pandas as pd
import datetime
import os
import random
import math

# Import the get_label function
from get_label import get_label_combined_maize

# Import function for getting metadata.
from temp_from_metadata import get_required_info_from_metadata, calculate_lst
# Set working directory.
os.chdir("C:/Users/Tim/Desktop/GIS/GISproject")

# Constants.
field_areas = ["EC1", "EC2", "EC3", "EC4", "EC5", "EC6", "1_1", "3", "4", "5",
               "6", "8", "9", "11", "13", "15", "17", "18_1", "18_2", "18_4", "19",
               "20", "21", "23", "25", "26", "27", "28", "29", "30", "33"]
summer_crops = ["B", "M", "SB", "GM", "SM"]
winter_crops = ["WW", "WB", "WR", "P", "SP" "TC", "O"]

###############################################################################
# Landsat 8
###############################################################################
# Function for determining which band the image is.
def determine_band(img_path):
    '''
    Takes the path of an images and determines its band.
    '''
    split_file = img_path.split(".")
    split_file = split_file[0]
    split_file = split_file.split("_")
    band_number = split_file[-1]
    band_number = int(band_number[1:])
    return band_number

# Column order for the df.
'''
Where: 
    date: date of image capture.
    day_of_year: day of the year out of 365/366.
    month: month of the year out of 12.
    last_crop: the last crop that was farmed in this area.
    ca: coastal aerosol band.
    b: blue band.
    g: green band.
    r: red band.
    nir: near infrared band.
    swir1: short wave infrared 1 band.
    swir2: short wave infrared 2 band.
    pan: panchromatic band.
    cir: cirrus band.
    tirs1: thermal infrared 1 band.
    tirs2: thermal infrared 2 band.
    ndvi: normalised vegetation index.
    lst: land surface temperature.
'''
columns = ["date", "day_of_year", "month", "year", "last_crop", "ca", "b", "g",
           "r", "nir", "swir1",  "swir2", "pan", "cir", "tirs1", "tirs2", "ndvi",
           "last_ca", "last_b", "last_g", "last_r", "last_nir", "last_swir1", 
           "last_swir2", "last_pan", "last_cir", "last_tirs1", "last_tirs2", 
           "lst", "label", "binary_label"]
train_df = pd.DataFrame(columns=columns)
test_df = pd.DataFrame(columns=columns)
test_field_areas = ["EC3", "4", "5", "18_1", "33"]

# Values for the first iteration only.
last_ca = 0
last_b = 0
last_g = 0
last_r = 0
last_nir = 0
last_swir1 = 0
last_swir2 = 0
last_pan = 0
last_cir = 0
last_tirs1 = 0
last_tirs2 = 0

# Loop for getting all files into the dataframe, labelled.
for field in field_areas:
    path = "C:/Users/Tim/Desktop/GIS/GISproject/landsat_8/" + field
    for file in os.listdir(path + "/5/"):
        if file.endswith('.tif'):
            # The year, month and day time.
            date = file[:-4]
            print(file, path)
            date = pd.to_datetime(date, format="%Y%m%d")
            # What day of the year is it out of 365/366.
            day_of_year = datetime.datetime.timetuple(date).tm_yday
            month = datetime.datetime.date(date).month
            year = datetime.datetime.date(date).year
            # Get the label based on csv.
            label, last_crop = get_label_combined_maize(field, date)
            # Load the landsat metadata.
            for meta_file in os.listdir(r"C:/Users/Tim/Desktop/GIS/GISproject/landsat/metadata/"):
                split_meta = meta_file.split("_")
                meta_date = split_meta[3]
                meta_date = pd.to_datetime(meta_date, format="%Y%m%d")
                if meta_date == date:
                    mult, add, k1, k2 = get_required_info_from_metadata(r"C:/Users/Tim/Desktop/GIS/GISproject/landsat/metadata/" + meta_file,
                                                                        8)
            # Load the training image.
            ca = cv2.imread(path +  "\\1\\" + file, 0)
            b = cv2.imread(path +  "\\2\\" + file, 0)
            g = cv2.imread(path +  "\\3\\" + file, 0)
            r = cv2.imread(path +  "\\4\\" + file, 0)
            nir = cv2.imread(path +  "\\5\\" + file, 0)
            swir1 = cv2.imread(path +  "\\6\\" + file, 0)
            swir2 = cv2.imread(path +  "\\7\\" + file, 0)
            pan = cv2.imread(path +  "\\8\\" + file, 0)
            cir = cv2.imread(path +  "\\9\\" + file, 0)
            tirs1 = cv2.imread(path +  "\\10\\" + file, 0)
            tirs2 = cv2.imread(path +  "\\11\\" + file, 0)
            
            # Reshape the panchromatic image to match others.
            img_shape = b.shape[:2]
            pan = cv2.resize(pan, dsize=(img_shape[1], img_shape[0]), 
                             interpolation=cv2.INTER_NEAREST)
            
            # Labels for early or late planting or crop.
            if label in summer_crops:
                summer_or_winter_crop = 'summer'
            elif label in winter_crops:
                summer_or_winter_crop = 'winter'
            else:
                summer_or_winter_crop = 'other'
                
            # Display the images for checking.
            plt.imshow(b, cmap="binary")
            plt.clim(30, 150)
            plt.show()
            plt.imshow(tirs2, cmap="binary")
            plt.clim(30, 150)
            plt.show()
            
            # Turn the 2D image data into a 1D series.
            ca = ca.ravel()
            b = b.ravel()
            g = g.ravel()
            r = r.ravel()
            nir = nir.ravel()
            swir1 = swir1.ravel()
            swir2 = swir2.ravel()
            pan = pan.ravel()
            cir = cir.ravel()
            tirs1 = tirs1.ravel()
            tirs2 = tirs2.ravel()
            
            # Calculate NDVI.
            ndvi = (nir - r) / (nir + r)

            # Create the secondary dataframe to append to the full dataframe.
            data = {"date": date, "day_of_year": day_of_year, "month": month,
                    "year": year, "last_crop": last_crop, "ca": ca, "b": b, 
                    "g": g, "r": r, "nir": nir, "swir1": swir1,  
                    "swir2": swir2, "pan": pan, "cir": cir, "tirs1": tirs1, 
                    "tirs2": tirs2, "ndvi": ndvi, "last_ca": last_ca,
                    "last_b": last_b, "last_g": last_g, "last_r": last_r,
                    "last_nir": last_nir, "last_swir1": last_swir1, 
                    "last_swir2": last_swir2, "last_pan": last_pan,
                    "last_cir": last_cir,
                    "last_tirs1": last_tirs1, "last_tirs2": last_tirs2,
                    "lst": 0, "label": label, 
                    "binary_label": summer_or_winter_crop}
            
            df_iter = pd.DataFrame(data=data)
            df_iter = df_iter[columns]
            df_iter["lst"] = calculate_lst(df_iter["ndvi"], df_iter["ndvi"].max(), 
                   df_iter["ndvi"].min(), tirs1, 8, mult, add, k1, k2)
            # Append secondary dataframe to the full dataframe.
            # Choose which df to add to.
#                random_number = random.randint(1, 100)
#                if random_number <= 20:
#                    test_df = test_df.append(df_iter)
#                else:
#                    train_df = train_df.append(df_iter)
            if field in test_field_areas:
                test_df = test_df.append(df_iter)
            else:
                train_df = train_df.append(df_iter)       
            
            # Get last values to pass to next iteration.
            b = b[b != 0]
            g = g[g != 0]
            r = r[r != 0]
            nir = nir[nir != 0]
            swir1 = swir1[swir1 != 0]
            swir2 = swir2[swir2 != 0]
            tirs1 = tirs1[tirs1 != 0]
            pan = pan[pan != 0]
            ca = ca[ca != 0]
            tirs2 = tirs2[tirs2 != 0]
            cir = cir[cir != 0]
            
            last_ca = ca.mean()
            last_b = b.mean()
            last_g = g.mean()
            last_r = r.mean()
            last_nir = nir.mean()
            last_swir1 = swir1.mean()
            last_swir2 = swir2.mean()
            last_pan = pan.mean()
            last_cir = cir.mean()
            last_tirs1 = tirs1.mean()
            last_tirs2 = tirs2.mean()
            
            
            
###############################################################################
# Landsat 7.
###############################################################################
# Loop for getting all files into the dataframe, labelled.
for field in field_areas:
    path = "C:/Users/Tim/Desktop/GIS/GISproject/landsat_7/" + field
    for file in os.listdir(path + "/5/"):
        if file.endswith('.tif'):
            # The year, month and day time.
            date = file[:-4]
            print(file, path)
            date = pd.to_datetime(date, format="%Y%m%d")
            # What day of the year is it out of 365/366.
            day_of_year = datetime.datetime.timetuple(date).tm_yday
            month = datetime.datetime.date(date).month
            year = datetime.datetime.date(date).year
            # Get the label based on csv.
            label, last_crop = get_label_combined_maize(field, date)
            # Load the training image.
            b = cv2.imread(path +  "\\1\\" + file, 0)
            g = cv2.imread(path +  "\\2\\" + file, 0)
            r = cv2.imread(path +  "\\3\\" + file, 0)
            nir = cv2.imread(path +  "\\4\\" + file, 0)
            swir1 = cv2.imread(path +  "\\5\\" + file, 0)
            tirs1 = cv2.imread(path +  "\\6\\" + file, 0)
            swir2 = cv2.imread(path +  "\\7\\" + file, 0)
            pan = cv2.imread(path +  "\\8\\" + file, 0)
                            
            # Reshape the panchromatic image to match others.
            img_shape = b.shape[:2]
            pan = cv2.resize(pan, dsize=(img_shape[1], img_shape[0]), 
                             interpolation=cv2.INTER_NEAREST)
            
            # Labels for early or late planting or crop.
            if label in summer_crops:
                summer_or_winter_crop = 'summer'
            elif label in winter_crops:
                summer_or_winter_crop = 'winter'
            else:
                summer_or_winter_crop = 'other'
                
            # Display the images for checking.
            plt.imshow(b, cmap="binary")
            plt.clim(30, 150)
            plt.show()
            
            # Load the landsat metadata.
            for meta_file in os.listdir(r"C:/Users/Tim/Desktop/GIS/GISproject/landsat/metadata/"):
                split_meta = meta_file.split("_")
                meta_date = split_meta[3]
                meta_date = pd.to_datetime(meta_date, format="%Y%m%d")
                if meta_date == date:
                    mult, add, k1, k2 = get_required_info_from_metadata(r"C:/Users/Tim/Desktop/GIS/GISproject/landsat/metadata/" + meta_file,
                                                                        7)
            # Turn the 2D image data into a 1D series.
            b = b.ravel()
            g = g.ravel()
            r = r.ravel()
            nir = nir.ravel()
            swir1 = swir1.ravel()
            swir2 = swir2.ravel()
            pan = pan.ravel()
            tirs1 = tirs1.ravel()
            
            # Calculate NDVI.
            ndvi = (nir - r) / (nir + r)
            
            # Create the secondary dataframe to append to the full dataframe.
            data = {"date": date, "day_of_year": day_of_year, "month": month,
                    "year": year, "last_crop": last_crop, "ca": 0, "b": b, 
                    "g": g, "r": r, "nir": nir, "swir1": swir1,  
                    "swir2": swir2, "pan": pan, "cir": 0, "tirs1": tirs1, 
                    "tirs2": 0, "ndvi": ndvi, "last_ca": 0,
                    "last_b": last_b, "last_g": last_g, "last_r": last_r,
                    "last_nir": last_nir, "last_swir1": last_swir1, 
                    "last_swir2": last_swir2, "last_pan": last_pan,
                    "last_cir": 0,
                    "last_tirs1": last_tirs1, "last_tirs2": 0,
                    "lst": 0, "label": label, 
                    "binary_label": summer_or_winter_crop}
            
            df_iter = pd.DataFrame(data=data)
            df_iter = df_iter[columns]
            df_iter["lst"] = calculate_lst(df_iter["ndvi"], df_iter["ndvi"].max(), 
                               df_iter["ndvi"].min(), tirs1, 7, mult, add, k1, k2)
            # Append secondary dataframe to the full dataframe.
            # Choose which df to add to.
#                random_number = random.randint(1, 100)
#                if random_number <= 20:
#                    test_df = test_df.append(df_iter)
#                else:
#                    train_df = train_df.append(df_iter)
            if field in test_field_areas:
                test_df = test_df.append(df_iter)
            else:
                train_df = train_df.append(df_iter)  

            # Get last values to pass to next iteration.
            b = b[b != 0]
            g = g[g != 0]
            r = r[r != 0]
            nir = nir[nir != 0]
            swir1 = swir1[swir1 != 0]
            swir2 = swir2[swir2 != 0]
            tirs1 = tirs1[tirs1 != 0]
            pan = pan[pan != 0]
            
            last_b = b.mean()
            last_g = g.mean()
            last_r = r.mean()
            last_nir = nir.mean()
            last_swir1 = swir1.mean()
            last_swir2 = swir2.mean()
            last_pan = pan.mean()
            last_tirs1 = tirs1.mean()
                  
###############################################################################
# Landsat 5.
###############################################################################
# Loop for getting all files into the dataframe, labelled.
for field in field_areas:
    path = "C:/Users/Tim/Desktop/GIS/GISproject/landsat_5/" + field
    for file in os.listdir(path + "/5/"):
        if file.endswith('.tif'):
            # The year, month and day time.
            date = file[:-4]
            print(file, path)
            date = pd.to_datetime(date, format="%Y%m%d")
            # What day of the year is it out of 365/366.
            day_of_year = datetime.datetime.timetuple(date).tm_yday
            month = datetime.datetime.date(date).month
            year = datetime.datetime.date(date).year
            # Get the label based on csv.
            label, last_crop = get_label_combined_maize(field, date)
            
            # Load the landsat metadata.
            for meta_file in os.listdir(r"C:/Users/Tim/Desktop/GIS/GISproject/landsat/metadata/"):
                split_meta = meta_file.split("_")
                meta_date = split_meta[3]
                meta_date = pd.to_datetime(meta_date, format="%Y%m%d")
                if meta_date == date:
                    mult, add, k1, k2 = get_required_info_from_metadata(r"C:/Users/Tim/Desktop/GIS/GISproject/landsat/metadata/" + meta_file,
                                                                        5)
            # Load the training image.
            b = cv2.imread(path +  "\\1\\" + file, 0)
            g = cv2.imread(path +  "\\2\\" + file, 0)
            r = cv2.imread(path +  "\\3\\" + file, 0)
            nir = cv2.imread(path +  "\\4\\" + file, 0)
            swir1 = cv2.imread(path +  "\\5\\" + file, 0)
            tirs1 = cv2.imread(path +  "\\6\\" + file, 0)
            swir2 = cv2.imread(path +  "\\7\\" + file, 0)
                   
            # Labels for early or late planting or crop.
            if label in summer_crops:
                summer_or_winter_crop = 'summer'
            elif label in winter_crops:
                summer_or_winter_crop = 'winter'
            else:
                summer_or_winter_crop = 'other'
                
            # Display the images for checking.
            plt.imshow(b, cmap="binary")
            plt.clim(30, 150)
            plt.show()
            
            # Turn the 2D image data into a 1D series.
            b = b.ravel()
            g = g.ravel()
            r = r.ravel()
            nir = nir.ravel()
            swir1 = swir1.ravel()
            swir2 = swir2.ravel()
            tirs1 = tirs1.ravel()
            
            # Calculate NDVI.
            ndvi = (nir - r) / (nir + r)
            
            # Create the secondary dataframe to append to the full dataframe.
            data = {"date": date, "day_of_year": day_of_year, "month": month,
                    "year": year, "last_crop": last_crop, "ca": 0, "b": b, 
                    "g": g, "r": r, "nir": nir, "swir1": swir1,  
                    "swir2": swir2, "pan": 0, "cir": 0, "tirs1": tirs1, 
                    "tirs2": 0, "ndvi": ndvi, "last_ca": 0,
                    "last_b": last_b, "last_g": last_g, "last_r": last_r,
                    "last_nir": last_nir, "last_swir1": last_swir1, 
                    "last_swir2": last_swir2, "last_pan": 0,
                    "last_cir": 0,
                    "last_tirs1": last_tirs1, "last_tirs2": 0,
                    "lst": 0, "label": label, 
                    "binary_label": summer_or_winter_crop}
            
            df_iter = pd.DataFrame(data=data)
            df_iter = df_iter[columns]
            df_iter["lst"] = calculate_lst(df_iter["ndvi"], df_iter["ndvi"].max(), 
                               df_iter["ndvi"].min(), tirs1, 5, mult, add, k1, k2)
            # Append secondary dataframe to the full dataframe.
            # Choose which df to add to.
#                random_number = random.randint(1, 100)
#                if random_number <= 20:
#                    test_df = test_df.append(df_iter)
#                else:
#                    train_df = train_df.append(df_iter)
            if field in test_field_areas:
                test_df = test_df.append(df_iter)
            else:
                train_df = train_df.append(df_iter) 
                
            # Get last values to pass to next iteration.
            b = b[b != 0]
            g = g[g != 0]
            r = r[r != 0]
            nir = nir[nir != 0]
            swir1 = swir1[swir1 != 0]
            swir2 = swir2[swir2 != 0]
            tirs1 = tirs1[tirs1 != 0]

            last_b = b.mean()
            last_g = g.mean()
            last_r = r.mean()
            last_nir = nir.mean()
            last_swir1 = swir1.mean()
            last_swir2 = swir2.mean()
            last_tirs1 = tirs1.mean()
                
###############################################################################
# Prepare dataframe for training.
###############################################################################

# Drop the black pixels in the dataframe.
train_df = train_df[(train_df.r != 0) & (train_df.g != 0) & (train_df.b != 0)]
test_df = test_df[(test_df.r != 0) & (test_df.g != 0) & (test_df.b != 0)]

# Remove infrared with no value.
train_df = train_df[train_df.tirs1 != 0]
test_df = test_df[test_df.tirs1 != 0]

# Drop values whose date is between September and March.
train_df = train_df[train_df.label != "irrelevant"]
test_df = test_df[test_df.label != "irrelevant"]

# Sort the dataframe by date.
train_df = train_df.sort_values(by="date")
test_df = test_df.sort_values(by="date")

# Replace infinity values.
train_df = train_df.replace([np.inf, -np.inf], np.nan)
test_df = test_df.replace([np.inf, -np.inf], np.nan)

# Drop the NaN values.
train_df = train_df.dropna()
test_df = test_df.dropna()

# Drop the non summer or winter crops.
train_df = train_df[train_df.binary_label != "other"]
test_df = test_df[test_df.binary_label != "other"]

# Drop all months but august.
#train_df_aug = train_df[(train_df.month == 3)]
#test_df_aug = test_df[(test_df.month == 3)]

# Drop everything but late march.
train_df_aug = train_df[(train_df.day_of_year >= 75) & (train_df.day_of_year <= 90)]
test_df_aug = test_df[(test_df.day_of_year >= 75) & (test_df.day_of_year <= 90)]

## Drop all years but 2016, 2017.
#test_df_aug = test_df_aug[test_df_aug.year == 2015]

## Save the dataframe as a csv.
#train_df.to_csv(r"C:/Users/Tim/Desktop/GIS/GISproject/train_df.csv")
#test_df.to_csv(r"C:/Users/Tim/Desktop/GIS/GISproject/test_df.csv")

## Load the dfs back
#train_df = pd.read_csv(r"C:/Users/Tim/Desktop/GIS/GISproject/train_df.csv")
#test_df = pd.read_csv(r"C:/Users/Tim/Desktop/GIS/GISproject/test_df.csv")

# Encode labels.
from sklearn.preprocessing import LabelEncoder
# Get X and y training data
X_train = train_df_aug.iloc[:, 1:-2]
y_train = train_df_aug.iloc[:, -1]

# Encode train labels.
encoder = LabelEncoder()
encoder.fit(y_train)
y_train = encoder.transform(y_train)

X_encoder = LabelEncoder()
X_encode_values = X_train.last_crop
X_encoder.fit(X_encode_values)
X_encode_values = X_encoder.transform(X_encode_values)
X_train.last_crop = X_encode_values

# Encode test labels, prepare test set.
# Get X and y training data.
X_test = test_df_aug.iloc[:, 1:-2]
y_test = test_df_aug.iloc[:, -1]
# Encode the y_test labels.
y_test = encoder.transform(y_test)

# Encode the X_test labels to onehotencoded values.
X_test_encoder = LabelEncoder()
X_test_encode_values = X_test.last_crop
X_test_encoder.fit(X_test_encode_values)
X_test_encode_values = X_test_encoder.transform(X_test_encode_values)
X_test.last_crop = X_test_encode_values




###############################################################################
# Create classifier.
###############################################################################
# Train a random forest.
from sklearn.ensemble import RandomForestClassifier
rand_for = RandomForestClassifier()

# Fit the classifier.
rand_for = RandomForestClassifier(bootstrap=True, n_estimators=500, random_state=42,
                                  criterion="entropy")
rand_for.fit(X_train, y_train)

# Feature importances.
print(rand_for.feature_importances_)

# Predict values
y_pred = rand_for.predict(X_test)

# Get scores of the classifier.
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted")
recall = recall_score(y_test, y_pred, average="weighted")

# Confusion matrix.
cm = confusion_matrix(y_test, y_pred)

# Turn encoded values back to non-encoded for comparison
#y_test_text = list(encoder.inverse_transform(y_test))
#y_pred_text = list(encoder.inverse_transform(y_pred))

# Save the model.
from sklearn.externals import joblib
joblib.dump(rand_for, "random_forest_2.pkl")
#joblib.dump(svm, "svm.pkl")

# Random forest for without last crop info (1st classification)
rand_for_no_last_crop = RandomForestClassifier(bootstrap=True, n_estimators=500, random_state=42)
X_train_no_last_crop = X_train[["month", "ndvi"]]
X_test_no_last_crop = X_test[["month", "ndvi"]]
X_train_no_last_crop["ndvi_ratio"] = X_train["month"] * X_train["ndvi"]
X_test_no_last_crop["ndvi_ratio"] = X_test["month"] * X_test["ndvi"]
rand_for_no_last_crop.fit(X_train_no_last_crop, y_train)
y_pred_no_last = rand_for_no_last_crop.predict(X_test_no_last_crop)
accuracy_no_last = accuracy_score(y_test, y_pred_no_last)
print(rand_for_no_last_crop.feature_importances_)

# Confusion matrix for no_last
cm_no_last = confusion_matrix(y_test, y_pred_no_last)
# Save this model
joblib.dump(rand_for_no_last_crop, "no_last_crop_forest.pkl")

# Try XGBoost instead
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

X_train_no_last_crop = X_train_no_last_crop.reset_index()
X_train_no_last_crop = X_train_no_last_crop.iloc[:, 1:]

# Define classifier.
XGB_clf = XGBClassifier(learning_rate=0.02, n_estimators=200, silent=True,
                        objective="binary:logistic", scoring="roc_auc")

# Fit the classifier to the training set.
XGB_clf.fit(X_train_no_last_crop, y_train)
y_pred_XGB = XGB_clf.predict(X_test_no_last_crop)
XGB_accuracy = accuracy_score(y_test, y_pred_XGB)
XGB_precision = precision_score(y_test, y_pred_XGB, average="weighted")
XGB_recall = recall_score(y_test, y_pred_XGB, average="weighted")
cm_XGB = confusion_matrix(y_test, y_pred_XGB)

# Get classification accuracy by class for xgboost.
# Approx. 93% accuracy.
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred_XGB))

# Add the prediction to the column.
X_test_no_last_crop["summer_or_winter"] = y_pred_XGB

###############################################################################
# Create columns for X_train.
###############################################################################
# For step by step classification.
train_df["2a"] = 0
train_df["2b"] = 0
train_df["3c"] = 0
train_df["4a"] = 0

# Give the columns their corresponding values.
# Labels for step 2a.
train_df["2a"] = (train_df["label"] == "WW") | (train_df["label"] == "WR") | (train_df["label"] == "WB") | (train_df["label"] == "TC") | (train_df["label"] == "SP")
# Labels for step 2b.
train_df["2b"] = (train_df["label"] == "WR") | (train_df["label"] == "TC")
# Labels for step 3c.
train_df["3c"] = (train_df["label"] == "M") | (train_df["label"] == "B") | (train_df["label"] == "SB")
# Labels for step 4a.
train_df["4a"] = (train_df["label"] == "WB") | (train_df["label"] == "WW")

# Encode df values.
# Encode labels.
from sklearn.preprocessing import LabelEncoder
# Get X and y training data
train_final = train_df[["day_of_year", "month", "g", "nir", "ndvi", "2a", "2b",
                          "3c", "4a", "label"]]

# Encode train labels.
encoder = LabelEncoder()
encoder.fit(train_final["label"])
train_final["label"] = encoder.transform(train_final["label"])

###############################################################################
# Do the same as above for the test dataset.
###############################################################################
# For step by step classification.
test_df["2a"] = 0
test_df["2b"] = 0
test_df["3c"] = 0
test_df["4a"] = 0

# Give the columns their corresponding values.
# Labels for step 2a.
test_df["2a"] = (test_df["label"] == "WW") | (test_df["label"] == "WR") | (test_df["label"] == "WB") | (test_df["label"] == "TC") | (test_df["label"] == "SP")
# Labels for step 2b.
test_df["2b"] = (test_df["label"] == "WR") | (test_df["label"] == "TC")
# Labels for step 3c.
test_df["3c"] = (test_df["label"] == "M") | (test_df["label"] == "B") | (test_df["label"] == "SB")
# Labels for step 4a.
test_df["4a"] = (test_df["label"] == "WB") | (test_df["label"] == "WW")

# Encode labels.
from sklearn.preprocessing import LabelEncoder
# Get X and y training data
test_final = test_df[["day_of_year", "month", "g", "nir", "ndvi", "2a", "2b",
                          "3c", "4a", "label"]]
# Encode train labels.
test_final["label"] = encoder.transform(test_final["label"])

# Labels are as follows:
'''
0: B
1: M
2: O
3: P
4: SB
5: WB
6: WR
7: WW
'''
###############################################################################
# Train the classifier for 2a (WR vs WW and WB)
###############################################################################
for_clf_2a = RandomForestClassifier(bootstrap=True, n_estimators=500, random_state=42)
train_2a = train_final.copy()
# Create training data only with relevant crops.
train_2a = train_2a[train_2a["2a"] == True]
train_2a["label"] = train_2a["label"].where(train_2a["label"] == 6, 0)

# Limit the dates.
train_2a = train_2a[(train_2a["day_of_year"] >= 120) & (train_2a["day_of_year"] <= 160)]
X_train_2a = train_2a.iloc[:, :5]
y_train_2a = train_2a.iloc[:, -1]
# Fit the classifier.
for_clf_2a.fit(X_train_2a, y_train_2a[:, np.newaxis])
# Prepare test dataset.
test_2a = test_final.copy()
# Create training data only with relevant crops.
test_2a = test_2a[test_2a["2a"] == True]
test_2a["label"] = test_2a["label"].where(test_2a["label"] == 6, 1)

# Limit the dates.
test_2a = test_2a[(test_2a["day_of_year"] >= 120) & (test_2a["day_of_year"] <= 180)]
X_test_2a = test_2a.iloc[:, :5]
y_test_2a = test_2a.iloc[:, -1]
# Predict y for 2a.
y_pred_2a = for_clf_2a.predict(X_test_2a)
# Accuracy.
accuracy_2a = accuracy_score(y_test_2a, y_pred_2a)
cm_2a = confusion_matrix(y_test_2a, y_pred_2a)
# 90% accuracy.

###############################################################################
# Train the classifier for 3c (B from M)
###############################################################################
for_clf_3c = RandomForestClassifier(bootstrap=True, n_estimators=500, random_state=42)
train_3c = train_final.copy()
# Create training data only with relevant crops.
train_3c = train_3c[train_3c["3c"] == True]
train_3c["label"] = train_3c["label"].where(train_3c["label"] == 1, 0)

# Limit the dates.
train_3c = train_3c[(train_3c["day_of_year"] >= 180) & (train_3c["day_of_year"] <= 210)]
X_train_3c = train_3c.iloc[:, 3:5]
y_train_3c = train_3c.iloc[:, -1]
# Fit the classifier.
for_clf_3c.fit(X_train_3c, y_train_3c[:, np.newaxis])
# Prepare test dataset.
test_3c = test_final.copy()
# Create training data only with relevant crops.
test_3c = test_3c[test_3c["3c"] == True]
test_3c["label"] = test_3c["label"].where(test_3c["label"] == 1, 0)

# Limit the dates.
test_3c = test_3c[(test_3c["day_of_year"] >= 180) & (test_3c["day_of_year"] <= 210)]
X_test_3c = test_3c.iloc[:, 3:5]
y_test_3c = test_3c.iloc[:, -1]
# Predict y for 2a.
y_pred_3c = for_clf_3c.predict(X_test_3c)
# Accuracy.
accuracy_3c = accuracy_score(y_test_3c, y_pred_3c)
cm_3c = confusion_matrix(y_test_3c, y_pred_3c)
# 69% Accuracy.

###############################################################################
# Train the classifier for 4a (WB, WW)
###############################################################################
for_clf_4a = RandomForestClassifier(bootstrap=True, n_estimators=500, random_state=42)
train_4a = train_final.copy()
# Create training data only with relevant crops.
train_4a = train_4a[train_4a["4a"] == True]
train_4a["label"] = train_4a["label"].where(train_4a["label"] == 7, 0)

# Limit the dates.
train_4a = train_4a[(train_4a["day_of_year"] >= 90) & (train_4a["day_of_year"] <= 115)]
X_train_4a = train_4a.iloc[:, 3:5]
y_train_4a = train_4a.iloc[:, -1]
# Fit the classifier.
for_clf_4a.fit(X_train_4a, y_train_4a[:, np.newaxis])
# Prepare test dataset.
test_4a = test_final.copy()
# Create training data only with relevant crops.
test_4a = test_4a[test_4a["4a"] == True]
test_4a["label"] = test_4a["label"].where(test_4a["label"] == 7, 0)

# Limit the dates.
test_4a = test_4a[(test_4a["day_of_year"] >= 90) & (test_4a["day_of_year"] <= 115)]
X_test_4a = test_4a.iloc[:, 3:5]
y_test_4a = test_4a.iloc[:, -1]
# Predict y for 2a.
y_pred_4a = for_clf_4a.predict(X_test_4a)
# Accuracy.
accuracy_4a = accuracy_score(y_test_4a, y_pred_4a)
cm_4a = confusion_matrix(y_test_4a, y_pred_4a)
# 90% accuracy (very wheat bias).