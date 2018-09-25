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
from get_label import get_label

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

# Set working directory.
os.chdir("C:/Users/Tim/Desktop/GIS/GISproject")
# Constants.
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
           "r", "nir", "swir1",  "swir2", "pan", "cir", "tirs1", "tirs2", 
           "label", "binary_label"]
train_df = pd.DataFrame(columns=columns)
test_df = pd.DataFrame(columns=columns)

field_areas = ["EC1", "EC2", "EC3", "EC4", "EC5", "EC6"]
summer_crops = ["SM, CC-SM", "CC-SB", "SP", "CC-GM"]
winter_crops = ["WW", "WB", "WR"]
              
# Loop for getting all files into the dataframe, labelled.
for field in field_areas:
    path = "C:/Users/Tim/Desktop/GIS/GISproject/landsat_8/" + field
    for band in os.listdir(path):
        for file in os.listdir(path + "/" + band):
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
                label, last_crop = get_label(field, date)
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
                plt.imshow(b)
                plt.show()
                plt.imshow(tirs2)
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
                        "tirs2": tirs2, "label": label, 
                        "binary_label": summer_or_winter_crop}
                
                df_iter = pd.DataFrame(data=data)
                df_iter = df_iter[columns]
                # Append secondary dataframe to the full dataframe.
                # Choose which df to add to.
                random_number = random.randint(1, 100)
                if random_number <= 20:
                    test_df = test_df.append(df_iter)
                else:
                    train_df = train_df.append(df_iter)
    #            if field in test_field_areas:
    #                test_df = test_df.append(df_iter)
    #            else:
    #                train_df = train_df.append(df_iter)
                        
# Drop the black pixels in the dataframe.
train_df = train_df[(train_df.r != 0) & (train_df.g != 0) & (train_df.b != 0)]
test_df = test_df[(test_df.r != 0) & (test_df.g != 0) & (test_df.b != 0)]

# Drop values whose date is between September and March.
train_df = train_df[train_df.label != "irrelevant"]
test_df = test_df[test_df.label != "irrelevant"]

## Downsample WW, SM and Cloud.
#from sklearn.utils import resample
#train_df_cloud = train_df[train_df.label == "Cloud"]
#train_df_SM = train_df[train_df.label == "SM"]
#train_df_WW = train_df[train_df.label == "WW"]
#
## Downsample Cloud.
#train_df_cloud = resample(train_df_cloud,
#                          replace=False,
#                          n_samples=3000,
#                          random_state=42)
## Downsample SM.
#train_df_SM = resample(train_df_SM,
#                          replace=False,
#                          n_samples=3000,
#                          random_state=42)
## Downsample WW.
#train_df_WW = resample(train_df_WW,
#                          replace=False,
#                          n_samples=3000,
#                          random_state=42)
#
## Remove the above from train df.
#train_df = train_df[train_df.label != "Cloud"]
#train_df = train_df[train_df.label != "SM"]
#train_df = train_df[train_df.label != "WW"]
#
## Append the resampled dfs to train df.
#train_df = pd.concat([train_df, train_df_cloud])
#train_df = pd.concat([train_df, train_df_SM])
#train_df = pd.concat([train_df, train_df_WW])


# Import libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

# Get X and y training data
X_train = train_df.iloc[:, 1:-2]
y_train = train_df.iloc[:, -2]

# Encode the y labels to onehotencoded values.
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
# Get X and y training data
X_test = test_df.iloc[:, 1:-2]
y_test = test_df.iloc[:, -2]
y_test = encoder.transform(y_test)

# Encode the y labels to onehotencoded values.
X_test_encoder = LabelEncoder()
X_test_encode_values = X_test.last_crop
X_test_encoder.fit(X_test_encode_values)
X_test_encode_values = X_test_encoder.transform(X_test_encode_values)
X_test.last_crop = X_test_encode_values

# Feature scale data.
#from sklearn.preprocessing import StandardScaler
#sc_X = StandardScaler()
#X = sc_X.fit_transform(X)


# Drop the NaN values.
X_train = X_train.dropna()
X_test.dropna()

# Train a random forest.
rand_for = RandomForestClassifier()

# This has already been done
# Optimise the classifier with a grid search.
#from sklearn.model_selection import GridSearchCV
#param_grid = [{'n_estimators': [3, 5, 10, 30, 100, 200, 500], 
#             'bootstrap':[False, True]}]
#grid_search = GridSearchCV(rand_for, param_grid, cv=5, scoring="accuracy")
#grid_search.fit(X_train, y_train)

# Best params?
#best_params = grid_search.best_params_


# Fit the classifier.
rand_for = RandomForestClassifier(bootstrap=True, n_estimators=10, random_state=42,
                                  criterion="entropy")
rand_for.fit(X_train, y_train)

# Feature importances.
print(rand_for.feature_importances_)

# Predict values
y_pred = rand_for.predict(X_test)

# Get scores of the classifier.
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
rand_for_no_last_crop = RandomForestClassifier(bootstrap=True, n_estimators=10, random_state=42)
X_train_no_last_crop = X_train[["month", "year", "ca", "b", "g",
           "r", "nir", "swir1",  "swir2", "pan", "cir", "tirs1", "tirs2"]]
X_test_no_last_crop = X_test[["month", "year", "ca", "b", "g",
           "r", "nir", "swir1",  "swir2", "pan", "cir", "tirs1", "tirs2"]]
rand_for_no_last_crop.fit(X_train_no_last_crop, y_train)
y_pred_no_last = rand_for_no_last_crop.predict(X_test_no_last_crop)
accuracy_no_last = accuracy_score(y_test, y_pred_no_last)
print(rand_for_no_last_crop.feature_importances_)

# Confusion matrix for no_last
cm_no_last = confusion_matrix(y_test, y_pred_no_last)
# Save this model
joblib.dump(rand_for_no_last_crop, "no_last_crop_forest.pkl")

# Train and test a naive bayes classifier.
NB_classifier = GaussianNB()
NB_classifier.fit(X_train_no_last_crop, y_train)
y_pred_NB = NB_classifier.predict(X_test_no_last_crop)
NB_accuracy = accuracy_score(y_test, y_pred_NB)
cm_NB = confusion_matrix(y_test, y_pred_NB)
joblib.dump(NB_classifier, "NB_classifier.pkl")

# Try XGBoost instead
from xgboost import XGBClassifier
XGB_clf = XGBClassifier(objective="multi:softmax")
XGB_clf.fit(X_train_no_last_crop, y_train)
y_pred_XGB = XGB_clf.predict(X_test_no_last_crop)
XGB_accuracy = accuracy_score(y_test, y_pred_XGB)
XGB_precision = precision_score(y_test, y_pred_XGB, average="weighted")
XGB_recall = recall_score(y_test, y_pred_XGB, average="weighted")
cm_XGB = confusion_matrix(y_test, y_pred_XGB)

# Try Adaboost
from sklearn.ensemble import AdaBoostClassifier
ada_clf = AdaBoostClassifier()
ada_clf.fit(X_train_no_last_crop, y_train)
y_pred_ada = ada_clf.predict(X_test_no_last_crop)
ada_accuracy = accuracy_score(y_test, y_pred_ada)
joblib.dump(ada_clf, "ada_clf.pkl")
