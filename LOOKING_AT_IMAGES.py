# Import libraries.
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pandas as pd
import datetime
import os

# Set working directory.
os.chdir("C:/Users/Tim/Desktop/GIS/GISproject")
# Constants.
# Column order for the df.
columns = ["date", "day_of_year", "month", "year", "last_crop", "r", "g", "b", "label"]
df = pd.DataFrame(columns=columns)
field_areas = ["EC1", "EC2", "EC3", "EC4", "EC5", "EC6", "Cloud", "CloudShadow"]

# Function for getting the label.
def get_label(field_area, date):
    '''
    Returns the label of an image based on the csv that shows which label is in
    which area by time.
    '''
    # Field EC1 - Kraichgau.
    if field_area == "EC1":
        if date < datetime.datetime(2010, 12, 1):
            label = "SM"
            last_crop = "unknown"
        elif date < datetime.datetime(2011, 7, 1):
            label = "WW"
            last_crop = "SM"
        elif date < datetime.datetime(2012, 4, 1):
            label = "WR"
            last_crop = "WW"
        elif date < datetime.datetime(2013, 7, 1):
            label = "WW"
            last_crop = "WR"
        elif date < datetime.datetime(2014, 12, 1):
            label = "CC-SM"
            last_crop = "WW"
        elif date < datetime.datetime(2015, 7, 1):
            label = "WW"
            last_crop = "CC-SM"
        elif date < datetime.datetime(2016, 12, 1):
            label = "CC-GM"
            last_crop = "WW"
        elif date < datetime.datetime(2017, 7, 1):
            label = "WW"
            last_crop = "CC-GM"
        elif date < datetime.datetime(2018, 4, 1):
            label = "WR"
            last_crop = "WW"
    
    # Field EC2 - Kraichgau.
    elif field_area == "EC2":
        if date < datetime.datetime(2010, 4, 1):
            label = "WR"
            last_crop = ""
        elif date < datetime.datetime(2011, 7, 1):
            label = "WW"
            last_crop = "WR"
        elif date < datetime.datetime(2012, 12, 1):
            label = "CC-SM"
            last_crop = "WW"
        elif date < datetime.datetime(2013, 7, 1):
            label = "WW"
            last_crop = "CC-SM"
        elif date < datetime.datetime(2014, 12, 1):
            label = "CC-SM"
            last_crop = "WW"
        elif date < datetime.datetime(2015, 7, 1):
            label = "WW"
            last_crop = "CC-SM"
        elif date < datetime.datetime(2016, 4, 1):
            label = "WR"
            last_crop = "WW"
        elif date < datetime.datetime(2017, 7, 1):
            label = "WW"
            last_crop = "WR"
        elif date < datetime.datetime(2018, 7, 1):
            label = "WW"
            last_crop = "WW"
    
    # Field EC3 - Kraichgau.
    elif field_area == "EC3":
        if date < datetime.datetime(2010, 7, 1):
            label = "WW"
            last_crop = "unknown"
        elif date < datetime.datetime(2011, 12, 1):
            label = "CC-SM"
            last_crop = "WW"
        elif date < datetime.datetime(2012, 7, 1):
            label = "WW"
            last_crop = "CC-SM"
        elif date < datetime.datetime(2013, 4, 1):
            label = "WR"
            last_crop = "WW"
        elif date < datetime.datetime(2014, 7, 1):
            label = "WW"
            last_crop = "WR"
        elif date < datetime.datetime(2015, 12, 1):
            label = "CC-SM"
            last_crop = "WW"
        elif date < datetime.datetime(2016, 7, 1):
            label = "WW"
            last_crop = "CC-SM"
        elif date < datetime.datetime(2017, 7, 1):
            label = "WW"
            last_crop = "WW"
        elif date < datetime.datetime(2018, 12, 1):
            label = "CC-SM"
            last_crop = "WW"
         
    # Field EC4 - Swaebisch Alp.
    elif field_area == "EC4":
        if date < datetime.datetime(2010, 1, 1):
            label = "WR"
            last_crop = "unknown"
        elif date < datetime.datetime(2011, 1, 1):
            label = "WW"
            last_crop = "WR"
        elif date < datetime.datetime(2012, 1, 1):
            label = "CC-SB"
            last_crop = "WW"
        elif date < datetime.datetime(2013, 1, 1):
            label = "WR"
            last_crop = "CC-SB"
        elif date < datetime.datetime(2014, 1, 1):
            label = "WW"
            last_crop = "WR"
        elif date < datetime.datetime(2015, 1, 1):
            label = "WW" 
            last_crop = "WW"
        elif date < datetime.datetime(2016, 1, 1):
            label = "CC-SB"
            last_crop = "WW"
        elif date < datetime.datetime(2017, 1, 1):
            label = "CC-SM"
            last_crop = "CC-SB"
        elif date < datetime.datetime(2018, 1, 1):
            label = "WW"
            last_crop = "CC-SM"
            
    # Field EC5 - Swaebisch Alp.
    elif field_area == "EC5":
        if date < datetime.datetime(2010, 1, 1):
            label = "WW"
            last_crop = "unknown"
        elif date < datetime.datetime(2011, 1, 1):
            label = "CC-SM"
            last_crop = "WW"
        elif date < datetime.datetime(2012, 1, 1):
            label = "SM"
            last_crop = "CC-SM"
        elif date < datetime.datetime(2013, 1, 1):
            label = "WB"
            last_crop = "SM"
        elif date < datetime.datetime(2014, 1, 1):
            label = "SP"
            last_crop = "WB"
        elif date < datetime.datetime(2015, 1, 1):
            label = "CC-SM"
            last_crop = "SP"
        elif date < datetime.datetime(2016, 1, 1):
            label = "CC-SM"
            last_crop = "CC-SM"
        elif date < datetime.datetime(2017, 1, 1):
            label = "WB"
            last_crop = "CC-SM"
        elif date < datetime.datetime(2018, 1, 1):
            label = "WR"
            last_crop = "WB"
            
    # Field EC6 - Swaebisch Alp.
    elif field_area == "EC6":
        if date < datetime.datetime(2010, 1, 1):
            label = "CC-SM"
            last_crop = "unknown"
        elif date < datetime.datetime(2011, 1, 1):
            label = "WW"
            last_crop = "CC-SM"
        elif date < datetime.datetime(2012, 1, 1):
            label = "WB"
            last_crop = "WW"
        elif date < datetime.datetime(2013, 1, 1):
            label = "CC-SM"
            last_crop = "WB"
        elif date < datetime.datetime(2014, 1, 1):
            label = "WW"
            last_crop = "CC-SM"
        elif date < datetime.datetime(2015, 1, 1):
            label = "WB"
            last_crop = "WW"
        elif date < datetime.datetime(2016, 1, 1):
            label = "CC-SM"
            last_crop = "WB"
        elif date < datetime.datetime(2017, 1, 1):
            label = "WW"
            last_crop = "CC-SM"
        elif date < datetime.datetime(2018, 1, 1):
            label = "WB"
            last_crop = "WW"
            
    elif field_area == "Cloud":
        label = "Cloud"
        last_crop = "unknown"
    elif field_area == "CloudShadow":
        label = "CloudShadow"
        last_crop = "unknown"


            
            
    return label, last_crop
            
            
    
    
    
# Loop for getting all files into the dataframe, labelled.
for field in field_areas:
    path = "C:/Users/Tim/Desktop/GIS/GISproject/" + field
    for file in os.listdir(path):
        if file.endswith('.tif'):
            # The year, month and day time.
            date = file[:-4]
            date = pd.to_datetime(date, format="%Y%m%d")
            # What day of the year is it out of 365/366.
            day_of_year = datetime.datetime.timetuple(date).tm_yday
            month = datetime.datetime.date(date).month
            year = datetime.datetime.date(date).year
            # Get the label based on csv.
            label, last_crop = get_label(field, date)
            # Load the training image.
            im = cv2.imread(r'C:\Users\Tim\Desktop\GIS\GISproject\\' + field + "\\" + file, 1)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            img = np.array(im)
            # Get the rgb values out of the image.
            r = img[:, :, 0]
            g = img[:, :, 1]
            b = img[:, :, 2]
            # Turn the 2D image data into a 1D series.
            r = r.ravel()
            g = g.ravel()
            b = b.ravel()
            plt.imshow(im)
            plt.show()
            
            # Create the secondary dataframe to append to the full dataframe.
            data = {"date": date, "day_of_year": day_of_year, "month": month, 
            "year": year, "last_crop": last_crop, "r": r, "g": g, "b": b, "label": label}
            df_iter = pd.DataFrame(data=data)
            df_iter = df_iter[columns]
            # Append secondary dataframe to the full dataframe.
            df = df.append(df_iter)
            
# Drop the black pixels in the dataframe.
df = df[(df.r != 0) & (df.g != 0) & (df.b != 0)]


# Predict with SVM.
# Import libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
# Get X and y data.
X = df.iloc[:, 2:-1]
y = df.iloc[:, -1]

# Encode the y labels to onehotencoded values.
# Encode.
encoder = LabelEncoder()
encoder.fit(y)
y = encoder.transform(y)

X_encoder = LabelEncoder()
X_encode_values = X.last_crop
X_encoder.fit(X_encode_values)
X_encode_values = X_encoder.transform(X_encode_values)
X.last_crop = X_encode_values

# Feature scale data.
#from sklearn.preprocessing import StandardScaler
#sc_X = StandardScaler()
#X = sc_X.fit_transform(X)


# Train-test split.
X_train, X_test, y_train, y_test = train_test_split(X, y)

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
rand_for = RandomForestClassifier(bootstrap=True, n_estimators=10, random_state=42)
rand_for.fit(X_train, y_train)



# SVM?
svm = SVC(kernel="rbf")
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)

# Feature importances.
print(rand_for.feature_importances_)

# Predict values
y_pred = rand_for.predict(X_test)

# Get scores of the classifier.
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted")
recall = recall_score(y_test, y_pred, average="weighted")

# Get scores of svm.
svm_acc = accuracy_score(y_test, y_pred_svm)
svm_precision = precision_score(y_test, y_pred_svm, average="weighted")
svm_recall = recall_score(y_test, y_pred_svm, average="weighted")
# Confusion matrix.
cm = confusion_matrix(y_test, y_pred)

# Turn encoded values back to non-encoded for comparison
y_test_text = list(encoder.inverse_transform(y_test))
y_pred_text = list(encoder.inverse_transform(y_pred))

# Save the model.
from sklearn.externals import joblib
joblib.dump(rand_for, "random_forest_2.pkl")
joblib.dump(svm, "svm.pkl")

# Random forest for without last crop info (1st classification)
rand_for_no_last_crop = RandomForestClassifier(bootstrap=True, n_estimators=10, random_state=42)
X_no_last_crop = X[["month", "year", "r", "g", "b"]]
X_no_last_crop_train, X_no_last_crop_test, y_train, y_test = train_test_split(X_no_last_crop, y)
rand_for_no_last_crop.fit(X_no_last_crop_train, y_train)
y_pred_no_last = rand_for_no_last_crop.predict(X_no_last_crop_test)
accuracy_no_last = accuracy_score(y_test, y_pred_no_last)

# Save this model
joblib.dump(rand_for_no_last_crop, "no_last_crop_forest.pkl")