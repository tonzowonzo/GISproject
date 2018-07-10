# Import libraries.
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pandas as pd
import datetime
import os

# Constants.
# Column order for the df.
columns = ["date", "day_of_year", "r", "g", "b", "label"]
df = pd.DataFrame(columns=columns)
field_areas = ["EC1", "EC2", "EC3", "Cloud", "CloudShadow", "Urban", "Water"]

# Function for getting the label.
def get_label(field_area, date):
    '''
    Returns the label of an image based on the csv that shows which label is in
    which area by time.
    '''
    # Field EC1 - Kraichgau.
    if field_area == "EC1":
        if date < datetime.datetime(2010, 1, 1):
            label = "SM"
        elif date < datetime.datetime(2011, 1, 1):
            label = "WW"
        elif date < datetime.datetime(2012, 1, 1):
            label = "WR"
        elif date < datetime.datetime(2013, 1, 1):
            label = "WW"
        elif date < datetime.datetime(2014, 1, 1):
            label = "CC-SM"
        elif date < datetime.datetime(2015, 1, 1):
            label = "WW"
        elif date < datetime.datetime(2016, 1, 1):
            label = "CC-GM"
        elif date < datetime.datetime(2017, 1, 1):
            label = "WW"
        elif date < datetime.datetime(2018, 1, 1):
            label = "WR"
    
    # Field EC2 - Kraichgau.
    elif field_area == "EC2":
        if date < datetime.datetime(2010, 1, 1):
            label = "WR"
        elif date < datetime.datetime(2011, 1, 1):
            label = "WW"
        elif date < datetime.datetime(2012, 1, 1):
            label = "CC-SM"
        elif date < datetime.datetime(2013, 1, 1):
            label = "WW"
        elif date < datetime.datetime(2014, 1, 1):
            label = "CC-SM"
        elif date < datetime.datetime(2015, 1, 1):
            label = "WW"
        elif date < datetime.datetime(2016, 1, 1):
            label = "WR"
        elif date < datetime.datetime(2017, 1, 1):
            label = "WW"
        elif date < datetime.datetime(2018, 1, 1):
            label = "WW"
    
    # Field EC3 - Kraichgau.
    elif field_area == "EC3":
        if date < datetime.datetime(2010, 1, 1):
            label = "WW"
        elif date < datetime.datetime(2011, 1, 1):
            label = "CC-SM"
        elif date < datetime.datetime(2012, 1, 1):
            label = "WW"
        elif date < datetime.datetime(2013, 1, 1):
            label = "WR"
        elif date < datetime.datetime(2014, 1, 1):
            label = "WW"
        elif date < datetime.datetime(2015, 1, 1):
            label = "CC-SM"
        elif date < datetime.datetime(2016, 1, 1):
            label = "WW"
        elif date < datetime.datetime(2017, 1, 1):
            label = "WW"
        elif date < datetime.datetime(2018, 1, 1):
            label = "CC-SM"
         
    # Field EC4 - Swaebisch Alp.
    elif field_area == "EC4":
        if date < datetime.datetime(2010, 1, 1):
            label = "WR"
        elif date < datetime.datetime(2011, 1, 1):
            label = "WW"
        elif date < datetime.datetime(2012, 1, 1):
            label = "CC-SB"
        elif date < datetime.datetime(2013, 1, 1):
            label = "WR"
        elif date < datetime.datetime(2014, 1, 1):
            label = "WW"
        elif date < datetime.datetime(2015, 1, 1):
            label = "WW"
        elif date < datetime.datetime(2016, 1, 1):
            label = "CC-SB"
        elif date < datetime.datetime(2017, 1, 1):
            label = "CC-SM"
        elif date < datetime.datetime(2018, 1, 1):
            label = "WW"
            
    # Field EC5 - Swaebisch Alp.
    elif field_area == "EC5":
        if date < datetime.datetime(2010, 1, 1):
            label = "WW"
        elif date < datetime.datetime(2011, 1, 1):
            label = "CC-SM"
        elif date < datetime.datetime(2012, 1, 1):
            label = "SM"
        elif date < datetime.datetime(2013, 1, 1):
            label = "WB"
        elif date < datetime.datetime(2014, 1, 1):
            label = "SP"
        elif date < datetime.datetime(2015, 1, 1):
            label = "CC-SM"
        elif date < datetime.datetime(2016, 1, 1):
            label = "CC-SM"
        elif date < datetime.datetime(2017, 1, 1):
            label = "WB"
        elif date < datetime.datetime(2018, 1, 1):
            label = "WR"
            
    # Field EC6 - Swaebisch Alp.
    elif field_area == "EC6":
        if date < datetime.datetime(2010, 1, 1):
            label = "CC-SM"
        elif date < datetime.datetime(2011, 1, 1):
            label = "WW"
        elif date < datetime.datetime(2012, 1, 1):
            label = "WB"
        elif date < datetime.datetime(2013, 1, 1):
            label = "CC-SM"
        elif date < datetime.datetime(2014, 1, 1):
            label = "WW"
        elif date < datetime.datetime(2015, 1, 1):
            label = "WB"
        elif date < datetime.datetime(2016, 1, 1):
            label = "CC-SM"
        elif date < datetime.datetime(2017, 1, 1):
            label = "WW"
        elif date < datetime.datetime(2018, 1, 1):
            label = "WB"
            
    elif field_area == "Cloud":
        label = "Cloud"
        
    elif field_area == "CloudShadow":
        label = "CloudShadow"
        
    elif field_area == "Urban":
        label = "Urban"
        
    elif field_area == "Water":
        label = "Water"
            
            
    return label
            
            
    
    
    
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
            # Get the label based on csv.
            label = get_label(field, date)
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
            data = {"date": date, "day_of_year": day_of_year, "r": r, "g": g, "b": b, "label": label}
            df_iter = pd.DataFrame(data=data)
            df_iter = df_iter[columns]
            # Append secondary dataframe to the full dataframe.
            df = df.append(df_iter)
            
            
# Predict with SVM.
# Import libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix
# Get X and y data.
X = df.iloc[:, 1:-1]
y = df.iloc[:, -1]

# Encode the y labels to onehotencoded values.
# Encode.
encoder = LabelEncoder()
encoder.fit(y)
y = encoder.transform(y)

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
rand_for = RandomForestClassifier(bootstrap=True, n_estimators=200)
rand_for.fit(X_train, y_train)

# Feature importances.
print(rand_for.feature_importances_)

# Predict values
y_pred = rand_for.predict(X_test)

# Get scores of the classifier.
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
# Confusion matrix.
cm = confusion_matrix(y_test, y_pred)

# Turn encoded values back to non-encoded for comparison
y_test = list(encoder.inverse_transform(y_test))
y_pred = list(encoder.inverse_transform(y_pred))