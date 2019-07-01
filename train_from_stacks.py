# Import libraries.
import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import os
os.chdir(r"C:/Users/Tim/Desktop/BOKU/GIS/GISproject/")
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from get_label import get_label
from xgboost import XGBClassifier
# Set path
path = r"C:/Users/Tim/Desktop/BOKU/GIS/GISproject/stacks/"
columns = [str(i+1) + "_" + str(j) for i in range(7) for j in range(14)]
columns = list(filter(lambda x: x.split("_")[0] != "6", columns))
columns = columns + ["Date", "Summer_Or_Winter", "Label", "Tile"]
years = ["2008", "2009", "2010", "2011", "2012", 
         "2013", "2014", "2015",
         "2016", "2017", "2018"]
bands = ["1", "2", "3", "4", "5", "7"]
tiles = ["EC1", "EC2", "EC3", "EC4", "EC5", "EC6", "1_1", "3", "5",
               "6", "8", "9", "11", "13", "15", "17", "18_1", "18_4", "19",
               "20", "21", "23", "25", "26", "27", "28", "29", "30", "33",
               #"final_farmland_polygon"
               ]
summer_crops = ["B", "M", "SB", "GM", "SM"]
winter_crops = ["WW", "WB", "WR", "P", "SP" "TC", "O"]

df = pd.DataFrame(columns=columns)
temp_df = pd.DataFrame(columns=columns)

for year in years:
    for tile in tiles:
        for band in bands:
            date = year + "0601"
            date = pd.to_datetime(date, format="%Y%m%d")
            array = np.load(path + "stack-" + tile + "-" + band + "-" + year 
                            + ".npy")
            mask = np.load(path + "stack-" + tile + "-8-" + year + ".npy")
            mask = np.resize(mask, array.shape)
            #print(mask.shape, array.shape)
            if tile == "final_farmland_polygon":
                array = np.resize(array, (300, 300, 14))
                mask = np.resize(mask, (300, 300, 14))
            array[mask >= 400] = 0
        
            for i in range(array.shape[2]):
                temp_df[band + "_" + str(i)] = array[:, :, i].ravel()
                
        temp_df["Date"] = date
        temp_df["Summer_Or_Winter"] = 0
        label = get_label(tile, date)
        temp_df["Label"] = label[0]
        if label[0] in summer_crops:
            temp_df["Summer_Or_Winter"] = 0
        else:
            temp_df["Summer_Or_Winter"] = 1
        temp_df["Tile"] = tile
        #temp_df = temp_df.dropna()
        df = df.append(temp_df)
        temp_df = pd.DataFrame(columns=columns)
        
df_ammertal = df[df["Tile"] == "final_farmland_polygon"]
df = df[df["Tile"] != "final_farmland_polygon"]

df = df.dropna(thresh=5)
#df = df.interpolate(axis=1, method='linear')
#df = df.interpolate(axis=1, limit_direction='backward',
#                    method='linear', order=2)
df = df[columns]
df = df.reset_index()
df = df[df["Label"] != "irrelevant"]
df = df.reset_index()
df = df.iloc[:, 1:]
encoder = LabelEncoder()
encoder.fit(df["Label"])
df["Label"] = encoder.transform(df["Label"])

y = df.iloc[:, -2]
X = df.iloc[:, :-4]
X_train, X_test, y_train, y_test = train_test_split(X, y)

rand_for = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)
rand_for.fit(X_train, y_train)
xgb = XGBClassifier(n_estimators=200, n_jobs=-1)
xgb.fit(X_train, y_train)
print(rand_for.feature_importances_)
# Predict values
y_pred = rand_for.predict(X_test)
y_pred_xgb = xgb.predict(X_test)

# Get scores of the classifier.
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
accuracy = accuracy_score(y_test, y_pred)
accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
precision = precision_score(y_test, y_pred, average="weighted")
precision_xgb = precision_score(y_test, y_pred_xgb, average="weighted")
recall = recall_score(y_test, y_pred, average="weighted")
recall_xgb = recall_score(y_test, y_pred_xgb, average="weighted")


# Confusion matrix.
cm = confusion_matrix(y_test, y_pred)
cm_xgb = confusion_matrix(y_test, y_pred_xgb)

example_wheat = df.iloc[304, :-4].values
example_maize = df.iloc[62, :-4].values
plt.figure(figsize=(12, 12))
plt.plot(example_wheat, color='green', marker='x')
plt.plot(example_maize, color='red', marker='o')
plt.show()

# Test summer or winter.
y = df.iloc[:, -3].astype("int")
X = df.iloc[:, :-4]
X_train, X_test, y_train, y_test = train_test_split(X, y)
rand_for_summer = RandomForestClassifier(n_estimators=200, 
                                         n_jobs=-1, 
                                         random_state=42)
xgb_summer = XGBClassifier(n_estimators=200,
                           n_jobs=-2)
rand_for_summer.fit(X_train, y_train)
xgb_summer.fit(X_train, y_train)
print(rand_for_summer.feature_importances_)
# Predict values
y_pred = rand_for_summer.predict(X_test)
y_pred_xgb = xgb_summer.predict(X_test)

# Get scores of the classifier.
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
accuracy_summer = accuracy_score(y_test, y_pred)
accuracy_summer_xgb = accuracy_score(y_test, y_pred_xgb)
precision_summer = precision_score(y_test, y_pred, average="weighted")
precision_summer_xgb = precision_score(y_test, y_pred_xgb, average="weighted")
recall_summer = recall_score(y_test, y_pred, average="weighted")
recall_summer_xgb = recall_score(y_test, y_pred_xgb, average="weighted")

# Confusion matrix.
cm_summer = confusion_matrix(y_test, y_pred)
cm_summer_xgb = confusion_matrix(y_test, y_pred_xgb)


# Prepare df for preds.
df_ammertal = df_ammertal.fillna(0)
preds_ammertal = rand_for.predict(df_ammertal.iloc[:, :-4])
preds_ammertal = preds_ammertal.reshape((330, 300))

preds_summer_ammertal = rand_for_summer.predict(df_ammertal.iloc[:, :-4])
counts_summer = np.unique(preds_summer_ammertal, return_counts=True)