# Import libraries.
import pandas as pd # For dataframes.
import numpy as np # For array manipulation.
import sklearn # For machine learning algorithms.
import matplotlib.pyplot as plt # For plotting data.
import os # For path management.
import cv2 # For loading images.
import random
os.chdir(r"C:/Users/Tim/Desktop/BOKU/GIS/GISproject/")

from sklearn.preprocessing import LabelEncoder # For encoding string labels into numbers.
from sklearn.model_selection import train_test_split # For splitting up data.
from sklearn.ensemble import RandomForestClassifier # Random Forest.
from get_label import get_label # Label getter for each field area.
from xgboost import XGBClassifier # XGBoost classifer.
from sklearn.externals import joblib # For saving and loading models.
# Set path
path = r"C:/Users/Tim/Desktop/BOKU/GIS/GISproject/stacks/"
# Create the required columns for the dataframe.
columns = [str(i+1) + "_" + str(j) for i in range(7) for j in range(14)]
columns = list(filter(lambda x: x.split("_")[0] != "6", columns))
columns = columns + ["Year", "Date", "Summer_Or_Winter", "Label", "Tile"]
# The years the algorithm will be trained on.
years = [#"2008", "2009", "2010", "2011", 
         "2013", "2014", "2015",
         "2016", "2017", 
         "2018", "2019"
         ]
#years = ["2016"]
# The bands it will be trained on.
bands = ["1", "2", "3", "4", "5", "7"]
# Field areas to be trained on.
tiles = ["EC1", "EC2", "EC3", "EC4", "EC5", "EC6",
               #"final_farmland_polygon"
               ] + [str(i + 1) + "_2019" for i in range(110)]

# Labels for summer and winter crop classification.
summer_crops = ["B", "M", "SB", "GM", "SM"]
winter_crops = ["WW", "WB", "WR", "P", "SP" "TC", "O"]

# Create the dataframes with the defined columns.
df = pd.DataFrame(columns=columns)
df_test = pd.DataFrame(columns=columns)
temp_df = pd.DataFrame(columns=columns)

# Fill the dataframe with data.
for year in years:
    for tile in tiles:
        for band in bands:
            # This date is required for the get_label function to work.
            date = year + "0601"
            date = pd.to_datetime(date, format="%Y%m%d")
            # Load in a stack for a certain tile, band and year.
            if os.path.exists(path + "stack-" + tile + "-" + band + "-" + year 
                            + ".npy"):
                array = np.load(path + "stack-" + tile + "-" + band + "-" + year 
                                + ".npy")
                # Load in the cloud mask for the same tile and year.
                mask = np.load(path + "stack-" + tile + "-8-" + year + ".npy")
                #mask = cv2.resize(mask, array.shape[:2])
                #print(mask.shape, array.shape)
    #            if tile == "final_farmland_polygon":
    #                array = cv2.resize(array, (300, 300))
    #                mask = cv2.resize(mask, (300, 300))
                # Where the mask is this value, turn the array to np.nan.
                array[mask >= 1000] = np.nan
                plt.imshow(array[:, :, 5])
                plt.show()
            
                for i in range(array.shape[2]):
                    # Add the image data to the dataframe.
                    temp_df[band + "_" + str(i)] = array[:, :, i].ravel()
            else:
                continue
                
        # Other labels for the dataframe.
        year_num = float(year)
        temp_df["Year"] = year_num
        temp_df["Date"] = date
        temp_df["Summer_Or_Winter"] = 0
        print(tile, date)
        label = get_label(tile, date)
        temp_df["Label"] = label[0]
        if label[0] in summer_crops:
            temp_df["Summer_Or_Winter"] = 0
        else:
            temp_df["Summer_Or_Winter"] = 1
        temp_df["Tile"] = tile
        #if tile != "final_farmland_polygon":
         #   temp_df = temp_df.dropna(thresh=8)
          #  temp_df = temp_df.fillna(axis=1, method="ffill")
          #  temp_df = temp_df.fillna(axis=1, method="bfill")
        
        # Append the temp df to the final df and then reset it so it's empty.
        rand = random.randint(0, 100)
        if rand >= 25:
            df = df.append(temp_df)
        else:
            df_test = df_test.append(temp_df)
        temp_df = pd.DataFrame(columns=columns)
        
# Split the ammertal area from the rest of the field areas.
df_ammertal = df[df["Tile"] == "final_farmland_polygon"]
df = df[df["Tile"] != "final_farmland_polygon"]

# Drop columns where more than 8 values are np.nan.
df = df.dropna(thresh=8)
df_test = df_test.dropna(thresh=8)

#df = df.interpolate(axis=1, method='linear')
#df = df.interpolate(axis=1, limit_direction='backward',
#                    method='nearest', order=2)

# Reorder the columns.
df = df[columns]
df_test = df_test[columns]

"""
0: Beet
1: Clover
2: Meadow
3: Oats
4: Rye
5: Spring Barley
6: Silomaise
7: Soy
8: Spelt
9: Triticale
10: Winter barley
11: Winter Rapeseed
12: Winter wheat
"""

# Renew the index so it starts at 0 again.
df = df.reset_index()
df_test = df_test.reset_index()

# Remove data without a label.
df = df[df["Label"] != "irrelevant"]
df = df.reset_index()
df_test = df_test[df_test["Label"] != "irrelevant"]
df_test = df_test.reset_index()
# Take data only past the first 2 columns.
df = df.iloc[:, 2:]
df_test = df_test.iloc[:, 2:]

# Encode the labels into numbers.
encoder = LabelEncoder()
encoder.fit(df["Label"])
df["Label"] = encoder.transform(df["Label"])
df_test["Label"] = encoder.transform(df_test["Label"])


# Get the labels and training data and split it into train and test sets.
#y = df.iloc[:, -2]
#X = df.iloc[:, :-5]
#X_train, X_test, y_train, y_test = train_test_split(X, y)
X_train = df.iloc[:, :-5]
X_test = df_test.iloc[:, :-5]

y_train = df.iloc[:, -2]
y_test = df_test.iloc[:, -2]

# Initialise the random forest.
rand_for = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)
# Train the random forest.
rand_for.fit(X_train, y_train)
# Initialise the XGB classifier.
xgb = XGBClassifier(n_estimators=200, n_jobs=-1)
# Train the XGB.
xgb.fit(X_train, y_train)
# Show feature importances.
print(rand_for.feature_importances_)
feat_imptc = rand_for.feature_importances_
feat_imptc = feat_imptc[:-1]
feat_x = [i for i in range(len(feat_imptc))]
colors = ["b", "g", "r", "c", "m", "y"]
labels = ["band 1", "band 2", "band 3", "band 4",
          "band 5", "band 7"]
val = 0
plt.style.use("fivethirtyeight")
plt.figure(figsize=(13, 13))
for count, i in enumerate(range(14, len(feat_x) + 10, 14)):
    print(val, i)
    print(feat_imptc[val:i])
    print(len(feat_imptc[val:i]))
    plt.plot(feat_x[val:i], feat_imptc[val:i], color=colors[count],
             label=labels[count], lw=4, marker="x", ms=10)
    val = i
plt.legend()
plt.ylabel("Feature importance")
plt.xlabel("Feature number")
plt.show()

# Predict values
y_pred = rand_for.predict(X_test)
y_pred_xgb = xgb.predict(X_test)

y_pred_probas = rand_for.predict_proba(X_test)
y_pred_xgb_probas = xgb.predict_proba(X_test)
y_pred_probas_combined = (y_pred_probas + y_pred_xgb_probas) / 2
y_pred_argmax = y_pred_probas_combined.argmax(axis=1)
# Get scores of the classifier.
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.utils.multiclass import unique_labels

accuracy = accuracy_score(y_test, y_pred)
accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
accuracy_combined = accuracy_score(y_test, y_pred_argmax)

precision = precision_score(y_test, y_pred, average="weighted")
precision_xgb = precision_score(y_test, y_pred_xgb, average="weighted")
recall = recall_score(y_test, y_pred, average="weighted")
recall_xgb = recall_score(y_test, y_pred_xgb, average="weighted")


# Confusion matrix.
cm = confusion_matrix(y_test, y_pred)
cm = cm / cm.sum(axis=0)
cm_xgb = confusion_matrix(y_test, y_pred_xgb)
cm_xgb = cm_xgb.astype("float") / cm_xgb.sum(axis=1)[:, np.newaxis]

cm_combined = confusion_matrix(y_test, y_pred_argmax)
example_wheat = df.iloc[304, :-4].values
example_maize = df.iloc[62, :-4].values
plt.figure(figsize=(12, 12))
plt.plot(example_wheat, color='green', marker='x', ms=5)
plt.plot(example_maize, color='red', marker='o', ms=5)
plt.show()

print(np.unique(y_pred))
print(y_test.unique())
uniques = set(sorted(list(y_test.unique()) + list(np.unique(y_pred_xgb))))
print(list(uniques))
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

# Save model.
joblib.dump(rand_for, r"C:/Users/Tim/Desktop/BOKU/GIS/GISproject/rand_for.pkl")
joblib.dump(rand_for_summer, r"C:/Users/Tim/Desktop/BOKU/GIS/GISproject/rand_for_summer.pkl")

# Load random forest.
rand_for = joblib.load(r"C:/Users/Tim/Desktop/BOKU/GIS/GISproject/rand_for.pkl")
# Prepare ammertal for preds.
mask_ammertal = df_ammertal.iloc[:, -10]
mask_ammertal = mask_ammertal.isnull().values
mask_ammertal = mask_ammertal.reshape((799, 930))
df_ammertal = df_ammertal.fillna(0)
# Predict on the ammertal data.
preds_ammertal = rand_for.predict(df_ammertal.iloc[:, :-4])
preds_ammertal = preds_ammertal.reshape((799 ,930)).astype(float)
# Mask out NaN data.
preds_ammertal[mask_ammertal] = 100

preds_summer_ammertal = rand_for_summer.predict(df_ammertal.iloc[:, :-4])
counts_summer = np.unique(preds_summer_ammertal, return_counts=True)

plt.imshow(preds_ammertal)
# Save the classification.
cv2.imwrite("ammertal_predictions_2016.tif", preds_ammertal)
# Try unsupervised.
from sklearn.cluster import KMeans
df_unsupervised = df.iloc[:, :-4]

kmeans = KMeans(n_clusters=5)
kmeans.fit(df_unsupervised)
joblib.dump(kmeans, r"C:/Users/Tim/Desktop/BOKU/GIS/GISproject/kmeans.pkl")
kmeans_pred = kmeans.predict(df_unsupervised)
accuracy = accuracy_score(y_test, kmeans_pred)
kmeans_pred = kmeans_pred.reshape((799 ,930))
plt.imshow(kmeans_pred)

