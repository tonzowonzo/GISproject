# 3D CNN example - Crop types.

# Set working directory.
import os
os.chdir(r"C:/Users/Tim/Desktop/GIS/GISproject")
path = r"C:/Users/Tim/Desktop/GIS/GISproject/"
# Import required libraries.
import numpy as np
import pandas as pd
import cv2
import keras

# Import the label grabber.
from get_label import get_label

# Load the image data.
# Define the dataframe.
columns = ["date", "image", "field_area", "same_season", "label"]
df = pd.DataFrame(columns=columns)

# Load in images (just green this time).
i = 0
field_areas = ["EC1", "EC2", "EC3", "EC4", "EC5", "EC6", "1_1", "2", "3", "4",
               "5", "6", "8", "9", "11", "13", "15", "17"]
landsats = ["landsat_5/", "landsat_7/", "landsat_8/"]
for field_area in field_areas:
    for landsat in landsats:
        # Green band is different in different landsats.
        if landsat == "landsat_5" or landsat == "landsat_7":
            band = "/1/"
        else:
            band = "/2/"
            
        for file in os.listdir(path + landsat + field_area + band):
            if file.endswith(".tif"):
                img = cv2.imread(path + landsat + field_area + band + file, 0)
                img = img.reshape((img.shape[0], img.shape[1], 1, 1))
                img = np.resize(img, (20, 20, 8, 1))
                date = file.split(".")[0]
                date = pd.to_datetime(date, format="%Y%m%d")
                label, last_crop = get_label("EC1", date)
                temp_df = pd.DataFrame(data={"date": date, "image": [img], 
                                             "field_area": field_area, "same_season": 0,
                                             "label": label}, index=[i])
                df = df.append(temp_df)
                i += 1
                
# Drop irrelevant data.
df = df[df["label"] != "irrelevant"]

# Create variable on whether the field area and date is the same.
for index in df.index:
    df["same_season"][index] = str(df["date"][index].year) + "_" + df["label"][index]
    
# Create year, crop image stacks.
stacked_df = pd.DataFrame(columns={"image", "label"})
for i, index in enumerate(df.index):
    
    # The initial image for stacking.
    if i == 0:
        image = df["image"][index]
        image_stack = image
        last_field_area = df["field_area"][index]
        last_same_season = df["same_season"][index]
        
    else:
        # Reset the stack if this is a new field area / crop.
        if df["field_area"][index] != last_field_area or df["same_season"][index] != last_same_season:
            stacked_df = stacked_df.append(pd.DataFrame(data={"image": [image_stack],
                                                 "label": df["label"][index]},
                                            index=[i]))
            image = df["image"][index]
            image_stack = image
            last_field_area = df["field_area"][index]
            last_same_season = df["same_season"][index]

        # Add to the stack if it is the same field area / crop.
        elif df["field_area"][index] == last_field_area and df["same_season"][index] == last_same_season:
            image = df["image"][index]
            print(image.shape)
            image_stack = np.dstack((image_stack, image))
            print(image_stack.shape)
            
            last_field_area = df["field_area"][index]
            last_same_season = df["same_season"][index]

# Separate X and y.
X = stacked_df.iloc[:, 0]
y = stacked_df.iloc[:, -1]

    
    
# Split into train and test set.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Save the numpy arrays.
# Train arrays.
#path = r"C:/Users/Tim/pythonscripts/crop_types/"
#for index in X_train.index:
#    cv2.imwrite(path + "train/" + y_train[index] + "/" + str(index) + ".tif",
#                X_train[index])
#
#    
## Save the numpy arrays.
## Test arrays.
#for index in X_test.index:
#    cv2.imwrite(path + "test/" + y_test[index] + "/" + str(index) + ".tif",
#                X_test[index])

    
# Create the network.
# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution3D
from keras.layers import MaxPooling3D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.models import load_model
from keras import metrics
from keras import losses

# Build the model
model = Sequential()  # Instantiate the model.

# Step 1 - Convolution
model.add(Convolution3D(32, (3, 3, 2), activation="relu", padding="same", input_shape=(50, 50, 2, 1)))

# Step 2 - Pooling
model.add(MaxPooling3D(pool_size = (2, 2, 1), strides=2))

# Dropout
model.add(Dropout(0.5))

# Batch norm.
model.add(BatchNormalization())

# Step 3 - Flattening
model.add(Flatten())

# Step 4 - Full connection
model.add(Dense(units=128, activation="relu"))
model.add(Dense(units=12, activation="softmax"))

# Compiling the CNN
model.compile(optimizer = 'SGD', loss = 'categorical_crossentropy'
              , metrics = ['accuracy'])

# Train the model on new data for a few epochs.
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.sequence import pad_sequences

# Reshape dataframes.
# Create the generators for datasets.
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.4,
                                   zoom_range = 0.4,
                                   horizontal_flip = True,
                                   rotation_range = 90,
                                   width_shift_range = 0.4,
                                   height_shift_range = 0.4)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow(X_train, batch_size=32)

test_set = test_datagen.flow_from_directory(path + "test/", batch_size = 32)

model.fit(training_set, steps_per_epoch=25, epochs=10, 
                    validation_data=test_set, validation_steps=10)


# Get the values from the generator
X_test = list(test_set.next())

# Predict from a batch

y_pred2 = model.predict((X_test[0]))

    

    
    
