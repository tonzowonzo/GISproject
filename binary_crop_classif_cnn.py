# CNN for classifying crops.
# Crop classification with pretrained algorithm.
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import metrics
from keras import losses
from keras import backend as K
import os

os.chdir(r"C:/Users/Tim/Desktop/GIS/GISproject/")

# Import the pretrained model
base_model = InceptionV3(weights='imagenet', include_top=False)

x = base_model.output
x = GlobalAveragePooling2D()(x)

# Add a fully connected layer.
x = Dense(128, activation='relu')(x)

# Add a classifying layer, 2 classes (Binary classification)
predictions = Dense(1, activation='sigmoid')(x)

# The model we'll train.
model = Model(inputs=base_model.input, outputs=predictions)

# Train only the top layer, freeze the weights of the others.
for layer in base_model.layers:
    layer.trainable = False
    
# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[metrics.hinge, 'accuracy', metrics.kullback_leibler_divergence])

# Train the model on new data for a few epochs.
from keras.preprocessing.image import ImageDataGenerator

# Create the generators for datasets.
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True,
                                   rotation_range = 20,
                                   width_shift_range = 0.2,
                                   height_shift_range = 0.2)

test_datagen = ImageDataGenerator(rescale = 1./255)

# Get the image from the directories.
training_set = train_datagen.flow_from_directory(r'temporal_stacks_summer_winter/train',
                                                 target_size = (299, 299),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory(r'temporal_stacks_summer_winter/test',
                                            target_size = (299, 299),
                                            batch_size = 32,
                                            class_mode = 'binary')

model.fit_generator(training_set, steps_per_epoch=10, epochs=10, 
                    validation_data=test_set, validation_steps=10)