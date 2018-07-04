# Import libraries.
import os
import numpy as np
import pandas as pd
import datetime
from PIL import Image
import sklearn
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing 
import matplotlib.pyplot as plt

# Path to array files.
path = ''

# Classify late vs early plant crops in landsat images.
# Fed a dataframe with preclassified crop pixels, dates and image arrays.
def classify_late_early_crops_train(training_array, path):
	'''
	Classify pixels in a given landsat image based on training data. First training needs to be done with
	input examples, then once trained we feed each pixel to the algorithm and try to classify it with an SVM
	classifier. There will also be a neither classifier, it takes care of ie buildings, lakes, trees etc.
	'''
	
	# Take in the training array.
	for file in os.listdir(path):
		# Load file.
		pass
	
	# Turn into Dataframe like shown in github.
	pass
	
	# Split dataframe into X and y data.
	# X is everything except the last column.
	X = df.iloc[:, :-1]
	# y is only the last column (classif column).
	y = df.iloc[:, -1]
	
	# Normalize pixel value data.
	pass
	
	# Encode y variables.
	pass
	
	# OnehotEncode y variables.
	pass
	
	# Split into training and test sets
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
	
	# Define the SVM classifier.
	classifier = SVC(kernel='linear')
	# Fit the classifier to the training data.
	classifier.fit(X_train, y_train)
	
	# Test classifier accuracy on test set.
	y_pred = classifier.predict(X_test)
	
	# Display accuracy of classifier.
	
	return y_test, y_pred, classifier
	
def classify_landsat_image(path):

	# Load in image in RGB format.
	img = Image.open(path).convert('RGB')
 
	# Turn date into day of year column DO IN PREPROCESSING.
	day_of_year = datetime.now().timetuple().tm_yday
	
	# Get the RGB and date values and put them into an algorithm.
	r,g,b = img.getpixel(int(str(i) + str(j)))
	pred = classifier.predict([day_of_year, r, g, b])
	