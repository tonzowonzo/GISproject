# Import libraries
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.externals import joblib
import pandas as pd
np.seterr(divide='ignore', invalid='ignore')
# Set current working directory.
os.chdir(r"C:/Users/Tim/Desktop/GIS/GISproject/")

# Load binary classifier.
binary_classifier = joblib.load("binary_classifier.pkl")
model_2a = joblib.load("for_clf_2a.pkl")
model_3c = joblib.load("for_clf_3c.pkl")
model_4a = joblib.load("for_clf_4a.pkl")
path = r"C:/Users/Tim/Desktop/GIS/GISproject/ammertal_clf_area/"
march = "20110321.tif"
april = "20110422.tif"
may = "20110508.tif"
july = "20110719.tif"

# Function for full classification of an image's crop types.
def crop_classification(march, may, july, april):
    '''
    Aims to classify a landsat image based on the method from Harfenmeister et
    al.
    All inputs are a date of the image required for classification.
    march: input for summer vs winter crop classification.
    may: input for rape from barley/rye separation.
    july: input for separation of sugarbeet vs corn.
    april: separation of wheat from barley and rye.
    '''
    # Create dataframe.
    columns = ["month", "b", "g", "r", "nir", "swir1", 
                      "tirs1", "ndvi", "ndvi_ratio", "2a", "2b",
                          "3c", "4a", "label"]
    image_date = march.split(".")[0]
    image_date = pd.to_datetime(image_date, format="%Y%m%d")
    month = image_date.month
    day_of_year = image_date.timetuple().tm_yday
    
    # Step one - classify summer vs winter crop.
    b_march = cv2.imread(path + "B/" + march, 0).ravel()
    g_march = cv2.imread(path + "G/" + march, 0).ravel()
    r_march = cv2.imread(path + "R/" + march, 0).ravel()
    nir_march = cv2.imread(path + "NIR/" + march, 0).ravel()
    swir1_march = cv2.imread(path + "swir1/" + march, 0).ravel()
    tirs1_march = cv2.imread(path + "tirs1/" + march, 0).ravel()
    ndvi_march = (nir_march - r_march) / (nir_march + r_march)
    ndvi_ratio_march = ndvi_march * month
    mask = np.where(nir_march == 0, 1, 0)
    data = {"day_of_year": day_of_year, "month": month, "b": b_march,
            "g": g_march, "r": r_march, "nir": nir_march, "ndvi": ndvi_march,
            "swir1": swir1_march,
            "ndvi_ratio": ndvi_ratio_march, "mask": mask}
    dataframe_full = pd.DataFrame(data=data)
    dataframe_full = dataframe_full.replace([np.inf, -np.inf], np.nan)
    dataframe_full = dataframe_full.fillna(-1)
#    dataframe_full["probs1"], dataframe_full["probs2"] = binary_classifier.predict_proba(
#            dataframe_full.iloc[:, :-1])
#    probas = binary_classifier.predict_proba(
#            dataframe_full.iloc[:, :-1])
#    probas = np.where(probas[:, 1] <= 0.05, 2, 3)
    dataframe_full["1"] = binary_classifier.predict(
            dataframe_full.iloc[:, :-1])
    return dataframe_full

    # Step two - classify rape vs barley where the crop is winter.
    
    
r_march = r"C:\Users\Tim\Desktop\GIS\GISproject\landsat\landsat_8_8bit\LC08_L1TP_194026_20140313_20170425_01_T1_B3.tif"
nir_march = r"C:\Users\Tim\Desktop\GIS\GISproject\landsat\landsat_8_8bit\LC08_L1TP_194026_20140313_20170425_01_T1_B4.tif"
df = crop_classification(march, may, july, april)

def plot_output(df, initial_shape=(808, 942)):
    
    preds = df.iloc[:, -1]
    preds[df["mask"] < 1] = 10
    preds = preds.values
    preds = preds.reshape(initial_shape)
    plt.imshow(preds)
    plt.show()
    return preds
    
preds = plot_output(df)
