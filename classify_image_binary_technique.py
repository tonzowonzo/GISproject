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
    image_date = march.split(".")[0]
    image_date = pd.to_datetime(image_date, format="%Y%m%d")
    month = image_date.month
    year = image_date.year
    day_of_year = image_date.timetuple().tm_yday
    
    # Step one - classify summer vs winter crop.
    b = cv2.imread(path + "B/" + march, 0).ravel()
    g = cv2.imread(path + "G/" + march, 0).ravel()
    r = cv2.imread(path + "R/" + march, 0).ravel()
    nir = cv2.imread(path + "NIR/" + march, 0).ravel()
    swirs1 = cv2.imread(path + "swir1/" + march, 0).ravel()
    tirs1 = cv2.imread(path + "tirs1/" + march, 0).ravel()
    numer = nir - r
    denom = nir + r
    ndvi = numer / denom
    ndvi_ratio = ndvi * month
    mask = np.where(nir == 0, 1, 0)
    data = {"month": month, "year": year,
            "b": b, "g": g, "r": r, "nir": nir,
            "swir1": swirs1, "tirs1": tirs1, "ndvi": ndvi,
            "ndvi_ratio": ndvi_ratio, "mask": mask}
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
    probas = binary_classifier.predict_proba(
            dataframe_full.iloc[:, :-2])
    
    # Classify winter rape vs winter wheat.
    # Remake dataframe.
    image_date = may.split(".")[0]
    image_date = pd.to_datetime(image_date, format="%Y%m%d")
    month = image_date.month
    year = image_date.year
    day_of_year = image_date.timetuple().tm_yday
    b = cv2.imread(path + "B/" + may, 0)
    b = cv2.resize(b, (808, 942)).ravel()
    g = cv2.imread(path + "G/" + may, 0)
    g = cv2.resize(g, (808, 942)).ravel()
    r = cv2.imread(path + "R/" + may, 0)
    r = cv2.resize(r, (808, 942)).ravel()
    nir = cv2.imread(path + "NIR/" + may, 0)
    nir = cv2.resize(nir, (808, 942)).ravel()
    swir1 = cv2.imread(path + "swir1/" + may, 0)
    swir1 = cv2.resize(swir1, (808, 942)).ravel()
    tirs1 = cv2.imread(path + "tirs1/" + may, 0)
    tirs1 = cv2.resize(tirs1, (808, 942)).ravel()
    numer = nir - r
    denom = nir + r
    ndvi = numer / denom
    ndvi_ratio = ndvi * month
    data = {"month": month, "year": year,
            "b": b, "g": g, "r": r, "nir": nir,
            "swir1": swir1, "tirs1": tirs1, "ndvi": ndvi,
            "ndvi_ratio": ndvi_ratio, "mask": mask}
    dataframe_2a = pd.DataFrame(data=data)
    dataframe_2a = dataframe_2a.replace([np.inf, -np.inf], np.nan)
    dataframe_2a = dataframe_2a.fillna(-1)
    dataframe_2a["2a"] = -1
    preds_2a = model_2a.predict(dataframe_2a.iloc[:, :-2])
    dataframe_full["2a"] = np.where(dataframe_full["1"] == 0,
                  -1, preds_2a)
    
    # Sugarbeet from corn.
    image_date = july.split(".")[0]
    image_date = pd.to_datetime(image_date, format="%Y%m%d")
    month = image_date.month
    year = image_date.year
    day_of_year = image_date.timetuple().tm_yday
    b = cv2.imread(path + "B/" + july, 0)
    b = cv2.resize(b, (808, 942)).ravel()
    g = cv2.imread(path + "G/" + july, 0)
    g = cv2.resize(g, (808, 942)).ravel()
    r = cv2.imread(path + "R/" + july, 0)
    r = cv2.resize(r, (808, 942)).ravel()
    nir = cv2.imread(path + "NIR/" + july, 0)
    nir = cv2.resize(nir, (808, 942)).ravel()
    swir1 = cv2.imread(path + "swir1/" + july, 0)
    swir1 = cv2.resize(swir1, (808, 942)).ravel()
    tirs1 = cv2.imread(path + "tirs1/" + july, 0)
    tirs1 = cv2.resize(tirs1, (808, 942)).ravel()
    numer = nir - r
    denom = nir + r
    ndvi = numer / denom
    ndvi_ratio = ndvi * month
    data = {"month": month, "year": year,
            "b": b, "g": g, "r": r, "nir": nir,
            "swir1": swir1, "tirs1": tirs1, "ndvi": ndvi,
            "ndvi_ratio": ndvi_ratio, "mask": mask}
    dataframe_3c = pd.DataFrame(data=data)
    dataframe_3c = dataframe_3c.replace([np.inf, -np.inf], np.nan)
    dataframe_3c = dataframe_3c.fillna(-1)
    dataframe_3c["3c"] = -1
    preds_3c = model_3c.predict(dataframe_3c.iloc[:, :-2])
    dataframe_full["3c"] = np.where(dataframe_full["1"] == 1,
                  -1, preds_3c)
    
    # Wheat from barley.
    image_date = april.split(".")[0]
    image_date = pd.to_datetime(image_date, format="%Y%m%d")
    month = image_date.month
    year = image_date.year
    day_of_year = image_date.timetuple().tm_yday
    b = cv2.imread(path + "B/" + april, 0)
    b = cv2.resize(b, (808, 942)).ravel()
    g = cv2.imread(path + "G/" + april, 0)
    g = cv2.resize(g, (808, 942)).ravel()
    r = cv2.imread(path + "R/" + april, 0)
    r = cv2.resize(r, (808, 942)).ravel()
    nir = cv2.imread(path + "NIR/" + april, 0)
    nir = cv2.resize(nir, (808, 942)).ravel()
    swir1 = cv2.imread(path + "swir1/" + april, 0)
    swir1 = cv2.resize(swir1, (808, 942)).ravel()
    tirs1 = cv2.imread(path + "tirs1/" + april, 0)
    tirs1 = cv2.resize(tirs1, (808, 942)).ravel()
    numer = nir - r
    denom = nir + r
    ndvi = numer / denom
    ndvi_ratio = ndvi * month
    data = {"month": month, "year": year,
            "b": b, "g": g, "r": r, "nir": nir,
            "swir1": swir1, "tirs1": tirs1, "ndvi": ndvi,
            "ndvi_ratio": ndvi_ratio, "mask": mask}
    dataframe_4a = pd.DataFrame(data=data)
    dataframe_4a = dataframe_4a.replace([np.inf, -np.inf], np.nan)
    dataframe_4a = dataframe_4a.fillna(-1)
    dataframe_4a["4a"] = -1
    preds_4c = model_4a.predict(dataframe_4a.iloc[:, :-2])
    dataframe_full["4a"] = np.where(dataframe_full["1"] == 1,
                  -1, preds_4c)
    
    return dataframe_full, probas

    # Step two - classify rape vs barley where the crop is winter.
    
    
r_march = r"C:\Users\Tim\Desktop\GIS\GISproject\landsat\landsat_8_8bit\LC08_L1TP_194026_20140313_20170425_01_T1_B3.tif"
nir_march = r"C:\Users\Tim\Desktop\GIS\GISproject\landsat\landsat_8_8bit\LC08_L1TP_194026_20140313_20170425_01_T1_B4.tif"
df, probas = crop_classification(march, may, july, april)

def plot_output(df, probas, initial_shape=(808, 942)):
    
    preds = df.iloc[:, -1]
    preds[df["mask"] == 1] = -1
    preds = preds.values
    preds = preds.reshape(initial_shape)
    cv2.imwrite("test_ammertal.png", preds)
    plt.imshow(preds)
    plt.show()
    preds = preds.reshape(initial_shape)
    
    # Plot probabilities.
    for i in range(probas.shape[1]):
        probas[df["mask"] == 1] = -1
        probs = probas[:, i].reshape(initial_shape)
        plt.imshow(probs)
        plt.colorbar()
        plt.show()
        
    return preds
    
preds = plot_output(df, probas)
