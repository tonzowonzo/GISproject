# Visualise the data from the farms.

# Import libraries.
import matplotlib.pyplot as plt
import pandas as pd
import seaborn

df_path = "C:/Users/Tim/Desktop/GIS/GISproject/Excel-data/20180702_Stations_Alb_safe.xls"

def visualise_data(df_path):
    
    # Read in data.
    df = pd.read_excel(df_path)
    fruits = df.iloc[:, 3:13]
    for fruit in fruits:
        print(df[fruit].unique())
    return fruits
    # Farm type distribution.
    
    # Farm type distribution with time.
    
    # Areas for each farm type.
     
    # Correlation plots.
    

df = visualise_data(df_path)