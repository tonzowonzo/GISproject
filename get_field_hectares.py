# Load in field hectares.
import pandas as pd

# Load in a csv for area.
areas = pd.read_csv(r"C:/Users/Tim/Desktop/GIS/GISproject/Excel-data/areas/201808_Ackerland_Hohenstein_RT_nach_Fruchtarten_BB_1999-2016.csv",
            skiprows=[0,1], delimiter=";")