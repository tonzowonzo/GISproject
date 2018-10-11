# Read in temperature data from metadata.
import os
import re
import math
import numpy as np

# File path to open in eg.
path = r"C:/Users/Tim/Desktop/GIS/GISproject/landsat/metadata/LC08_L1TP_195026_20140811_20170420_01_T1_MTL.txt"
num_regex = re.compile(r"\d+\.\d+")

def get_required_info_from_metadata(path, landsat):
    with open(path) as file:
        for line in file:
            if landsat == 7 or landsat == 5:
                if "RADIANCE_MULT_BAND_6" in line:
                    print(line)
                    mult = re.findall(num_regex, line)
                    mult = float(mult[0])
                elif "RADIANCE_ADD_BAND_6" in line:
                    print(line)
                    add = re.findall(num_regex, line)
                    add = float(add[0])
                elif " K1_CONSTANT_BAND_6" in line:
                    print(line)
                    k1 = re.findall(num_regex, line)
                    k1 = float(k1[0])
                elif " K2_CONSTANT_BAND_6" in line:
                    print(line)
                    k2 = re.findall(num_regex, line)
                    k2 = float(k2[0])
                    
            elif landsat == 8:
                if "RADIANCE_MULT_BAND_10" in line:
                    print(line)
                    mult = re.findall(num_regex, line)
                    mult = float(mult[0])
                elif "RADIANCE_ADD_BAND_10" in line:
                    print(line)
                    add = re.findall(num_regex, line)
                    add = float(add[0])
                elif " K1_CONSTANT_BAND_10" in line:
                    print(line)
                    k1 = re.findall(num_regex, line)
                    k1 = float(k1[0])
                elif " K2_CONSTANT_BAND_10" in line:
                    print(line)
                    k2 = re.findall(num_regex, line)
                    k2 = float(k2[0])

            
    return mult, add, k1, k2

#get_required_info_from_metadata(path, 8)

def calculate_lst(ndvi, ndvi_max, ndvi_min, thermal_band, landsat, mult, add, k1, k2):
    '''
    Calculate the landsurface temperature based on a thermal raster from landsat
    5, 7 or 8.
    Where:
        ndvi_max : the maximum ndvi value over the field area.
        ndvi_min : the minimum ndvi value over the field area.
        thermal_band : the value from the thermal band (band 6 or band 10)
        landsat : Which landsat is being used (5, 7, 8)
        mult: RADIANCE_MULT_BAND_6 or 10 from metadata
        add: RADIANCE_ADD_BAND_6 or 10 from metadata
        k1: K1_CONSTANT_BAND_6 or 10 from metadata
        k2: K2_CONSTANT_BAND_6 or 10 from metadata
    '''
    # Calculate top of atmosphere temperature.
    top_of_atmosphere = mult * thermal_band + add
    # Calculate the temperature caused by brightness.
    brightness_temperature = (k2/(np.log(k1/top_of_atmosphere) + 1)) - 273.15
    # Calculate proportion of vegetation.
    Pv = ((ndvi - ndvi_min)/(ndvi_max - ndvi_min))**2
    # Calculate emissivity.
    emissivity = 0.004 * Pv + 0.986
    # Calculate land surface temperature.
    at_sat_temp = (brightness_temperature/(1 + (0.00115*brightness_temperature/1.4388) * np.log(emissivity)))
    LST = (at_sat_temp / 1 + 11.5 * (at_sat_temp / 14380) * np.log(emissivity))
    LST = LST / 50
    return LST
