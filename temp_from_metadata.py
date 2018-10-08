# Read in temperature data from metadata.
import os
import re
import math

# File path to open in eg.
path = r"C:/Users/Tim/Desktop/GIS/GISproject/landsat/Bulk Order 948741/Landsat 4-5 TM C1 Level-1/LT05_L1GS_195026_20100613_20161015_01_T2_MTL.txt"
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

mult, add, k1, k2 = get_required_info_from_metadata(path, 7)

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
    mult = mult / 10
    print(mult)
    # Calculate top of atmosphere temperature.
    top_of_atmosphere = mult * thermal_band + add
    print(top_of_atmosphere)
    # Calculate the temperature caused by brightness.
    brightness_temperature = (k2/(math.log(k1/top_of_atmosphere) + 1)) - 273.15
    print(brightness_temperature)
    # Calculate proportion of vegetation.
    Pv = ((ndvi - ndvi_min)/(ndvi_max - ndvi_min))**2
    print(Pv)
    # Calculate emissivity.
    emissivity = 0.004 * Pv + 0.986
    print(emissivity)
    # Calculate land surface temperature.
    LST = (brightness_temperature/(1 + (0.00115*brightness_temperature/1.4388) * math.log(emissivity)))
    
    return LST

LST = calculate_lst(0.6, 0.9, 0.3, 54, 8, mult, add, k1, k2)
    
print(LST)