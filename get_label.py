# Get the label of a particular image.
# Function for getting the label.
import datetime

def get_label(field_area, date):
    '''
    Returns the label of an image based on the csv that shows which label is in
    which area by time.
    
    Where:
        SM: Silomaise
        WW: Winter wheat
        WR: Winter rape
        WB: Winter barley
        GM: Grain maise
        SB: Summer barley
        SP: Spelt / Dinkel
        B: Beet
        C: Clover / Kleegras
        TC: Triticale
        M: Mais
        O: Oats
        MU: Mustard
        R: Rye
        P: Peas
        LU: Luzerne
        SOY: Soja
        ME: Meadow
        Irrelevant: Crop not being classified
    '''
    # If the crop is either a cover crop or no cover (during winter).
    label = "irrelevant"
    last_crop = "unknown"
    if date.month > 8 or date.month < 3:
        label = "irrelevant"
        last_crop = "unknown"
    elif date.year >= 2019:
        label = "irrelevant"
        last_crop = "unknown"
        
    # Field EC1 - Kraichgau.
    elif field_area == "EC1":
        if date < datetime.datetime(2010, 12, 1):
            label = "SM"
            last_crop = "unknown"
        elif date < datetime.datetime(2011, 10, 1):
            label = "WW"
            last_crop = "SM"
        elif date < datetime.datetime(2012, 10, 1):
            label = "WR"
            last_crop = "WW"
        elif date < datetime.datetime(2013, 10, 1):
            label = "WW"
            last_crop = "WR"
        elif date < datetime.datetime(2014, 12, 1):
            label = "SM"
            last_crop = "WW"
        elif date < datetime.datetime(2015, 10, 1):
            label = "WW"
            last_crop = "SM"
        elif date < datetime.datetime(2016, 12, 1):
            label = "GM"
            last_crop = "WW"
        elif date < datetime.datetime(2017, 10, 1):
            label = "WW"
            last_crop = "GM"
        elif date < datetime.datetime(2018, 10, 1):
            label = "WR"
            last_crop = "WW"
    
    # Field EC2 - Kraichgau.
    elif field_area == "EC2":
        if date < datetime.datetime(2010, 10, 1):
            label = "WR"
            last_crop = "unknown"
        elif date < datetime.datetime(2011, 10, 1):
            label = "WW"
            last_crop = "WR"
        elif date < datetime.datetime(2012, 12, 1):
            label = "SM"
            last_crop = "WW"
        elif date < datetime.datetime(2013, 10, 1):
            label = "WW"
            last_crop = "SM"
        elif date < datetime.datetime(2014, 12, 1):
            label = "SM"
            last_crop = "WW"
        elif date < datetime.datetime(2015, 10, 1):
            label = "WW"
            last_crop = "SM"
        elif date < datetime.datetime(2016, 10, 1):
            label = "WR"
            last_crop = "WW"
        elif date < datetime.datetime(2017, 10, 1):
            label = "WW"
            last_crop = "WR"
        elif date < datetime.datetime(2018, 10, 1):
            label = "WW"
            last_crop = "WW"
    
    # Field EC3 - Kraichgau.
    elif field_area == "EC3":
        if date < datetime.datetime(2010, 10, 1):
            label = "WW"
            last_crop = "unknown"
        elif date < datetime.datetime(2011, 12, 1):
            label = "SM"
            last_crop = "WW"
        elif date < datetime.datetime(2012, 10, 1):
            label = "WW"
            last_crop = "SM"
        elif date < datetime.datetime(2013, 10, 1):
            label = "WR"
            last_crop = "WW"
        elif date < datetime.datetime(2014, 10, 1):
            label = "WW"
            last_crop = "WR"
        elif date < datetime.datetime(2015, 12, 1):
            label = "SM"
            last_crop = "WW"
        elif date < datetime.datetime(2016, 10, 1):
            label = "WW"
            last_crop = "SM"
        elif date < datetime.datetime(2017, 10, 1):
            label = "WW"
            last_crop = "WW"
        elif date < datetime.datetime(2018, 12, 1):
            label = "SM"
            last_crop = "WW"
         
    # Field EC4 - Swaebisch Alp.
    elif field_area == "EC4":
        if date < datetime.datetime(2010, 10, 1):
            label = "WR"
            last_crop = "unknown"
        elif date < datetime.datetime(2011, 10, 1):
            label = "WW"
            last_crop = "WR"
        elif date < datetime.datetime(2012, 12, 1):
            label = "SB"
            last_crop = "WW"
        elif date < datetime.datetime(2013, 10, 1):
            label = "WR"
            last_crop = "SB"
        elif date < datetime.datetime(2014, 10, 1):
            label = "WW"
            last_crop = "WR"
        elif date < datetime.datetime(2015, 10, 1):
            label = "WW" 
            last_crop = "WW"
        elif date < datetime.datetime(2016, 12, 1):
            label = "SB"
            last_crop = "WW"
        elif date < datetime.datetime(2017, 12, 1):
            label = "SM"
            last_crop = "SB"
        elif date < datetime.datetime(2018, 10, 1):
            label = "WW"
            last_crop = "SM"
            
    # Field EC5 - Swaebisch Alp.
    elif field_area == "EC5":
        if date < datetime.datetime(2010, 10, 1):
            label = "WW"
            last_crop = "unknown"
        elif date < datetime.datetime(2011, 12, 1):
            label = "SM"
            last_crop = "WW"
        elif date < datetime.datetime(2012, 12, 1):
            label = "SM"
            last_crop = "SM"
        elif date < datetime.datetime(2013, 10, 1):
            label = "WB"
            last_crop = "SM"
        elif date < datetime.datetime(2014, 12, 1):
            label = "SP"
            last_crop = "WB"
        elif date < datetime.datetime(2015, 12, 1):
            label = "SM"
            last_crop = "SP"
        elif date < datetime.datetime(2016, 12, 1):
            label = "SM"
            last_crop = "SM"
        elif date < datetime.datetime(2017, 10, 1):
            label = "WB"
            last_crop = "SM"
        elif date < datetime.datetime(2018, 10, 1):
            label = "WR"
            last_crop = "WB"
            
    # Field EC6 - Swaebisch Alp.
    elif field_area == "EC6":
        if date < datetime.datetime(2010, 12, 1):
            label = "SM"
            last_crop = "unknown"
        elif date < datetime.datetime(2011, 10, 1):
            label = "WW"
            last_crop = "SM"
        elif date < datetime.datetime(2012, 10, 1):
            label = "WB"
            last_crop = "WW"
        elif date < datetime.datetime(2013, 12, 1):
            label = "SM"
            last_crop = "WB"
        elif date < datetime.datetime(2014, 10, 1):
            label = "WW"
            last_crop = "SM"
        elif date < datetime.datetime(2015, 10, 1):
            label = "WB"
            last_crop = "WW"
        elif date < datetime.datetime(2016, 12, 1):
            label = "SM"
            last_crop = "WB"
        elif date < datetime.datetime(2017, 10, 1):
            label = "WW"
            last_crop = "SM"
        elif date < datetime.datetime(2018, 10, 1):
            label = "WB"
            last_crop = "WW"
    
    # Field area 1-1 - Kraichgau.
    elif field_area == "1_1":
        if date < datetime.datetime(2010, 10, 1):
            label = "WR"
            last_crop = "unknown"
        elif date < datetime.datetime(2011, 10, 1):
            label = "WW"
            last_crop = "WR"
        elif date < datetime.datetime(2011, 12, 1):
            label = "GM"
            last_crop= "WW"
        elif date < datetime.datetime(2012, 12, 1):
            label = "SB"
            last_crop = "GM"
        elif date < datetime.datetime(2013, 10, 1):
            label = "WR"
            last_crop = "SB"
        elif date < datetime.datetime(2014, 10, 1):
            label = "WW"
            last_crop = "WR"
        elif date < datetime.datetime(2015, 12, 1):
            label = "irrelevant"
            last_crop = "WW"
        elif date < datetime.datetime(2016, 10, 1):
            label = "WW"
            last_crop = "unknown"
        elif date < datetime.datetime(2017, 12, 1):
            label = "irrelevant"
            last_crop = "WW"

    # Field area 1-4 - Kraichgau.
    elif field_area == "1_4":
        if date < datetime.datetime(2010, 10, 1):
            label = "B"
            last_crop = "SB"
        elif date < datetime.datetime(2011, 10, 1):
            label = "GM"
            last_crop = "B"
        elif date < datetime.datetime(2012, 12, 1):
            label = "WW"
            last_crop= "GM"
        elif date < datetime.datetime(2013, 10, 1):
            label = "SB"
            last_crop = "WW"
        elif date < datetime.datetime(2014, 10, 1):
            label = "B"
            last_crop = "SB"
        elif date < datetime.datetime(2015, 12, 1):
            label = "WW"
            last_crop = "B"
        elif date < datetime.datetime(2016, 10, 1):
            label = "SM"
            last_crop = "WW"
        elif date < datetime.datetime(2017, 10, 1):
            label = "WW"
            last_crop = "SB"
            
    # Field area 2 - Kraichgau.
    elif field_area == "2":
        if date < datetime.datetime(2010, 10, 1):
            label = "WB"
            last_crop = "WW"
        elif date < datetime.datetime(2011, 10, 1):
            label = "irrelevant"
            last_crop = "WB"
        elif date < datetime.datetime(2012, 10, 1):
            label = "C"
            last_crop = "unknown"
        elif date < datetime.datetime(2013, 10, 1):
            label = "SP"
            last_crop = "C"
        elif date < datetime.datetime(2014, 10, 1):
            label = "irrelevant"
            last_crop = "SP"
        elif date < datetime.datetime(2015, 12, 1):
            label = "SB"
            last_crop = "unknown"
        elif date < datetime.datetime(2016, 10, 1):
            label = "LU"
            last_crop = "SB"
        elif date < datetime.datetime(2017, 10, 1):
            label = "LU"
            last_crop = "LU"
            
    # Field area 3 - Kraichgau.
    elif field_area == "3":
        if date < datetime.datetime(2010, 10, 1):
            label = "WR"
            last_crop = "SB"
        elif date < datetime.datetime(2011, 10, 1):
            label = "WW"
            last_crop = "WR"
        elif date < datetime.datetime(2012, 12, 1):
            label = "irrelevant"
            last_crop = "WW"
        elif date < datetime.datetime(2013, 10, 1):
            label = "WR"
            last_crop = "unknown"
        elif date < datetime.datetime(2014, 10, 1):
            label = "WW"
            last_crop = "WR"
        elif date < datetime.datetime(2017, 12, 1):
            label = "irrelevant"
            last_crop = "SM"

    # Field area 4 - Kraichgau.
    elif field_area == "4":
        if date < datetime.datetime(2010, 10, 1):
            label = "B"
            last_crop = "SB"
        elif date < datetime.datetime(2011, 10, 1):
            label = "WW"
            last_crop = "B"
        elif date < datetime.datetime(2012, 10, 1):
            label = "WR"
            last_crop = "WW"
        elif date < datetime.datetime(2013, 10, 1):
            label = "WW"
            last_crop = "WR"
        elif date < datetime.datetime(2014, 12, 1):
            label = "irrelevant"
            last_crop = "WW"
        elif date < datetime.datetime(2015, 12, 1):
            label = "SB"
            last_crop = "unknown"
        elif date < datetime.datetime(2016, 10, 1):
            label = "B"
            last_crop = "SB"
        elif date < datetime.datetime(2017, 12, 1):
            label = "irrelevant"
            last_crop = "unknown"
            
    # Field area 5 - Kraichgau.
    elif field_area == "5":
        if date < datetime.datetime(2010, 10, 1):
            label = "WW"
            last_crop = "WW"
        elif date < datetime.datetime(2011, 12, 1):
            label = "SM"
            last_crop = "WW"
        elif date < datetime.datetime(2012, 12, 1):
            label = "B"
            last_crop = "SM"
        elif date < datetime.datetime(2013, 10, 1):
            label = "WW"
            last_crop = "B"
        elif date < datetime.datetime(2014, 12, 1):
            label = "SM"
            last_crop = "WW"
        elif date < datetime.datetime(2015, 10, 1):
            label = "WW"
            last_crop = "SM"
        elif date < datetime.datetime(2016, 10, 1):
            label = "WB"
            last_crop = "WW"
        elif date < datetime.datetime(2017, 12, 1):
            label = "irrelevant"
            last_crop = "WB"
            
    # Field area 6 - Kraichgau.
    elif field_area == "6":
        if date < datetime.datetime(2010, 10, 1):
            label = "WR"
            last_crop = "WW"
        elif date < datetime.datetime(2011, 10, 1):
            label = "WW"
            last_crop = "WR"
        elif date < datetime.datetime(2012, 10, 1):
            label = "WR"
            last_crop = "WW"
        elif date < datetime.datetime(2013, 10, 1):
            label = "WW"
            last_crop = "WR"
        elif date < datetime.datetime(2014, 12, 1):
            label = "irrelevant"
            last_crop = "WW"
        elif date < datetime.datetime(2015, 10, 1):
            label = "WW"
            last_crop = "unknown"
        elif date < datetime.datetime(2016, 10, 1):
            label = "WR"
            last_crop = "WW"
        elif date < datetime.datetime(2017, 10, 1):
            label = "WW"
            last_crop = "WR"
        elif date < datetime.datetime(2017, 12, 1):
            label = "irrelevant"
            last_crop = "unknown"
            
    # Field area 8 - Kraichgau.
    elif field_area == "8":
        if date < datetime.datetime(2010, 10, 1):
            label = "WW"
            last_crop = "B"
        elif date < datetime.datetime(2011, 10, 1):
            label = "WR"
            last_crop = "WW"
        elif date < datetime.datetime(2012, 10, 1):
            label = "WW"
            last_crop = "WR"
        elif date < datetime.datetime(2013, 10, 1):
            label = "R"
            last_crop = "WW"
        elif date < datetime.datetime(2014, 12, 1):
            label = "irrelevant"
            last_crop = "unknown"
        elif date < datetime.datetime(2015, 10, 1):
            label = "WW"
            last_crop = "unknown"
        elif date < datetime.datetime(2016, 10, 1):
            label = "R"
            last_crop = "WW"
        elif date < datetime.datetime(2017, 12, 1):
            label = "irrelevant"
            last_crop = "unknown"
            
    # Field area 9 - Kraichgau.
    elif field_area == "9":
        if date < datetime.datetime(2010, 10, 1):
            label = "WW"
            last_crop = "WR"
        elif date < datetime.datetime(2011, 12, 1):
            label = "GM"
            last_crop = "WW"
        elif date < datetime.datetime(2012, 10, 1):
            label = "WW"
            last_crop = "GM"
        elif date < datetime.datetime(2013, 12, 1):
            label = "SB"
            last_crop = "WW"
        elif date < datetime.datetime(2014, 10, 1):
            label = "WR"
            last_crop = "SB"
        elif date < datetime.datetime(2015, 10, 1):
            label = "WW"
            last_crop = "WR"
        elif date < datetime.datetime(2016, 12, 1):
            label = "irrelevant"
            last_crop = "WW"
        elif date < datetime.datetime(2017, 12, 1):
            label = "irrelevant"
            last_crop = "unknown"
            
    # Field area 11 - Kraichgau.
    elif field_area == "11":
        if date < datetime.datetime(2010, 10, 1):
            label = "B"
            last_crop = "SB"
        elif date < datetime.datetime(2011, 10, 1):
            label = "WW"
            last_crop = "B"
        elif date < datetime.datetime(2012, 12, 1):
            label = "SB"
            last_crop = "WW"
        elif date < datetime.datetime(2013, 12, 1):
            label = "irrelevant"
            last_crop = "SB"
        elif date < datetime.datetime(2014, 10, 1):
            label = "WW"
            last_crop = "unknown"
        elif date < datetime.datetime(2015, 12, 1):
            label = "SB"
            last_crop = "WW"
        elif date < datetime.datetime(2016, 12, 1):
            label = "irrelevant"
            last_crop = "SB"
        elif date < datetime.datetime(2017, 12, 1):
            label = "irrelevant"
            last_crop = "unknown"
            
    # Field area 13 - Kraichgau.
    elif field_area == "13":
        if date < datetime.datetime(2010, 12, 1):
            label = "SB"
            last_crop = "WW"
        elif date < datetime.datetime(2011, 10, 1):
            label = "B"
            last_crop = "SB"
        elif date < datetime.datetime(2012, 10, 1):
            label = "WW"
            last_crop = "B"
        elif date < datetime.datetime(2013, 10, 1):
            label = "WR"
            last_crop = "WW"
        elif date < datetime.datetime(2014, 10, 1):
            label = "WW"
            last_crop = "WR"
        elif date < datetime.datetime(2015, 12, 1):
            label = "SB"
            last_crop = "WW"
        elif date < datetime.datetime(2016, 12, 1):
            label = "irrelevant"
            last_crop = "SB"
        elif date < datetime.datetime(2017, 12, 1):
            label = "irrelevant"
            last_crop = "unknown"
            
    # Field area 15 - Kraichgau.
    elif field_area == "15":
        if date < datetime.datetime(2010, 12, 1):
            label = "SB"
            last_crop = "WW"
        elif date < datetime.datetime(2011, 10, 1):
            label = "WR"
            last_crop = "SB"
        elif date < datetime.datetime(2012, 10, 1):
            label = "WW"
            last_crop = "WR"
        elif date < datetime.datetime(2013, 12, 1):
            label = "irrelevant"
            last_crop = "WW"
        elif date < datetime.datetime(2014, 10, 1):
            label = "WR"
            last_crop = "SB"
        elif date < datetime.datetime(2015, 10, 1):
            label = "WW"
            last_crop = "WR"
        elif date < datetime.datetime(2016, 12, 1):
            label = "SOY"
            last_crop = "WW"
        elif date < datetime.datetime(2017, 12, 1):
            label = "irrelevant"
            last_crop = "SOY"
            
    # Field area 17 - Kraichgau.
    elif field_area == "17":
        if date < datetime.datetime(2010, 12, 1):
            label = "SB"
            last_crop = "B"
        elif date < datetime.datetime(2011, 12, 1):
            label = "GM"
            last_crop = "SB"
        elif date < datetime.datetime(2012, 10, 1):
            label = "WW"
            last_crop = "GM"
        elif date < datetime.datetime(2013, 12, 1):
            label = "MU"
            last_crop = "WW"
        elif date < datetime.datetime(2014, 10, 1):
            label = "WW"
            last_crop = "MU"
        elif date < datetime.datetime(2015, 12, 1):
            label = "SB"
            last_crop = "WW"
        elif date < datetime.datetime(2016, 12, 1):
            label = "irrelevant"
            last_crop = "SB"
        elif date < datetime.datetime(2017, 12, 1):
            label = "irrelevant"
            last_crop = "unknown"

    # Field area 18_1 - Alps.
    elif field_area == "18_1":
        if date < datetime.datetime(2010, 12, 1):
            label = "SB"
            last_crop = "C"
        elif date < datetime.datetime(2011, 10, 1):
            label = "SB"
            last_crop = "SB"
        elif date < datetime.datetime(2012, 10, 1):
            label = "SB"
            last_crop = "SB"
        elif date < datetime.datetime(2013, 10, 1):
            label = "SM"
            last_crop = "SB"
        elif date < datetime.datetime(2014, 10, 1):
            label = "WW"
            last_crop = "SM"
        elif date < datetime.datetime(2015, 12, 1):
            label = "SB"
            last_crop = "WW"
        elif date < datetime.datetime(2016, 12, 1):
            label = "irrelevant"
            last_crop = "SB"
        elif date < datetime.datetime(2017, 12, 1):
            label = "irrelevant"
            last_crop = "unknown"

    # Field area 18_2 - Alps.
    elif field_area == "18_2":
        if date < datetime.datetime(2010, 12, 1):
            label = "SB"
            last_crop = "WW"
        elif date < datetime.datetime(2011, 10, 1):
            label = "TC"
            last_crop = "SB"
        elif date < datetime.datetime(2012, 10, 1):
            label = "M"
            last_crop = "TC"
        elif date < datetime.datetime(2013, 10, 1):
            label = "WW"
            last_crop = "M"
        elif date < datetime.datetime(2014, 10, 1):
            label = "irrelevant"
            last_crop = "WW"
        elif date < datetime.datetime(2015, 12, 1):
            label = "SB"
            last_crop = "MU"
        elif date < datetime.datetime(2016, 12, 1):
            label = "P"
            last_crop = "SB"
        elif date < datetime.datetime(2017, 12, 1):
            label = "WW"
            last_crop = "P"
            
    # Field area 18_4 - Alps.
    elif field_area == "18_4":
        if date < datetime.datetime(2010, 12, 1):
            label = "SB"
            last_crop = "unknown"
        elif date < datetime.datetime(2011, 10, 1):
            label = "TC"
            last_crop = "SB"
        elif date < datetime.datetime(2012, 10, 1):
            label = "irrelevant"
            last_crop = "TC"
        elif date < datetime.datetime(2013, 10, 1):
            label = "SB"
            last_crop = "unknown"
        elif date < datetime.datetime(2014, 10, 1):
            label = "irrelevant"
            last_crop = "SB"
        elif date < datetime.datetime(2015, 12, 1):
            label = "irrelevant"
            last_crop = "unknown"
        elif date < datetime.datetime(2016, 12, 1):
            label = "WW"
            last_crop = "unknown"
        elif date < datetime.datetime(2017, 12, 1):
            label = "irrelevant"
            last_crop = "WW"
            
    # Field area 19 - Alps.
    elif field_area == "19":
        if date < datetime.datetime(2010, 12, 1):
            label = "SP"
            last_crop = "SM"
        elif date < datetime.datetime(2011, 10, 1):
            label = "SB"
            last_crop = "SP"
        elif date < datetime.datetime(2012, 10, 1):
            label = "WB"
            last_crop = "SB"
        elif date < datetime.datetime(2013, 10, 1):
            label = "WR"
            last_crop = "WB"
        elif date < datetime.datetime(2014, 10, 1):
            label = "WW"
            last_crop = "WR"
        elif date < datetime.datetime(2015, 12, 1):
            label = "irrelevant"
            last_crop = "WW"
        elif date < datetime.datetime(2016, 12, 1):
            label = "WW"
            last_crop = "unknown"
        elif date < datetime.datetime(2017, 12, 1):
            label = "WW"
            last_crop = "WW"
            
    # Field area 20 - Alps.
    elif field_area == "20":
        if date < datetime.datetime(2010, 12, 1):
            label = "SP"
            last_crop = "SP"
        elif date < datetime.datetime(2011, 10, 1):
            label = "WB"
            last_crop = "SP"
        elif date < datetime.datetime(2012, 10, 1):
            label = "WR"
            last_crop = "WB"
        elif date < datetime.datetime(2013, 10, 1):
            label = "SP"
            last_crop = "WR"
        elif date < datetime.datetime(2014, 10, 1):
            label = "irrelevant"
            last_crop = "SP"
        elif date < datetime.datetime(2015, 12, 1):
            label = "WB"
            last_crop = "unknown"
        elif date < datetime.datetime(2016, 12, 1):
            label = "WR"
            last_crop = "WB"
        elif date < datetime.datetime(2017, 12, 1):
            label = "WW"
            last_crop = "WR"
            
    # Field area 21 - Alps.
    elif field_area == "21":
        if date < datetime.datetime(2010, 12, 1):
            label = "TC"
            last_crop = "WB"
        elif date < datetime.datetime(2011, 10, 1):
            label = "irrelevant"
            last_crop = "TC"
        elif date < datetime.datetime(2012, 10, 1):
            label = "C"
            last_crop = "unknown"
        elif date < datetime.datetime(2013, 10, 1):
            label = "irrelevant"
            last_crop = "C"
        elif date < datetime.datetime(2014, 10, 1):
            label = "C"
            last_crop = "unknown"
        elif date < datetime.datetime(2015, 12, 1):
            label = "SP"
            last_crop = "C"
        elif date < datetime.datetime(2016, 12, 1):
            label = "irrelevant"
            last_crop = "SP"
        elif date < datetime.datetime(2017, 12, 1):
            label = "SB"
            last_crop = "unknown"
            
    # Field area 23 - Alps.
    elif field_area == "23":
        if date < datetime.datetime(2010, 12, 1):
            label = "P"
            last_crop = "SB"
        elif date < datetime.datetime(2011, 10, 1):
            label = "WW"
            last_crop = "P"
        else:
            label = "irrelevant"
            last_crop = "unknown"
            
    # Field area 25 - Alps.
    elif field_area == "25":
        if date < datetime.datetime(2010, 12, 1):
            label = "O"
            last_crop = "WB"
        elif date < datetime.datetime(2011, 10, 1):
            label = "SB"
            last_crop = "O"
        elif date < datetime.datetime(2012, 10, 1):
            label = "C"
            last_crop = "SB"
        elif date < datetime.datetime(2013, 10, 1):
            label = "C"
            last_crop = "C"
        elif date < datetime.datetime(2014, 10, 1):
            label = "C"
            last_crop = "C"
        elif date < datetime.datetime(2015, 12, 1):
            label = "C"
            last_crop = "C"
        elif date < datetime.datetime(2016, 12, 1):
            label = "C"
            last_crop = "C"
        elif date < datetime.datetime(2017, 12, 1):
            label = "C"
            last_crop = "C"
            
    # Field area 26 - Alps.
    elif field_area == "26":
        if date < datetime.datetime(2010, 12, 1):
            label = "WW"
            last_crop = "unknown"
        elif date < datetime.datetime(2011, 10, 1):
            label = "SB"
            last_crop = "WW"
        elif date < datetime.datetime(2012, 10, 1):
            label = "SB"
            last_crop = "SB"
        elif date < datetime.datetime(2013, 10, 1):
            label = "WR"
            last_crop = "SB"
        elif date < datetime.datetime(2014, 10, 1):
            label = "WW"
            last_crop = "WR"
        elif date < datetime.datetime(2015, 12, 1):
            label = "C"
            last_crop = "WW"
        elif date < datetime.datetime(2016, 12, 1):
            label = "SP"
            last_crop = "C"
        elif date < datetime.datetime(2017, 12, 1):
            label = "irrelevant"
            last_crop = "SP"
            
    # Field area 27 - Alps.
    elif field_area == "27":
        if date < datetime.datetime(2010, 12, 1):
            label = "irrelevant"
            last_crop = "unknown"
        elif date < datetime.datetime(2011, 10, 1):
            label = "WW"
            last_crop = "unknown"
        elif date < datetime.datetime(2012, 10, 1):
            label = "irrelevant"
            last_crop = "WW"
        elif date < datetime.datetime(2013, 10, 1):
            label = "irrelevant"
            last_crop = "unknown"
        elif date < datetime.datetime(2014, 10, 1):
            label = "C"
            last_crop = "unknown"
        elif date < datetime.datetime(2015, 12, 1):
            label = "C"
            last_crop = "C"
        elif date < datetime.datetime(2016, 12, 1):
            label = "C"
            last_crop = "C"
        elif date < datetime.datetime(2017, 12, 1):
            label = "C"
            last_crop = "C"
            
    # Field area 28 - Alps.
    elif field_area == "28":
        if date < datetime.datetime(2010, 12, 1):
            label = "WB"
            last_crop = "SP"
        elif date < datetime.datetime(2011, 10, 1):
            label = "WR"
            last_crop = "WB"
        elif date < datetime.datetime(2012, 10, 1):
            label = "WW"
            last_crop = "WR"
        elif date < datetime.datetime(2013, 10, 1):
            label = "WW"
            last_crop = "WW"
        elif date < datetime.datetime(2014, 10, 1):
            label = "irrelevant"
            last_crop = "WW"
        elif date < datetime.datetime(2015, 12, 1):
            label = "SP"
            last_crop = "unknown"
        elif date < datetime.datetime(2016, 12, 1):
            label = "WB"
            last_crop = "SP"
        elif date < datetime.datetime(2017, 12, 1):
            label = "WR"
            last_crop = "WB"
            
    # Field area 210 - Alps.
    elif field_area == "210":
        if date < datetime.datetime(2010, 12, 1):
            label = "SB"
            last_crop = "O"
        elif date < datetime.datetime(2011, 10, 1):
            label = "SM"
            last_crop = "SB"
        elif date < datetime.datetime(2012, 10, 1):
            label = "WW"
            last_crop = "SM"
        elif date < datetime.datetime(2013, 10, 1):
            label = "M"
            last_crop = "WW"
        elif date < datetime.datetime(2014, 10, 1):
            label = "WW"
            last_crop = "M"
        elif date < datetime.datetime(2015, 12, 1):
            label = "SB"
            last_crop = "WW"
        elif date < datetime.datetime(2017, 12, 1):
            label = "irrelevant"
            last_crop = "unknown"
            
    # Field area 30 - Alps.
    elif field_area == "30":
        if date < datetime.datetime(2010, 12, 1):
            label = "O"
            last_crop = "SB"
        elif date < datetime.datetime(2011, 10, 1):
            label = "SP"
            last_crop = "O"
        elif date < datetime.datetime(2012, 10, 1):
            label = "SB"
            last_crop = "SP"
        elif date < datetime.datetime(2013, 10, 1):
            label = "irrelevant"
            last_crop = "SB"
        elif date < datetime.datetime(2014, 10, 1):
            label = "C"
            last_crop = "unknown"
        elif date < datetime.datetime(2015, 12, 1):
            label = "WW"
            last_crop = "C"
        elif date < datetime.datetime(2016, 12, 1):
            label = "irrelevant"
            last_crop = "WW"
        elif date < datetime.datetime(2017, 12, 1):
            label = "irrelevant"
            last_crop = "unknown"
            
    # Field area 33 - Alps.
    elif field_area == "33":
        if date < datetime.datetime(2010, 12, 1):
            label = "irrelevant"
            last_crop = "TC"
        elif date < datetime.datetime(2011, 10, 1):
            label = "SM"
            last_crop = "unknown"
        elif date < datetime.datetime(2012, 10, 1):
            label = "SB"
            last_crop = "SM"
        elif date < datetime.datetime(2013, 10, 1):
            label = "O"
            last_crop = "SB"
        elif date < datetime.datetime(2014, 10, 1):
            label = "irrelevant"
            last_crop = "O"
        elif date < datetime.datetime(2015, 12, 1):
            label = "irrelevant"
            last_crop = "unknown"
        elif date < datetime.datetime(2016, 12, 1):
            label = "WW"
            last_crop = "unknown"
        elif date < datetime.datetime(2017, 12, 1):
            label = "irrelevant"
            last_crop = "unknown"
            
    if field_area == "1_2019":
        if date < datetime.datetime(2019, 12, 1):
            label = "WW"
            last_crop = "unknown"
    elif field_area == "2_2019":
        if date < datetime.datetime(2019, 12, 1):
            label = "WB"
            last_crop = "unknown"
    elif field_area == "3_2019":
        if date < datetime.datetime(2019, 12, 1):
            label = "WB"
            last_crop = "unknown"
    elif field_area == "4_2019":
        if date < datetime.datetime(2019, 12, 1):
            label = "SB"
            last_crop = "unknown"
    elif field_area == "5_2019":
        if date < datetime.datetime(2019, 12, 1):
            label = "SB"
            last_crop = "unknown"
    elif field_area == "7_2019":
        if date < datetime.datetime(2019, 12, 1):
            label = "SM"
            last_crop = "unknown"
    elif field_area == "8_2019":
        if date < datetime.datetime(2019, 12, 1):
            label = "WW"
            last_crop = "unknown"
    elif field_area == "10_2019":
        label = "WW"
        last_crop = "unknown"
    elif field_area == "11_2019":
        label = "WW"
        last_crop = "unknown"
    elif field_area == "12_2019":
        label = "SM"
        last_crop = "unknown"
    elif field_area == "13_2019":
        label = "WW"
        last_crop = "unknown"
    elif field_area == "14_2019":
        label = "WW"
        last_crop = "unknown"
    elif field_area == "15_2019":
        label = "WR"
        last_crop = "unknown"
    elif field_area == "16_2019":
        label = "WW"
        last_crop = "unknown"
    elif field_area == "17_2019":
        label = "SB"
        last_crop = "unknown"
    elif field_area == "18_2019":
        label = "C"
        last_crop = "unknown"
    elif field_area == "19_2019":
        label = "WW"
        last_crop = "unknown"
    elif field_area == "20_2019":
        label = "C"
        last_crop = "unknown"
    elif field_area == "21_2019":
        label = "TC"
        last_crop = "unknown"
    elif field_area == "22_2019":
        label = "WR"
        last_crop = "unknown"
    elif field_area == "23_2019":
        label = "WR"
        last_crop = "unknown"
    elif field_area == "24_2019":
        label = "B"
        last_crop = "unknown"
    elif field_area == "25_2019":
        label = "R"
        last_crop = "unknown"
    elif field_area == "26_2019":
        label = "SB"
        last_crop = "unknown"
    elif field_area == "27_2019":
        label = "B"
        last_crop = "unknown"
    elif field_area == "28_2019":
        label = "WW"
        last_crop = "unknown"
    elif field_area == "29_2019":
        label = "WB"
        last_crop = "unknown"
    elif field_area == "30_2019":
        label = "WR"
        last_crop = "unknown"
    elif field_area == "31_2019":
        label = "irrelevant"
        last_crop = "unknown"
    elif field_area == "32_2019":
        label = "WW"
        last_crop = "unknown"
    elif field_area == "33_2019":
        label = "WW"
        last_crop = "unknown"
    elif field_area == "34_2019":
        label = "WB"
        last_crop = "unknown"
    elif field_area == "35_2019":
        label = "ME"
        last_crop = "unknown"
    elif field_area == "36_2019":
        label = "ME"
        last_crop = "unknown"
    elif field_area == "37_2019":
        label = "WW"
        last_crop = "unknown"
    elif field_area == "38_2019":
        label = "WW"
        last_crop = "unknown"
    elif field_area == "39_2019":
        label = "SM"
        last_crop = "unknown"
    elif field_area == "40_2019":
        label = "SB"
        last_crop = "unknown"
    elif field_area == "41_2019":
        label = "irrelevant"
        last_crop = "unknown"
    elif field_area == "42_2019":
        label = "SM"
        last_crop = "unknown"
    elif field_area == "43_2019":
        label = "WR"
        last_crop = "unknown"
    elif field_area == "44_2019":
        label = "WR"
        last_crop = "unknown"
    elif field_area == "45_2019":
        label = "WB"
        last_crop = "unknown"
    elif field_area == "46_2019":
        label = "WW"
        last_crop = "unknown"
    elif field_area == "47_2019":
        label = "SB"
        last_crop = "unknown"
    elif field_area == "48_2019":
        label = "WW"
        last_crop = "unknown"
    elif field_area == "49_2019":
        label = "WW"
        last_crop = "unknown"
    elif field_area == "50_2019":
        label = "SM"
        last_crop = "unknown"
    elif field_area == "51_2019":
        label = "ME"
        last_crop = "unknown"
    elif field_area == "52_2019":
        label = "ME"
        last_crop = "unknown"
    elif field_area == "53_2019":
        label = "WW"
        last_crop = "unknown"
    elif field_area == "54_2019":
        label = "WB"
        last_crop = "unknown"
    elif field_area == "55_2019":
        label = "WW"
        last_crop = "unknown"
    elif field_area == "56_2019":
        label = "WW"
        last_crop = "unknown"
    elif field_area == "57_2019":
        label = "WR"
        last_crop = "unknown"
    elif field_area == "58_2019":
        label = "irrelevant"
        last_crop = "unknown"
    elif field_area == "59_2019":
        label = "WW"
        last_crop = "unknown"
    elif field_area == "60_2019":
        label = "C"
        last_crop = "unknown"
    elif field_area == "61_2019":
        label = "ME"
        last_crop = "unknown"
    elif field_area == "62_2019":
        label = "ME"
        last_crop = "unknown"
    elif field_area == "Cloud":
        label = "Cloud"
        last_crop = "unknown"
    elif field_area == "CloudShadow":
        label = "CloudShadow"
        last_crop = "unknown"

    last_crop = "unknown"
            
            
    return label, last_crop

def get_label_combined_maize(field_area, date):
    '''
    Returns the label of an image based on the csv that shows which label is in
    which area by time.
    
    Where:
        SM: Silomaise
        WW: Winter wheat
        WR: Winter rape
        WB: Winter barley
        GM: Grain maise
        SB: Summer barley
        SP: Spelt / Dinkel
        B: Beet
        C: Clover / Kleegras
        TC: Triticale
        M: Mais
        O: Oats
        MU: Mustard
        R: Rye
        P: Peas
        LU: Luzerne
        SOY: Soja
        Irrelevant: Crop not being classified
    '''
    # If the crop is either a cover crop or no cover (during winter).
    if date.month > 8 or date.month < 3:
        label = "irrelevant"
        last_crop = "unknown"
    elif date.year >= 2018:
        label = "irrelevant"
        last_crop = "unknown"
        
    # Field EC1 - Kraichgau.
    elif field_area == "EC1":
        if date < datetime.datetime(2010, 12, 1):
            label = "M"
            last_crop = "unknown"
        elif date < datetime.datetime(2011, 10, 1):
            label = "WW"
            last_crop = "M"
        elif date < datetime.datetime(2012, 10, 1):
            label = "WR"
            last_crop = "WW"
        elif date < datetime.datetime(2013, 10, 1):
            label = "WW"
            last_crop = "WR"
        elif date < datetime.datetime(2014, 12, 1):
            label = "M"
            last_crop = "WW"
        elif date < datetime.datetime(2015, 10, 1):
            label = "WW"
            last_crop = "M"
        elif date < datetime.datetime(2016, 12, 1):
            label = "M"
            last_crop = "WW"
        elif date < datetime.datetime(2017, 10, 1):
            label = "WW"
            last_crop = "M"
        elif date < datetime.datetime(2018, 10, 1):
            label = "WR"
            last_crop = "WW"
    
    # Field EC2 - Kraichgau.
    elif field_area == "EC2":
        if date < datetime.datetime(2010, 10, 1):
            label = "WR"
            last_crop = "unknown"
        elif date < datetime.datetime(2011, 10, 1):
            label = "WW"
            last_crop = "WR"
        elif date < datetime.datetime(2012, 12, 1):
            label = "M"
            last_crop = "WW"
        elif date < datetime.datetime(2013, 10, 1):
            label = "WW"
            last_crop = "M"
        elif date < datetime.datetime(2014, 12, 1):
            label = "M"
            last_crop = "WW"
        elif date < datetime.datetime(2015, 10, 1):
            label = "WW"
            last_crop = "M"
        elif date < datetime.datetime(2016, 10, 1):
            label = "WR"
            last_crop = "WW"
        elif date < datetime.datetime(2017, 10, 1):
            label = "WW"
            last_crop = "WR"
        elif date < datetime.datetime(2018, 10, 1):
            label = "WW"
            last_crop = "WW"
    
    # Field EC3 - Kraichgau.
    elif field_area == "EC3":
        if date < datetime.datetime(2010, 10, 1):
            label = "WW"
            last_crop = "unknown"
        elif date < datetime.datetime(2011, 12, 1):
            label = "M"
            last_crop = "WW"
        elif date < datetime.datetime(2012, 10, 1):
            label = "WW"
            last_crop = "M"
        elif date < datetime.datetime(2013, 10, 1):
            label = "WR"
            last_crop = "WW"
        elif date < datetime.datetime(2014, 10, 1):
            label = "WW"
            last_crop = "WR"
        elif date < datetime.datetime(2015, 12, 1):
            label = "M"
            last_crop = "WW"
        elif date < datetime.datetime(2016, 10, 1):
            label = "WW"
            last_crop = "M"
        elif date < datetime.datetime(2017, 10, 1):
            label = "WW"
            last_crop = "WW"
        elif date < datetime.datetime(2018, 12, 1):
            label = "M"
            last_crop = "WW"
         
    # Field EC4 - Swaebisch Alp.
    elif field_area == "EC4":
        if date < datetime.datetime(2010, 10, 1):
            label = "WR"
            last_crop = "unknown"
        elif date < datetime.datetime(2011, 10, 1):
            label = "WW"
            last_crop = "WR"
        elif date < datetime.datetime(2012, 12, 1):
            label = "SB"
            last_crop = "WW"
        elif date < datetime.datetime(2013, 10, 1):
            label = "WR"
            last_crop = "SB"
        elif date < datetime.datetime(2014, 10, 1):
            label = "WW"
            last_crop = "WR"
        elif date < datetime.datetime(2015, 10, 1):
            label = "WW" 
            last_crop = "WW"
        elif date < datetime.datetime(2016, 12, 1):
            label = "SB"
            last_crop = "WW"
        elif date < datetime.datetime(2017, 12, 1):
            label = "M"
            last_crop = "SB"
        elif date < datetime.datetime(2018, 10, 1):
            label = "WW"
            last_crop = "M"
            
    # Field EC5 - Swaebisch Alp.
    elif field_area == "EC5":
        if date < datetime.datetime(2010, 10, 1):
            label = "WW"
            last_crop = "unknown"
        elif date < datetime.datetime(2011, 12, 1):
            label = "M"
            last_crop = "WW"
        elif date < datetime.datetime(2012, 12, 1):
            label = "M"
            last_crop = "SM"
        elif date < datetime.datetime(2013, 10, 1):
            label = "WB"
            last_crop = "M"
        elif date < datetime.datetime(2014, 12, 1):
            label = "SP"
            last_crop = "WB"
        elif date < datetime.datetime(2015, 12, 1):
            label = "M"
            last_crop = "SP"
        elif date < datetime.datetime(2016, 12, 1):
            label = "M"
            last_crop = "M"
        elif date < datetime.datetime(2017, 10, 1):
            label = "WB"
            last_crop = "M"
        elif date < datetime.datetime(2018, 10, 1):
            label = "WR"
            last_crop = "WB"
            
    # Field EC6 - Swaebisch Alp.
    elif field_area == "EC6":
        if date < datetime.datetime(2010, 12, 1):
            label = "M"
            last_crop = "unknown"
        elif date < datetime.datetime(2011, 10, 1):
            label = "WW"
            last_crop = "M"
        elif date < datetime.datetime(2012, 10, 1):
            label = "WB"
            last_crop = "WW"
        elif date < datetime.datetime(2013, 12, 1):
            label = "M"
            last_crop = "WB"
        elif date < datetime.datetime(2014, 10, 1):
            label = "WW"
            last_crop = "M"
        elif date < datetime.datetime(2015, 10, 1):
            label = "WB"
            last_crop = "WW"
        elif date < datetime.datetime(2016, 12, 1):
            label = "M"
            last_crop = "WB"
        elif date < datetime.datetime(2017, 10, 1):
            label = "WW"
            last_crop = "SM"
        elif date < datetime.datetime(2018, 10, 1):
            label = "WB"
            last_crop = "WW"
    
    # Field area 1-1 - Kraichgau.
    elif field_area == "1_1":
        if date < datetime.datetime(2010, 10, 1):
            label = "WR"
            last_crop = "unknown"
        elif date < datetime.datetime(2011, 10, 1):
            label = "WW"
            last_crop = "WR"
        elif date < datetime.datetime(2011, 12, 1):
            label = "M"
            last_crop= "WW"
        elif date < datetime.datetime(2012, 12, 1):
            label = "SB"
            last_crop = "M"
        elif date < datetime.datetime(2013, 10, 1):
            label = "WR"
            last_crop = "SB"
        elif date < datetime.datetime(2014, 10, 1):
            label = "WW"
            last_crop = "WR"
        elif date < datetime.datetime(2015, 12, 1):
            label = "irrelevant"
            last_crop = "WW"
        elif date < datetime.datetime(2016, 10, 1):
            label = "WW"
            last_crop = "unknown"
        elif date < datetime.datetime(2017, 12, 1):
            label = "irrelevant"
            last_crop = "WW"

    # Field area 1-4 - Kraichgau.
    elif field_area == "1_4":
        if date < datetime.datetime(2010, 10, 1):
            label = "B"
            last_crop = "SB"
        elif date < datetime.datetime(2011, 10, 1):
            label = "M"
            last_crop = "B"
        elif date < datetime.datetime(2012, 12, 1):
            label = "WW"
            last_crop= "M"
        elif date < datetime.datetime(2013, 10, 1):
            label = "SB"
            last_crop = "WW"
        elif date < datetime.datetime(2014, 10, 1):
            label = "B"
            last_crop = "SB"
        elif date < datetime.datetime(2015, 12, 1):
            label = "WW"
            last_crop = "B"
        elif date < datetime.datetime(2016, 10, 1):
            label = "M"
            last_crop = "WW"
        elif date < datetime.datetime(2017, 10, 1):
            label = "WW"
            last_crop = "M"
            
    # Field area 2 - Kraichgau.
    elif field_area == "2":
        if date < datetime.datetime(2010, 10, 1):
            label = "WB"
            last_crop = "WW"
        elif date < datetime.datetime(2011, 10, 1):
            label = "irrelevant"
            last_crop = "WB"
        elif date < datetime.datetime(2012, 10, 1):
            label = "C"
            last_crop = "unknown"
        elif date < datetime.datetime(2013, 10, 1):
            label = "SP"
            last_crop = "C"
        elif date < datetime.datetime(2014, 10, 1):
            label = "irrelevant"
            last_crop = "SP"
        elif date < datetime.datetime(2015, 12, 1):
            label = "SB"
            last_crop = "unknown"
        elif date < datetime.datetime(2016, 10, 1):
            label = "LU"
            last_crop = "SB"
        elif date < datetime.datetime(2017, 10, 1):
            label = "LU"
            last_crop = "LU"
            
    # Field area 3 - Kraichgau.
    elif field_area == "3":
        if date < datetime.datetime(2010, 10, 1):
            label = "WR"
            last_crop = "SB"
        elif date < datetime.datetime(2011, 10, 1):
            label = "WW"
            last_crop = "WR"
        elif date < datetime.datetime(2012, 12, 1):
            label = "irrelevant"
            last_crop = "WW"
        elif date < datetime.datetime(2013, 10, 1):
            label = "WR"
            last_crop = "unknown"
        elif date < datetime.datetime(2014, 10, 1):
            label = "WW"
            last_crop = "WR"
        elif date < datetime.datetime(2017, 12, 1):
            label = "irrelevant"
            last_crop = "SM"

    # Field area 4 - Kraichgau.
    elif field_area == "4":
        if date < datetime.datetime(2010, 10, 1):
            label = "B"
            last_crop = "SB"
        elif date < datetime.datetime(2011, 10, 1):
            label = "WW"
            last_crop = "B"
        elif date < datetime.datetime(2012, 10, 1):
            label = "WR"
            last_crop = "WW"
        elif date < datetime.datetime(2013, 10, 1):
            label = "WW"
            last_crop = "WR"
        elif date < datetime.datetime(2014, 12, 1):
            label = "irrelevant"
            last_crop = "WW"
        elif date < datetime.datetime(2015, 12, 1):
            label = "SB"
            last_crop = "unknown"
        elif date < datetime.datetime(2016, 10, 1):
            label = "B"
            last_crop = "SB"
        elif date < datetime.datetime(2017, 12, 1):
            label = "irrelevant"
            last_crop = "unknown"
            
    # Field area 5 - Kraichgau.
    elif field_area == "5":
        if date < datetime.datetime(2010, 10, 1):
            label = "WW"
            last_crop = "WW"
        elif date < datetime.datetime(2011, 12, 1):
            label = "M"
            last_crop = "WW"
        elif date < datetime.datetime(2012, 12, 1):
            label = "B"
            last_crop = "M"
        elif date < datetime.datetime(2013, 10, 1):
            label = "WW"
            last_crop = "B"
        elif date < datetime.datetime(2014, 12, 1):
            label = "M"
            last_crop = "WW"
        elif date < datetime.datetime(2015, 10, 1):
            label = "WW"
            last_crop = "M"
        elif date < datetime.datetime(2016, 10, 1):
            label = "WB"
            last_crop = "WW"
        elif date < datetime.datetime(2017, 12, 1):
            label = "irrelevant"
            last_crop = "WB"
            
    # Field area 6 - Kraichgau.
    elif field_area == "6":
        if date < datetime.datetime(2010, 10, 1):
            label = "WR"
            last_crop = "WW"
        elif date < datetime.datetime(2011, 10, 1):
            label = "WW"
            last_crop = "WR"
        elif date < datetime.datetime(2012, 10, 1):
            label = "WR"
            last_crop = "WW"
        elif date < datetime.datetime(2013, 10, 1):
            label = "WW"
            last_crop = "WR"
        elif date < datetime.datetime(2014, 12, 1):
            label = "irrelevant"
            last_crop = "WW"
        elif date < datetime.datetime(2015, 10, 1):
            label = "WW"
            last_crop = "unknown"
        elif date < datetime.datetime(2016, 10, 1):
            label = "WR"
            last_crop = "WW"
        elif date < datetime.datetime(2017, 10, 1):
            label = "WW"
            last_crop = "WR"
        elif date < datetime.datetime(2017, 12, 1):
            label = "irrelevant"
            last_crop = "unknown"
            
    # Field area 8 - Kraichgau.
    elif field_area == "8":
        if date < datetime.datetime(2010, 10, 1):
            label = "WW"
            last_crop = "B"
        elif date < datetime.datetime(2011, 10, 1):
            label = "WR"
            last_crop = "WW"
        elif date < datetime.datetime(2012, 10, 1):
            label = "WW"
            last_crop = "WR"
        elif date < datetime.datetime(2013, 10, 1):
            label = "R"
            last_crop = "WW"
        elif date < datetime.datetime(2014, 12, 1):
            label = "irrelevant"
            last_crop = "unknown"
        elif date < datetime.datetime(2015, 10, 1):
            label = "WW"
            last_crop = "unknown"
        elif date < datetime.datetime(2016, 10, 1):
            label = "R"
            last_crop = "WW"
        elif date < datetime.datetime(2017, 12, 1):
            label = "irrelevant"
            last_crop = "unknown"
            
    # Field area 9 - Kraichgau.
    elif field_area == "9":
        if date < datetime.datetime(2010, 10, 1):
            label = "WW"
            last_crop = "WR"
        elif date < datetime.datetime(2011, 12, 1):
            label = "M"
            last_crop = "WW"
        elif date < datetime.datetime(2012, 10, 1):
            label = "WW"
            last_crop = "M"
        elif date < datetime.datetime(2013, 12, 1):
            label = "SB"
            last_crop = "WW"
        elif date < datetime.datetime(2014, 10, 1):
            label = "WR"
            last_crop = "SB"
        elif date < datetime.datetime(2015, 10, 1):
            label = "WW"
            last_crop = "WR"
        elif date < datetime.datetime(2016, 12, 1):
            label = "irrelevant"
            last_crop = "WW"
        elif date < datetime.datetime(2017, 12, 1):
            label = "irrelevant"
            last_crop = "unknown"
            
    # Field area 11 - Kraichgau.
    elif field_area == "11":
        if date < datetime.datetime(2010, 10, 1):
            label = "B"
            last_crop = "SB"
        elif date < datetime.datetime(2011, 10, 1):
            label = "WW"
            last_crop = "B"
        elif date < datetime.datetime(2012, 12, 1):
            label = "SB"
            last_crop = "WW"
        elif date < datetime.datetime(2013, 12, 1):
            label = "irrelevant"
            last_crop = "SB"
        elif date < datetime.datetime(2014, 10, 1):
            label = "WW"
            last_crop = "unknown"
        elif date < datetime.datetime(2015, 12, 1):
            label = "SB"
            last_crop = "WW"
        elif date < datetime.datetime(2016, 12, 1):
            label = "irrelevant"
            last_crop = "SB"
        elif date < datetime.datetime(2017, 12, 1):
            label = "irrelevant"
            last_crop = "unknown"
            
    # Field area 13 - Kraichgau.
    elif field_area == "13":
        if date < datetime.datetime(2010, 12, 1):
            label = "SB"
            last_crop = "WW"
        elif date < datetime.datetime(2011, 10, 1):
            label = "B"
            last_crop = "SB"
        elif date < datetime.datetime(2012, 10, 1):
            label = "WW"
            last_crop = "B"
        elif date < datetime.datetime(2013, 10, 1):
            label = "WR"
            last_crop = "WW"
        elif date < datetime.datetime(2014, 10, 1):
            label = "WW"
            last_crop = "WR"
        elif date < datetime.datetime(2015, 12, 1):
            label = "SB"
            last_crop = "WW"
        elif date < datetime.datetime(2016, 12, 1):
            label = "irrelevant"
            last_crop = "SB"
        elif date < datetime.datetime(2017, 12, 1):
            label = "irrelevant"
            last_crop = "unknown"
            
    # Field area 15 - Kraichgau.
    elif field_area == "15":
        if date < datetime.datetime(2010, 12, 1):
            label = "SB"
            last_crop = "WW"
        elif date < datetime.datetime(2011, 10, 1):
            label = "WR"
            last_crop = "SB"
        elif date < datetime.datetime(2012, 10, 1):
            label = "WW"
            last_crop = "WR"
        elif date < datetime.datetime(2013, 12, 1):
            label = "irrelevant"
            last_crop = "WW"
        elif date < datetime.datetime(2014, 10, 1):
            label = "WR"
            last_crop = "SB"
        elif date < datetime.datetime(2015, 10, 1):
            label = "WW"
            last_crop = "WR"
        elif date < datetime.datetime(2016, 12, 1):
            label = "SOY"
            last_crop = "WW"
        elif date < datetime.datetime(2017, 12, 1):
            label = "irrelevant"
            last_crop = "SOY"
            
    # Field area 17 - Kraichgau.
    elif field_area == "17":
        if date < datetime.datetime(2010, 12, 1):
            label = "SB"
            last_crop = "B"
        elif date < datetime.datetime(2011, 12, 1):
            label = "M"
            last_crop = "SB"
        elif date < datetime.datetime(2012, 10, 1):
            label = "WW"
            last_crop = "M"
        elif date < datetime.datetime(2013, 12, 1):
            label = "MU"
            last_crop = "WW"
        elif date < datetime.datetime(2014, 10, 1):
            label = "WW"
            last_crop = "MU"
        elif date < datetime.datetime(2015, 12, 1):
            label = "SB"
            last_crop = "WW"
        elif date < datetime.datetime(2016, 12, 1):
            label = "irrelevant"
            last_crop = "SB"
        elif date < datetime.datetime(2017, 12, 1):
            label = "irrelevant"
            last_crop = "unknown"

    # Field area 18_1 - Alps.
    elif field_area == "18_1":
        if date < datetime.datetime(2010, 12, 1):
            label = "SB"
            last_crop = "C"
        elif date < datetime.datetime(2011, 10, 1):
            label = "SB"
            last_crop = "SB"
        elif date < datetime.datetime(2012, 10, 1):
            label = "SB"
            last_crop = "SB"
        elif date < datetime.datetime(2013, 10, 1):
            label = "M"
            last_crop = "SB"
        elif date < datetime.datetime(2014, 10, 1):
            label = "WW"
            last_crop = "M"
        elif date < datetime.datetime(2015, 12, 1):
            label = "SB"
            last_crop = "WW"
        elif date < datetime.datetime(2016, 12, 1):
            label = "irrelevant"
            last_crop = "SB"
        elif date < datetime.datetime(2017, 12, 1):
            label = "irrelevant"
            last_crop = "unknown"

    # Field area 18_2 - Alps.
    elif field_area == "18_2":
        if date < datetime.datetime(2010, 12, 1):
            label = "SB"
            last_crop = "WW"
        elif date < datetime.datetime(2011, 10, 1):
            label = "TC"
            last_crop = "SB"
        elif date < datetime.datetime(2012, 10, 1):
            label = "M"
            last_crop = "TC"
        elif date < datetime.datetime(2013, 10, 1):
            label = "WW"
            last_crop = "M"
        elif date < datetime.datetime(2014, 10, 1):
            label = "irrelevant"
            last_crop = "WW"
        elif date < datetime.datetime(2015, 12, 1):
            label = "SB"
            last_crop = "MU"
        elif date < datetime.datetime(2016, 12, 1):
            label = "P"
            last_crop = "SB"
        elif date < datetime.datetime(2017, 12, 1):
            label = "WW"
            last_crop = "P"
            
    # Field area 18_4 - Alps.
    elif field_area == "18_4":
        if date < datetime.datetime(2010, 12, 1):
            label = "SB"
            last_crop = "unknown"
        elif date < datetime.datetime(2011, 10, 1):
            label = "TC"
            last_crop = "SB"
        elif date < datetime.datetime(2012, 10, 1):
            label = "irrelevant"
            last_crop = "TC"
        elif date < datetime.datetime(2013, 10, 1):
            label = "SB"
            last_crop = "unknown"
        elif date < datetime.datetime(2014, 10, 1):
            label = "irrelevant"
            last_crop = "SB"
        elif date < datetime.datetime(2015, 12, 1):
            label = "irrelevant"
            last_crop = "unknown"
        elif date < datetime.datetime(2016, 12, 1):
            label = "WW"
            last_crop = "unknown"
        elif date < datetime.datetime(2017, 12, 1):
            label = "irrelevant"
            last_crop = "WW"
            
    # Field area 19 - Alps.
    elif field_area == "19":
        if date < datetime.datetime(2010, 12, 1):
            label = "SP"
            last_crop = "M"
        elif date < datetime.datetime(2011, 10, 1):
            label = "SB"
            last_crop = "SP"
        elif date < datetime.datetime(2012, 10, 1):
            label = "WB"
            last_crop = "SB"
        elif date < datetime.datetime(2013, 10, 1):
            label = "WR"
            last_crop = "WB"
        elif date < datetime.datetime(2014, 10, 1):
            label = "WW"
            last_crop = "WR"
        elif date < datetime.datetime(2015, 12, 1):
            label = "irrelevant"
            last_crop = "WW"
        elif date < datetime.datetime(2016, 12, 1):
            label = "WW"
            last_crop = "unknown"
        elif date < datetime.datetime(2017, 12, 1):
            label = "WW"
            last_crop = "WW"
            
    # Field area 20 - Alps.
    elif field_area == "20":
        if date < datetime.datetime(2010, 12, 1):
            label = "SP"
            last_crop = "SP"
        elif date < datetime.datetime(2011, 10, 1):
            label = "WB"
            last_crop = "SP"
        elif date < datetime.datetime(2012, 10, 1):
            label = "WR"
            last_crop = "WB"
        elif date < datetime.datetime(2013, 10, 1):
            label = "SP"
            last_crop = "WR"
        elif date < datetime.datetime(2014, 10, 1):
            label = "irrelevant"
            last_crop = "SP"
        elif date < datetime.datetime(2015, 12, 1):
            label = "WB"
            last_crop = "unknown"
        elif date < datetime.datetime(2016, 12, 1):
            label = "WR"
            last_crop = "WB"
        elif date < datetime.datetime(2017, 12, 1):
            label = "WW"
            last_crop = "WR"
            
    # Field area 21 - Alps.
    elif field_area == "21":
        if date < datetime.datetime(2010, 12, 1):
            label = "TC"
            last_crop = "WB"
        elif date < datetime.datetime(2011, 10, 1):
            label = "irrelevant"
            last_crop = "TC"
        elif date < datetime.datetime(2012, 10, 1):
            label = "C"
            last_crop = "unknown"
        elif date < datetime.datetime(2013, 10, 1):
            label = "irrelevant"
            last_crop = "C"
        elif date < datetime.datetime(2014, 10, 1):
            label = "C"
            last_crop = "unknown"
        elif date < datetime.datetime(2015, 12, 1):
            label = "SP"
            last_crop = "C"
        elif date < datetime.datetime(2016, 12, 1):
            label = "irrelevant"
            last_crop = "SP"
        elif date < datetime.datetime(2017, 12, 1):
            label = "SB"
            last_crop = "unknown"
            
    # Field area 23 - Alps.
    elif field_area == "23":
        if date < datetime.datetime(2010, 12, 1):
            label = "P"
            last_crop = "SB"
        elif date < datetime.datetime(2011, 10, 1):
            label = "WW"
            last_crop = "P"
        else:
            label = "irrelevant"
            last_crop = "unknown"
            
    # Field area 25 - Alps.
    elif field_area == "25":
        if date < datetime.datetime(2010, 12, 1):
            label = "O"
            last_crop = "WB"
        elif date < datetime.datetime(2011, 10, 1):
            label = "SB"
            last_crop = "O"
        elif date < datetime.datetime(2012, 10, 1):
            label = "C"
            last_crop = "SB"
        elif date < datetime.datetime(2013, 10, 1):
            label = "C"
            last_crop = "C"
        elif date < datetime.datetime(2014, 10, 1):
            label = "C"
            last_crop = "C"
        elif date < datetime.datetime(2015, 12, 1):
            label = "C"
            last_crop = "C"
        elif date < datetime.datetime(2016, 12, 1):
            label = "C"
            last_crop = "C"
        elif date < datetime.datetime(2017, 12, 1):
            label = "C"
            last_crop = "C"
            
    # Field area 26 - Alps.
    elif field_area == "26":
        if date < datetime.datetime(2010, 12, 1):
            label = "WW"
            last_crop = "unknown"
        elif date < datetime.datetime(2011, 10, 1):
            label = "SB"
            last_crop = "WW"
        elif date < datetime.datetime(2012, 10, 1):
            label = "SB"
            last_crop = "SB"
        elif date < datetime.datetime(2013, 10, 1):
            label = "WR"
            last_crop = "SB"
        elif date < datetime.datetime(2014, 10, 1):
            label = "WW"
            last_crop = "WR"
        elif date < datetime.datetime(2015, 12, 1):
            label = "C"
            last_crop = "WW"
        elif date < datetime.datetime(2016, 12, 1):
            label = "SP"
            last_crop = "C"
        elif date < datetime.datetime(2017, 12, 1):
            label = "irrelevant"
            last_crop = "SP"
            
    # Field area 27 - Alps.
    elif field_area == "27":
        if date < datetime.datetime(2010, 12, 1):
            label = "irrelevant"
            last_crop = "unknown"
        elif date < datetime.datetime(2011, 10, 1):
            label = "WW"
            last_crop = "unknown"
        elif date < datetime.datetime(2012, 10, 1):
            label = "irrelevant"
            last_crop = "WW"
        elif date < datetime.datetime(2013, 10, 1):
            label = "irrelevant"
            last_crop = "unknown"
        elif date < datetime.datetime(2014, 10, 1):
            label = "C"
            last_crop = "unknown"
        elif date < datetime.datetime(2015, 12, 1):
            label = "C"
            last_crop = "C"
        elif date < datetime.datetime(2016, 12, 1):
            label = "C"
            last_crop = "C"
        elif date < datetime.datetime(2017, 12, 1):
            label = "C"
            last_crop = "C"
            
    # Field area 28 - Alps.
    elif field_area == "28":
        if date < datetime.datetime(2010, 12, 1):
            label = "WB"
            last_crop = "SP"
        elif date < datetime.datetime(2011, 10, 1):
            label = "WR"
            last_crop = "WB"
        elif date < datetime.datetime(2012, 10, 1):
            label = "WW"
            last_crop = "WR"
        elif date < datetime.datetime(2013, 10, 1):
            label = "WW"
            last_crop = "WW"
        elif date < datetime.datetime(2014, 10, 1):
            label = "irrelevant"
            last_crop = "WW"
        elif date < datetime.datetime(2015, 12, 1):
            label = "SP"
            last_crop = "unknown"
        elif date < datetime.datetime(2016, 12, 1):
            label = "WB"
            last_crop = "SP"
        elif date < datetime.datetime(2017, 12, 1):
            label = "WR"
            last_crop = "WB"
            
    # Field area 210 - Alps.
    elif field_area == "210":
        if date < datetime.datetime(2010, 12, 1):
            label = "SB"
            last_crop = "O"
        elif date < datetime.datetime(2011, 10, 1):
            label = "M"
            last_crop = "SB"
        elif date < datetime.datetime(2012, 10, 1):
            label = "WW"
            last_crop = "M"
        elif date < datetime.datetime(2013, 10, 1):
            label = "M"
            last_crop = "WW"
        elif date < datetime.datetime(2014, 10, 1):
            label = "WW"
            last_crop = "M"
        elif date < datetime.datetime(2015, 12, 1):
            label = "SB"
            last_crop = "WW"
        elif date < datetime.datetime(2017, 12, 1):
            label = "irrelevant"
            last_crop = "unknown"
            
    # Field area 30 - Alps.
    elif field_area == "30":
        if date < datetime.datetime(2010, 12, 1):
            label = "O"
            last_crop = "SB"
        elif date < datetime.datetime(2011, 10, 1):
            label = "SP"
            last_crop = "O"
        elif date < datetime.datetime(2012, 10, 1):
            label = "SB"
            last_crop = "SP"
        elif date < datetime.datetime(2013, 10, 1):
            label = "irrelevant"
            last_crop = "SB"
        elif date < datetime.datetime(2014, 10, 1):
            label = "C"
            last_crop = "unknown"
        elif date < datetime.datetime(2015, 12, 1):
            label = "WW"
            last_crop = "C"
        elif date < datetime.datetime(2016, 12, 1):
            label = "irrelevant"
            last_crop = "WW"
        elif date < datetime.datetime(2017, 12, 1):
            label = "irrelevant"
            last_crop = "unknown"
            
    # Field area 33 - Alps.
    elif field_area == "33":
        if date < datetime.datetime(2010, 12, 1):
            label = "irrelevant"
            last_crop = "TC"
        elif date < datetime.datetime(2011, 10, 1):
            label = "M"
            last_crop = "unknown"
        elif date < datetime.datetime(2012, 10, 1):
            label = "SB"
            last_crop = "M"
        elif date < datetime.datetime(2013, 10, 1):
            label = "O"
            last_crop = "SB"
        elif date < datetime.datetime(2014, 10, 1):
            label = "irrelevant"
            last_crop = "O"
        elif date < datetime.datetime(2015, 12, 1):
            label = "irrelevant"
            last_crop = "unknown"
        elif date < datetime.datetime(2016, 12, 1):
            label = "WW"
            last_crop = "unknown"
        elif date < datetime.datetime(2017, 12, 1):
            label = "irrelevant"
            last_crop = "unknown"
            
    elif field_area == "Cloud":
        label = "Cloud"
        last_crop = "unknown"
    elif field_area == "CloudShadow":
        label = "CloudShadow"
        last_crop = "unknown"
    else:
        label = "irrelevant"
        last_crop = "unknown"


            
            
    return label, last_crop