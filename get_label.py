# Get the label of a particular image.
# Function for getting the label.
import datetime

def get_label(field_area, date):
    '''
    Returns the label of an image based on the csv that shows which label is in
    which area by time.
    '''
    # If the crop is either a cover crop or no cover (during winter).
    if date.month >= 9 or date.month <= 3:
        label = "irrelevant"
        last_crop = "unknown"
    elif date.year >= 2018:
        label = "irrelevant"
        last_crop = "unknown"
        
    # Field EC1 - Kraichgau.
    elif field_area == "EC1":
        if date < datetime.datetime(2010, 12, 1):
            label = "SM"
            last_crop = "unknown"
        elif date < datetime.datetime(2011, 9, 1):
            label = "WW"
            last_crop = "SM"
        elif date < datetime.datetime(2012, 9, 1):
            label = "WR"
            last_crop = "WW"
        elif date < datetime.datetime(2013, 9, 1):
            label = "WW"
            last_crop = "WR"
        elif date < datetime.datetime(2014, 12, 1):
            label = "SM"
            last_crop = "WW"
        elif date < datetime.datetime(2015, 9, 1):
            label = "WW"
            last_crop = "SM"
        elif date < datetime.datetime(2016, 12, 1):
            label = "GM"
            last_crop = "WW"
        elif date < datetime.datetime(2017, 9, 1):
            label = "WW"
            last_crop = "GM"
        elif date < datetime.datetime(2018, 9, 1):
            label = "WR"
            last_crop = "WW"
    
    # Field EC2 - Kraichgau.
    elif field_area == "EC2":
        if date < datetime.datetime(2010, 9, 1):
            label = "WR"
            last_crop = "unknown"
        elif date < datetime.datetime(2011, 9, 1):
            label = "WW"
            last_crop = "WR"
        elif date < datetime.datetime(2012, 12, 1):
            label = "SM"
            last_crop = "WW"
        elif date < datetime.datetime(2013, 9, 1):
            label = "WW"
            last_crop = "SM"
        elif date < datetime.datetime(2014, 12, 1):
            label = "SM"
            last_crop = "WW"
        elif date < datetime.datetime(2015, 9, 1):
            label = "WW"
            last_crop = "SM"
        elif date < datetime.datetime(2016, 9, 1):
            label = "WR"
            last_crop = "WW"
        elif date < datetime.datetime(2017, 9, 1):
            label = "WW"
            last_crop = "WR"
        elif date < datetime.datetime(2018, 9, 1):
            label = "WW"
            last_crop = "WW"
    
    # Field EC3 - Kraichgau.
    elif field_area == "EC3":
        if date < datetime.datetime(2010, 9, 1):
            label = "WW"
            last_crop = "unknown"
        elif date < datetime.datetime(2011, 12, 1):
            label = "SM"
            last_crop = "WW"
        elif date < datetime.datetime(2012, 9, 1):
            label = "WW"
            last_crop = "SM"
        elif date < datetime.datetime(2013, 9, 1):
            label = "WR"
            last_crop = "WW"
        elif date < datetime.datetime(2014, 9, 1):
            label = "WW"
            last_crop = "WR"
        elif date < datetime.datetime(2015, 12, 1):
            label = "SM"
            last_crop = "WW"
        elif date < datetime.datetime(2016, 9, 1):
            label = "WW"
            last_crop = "SM"
        elif date < datetime.datetime(2017, 9, 1):
            label = "WW"
            last_crop = "WW"
        elif date < datetime.datetime(2018, 12, 1):
            label = "SM"
            last_crop = "WW"
         
    # Field EC4 - Swaebisch Alp.
    elif field_area == "EC4":
        if date < datetime.datetime(2010, 9, 1):
            label = "WR"
            last_crop = "unknown"
        elif date < datetime.datetime(2011, 9, 1):
            label = "WW"
            last_crop = "WR"
        elif date < datetime.datetime(2012, 12, 1):
            label = "SB"
            last_crop = "WW"
        elif date < datetime.datetime(2013, 9, 1):
            label = "WR"
            last_crop = "SB"
        elif date < datetime.datetime(2014, 9, 1):
            label = "WW"
            last_crop = "WR"
        elif date < datetime.datetime(2015, 9, 1):
            label = "WW" 
            last_crop = "WW"
        elif date < datetime.datetime(2016, 12, 1):
            label = "SB"
            last_crop = "WW"
        elif date < datetime.datetime(2017, 12, 1):
            label = "SM"
            last_crop = "SB"
        elif date < datetime.datetime(2018, 9, 1):
            label = "WW"
            last_crop = "SM"
            
    # Field EC5 - Swaebisch Alp.
    elif field_area == "EC5":
        if date < datetime.datetime(2010, 9, 1):
            label = "WW"
            last_crop = "unknown"
        elif date < datetime.datetime(2011, 12, 1):
            label = "SM"
            last_crop = "WW"
        elif date < datetime.datetime(2012, 12, 1):
            label = "SM"
            last_crop = "SM"
        elif date < datetime.datetime(2013, 9, 1):
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
        elif date < datetime.datetime(2017, 9, 1):
            label = "WB"
            last_crop = "SM"
        elif date < datetime.datetime(2018, 9, 1):
            label = "WR"
            last_crop = "WB"
            
    # Field EC6 - Swaebisch Alp.
    elif field_area == "EC6":
        if date < datetime.datetime(2010, 12, 1):
            label = "SM"
            last_crop = "unknown"
        elif date < datetime.datetime(2011, 9, 1):
            label = "WW"
            last_crop = "SM"
        elif date < datetime.datetime(2012, 9, 1):
            label = "WB"
            last_crop = "WW"
        elif date < datetime.datetime(2013, 12, 1):
            label = "SM"
            last_crop = "WB"
        elif date < datetime.datetime(2014, 9, 1):
            label = "WW"
            last_crop = "SM"
        elif date < datetime.datetime(2015, 9, 1):
            label = "WB"
            last_crop = "WW"
        elif date < datetime.datetime(2016, 12, 1):
            label = "SM"
            last_crop = "WB"
        elif date < datetime.datetime(2017, 9, 1):
            label = "WW"
            last_crop = "SM"
        elif date < datetime.datetime(2018, 9, 1):
            label = "WB"
            last_crop = "WW"
    
    # Field area 1-1 - Kraichgau.
    elif field_area == "1_1":
        if date < datetime.datetime(2010, 9, 1):
            label = "WR"
            last_crop = "unknown"
        elif date < datetime.datetime(2011, 9, 1):
            label = "WW"
            last_crop = "WR"
        elif date < datetime.datetime(2011, 12, 1):
            label = "GM"
            last_crop= "WW"
        elif date < datetime.datetime(2012, 12, 1):
            label = "SB"
            last_crop = "GM"
        elif date < datetime.datetime(2013, 9, 1):
            label = "WR"
            last_crop = "SB"
        elif date < datetime.datetime(2014, 9, 1):
            label = "WW"
            last_crop = "WR"
        elif date < datetime.datetime(2015, 12, 1):
            label = "irrelevant"
            last_crop = "WW"
        elif date < datetime.datetime(2016, 9, 1):
            label = "WW"
            last_crop = "unknown"
        elif date < datetime.datetime(2017, 12, 1):
            label = "irrelevant"
            last_crop = "WW"

#    # Field area 1-4 - Kraichgau.
#    # CHANGE THIS ONE LATER, IT'S WRONG.
#    elif field_area == "1_4":
#        if date < datetime.datetime(2010, 7, 1):
#            label = "WW"
#            last_crop = "WW"
#        elif date < datetime.datetime(2011, 6, 1):
#            label = "WR"
#            last_crop = "SB"
#        elif date < datetime.datetime(2012, 12, 1):
#            label = "SB"
#            last_crop= "WR"
#        elif date < datetime.datetime(2013, 6, 1):
#            label = "WR"
#            last_crop = "SB"
#        elif date < datetime.datetime(2014, 7, 1):
#            label = "WW"
#            last_crop = "WR"
#        elif date < datetime.datetime(2015, 12, 1):
#            label = "SB"
#            last_crop = "WW"
#        elif date < datetime.datetime(2016, 6, 1):
#            label = "WR"
#            last_crop = "SB"
#        elif date < datetime.datetime(2017, 7, 1):
#            label = "WW"
#            last_crop = "SB"
            
    # Field area 2 - Kraichgau.
    elif field_area == "2":
        if date < datetime.datetime(2010, 9, 1):
            label = "WB"
            last_crop = "WW"
        elif date < datetime.datetime(2011, 9, 1):
            label = "WR"
            last_crop = "SB"
        elif date < datetime.datetime(2013, 9, 1):
            label = "WR"
            last_crop = "SB"
        elif date < datetime.datetime(2014, 9, 1):
            label = "WW"
            last_crop = "WR"
        elif date < datetime.datetime(2015, 12, 1):
            label = "SB"
            last_crop = "WW"
        elif date < datetime.datetime(2016, 9, 1):
            label = "WR"
            last_crop = "SB"
        elif date < datetime.datetime(2017, 9, 1):
            label = "WW"
            last_crop = "SB"
            
    # Field area 3 - Kraichgau.
    elif field_area == "3":
        if date < datetime.datetime(2010, 9, 1):
            label = "WR"
            last_crop = "SB"
        elif date < datetime.datetime(2011, 9, 1):
            label = "WW"
            last_crop = "WR"
        elif date < datetime.datetime(2012, 12, 1):
            label = "SB"
            last_crop = "WW"
        elif date < datetime.datetime(2013, 9, 1):
            label = "WR"
            last_crop = "SB"
        elif date < datetime.datetime(2014, 9, 1):
            label = "WW"
            last_crop = "WR"
        elif date < datetime.datetime(2015, 12, 1):
            label = "SB"
            last_crop = "WW"
        elif date < datetime.datetime(2016, 12, 1):
            label = "SM"
            last_crop = "SB"
        elif date < datetime.datetime(2017, 12, 1):
            label = "irrelevant"
            last_crop = "SM"

    # Field area 4 - Kraichgau.
    elif field_area == "4":
        if date < datetime.datetime(2010, 9, 1):
            label = "irrelevant"
            last_crop = "SB"
        elif date < datetime.datetime(2011, 9, 1):
            label = "WW"
            last_crop = "unknown"
        elif date < datetime.datetime(2012, 9, 1):
            label = "WR"
            last_crop = "WW"
        elif date < datetime.datetime(2013, 9, 1):
            label = "WW"
            last_crop = "WR"
        elif date < datetime.datetime(2014, 12, 1):
            label = "SB"
            last_crop = "WW"
        elif date < datetime.datetime(2015, 12, 1):
            label = "SB"
            last_crop = "SB"
        elif date < datetime.datetime(2016, 9, 1):
            label = "irrelevant"
            last_crop = "SB"
        elif date < datetime.datetime(2017, 12, 1):
            label = "irrelevant"
            last_crop = "unknown"
            
    # Field area 5 - Kraichgau.
    elif field_area == "5":
        if date < datetime.datetime(2010, 9, 1):
            label = "WW"
            last_crop = "WW"
        elif date < datetime.datetime(2011, 12, 1):
            label = "SM"
            last_crop = "WW"
        elif date < datetime.datetime(2012, 12, 1):
            label = "irrelevant"
            last_crop = "SM"
        elif date < datetime.datetime(2013, 9, 1):
            label = "WW"
            last_crop = "unknown"
        elif date < datetime.datetime(2014, 12, 1):
            label = "SM"
            last_crop = "WW"
        elif date < datetime.datetime(2015, 9, 1):
            label = "WW"
            last_crop = "SM"
        elif date < datetime.datetime(2016, 9, 1):
            label = "WB"
            last_crop = "WW"
        elif date < datetime.datetime(2017, 12, 1):
            label = "SM"
            last_crop = "WB"
            
    # Field area 6 - Kraichgau.
    elif field_area == "6":
        if date < datetime.datetime(2010, 9, 1):
            label = "WR"
            last_crop = "WW"
        elif date < datetime.datetime(2011, 9, 1):
            label = "WW"
            last_crop = "WR"
        elif date < datetime.datetime(2012, 9, 1):
            label = "WR"
            last_crop = "WW"
        elif date < datetime.datetime(2013, 9, 1):
            label = "WW"
            last_crop = "WR"
        elif date < datetime.datetime(2014, 12, 1):
            label = "irrelevant"
            last_crop = "WW"
        elif date < datetime.datetime(2015, 9, 1):
            label = "WW"
            last_crop = "unknown"
        elif date < datetime.datetime(2016, 9, 1):
            label = "WR"
            last_crop = "WW"
        elif date < datetime.datetime(2017, 9, 1):
            label = "WW"
            last_crop = "WR"
        elif date < datetime.datetime(2017, 12, 1):
            label = "irrelevant"
            last_crop = "unknown"
            
    # Field area 8 - Kraichgau.
    elif field_area == "8":
        if date < datetime.datetime(2010, 9, 1):
            label = "WW"
            last_crop = "unknown"
        elif date < datetime.datetime(2011, 9, 1):
            label = "WR"
            last_crop = "WW"
        elif date < datetime.datetime(2012, 9, 1):
            label = "WW"
            last_crop = "WR"
        elif date < datetime.datetime(2013, 9, 1):
            label = "irrelevant"
            last_crop = "WW"
        elif date < datetime.datetime(2014, 12, 1):
            label = "irrelevant"
            last_crop = "unknown"
        elif date < datetime.datetime(2015, 9, 1):
            label = "WW"
            last_crop = "unknown"
        elif date < datetime.datetime(2016, 9, 1):
            label = "irrelevant"
            last_crop = "WW"
        elif date < datetime.datetime(2017, 12, 1):
            label = "irrelevant"
            last_crop = "unknown"
            
    # Field area 9 - Kraichgau.
    elif field_area == "9":
        if date < datetime.datetime(2010, 9, 1):
            label = "WW"
            last_crop = "WR"
        elif date < datetime.datetime(2011, 12, 1):
            label = "GM"
            last_crop = "WW"
        elif date < datetime.datetime(2012, 9, 1):
            label = "WW"
            last_crop = "GM"
        elif date < datetime.datetime(2013, 12, 1):
            label = "SB"
            last_crop = "WW"
        elif date < datetime.datetime(2014, 9, 1):
            label = "WR"
            last_crop = "SB"
        elif date < datetime.datetime(2015, 9, 1):
            label = "WW"
            last_crop = "WR"
        elif date < datetime.datetime(2016, 12, 1):
            label = "GM"
            last_crop = "WW"
        elif date < datetime.datetime(2017, 12, 1):
            label = "irrelevant"
            last_crop = "unknown"
            
    # Field area 11 - Kraichgau.
    elif field_area == "11":
        if date < datetime.datetime(2010, 9, 1):
            label = "irrelevant"
            last_crop = "SB"
        elif date < datetime.datetime(2011, 9, 1):
            label = "WW"
            last_crop = "unknown"
        elif date < datetime.datetime(2012, 12, 1):
            label = "SB"
            last_crop = "WW"
        elif date < datetime.datetime(2013, 12, 1):
            label = "irrelevant"
            last_crop = "SB"
        elif date < datetime.datetime(2014, 9, 1):
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
        elif date < datetime.datetime(2011, 9, 1):
            label = "irrelevant"
            last_crop = "SB"
        elif date < datetime.datetime(2012, 9, 1):
            label = "WW"
            last_crop = "unknown"
        elif date < datetime.datetime(2013, 9, 1):
            label = "WR"
            last_crop = "WW"
        elif date < datetime.datetime(2014, 9, 1):
            label = "WW"
            last_crop = "WR"
        elif date < datetime.datetime(2015, 12, 1):
            label = "SB"
            last_crop = "WW"
        elif date < datetime.datetime(2016, 12, 1):
            label = "SB"
            last_crop = "SB"
        elif date < datetime.datetime(2017, 12, 1):
            label = "irrelevant"
            last_crop = "unknown"
            
    # Field area 15 - Kraichgau.
    elif field_area == "15":
        if date < datetime.datetime(2010, 12, 1):
            label = "SB"
            last_crop = "WW"
        elif date < datetime.datetime(2011, 9, 1):
            label = "WR"
            last_crop = "SB"
        elif date < datetime.datetime(2012, 9, 1):
            label = "irrelevant"
            last_crop = "WR"
        elif date < datetime.datetime(2013, 12, 1):
            label = "SB"
            last_crop = "unknown"
        elif date < datetime.datetime(2014, 9, 1):
            label = "WR"
            last_crop = "SB"
        elif date < datetime.datetime(2015, 9, 1):
            label = "WW"
            last_crop = "WR"
        elif date < datetime.datetime(2017, 12, 1):
            label = "irrelevant"
            last_crop = "unknown"
            
    # Field area 17 - Kraichgau.
    elif field_area == "17":
        if date < datetime.datetime(2010, 12, 1):
            label = "SB"
            last_crop = "unknown"
        elif date < datetime.datetime(2011, 12, 1):
            label = "GM"
            last_crop = "SB"
        elif date < datetime.datetime(2012, 9, 1):
            label = "WW"
            last_crop = "GM"
        elif date < datetime.datetime(2013, 12, 1):
            label = "irrelevant"
            last_crop = "WW"
        elif date < datetime.datetime(2014, 9, 1):
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