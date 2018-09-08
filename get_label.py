# Get the label of a particular image.
# Function for getting the label.
import datetime

def get_label(field_area, date):
    '''
    Returns the label of an image based on the csv that shows which label is in
    which area by time.
    '''
    # Field EC1 - Kraichgau.
    if field_area == "EC1":
        if date < datetime.datetime(2010, 12, 1):
            label = "SM"
            last_crop = "unknown"
        elif date < datetime.datetime(2011, 7, 1):
            label = "WW"
            last_crop = "SM"
        elif date < datetime.datetime(2012, 4, 1):
            label = "WR"
            last_crop = "WW"
        elif date < datetime.datetime(2013, 7, 1):
            label = "WW"
            last_crop = "WR"
        elif date < datetime.datetime(2014, 12, 1):
            label = "SM"
            last_crop = "WW"
        elif date < datetime.datetime(2015, 7, 1):
            label = "WW"
            last_crop = "SM"
        elif date < datetime.datetime(2016, 12, 1):
            label = "GM"
            last_crop = "WW"
        elif date < datetime.datetime(2017, 7, 1):
            label = "WW"
            last_crop = "GM"
        elif date < datetime.datetime(2018, 4, 1):
            label = "WR"
            last_crop = "WW"
    
    # Field EC2 - Kraichgau.
    elif field_area == "EC2":
        if date < datetime.datetime(2010, 4, 1):
            label = "WR"
            last_crop = "unknown"
        elif date < datetime.datetime(2011, 7, 1):
            label = "WW"
            last_crop = "WR"
        elif date < datetime.datetime(2012, 12, 1):
            label = "SM"
            last_crop = "WW"
        elif date < datetime.datetime(2013, 7, 1):
            label = "WW"
            last_crop = "SM"
        elif date < datetime.datetime(2014, 12, 1):
            label = "SM"
            last_crop = "WW"
        elif date < datetime.datetime(2015, 7, 1):
            label = "WW"
            last_crop = "SM"
        elif date < datetime.datetime(2016, 4, 1):
            label = "WR"
            last_crop = "WW"
        elif date < datetime.datetime(2017, 7, 1):
            label = "WW"
            last_crop = "WR"
        elif date < datetime.datetime(2018, 7, 1):
            label = "WW"
            last_crop = "WW"
    
    # Field EC3 - Kraichgau.
    elif field_area == "EC3":
        if date < datetime.datetime(2010, 7, 1):
            label = "WW"
            last_crop = "unknown"
        elif date < datetime.datetime(2011, 12, 1):
            label = "SM"
            last_crop = "WW"
        elif date < datetime.datetime(2012, 7, 1):
            label = "WW"
            last_crop = "SM"
        elif date < datetime.datetime(2013, 4, 1):
            label = "WR"
            last_crop = "WW"
        elif date < datetime.datetime(2014, 7, 1):
            label = "WW"
            last_crop = "WR"
        elif date < datetime.datetime(2015, 12, 1):
            label = "SM"
            last_crop = "WW"
        elif date < datetime.datetime(2016, 7, 1):
            label = "WW"
            last_crop = "SM"
        elif date < datetime.datetime(2017, 7, 1):
            label = "WW"
            last_crop = "WW"
        elif date < datetime.datetime(2018, 12, 1):
            label = "SM"
            last_crop = "WW"
         
    # Field EC4 - Swaebisch Alp.
    elif field_area == "EC4":
        if date < datetime.datetime(2010, 1, 1):
            label = "WR"
            last_crop = "unknown"
        elif date < datetime.datetime(2011, 1, 1):
            label = "WW"
            last_crop = "WR"
        elif date < datetime.datetime(2012, 1, 1):
            label = "SB"
            last_crop = "WW"
        elif date < datetime.datetime(2013, 1, 1):
            label = "WR"
            last_crop = "SB"
        elif date < datetime.datetime(2014, 1, 1):
            label = "WW"
            last_crop = "WR"
        elif date < datetime.datetime(2015, 1, 1):
            label = "WW" 
            last_crop = "WW"
        elif date < datetime.datetime(2016, 1, 1):
            label = "SB"
            last_crop = "WW"
        elif date < datetime.datetime(2017, 1, 1):
            label = "SM"
            last_crop = "SB"
        elif date < datetime.datetime(2018, 1, 1):
            label = "WW"
            last_crop = "SM"
            
    # Field EC5 - Swaebisch Alp.
    elif field_area == "EC5":
        if date < datetime.datetime(2010, 1, 1):
            label = "WW"
            last_crop = "unknown"
        elif date < datetime.datetime(2011, 1, 1):
            label = "SM"
            last_crop = "WW"
        elif date < datetime.datetime(2012, 1, 1):
            label = "SM"
            last_crop = "SM"
        elif date < datetime.datetime(2013, 1, 1):
            label = "WB"
            last_crop = "SM"
        elif date < datetime.datetime(2014, 1, 1):
            label = "SP"
            last_crop = "WB"
        elif date < datetime.datetime(2015, 1, 1):
            label = "SM"
            last_crop = "SP"
        elif date < datetime.datetime(2016, 1, 1):
            label = "SM"
            last_crop = "SM"
        elif date < datetime.datetime(2017, 1, 1):
            label = "WB"
            last_crop = "SM"
        elif date < datetime.datetime(2018, 1, 1):
            label = "WR"
            last_crop = "WB"
            
    # Field EC6 - Swaebisch Alp.
    elif field_area == "EC6":
        if date < datetime.datetime(2010, 1, 1):
            label = "SM"
            last_crop = "unknown"
        elif date < datetime.datetime(2011, 1, 1):
            label = "WW"
            last_crop = "SM"
        elif date < datetime.datetime(2012, 1, 1):
            label = "WB"
            last_crop = "WW"
        elif date < datetime.datetime(2013, 1, 1):
            label = "SM"
            last_crop = "WB"
        elif date < datetime.datetime(2014, 1, 1):
            label = "WW"
            last_crop = "SM"
        elif date < datetime.datetime(2015, 1, 1):
            label = "WB"
            last_crop = "WW"
        elif date < datetime.datetime(2016, 1, 1):
            label = "SM"
            last_crop = "WB"
        elif date < datetime.datetime(2017, 1, 1):
            label = "WW"
            last_crop = "SM"
        elif date < datetime.datetime(2018, 1, 1):
            label = "WB"
            last_crop = "WW"
            
    elif field_area == "Cloud":
        label = "Cloud"
        last_crop = "unknown"
    elif field_area == "CloudShadow":
        label = "CloudShadow"
        last_crop = "unknown"


            
            
    return label, last_crop