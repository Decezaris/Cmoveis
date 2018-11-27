import pandas as pd
import numpy as np
from sklearn import neighbors
from math import sin, cos, sqrt, atan2, radians
#from sklearn.model_selection import GridSearchCV
#import matplotlib.pyplot as plt
#from math import sin, cos, sqrt, atan2, radians
#import seaborn as sns


def get_distance_in_meters(lat1, lat2, long1, long2):
    # approximate radius of earth in km
    R = 6371.0

    lat1 = radians(lat1)
    lon1 = radians(long1)
    lat2 = radians(lat2)
    lon2 = radians(long2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c


    return distance * 1000


