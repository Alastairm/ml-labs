from typing import (
    Any,
    Dict,
    List,
    Tuple,
    Union
)

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd


solar_data = pd.read_csv('SolarExposure_2018_Data.csv')
temp_data = pd.read_csv('Temperature_2018_Data.csv')

day_of_year = [i for i in range(365)]
solar = list(solar_data['Daily global solar exposure (MJ/m*m)'])
temp = list(temp_data['Maximum temperature (Degree C)'])

data = []
for i in day_of_year:
    data.append([i, solar[i], temp[i]])

headers = ('Day of year', 'solar', 'temp')
data = pd.DataFrame(data)

