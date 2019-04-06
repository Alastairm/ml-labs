"""
Lab 3 Project 2
"""
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
from sklearn import (
    metrics,
    model_selection,
    preprocessing,
    svm,
    tree
)


solar_data = pd.read_csv('SolarExposure_2018_Data.csv')
temp_data = pd.read_csv('Temperature_2018_Data.csv')

solar_data.head()
temp_data.head()


"""Arrange Data into X & y train & test sets."""
solar_columns = solar_data.columns.values.tolist()
temp_columns = temp_data.columns.values.tolist()

temp_values = temp_data[temp_columns[-3]]
solar_values = solar_data[solar_columns[-1]]
month_values = solar_data[solar_columns[3]]
day_values = solar_data[solar_columns[4]]

y = temp_values
data_attributes = [solar_values, month_values, day_values]
X = pd.concat(data_attributes, axis=1)

split_data = model_selection.train_test_split(y, X, test_size=0.2)
train_y, test_y, train_X, test_X = split_data


svm_reg = svm.SVR().fit(train_X, train_y)

train_pred = svm_reg.predict(train_X)
test_pred = svm_reg.predict(test_X)

train_mse = metrics.mean_squared_error(train_y, train_pred)
test_mse = metrics.mean_squared_error(test_y, test_pred)

print(f"train MSE: {train_mse}, test MSE: {test_mse}")


def srt(**kwargs):
    svm_reg = svm.SVR(**kwargs).fit(train_X, train_y)

    train_pred = svm_reg.predict(train_X)
    test_pred = svm_reg.predict(test_X)

    train_mse = metrics.mean_squared_error(train_y, train_pred)
    test_mse = metrics.mean_squared_error(test_y, test_pred)
    print(kwargs)
    print(f"train MSE: {train_mse:0.2f}, test MSE: {test_mse:0.2f}")