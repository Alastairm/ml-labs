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

Array1D = Union[List, pd.core.series.Series]
Array2D = Union[List[List], np.ndarray, pd.DataFrame]

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


def reg_mse(regression: Any, train_y: Array1D, train_X: Array2D,
            test_y: Array1D, test_X: Array2D) -> Tuple[float, float]:
    """
    Mean squared error loss of both test and training data.
    """
    reg = regression.fit(train_X, train_y)
    train_pred = reg.predict(train_X)
    test_pred = reg.predict(test_X)
    train_mse = metrics.mean_squared_error(train_y, train_pred)
    test_mse = metrics.mean_squared_error(test_y, test_pred)
    return (train_mse, test_mse)

results = []
for kernel in ['rbf']:
    for C in [10]:
        for gamma in [0.0001, 0.001, 0.01, 0.1, 1]:
            result = [kernel, gamma]
            svm_reg = svm.SVR(kernel=kernel, C=C, gamma=gamma)
            result += list(reg_mse(svm_reg, train_y, train_X, test_y, test_X))
            results.append(result)
columns = ['Kernel', 'gamma', 'Train MSE', 'Test MSE']
results_df = pd.DataFrame(results, columns=columns)
print(results_df)


# Todo Improve search & explanaiton