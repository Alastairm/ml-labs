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

DAYS_IN_MONTH = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]


def get_day_of_year(month: int, day_of_month: int) -> int:
    day_of_year = sum(DAYS_IN_MONTH[:month - 1])
    day_of_year += day_of_month
    return day_of_year


def get_month_and_day(day_of_year: int) -> Tuple[int, int]:
    month = 1
    for i in range(12):
        if day_of_year > DAYS_IN_MONTH[i]:
            month += 1
            day_of_year -= DAYS_IN_MONTH[i]
        else:
            return (month, day_of_year)

solar_data = pd.read_csv('SolarExposure_2018_Data.csv')
temp_data = pd.read_csv('Temperature_2018_Data.csv')

day_of_year = [i for i in range(365)]
solar = list(solar_data['Daily global solar exposure (MJ/m*m)'])
temp = list(temp_data['Maximum temperature (Degree C)'])

data = []
for i in day_of_year:
    data.append([i, solar[i], temp[i]])

headers = ('day', 'solar', 'temp')
data = pd.DataFrame(data, columns=headers)

# plt.scatter(data['day'], data['temp'])
# plt.scatter(data['day'], data['solar'])
# plt.show()

train, test = model_selection.train_test_split(data,
                                               test_size=0.2,
                                               train_size=0.8)
train = train.reset_index()
test = test.reset_index()


# scaler = preprocessing.StandardScaler().fit(train_X)


train_y = train['temp']
train_X = train.drop(columns=['temp'])
test_y = test['temp']
test_X = test.drop(columns=['temp'])

svm_regressor = svm.SVR(C=100)

dt_regressor = tree.DecisionTreeRegressor()

svm_pred_y = svm_regressor.fit(train_X, train_y).predict(test_X)
dt_pred_y = dt_regressor.fit(train_X, train_y).predict(test_X)

print(metrics.mean_squared_error(test_y, svm_pred_y))
print(metrics.mean_squared_error(test_y, dt_pred_y))

plt.scatter(test_X['day'], test_y)
plt.scatter(test_X['day'], svm_pred_y)
plt.scatter(test_X['day'], dt_pred_y)
plt.show()