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
import warnings

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


# Disable Sklearn Warnings
def warn(*args, **kwargs):
    pass
warnings.warn = warn


Array1D = Union[List, pd.core.series.Series]
Array2D = Union[List[List], np.ndarray, pd.DataFrame]


solar_data = pd.read_csv('SolarExposure_2018_Data.csv')
temp_data = pd.read_csv('Temperature_2018_Data.csv')

solar_data.head()
temp_data.head()


"""Plot temperature and solar exposure vs day of year"""
days = [i for i in range(1, 366)]
fig, ax1 = plt.subplots()
ax1.set_xlabel("Day of year", size=12)
ax1.plot(days, temp_data['Maximum temperature (Degree C)'], 'r.')
ax1.set_ylabel(r"Solar exposure ($\frac{MJ}{m^2}$)", size=12)

ax2 = ax1.twinx()
ax2.plot(days, solar_data['Daily global solar exposure (MJ/m*m)'], 'g.')
ax2.set_ylabel(r"Temperature ($^\circ$C)", size=12)

fig.tight_layout()
ax2.set_title("Perth Daily Temperature & Solar Exposure", size=20)
fig.legend(('Temp', 'Solar'), loc=4)
plt.show()


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


def reg_mse(regressor: Any, train_y: Array1D, train_X: Array2D,
            test_y: Array1D, test_X: Array2D) -> Tuple[float, float]:
    """
    Calculatee mean squared error loss of both test and training data
    on given regressor.
    """
    reg = regressor.fit(train_X, train_y)
    train_pred = reg.predict(train_X)
    test_pred = reg.predict(test_X)
    train_mse = metrics.mean_squared_error(train_y, train_pred)
    test_mse = metrics.mean_squared_error(test_y, test_pred)
    return (train_mse, test_mse)


def test_params(hyperparams, model='SVM', normalise=True):
    """
    test either a SVM or DT Regressor with given hyperparameters and
    print the results in a table.

    Parameters
    ----------
    model: str, 'SVM' or 'DT'
    Selects regressor type.

    normalise: bool
    Whether to normalise the data prior to regression or not.
    """
    results = []
    columns = list(hyperparams[0].keys()) + ['Train MSE', 'Test MSE']
    for params in hyperparams:
        result = list(params.values())
        if 'model' in params:
            model = params.pop('model')
        if 'normalise' in params:
            normalise = params.pop('normalise')
        if model == 'SVM':
            reg = svm.SVR(**params)
        elif model == 'DT':
            reg = tree.DecisionTreeRegressor(**params)
        else:
            raise NotImplementedError(f"type '{model}' not implemented")
        if normalise:
            xscale = preprocessing.StandardScaler().fit(train_X)
            train_X_norm = xscale.transform(train_X)
            test_X_norm = xscale.transform(test_X)
            result += list(reg_mse(reg, train_y, train_X_norm,
                                   test_y, test_X_norm))
        else:
            result += list(reg_mse(reg, train_y, train_X, test_y, test_X))
        results.append(result)
    results_df = pd.DataFrame(results, columns=columns)
    print(results_df)

hyperparams = [
    {'model': 'SVM', 'normalise': False},
    {'model': 'SVM', 'normalise': True},
    {'model': 'DT', 'normalise': False},
    {'model': 'DT', 'normalise': True}
]
test_params(hyperparams)

"""
hyperparams = [
    {'model': 'SVM'},
    {'model': 'DT'}
]
test_params(hyperparams)

hyperparams = [
    {'kernel': 'linear'},
    {'kernel': 'rbf'},
#    {'kernel': 'poly'},  # Removed from tests, too slow.
    {'kernel': 'sigmoid'}
]
test_params(hyperparams)

hyperparams = [
    {'kernel': 'linear', 'C': 0.1},
    {'kernel': 'linear', 'C': 1},
    {'kernel': 'linear', 'C': 10},
    {'kernel': 'rbf', 'C': 0.1},
    {'kernel': 'rbf', 'C': 1},
    {'kernel': 'rbf', 'C': 10},
]
test_params(hyperparams)

hyperparams = [
    {'kernel': 'linear', 'C': 1},
    {'kernel': 'rbf', 'C': 1},
    {'kernel': 'rbf', 'C': 10},
    {'kernel': 'rbf', 'C': 100},
]
test_params(hyperparams)

hyperparams = [
    {'min_samples_split': 2},
    {'min_samples_split': 0.01},
    {'min_samples_split': 0.05},
    {'min_samples_split': 0.1},
    {'min_samples_split': 0.2},
    {'min_samples_split': 0.3},
    {'min_samples_split': 0.4},
    {'min_samples_split': 0.5},
]
test_params(hyperparams, 'DT')

hyperparams = [
    {'min_samples_split': 0.2, 'criterion': 'mse', 'splitter': 'best'},
    {'min_samples_split': 0.2, 'criterion': 'mae', 'splitter': 'best'},
    {'min_samples_split': 0.2, 'criterion': 'mse', 'splitter': 'random'},
    {'min_samples_split': 0.2, 'criterion': 'mae', 'splitter': 'random'}
]
test_params(hyperparams, 'DT')
"""
