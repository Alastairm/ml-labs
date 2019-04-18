"""
Lab 3 Project 1
"""
from copy import deepcopy as dc
from typing import (Any, List, Tuple)
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn as sk
from sklearn import (
    decomposition,
    ensemble,
    linear_model,
    metrics,
    model_selection,
    preprocessing,
    pipeline,
    svm
)


# Disable Sklearn Warnings
def warn(*args, **kwargs):
    pass
warnings.warn = warn


data_url = 'abalone.data'

attribute_names = [
    'Sex',
    'Length',
    'Diameter',
    'Height',
    'Whole weight',
    'Shucked weight',
    'Viscera weight',
    'Shell weight',
    'Rings',
]
data = pd.read_csv(data_url, names=attribute_names)

# data.hist()
# plt.show()


def plot_rings_vs_sex(data):
    """
    Creates separate plots of infant, female and male ring count frequencies.

    Returns
    -------
        
    """
    infant_rings = []
    female_rings = []
    male_rings = []
    for i in range(len(data)):
        if data.loc[i, 'Sex'] == 'I':
            infant_rings.append(data.loc[i, 'Rings'])
        elif data.loc[i, 'Sex'] == 'F':
            female_rings.append(data.loc[i, 'Rings'])
        elif data.loc[i, 'Sex'] == 'M':
            male_rings.append(data.loc[i, 'Rings'])
    
    ages = range(0, 30)
    infant_ages = [infant_rings.count(i) for i in ages]
    female_ages = [female_rings.count(i) for i in ages]
    male_ages = [male_rings.count(i) for i in ages]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(ages, infant_ages, label='Infant')
    ax.plot(ages, female_ages, label='Female')
    ax.plot(ages, male_ages, label='Male')

    ax.set_ylabel('Number of abalone')
    ax.set_xlabel('Ring count')
    plt.legend(loc='upper right')

    return fig

# plot_rings_vs_sex(data)


def handle_ring_outliers(data: Any, method: str) -> pd.DataFrame:

    def group(data: pd.DataFrame) -> pd.DataFrame:
        for i in range(len(data)):
            if 5 > data.loc[i, 'Rings']:
                data.loc[i, 'Rings'] = 4
            elif 20 < data.loc[i, 'Rings']:
                data.loc[i, 'Rings'] = 21
        return data

    def drop(data: pd.DataFrame) -> pd.DataFrame:
        for i in range(len(data)):
            if not 5 <= data.loc[i, 'Rings'] <= 20:
                data = data.drop(i)
        return data
    
    if method == 'group':
        return group(data)

    elif method == 'drop':
        return drop(data)
    else:
        return data


def handle_ring_outliers_xy(x: pd.DataFrame, y: pd.Series, method: str)\
 -> Tuple[pd.DataFrame, pd.Series]:
    data = pd.concat([x, y], axis=1)
    data = handle_ring_outliers(data, method=method)
    y = data['Rings']
    x = data.drop(columns=['Rings'])
    return (x, y)


def arrange_data(data: pd.DataFrame, ring_handler='drop')\
 -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Clean data then split into test & training sets.

    Returns:
        data (pd.Dataframe):
        ring_handler (str):

    Returns:
        Tuple[pd.Series, pd.Series, pd.DataFrame, pd.DataFrame]:
            Tuple containing class labels & class attributes.
    """
    def handle_sex_attribute(data: pd.DataFrame) -> pd.DataFrame:
        is_adult = []
        for i in range(len(data)):
            if data.loc[i, 'Sex'] == 'I':
                is_adult.append(0)
            else:
                is_adult.append(1)
        is_adult_df = pd.DataFrame(is_adult, columns=['Is adult'])
        data = pd.concat([data, is_adult_df], axis=1)
        return data

    # Avoid editing original data
    data = dc(data)

    # Replace Sex attribute with 'Is adult' value.
    data = handle_sex_attribute(data)

    # Drop, group, or do nothing with outlier ring counts.
    data = handle_ring_outliers(data, 'drop')

    y = data['Rings']
    x = data.drop(columns=['Sex', 'Rings'])

    split_data = model_selection.train_test_split(y, x, test_size=0.1)
    train_y, test_y, train_x, test_x = split_data

    train_y = train_y.reset_index()['Rings']
    test_y = test_y.reset_index()['Rings']
    train_x = train_x.reset_index().drop(columns=['index'])
    test_x = test_x.reset_index().drop(columns=['index'])

    return (train_x, test_x, train_y, test_y)


def reg_mse(regressor: Any,
            train_x: pd.DataFrame, test_x: pd.DataFrame,
            train_y: pd.Series, test_y: pd.Series)\
 -> Tuple[float, float]:
    """
    Calculatee mean squared error loss of both test and training data
    on given regressor.
    """
    reg = regressor.fit(train_x, train_y)
    train_pred = reg.predict(train_x)
    test_pred = reg.predict(test_x)
    train_mse = metrics.mean_squared_error(train_y, train_pred)
    test_mse = metrics.mean_squared_error(test_y, test_pred)
    return (train_mse, test_mse)


def test_aba_regressor_ring_handling(data):
    train_x, test_x, train_y, test_y = arrange_data(data)
    results = []
    for handler in ['none', 'drop', 'group']:
        _train_x, _train_y =\
            handle_ring_outliers_xy(train_x, train_y, handler)
        _test_x, _test_y =\
            handle_ring_outliers_xy(test_x, test_y, handler)

        reg = sk.ensemble.RandomForestRegressor(n_estimators=100)\
            .fit(_train_x, _train_y)
        mse = reg_mse(reg, _train_x, _test_x, _train_y, _test_y)
        results.append([handler, *mse])
    headers = ['Ring Handler', 'Train MSE', 'Test MSE']
    return pd.DataFrame(results, columns=headers)

# print(test_aba_regressor_ring_handling(data))


train_x, test_x, train_y, test_y = arrange_data(data)


rf_reg = sk.ensemble.RandomForestRegressor(n_estimators=100)
rf_reg_fit = dc(rf_reg).fit(train_x, train_y)


def rank_feature_importance(reg, columns, reverse=False):
    """ Return a list of (feature, importance) tuples."""
    _reg = dc(rf_reg_fit)
    importance = _reg.feature_importances_
    if not (len(importance) == len(columns)):
        raise ValueError()
    feat_impo = []
    for i, feature in enumerate(columns):
        feat_impo.append((feature, importance[i]))
    feat_impo.sort(key=lambda x: x[1], reverse=reverse)
    return feat_impo

# feature_importances = rank_feature_importance(rf_reg, train_x.columns, True)
# names = ['Feature', 'Importance']
# print(pd.DataFrame(feature_importances, columns=names))


def drop_features(data: List[pd.DataFrame], features: List[str]) -> List:
    """ Remove given feature columns from DataFrames."""
    return [d.drop(columns=features) for d in dc(data)]


def drop_n_features(reg, train_x, test_X, n=1)\
 -> Tuple[pd.DataFrame, pd.DataFrame]:
    """ Remove the n least important features.

    Args:
        clf: sklearn classifier used to determine feature importance.
        train: training data set.
        test: testing data set.
        n: number of features to remove
    
    Returns:
        (Tuple[pd.DataFrame, pd.DataFrame, List[str], pd.DataFrame])
        
    """
    _reg = dc(rf_reg_fit)
    ranked_feat = rank_feature_importance(_reg, train_x.columns)
    feat_to_drop = []
    for i in range(n):
        feat_to_drop.append(ranked_feat[i][0])
    train_d, test_d = drop_features([train_x, test_x], feat_to_drop)
    return (train_d, test_d)


def test_manual_feature_reduction():
    results = []
    _reg = dc(rf_reg_fit)
    for i in range(8):
        _train_x, _test_x = drop_n_features(_reg, train_x, test_x, n=i)
        _reg2 = rf_reg.fit(_train_x, train_y)
        train_mse, test_mse = reg_mse(_reg2, _train_x, _test_x, train_y, test_y)
        results.append([i, train_mse, test_mse])
    names = ['Features Dropped', 'Train MSE', 'Test MSE']
    return pd.DataFrame(results, columns=names)

# print(test_manual_feature_reduction())

pca = sk.decomposition.PCA(.99)
pca.fit(train_x, train_y)
train_x_pca = pca.transform(train_x)
test_x_pca = pca.transform(test_x)
results = reg_mse(rf_reg, train_x_pca, test_x_pca, train_y, test_y)
print(results)
results = reg_mse(rf_reg, train_x, test_x, train_y, test_y)
print(results)
