"""
Lab 3 Project 1
"""
from typing import Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn as sk
from sklearn import (
    ensemble,
    linear_model,
    metrics,
    model_selection,
    preprocessing,
    pipeline,
    svm
)


data_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data'
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
    plt.show()


plot_rings_vs_sex(data)


def arrange_data(data, ring_handler=None)\
 -> Tuple[pd.Series, pd.Series, pd.DataFrame, pd.DataFrame]:
    """
        Clean data then split into test & training sets.
    """
    # Replace Sex attribute with 'Is adult' value.
    is_adult = []
    for i in range(len(data)):
        if data.loc[i, 'Sex'] == 'I':
            is_adult.append(0)
        else:
            is_adult.append(1)
    is_adult_df = pd.DataFrame(is_adult, columns=['Is adult'])
    pd.concat([data, is_adult_df], axis=1)

    # Drop, greoup, or do nothing with outlier ring counts.
    if ring_handler == 'drop':
        for i in range(len(data)):
            if data.loc[i, 'Rings'] < 5 or data.loc[i, 'Rings'] > 20:
                data.drop(i)
    elif ring_handler == 'group':
        for i in range(len(data)):
            if data.loc[i, 'Rings'] < 5:
                data.loc[i, 'Rings'] = 4
            elif data.loc[i, 'Rings'] > 20:
                data.loc[i, 'Rings'] = 21
    
    y = data['Rings']
    X = data.drop(columns=['Sex', 'Rings'])
    
    return model_selection.train_test_split(y, X, test_size=0.1)