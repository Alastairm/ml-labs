from typing import (
    Dict,
    List,
    Union
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import (
    linear_model,
    metrics,
    model_selection,
    preprocessing,
    svm,
    tree
)

CLASS_LABELS = ['cp', 'im', 'pp', 'imU', 'om']
DROP_CLASSES = ['omL', 'imL', 'imS']
HEADER = ['Sequence Name', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'class']

# 1. Read & explore data
"""
read_csv() defaults to inferring the data column headers and
using a comma as the delimeter so we need to set the attributes
delim_whitespace=True &
names=[list of the column names we want to use]
"""
data_file = pd.read_csv('./ecoli.data', delim_whitespace=True, names=HEADER)
# We'll immediately split our data to avoid a data snooping bias.
train, test = model_selection.train_test_split(data_file, test_size=0.2, train_size=0.8)
train = train.reset_index()
test = test.reset_index()
print(f"{len(train)}, {len(test)}")
# Now lets explore our data
# train.head()
# train.describe()
# train.info()
# train.hist()
# plt.show()
"""
from the above it's clear the columns 'Sequence Name', 'lip' and 'chg' will not
be very helpful in training our classifiers. Here's a function that will split
the data into classes and training data, and also drop those three columns.
"""
def preprocess_data(data):
    drop_rows = []
    for i in range(len(data)):
        if data.loc[i,'class'] in DROP_CLASSES:
            drop_rows.append(i)
    data = data.drop(index=drop_rows)
    y = data['class']
    drop_columns = ['class', 'Sequence Name', 'lip', 'chg']
    X = data.drop(columns=drop_columns)
    return y, X


train_y, train_X = preprocess_data(train)
test_y, test_X = preprocess_data(test)



SVM_clf = svm.SVC()

DT_clf = tree.DecisionTreeClassifier()


def test_classifier(clf, train_X, train_y, test_X, test_y,
                    normalise=False):
    # Scale data if SVM:
    if normalise:
        scaler = preprocessing.StandardScaler().fit(train_X)
        train_X = scaler.transform(train_X)
        test_X = scaler.transform(test_X)

    pred_y = clf.fit(train_X, train_y).predict(test_X)
    confusion_matrix = metrics.confusion_matrix(test_y, pred_y)
    print(confusion_matrix)


test_classifier(SVM_clf, train_X, train_y, test_X, test_y)
test_classifier(SVM_clf, train_X, train_y, test_X, test_y, normalise=True)
test_classifier(DT_clf, train_X, train_y, test_X, test_y)

