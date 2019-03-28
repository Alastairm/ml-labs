"""
Lab 3 Project 1
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
    linear_model,
    metrics,
    model_selection,
    preprocessing,
    svm,
    tree
)

Array = Union[List[List], np.ndarray, pd.DataFrame]
Series = Union[List, pd.core.series.Series]

CLASS_LABELS = ['cp', 'im', 'pp', 'imU', 'om']
DROP_CLASSES = ['omL', 'imL', 'imS']

HEADER = ['Sequence Name', 'mcg', 'gvh', 'lip',
          'chg', 'aac', 'alm1', 'alm2', 'class']
DROP_COLUMNS = ['class', 'Sequence Name', 'lip', 'chg']

# 1. Read & explore data
"""
read_csv() defaults to inferring the data column headers and
using a comma as the delimeter so we need to set the attributes
delim_whitespace=True &
names=[list of the column names we want to use]
"""
data_file = pd.read_csv('./ecoli.data', delim_whitespace=True, names=HEADER)
# We'll immediately split our data to avoid a data snooping bias.
train, test = model_selection.train_test_split(data_file,
                                               test_size=0.2,
                                               train_size=0.8)
train = train.reset_index()
test = test.reset_index()
# Now lets explore our data
# train.head()
# train.info()
"""
It looks like there's a second column of text in addition to the class labels:
'Sequence Name', if there's only a few unique strings in this column it may be
a useful feature variable. 
"""
len(set(list(train['Sequence Name'])))
"""
268 unique strings for 268 rows of data, 
"""
# train.describe()
# train.hist()
# plt.show()
"""
From the above it's clear the columns 'Sequence Name', 'lip' and 'chg' will not
be very helpful in training our classifiers. Here's a function that will drop
those three columns and remove any classes we're not interested in. Using a
function to preprocess the data ensures both the training and testing sets are
treated exactly the same.
"""


def preprocess_data(data: pd.DataFrame,
                    drop_classes: List[str] = DROP_CLASSES,
                    drop_columns: List[str] = DROP_COLUMNS
                    ) -> Tuple[Series, Array]:
    """
    Drop rows containing data for classes in drop_classes and remove columns
    contained in drop_columns, returns the data's class labels (y) and the
    remaining data as x.
    """
    drop_rows = []
    for i in range(len(data)):
        if data.loc[i, 'class'] in drop_classes:
            drop_rows.append(i)
    data = data.drop(index=drop_rows)
    data.reset_index()
    y = data['class']
    X = data.drop(columns=drop_columns)
    return y, X


train_y, train_X = preprocess_data(train)
test_y, test_X = preprocess_data(test)


SVM_clf = svm.SVC()

DT_clf = tree.DecisionTreeClassifier()


def test_classifier(clf: Any,  # Any SKLearn classifier object
                    train_X: Array, train_y: Series,
                    test_X: Array, test_y: Series,
                    normalise_data: bool = False,
                    normalise_accuracy_score: bool = True,
                    f1_average_method: str = 'weighted'
                    ) -> Tuple[Array, bool]:
    """
    Fit clf against training data (train_X & train_y), run predictions on test
    data (test_X) and compare predictions to test_y, return confusion matrix
    of results.
    """
    if normalise_data:
        scaler = preprocessing.StandardScaler().fit(train_X)
        train_X = scaler.transform(train_X)
        test_X = scaler.transform(test_X)

    pred_y = clf.fit(train_X, train_y).predict(test_X)  # Do the classification
    confusion_matrix = metrics.confusion_matrix(test_y, pred_y)
    accuracy = metrics.accuracy_score(test_y, pred_y,
                                      normalize=normalise_accuracy_score)
    f1 = metrics.f1_score(test_y, pred_y, average=f1_average_method)
    return confusion_matrix, accuracy, f1


(test_classifier(SVM_clf, train_X, train_y, test_X, test_y))
(test_classifier(SVM_clf, train_X, train_y, test_X, test_y, normalise_data=True))
cm, acc, f1 = (test_classifier(DT_clf, train_X, train_y, test_X, test_y))


"""
The code in this cell is used subject to the terms below:

New BSD License

Copyright (c) 2007–2019 The scikit-learn developers.
All rights reserved.


Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

  a. Redistributions of source code must retain the above copyright notice,
     this list of conditions and the following disclaimer.
  b. Redistributions in binary form must reproduce the above copyright
     notice, this list of conditions and the following disclaimer in the
     documentation and/or other materials provided with the distribution.
  c. Neither the name of the Scikit-learn Developers  nor the names of
     its contributors may be used to endorse or promote products
     derived from this software without specific prior written
     permission. 


THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
DAMAGE.
"""


def plot_confusion_matrix(confusion_matrix, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    cm = confusion_matrix
    if not title:
        title = "Confusion Matrix"

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


plot_confusion_matrix(cm, CLASS_LABELS)
plt.show()


