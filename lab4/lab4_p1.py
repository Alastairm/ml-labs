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


data_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data'
data = pd.read_csv(data_url)


def data_clean() -> Tuple[pd.Series, pd.Series, pd.DataFrame, pd.DataFrame]:
    # Replace 1 & 0 with 'parkinsons' & 'healthy' respectively.
    y = ['parkinsons' if s == 1 else 'healthy' for s in data['status']]
    # Remove name (string value) & status (class label) from attributes.
    X = data.drop(columns=['name', 'status'])
    return sk.model_selection.train_test_split(y, X, test_size=0.2)

def data_clean() -> Tuple[pd.Series, pd.Series, pd.DataFrame, pd.DataFrame]:
    # Replace 1 & 0 with 'parkinsons' & 'healthy' respectively.
    y = data['status']
    for i, status in enumerate(y):
        y[i] = 'parkinsons' if status == 1 else 'healthy'
    y.cl
    # Remove name (string value) & status (class label) from attributes.
    X = data.drop(columns=['name', 'status'])
    return sk.model_selection.train_test_split(y, X, test_size=0.2)


train_y, test_y, train_X, test_X = data_clean()
train_y, test_y, train_X, test_X = data_clean()

standard_scaler = sk.preprocessing.StandardScaler()

svm_clf = sk.pipeline.Pipeline(steps=[
    ('scale', standard_scaler),
    ('svc', sk.svm.SVC())
])

logit_clf = sk.pipeline.Pipeline(steps=[
    ('scale', standard_scaler),
    ('logit', sk.linear_model.LogisticRegression())
])

ensemble_clf = sk.ensemble.VotingClassifier(
    estimators=[('svc', svm_clf), ('logit', logit_clf)]
)


svm_pred = svm_clf.fit(train_X, train_y).predict(test_X)
logit_pred = logit_clf.fit(train_X, train_y).predict(test_X)
ensemble_pred = ensemble_clf.fit(train_X, train_y).predict(test_X)

print(f"SVM: {sk.metrics.f1_score(test_y, svm_pred):05}")
print(f"Logit: {sk.metrics.f1_score(test_y, logit_pred):05}")
print(f"Ensemble: {sk.metrics.f1_score(test_y, ensemble_pred):05}")


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues) -> mpl.axes.Axes:
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Copyright (c) 2007–2019 The scikit-learn developers.
    Used under license.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = sk.metrics.confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    # classes = classes[sk.utils.multiclass.unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

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


plot_confusion_matrix(test_y, ensemble_pred, [1, 0])
# plt.show()
