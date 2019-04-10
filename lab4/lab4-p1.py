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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn as sk
from sklearn import (
    ensemble,
    model_selection,
    pipeline,
    svm
)


data_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data'
data = pd.read_csv(data_url)


def data_clean() -> Tuple[pd.Series, pd.Series, pd.DataFrame, pd.DataFrame]:
    y = data['status']
    X = data.drop(columns=['name', 'status'])
    return model_selection.train_test_split(y, X, test_size=0.2)


train_y, test_y, train_X, test_X = data_clean()


svm_clf = sk.pipeline.Pipeline(steps=[
    ('scale', sk.preprocessing.StandardScaler()),
    ('svc', sk.svm.SVC())
])

logit_clf = sk.linear_model.LogisticRegression()

ensemble_clf = sk.pipeline.Pipeline(
    steps=[
        ('scale', sk.preprocessing.StandardScaler()),
        ('clf', sk.ensemble.VotingClassifier(
            estimators=[('svc', svm_clf), ('logit', logit_clf)]
        ))
    ]
)


svm_pred = svm_clf.fit(train_X, train_y).predict(test_X)
logit_pred = logit_clf.fit(train_X, train_y).predict(test_X)

ensemble_pred = ensemble_clf.fit(train_X, train_y).predict(test_X)

print(sk.metrics.f1_score(test_y, svm_pred))
print(sk.metrics.f1_score(test_y, logit_pred))
print(sk.metrics.f1_score(test_y, ensemble_pred))


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
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
