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
from sklearn import (
    ensemble,
    linear_model,
    model_selection,
    pipeline,
    preprocessing,
    svm
)


data_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data'
data = pd.read_csv(data_url)


def data_clean() -> Tuple[pd.Series, pd.Series, pd.DataFrame, pd.DataFrame]:
    data.drop(columns='name')
    y = data['status']
    data.drop(columns='status')
    X = data
    return model_selection.train_test_split(y, X, test_size=0.2)


train_y, test_y, train_X, test_X = data_clean()


svm_clf = pipeline.Pipeline(steps=[
    ('scale', preprocessing.StandardScaler()),
    ('svc', svm.SVC())
])

logit_clf = linear_model.LogisticRegression()

ensemble_clf = ensemble.VotingClassifier(estimators=[
    ('svc', svm_clf), ('logit', logit_clf)
])

pred = ensemble_clf.fit(train_X, train_y).predict(test_X)