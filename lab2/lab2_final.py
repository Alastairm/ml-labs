"""lab_2"""

import typing
from typing import Any, Dict, List  # Types make method defs more useful.
import warnings  # Used to clearup some sklearn output.

from matplotlib import pyplot
import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV, SGDClassifier

# Constants
CLASS_LABELS = ['s', 'h', 'd', 'o']
DATA_LABELS = ["b1", "b2", "b3", "b4", "b5", "b6", "b7", "b8", "b9"]

# Types
Array = typing.Union[List, np.array, pd.DataFrame]  # np "array_like"


#######################################################################

"""Task 1: Fetch and Explore the data.

1.1 Read in the contents of both csv files.
1.2 Inspect what the columns are by displaying the first few lines
    of the file.
1.3 Use appropriate functions to display (visualize) the different
    features (or attributes / columns).
1.4 Display some plots for visualizing the data.
1.5 Describe what you see.
"""
# Task 1.1 #
DEBUG_FILE_PATH = "/home/alastair/Documents/ml/"
training = pd.read_csv(DEBUG_FILE_PATH+"training.csv")
testing = pd.read_csv(DEBUG_FILE_PATH+"testing.csv")

# Task 1.2
training.columns
#
training.head()
#
testing.head()

# Task 1.2
training.describe()

# print(training.info())
# print(training.describe())
# training.hist(bins=50, figsize=(20, 15))
# pyplot.show()


"""Task 2: Reorganise data to simplify classification task.

2.1 Remove all the columns whose names begin with pred minus obs.
2.2 Extract class labels from the first column of both data sets.
"""

columns_to_drop: List[str] = list()
# Task 2.1 Drop "pred_minus_obs_..." columns:
columns_to_drop += [f"pred_minus_obs_H_b{i}" for i in range(1, 10)]
columns_to_drop += [f"pred_minus_obs_S_b{i}" for i in range(1, 10)]
# Task 2.2 Extract class label columns:
training_labels = training["class"]
testing_labels = testing["class"]
columns_to_drop += ["class"]

training = training.drop(columns=columns_to_drop)
testing = testing.drop(columns=columns_to_drop)


""" Task 3: Count class labels.

3.1 Count the number of instance for each class label.
3.2 Do you have an imbalanced training set?
"""


def count_labels(data: List[str], class_labels: List[str]) -> List[int]:
    """Counts number of instances of each class label in data."""
    counts = []
    for label in class_labels:
        counts.append(sum([x == label for x in data]))
    return counts


def label_proportions(data: List[str], class_labels: List[str]) -> List[float]:
    """Get an array containing proportion each class is of total data."""
    counts = count_labels(data, class_labels)
    total = sum(counts)
    proportions = [counts[i] / total for i in range(len(counts))]
    return proportions


def print_table(data: Array,
                title=None,
                row_titles=None,
                column_titles=None,
                append_row=None,
                append_column=None) -> pd.DataFrame:
    """Convert data to a Dataframe and print it."""
    data = np.array(data)
    data = [list(row) for row in data]  # Data is now a list of lists.
    if append_row and append_column:
        data += append_row
        if len(data) != len(append_column):
            raise AttributeError(
                f"column length mismatch: {len(data)}!={len(append_column)}")
        else:
            for i in range(len(data)):
                data[i] += [append_column[i]]
    elif append_row:
        data += append_row
    elif append_column:
        if len(data) != len(append_column):
            raise AttributeError("column length mismatch")
        else:
            for i in range(len(data)):
                data[i] += [append_column[i]]
    data = np.array(data)  # Data is now ndarray
    data = pd.DataFrame(data, row_titles, column_titles)
    if title:
        print(title)
    print(data)
    return data


training_label_counts = count_labels(training_labels, CLASS_LABELS)
testing_label_counts = count_labels(testing_labels, CLASS_LABELS)
print_table([training_label_counts, testing_label_counts], "Class counts",
            ["Training", "Testing"], CLASS_LABELS)

print("\nIt is very hard to determine balance with the training and testing"
      " data class counts alone.\n")

total_labels = training_labels.append(testing_labels)
total_label_counts = count_labels(total_labels, CLASS_LABELS)
training_row_num = len(training)
testing_row_num = len(testing)
total_row_num = len(total_labels)
print_table([training_label_counts, testing_label_counts],
            "Class counts with totals",
            ["Training", "Testing", "Total"], CLASS_LABELS + ["Any"],
            [total_label_counts],
            [training_row_num, testing_row_num, total_row_num])

print("\nViewing what proportion each class is of the dataset would make it"
      " easier to comment on the balance of the training set.\n")

training_label_proportions = label_proportions(training_labels, CLASS_LABELS)
testing_label_proportions = label_proportions(testing_labels, CLASS_LABELS)
total_label_proportions = label_proportions(total_labels, CLASS_LABELS)
print_table([training_label_proportions, testing_label_proportions,
            total_label_proportions],
            "Class Label proportions",
            ["Training", "Testing", "Total"], CLASS_LABELS)

""" Task 4. Perform data normalization before performing classification.

4.1 Select an appropriate data normalisation method.
4.2 Normalise the data, ensuring the method is the same for both datasets.
"""

# Scaling of sample features
# We don't need to be told ints are turning into floats...
DataConversionWarning = sklearn.exceptions.DataConversionWarning
sklearn.warnings.filterwarnings("ignore", category=DataConversionWarning)
# Scale sample feature values
scaler = StandardScaler().fit(training)
training_scaled = scaler.transform(training)
testing_scaled = scaler.transform(testing)


""" Task 5. Stochastic gradient descent classification.

5.1 Use the stochastic gradient descent classifier to perform one-versus-all
    binary classification on the 4 class labels.
5.2 Show the confusion matrix on the test set.
5.3 Try experimenting with some hyperparameters to see if you can improve the
    performance of the classification.
"""


def print_confusion_matrix(predictions: Array, actual: Array, label: str):
    """Print a confusion matrix with labeled rows and columns."""
    confusion_matrix = sklearn.metrics.confusion_matrix(actual, predictions)
    tn, fp, fn, tp = confusion_matrix.ravel()
    print(f"Class {label} confusion matrix:")
    data_table = [[tp, tn], [fp, fn]]
    print_table(data_table,
                row_titles=["Correct", "Incorrect"],
                column_titles=[f"is {label}", f"not {label}"])


def binary_classify(classifier: Any, class_labels: List[str],
                    training_labels: Array, training_data: Dict,
                    testing_labels: Array, testing_data: Dict) -> Dict:
    """Runs an SGD Binary Classifier once for each class label."""
    return {}


# Create classifier
clf = SGDClassifier(loss="log",
                    penalty="l2",
                    max_iter=50,
                    tol=0.01)

print("Stochastic gradient descent (SGD) binary classification:")
sgd_results: Dict[str, Dict[str, List[bool]]] = {}
for label in CLASS_LABELS:
    # Classification fit & predict parameters
    x_train = training_scaled
    y_train = [x is label for x in training_labels]
    x_test = testing_scaled
    # Get true and predicted labels
    y_true = [x is label for x in testing_labels]
    y_pred = clf.fit(x_train, y_train).predict(x_test)
    # Store results
    sgd_results[label] = {}
    sgd_results[label]["y_true"] = y_true
    sgd_results[label]["y_pred"] = y_pred

    print("")
    print_confusion_matrix(y_pred, y_true, label)

""" Task 6. Logistic Regression Classification

6.1 Use LogisticRegressionCV to perform multiclass classification on the 4
    classes.
6.2 Show the confusion matrix on the test set.
6.3 Try experimenting with some hyperparameters to see if you can improve the
    performance of the classification.
"""

clf = LogisticRegressionCV()

x_train = training_scaled
y_train = training_labels
x_test = testing_scaled

y_true = testing_labels
y_pred = clf.fit(x_train, y_train).predict(x_test)
cm = sklearn.metrics.confusion_matrix(y_pred, y_true)
print(cm)
""" Task 7.Conclusions
What is your conclusion? Which classifier gave better performance?
"""


"""
5
Presentation
A few tips on the presentation of your ipynb files:
• Present your ipynb file as a portfolio, with Markdown cells inserted
appropriately to
explain your code. See the following links if you are not familiar with
Markdown:
https://www.markdownguide.org/cheat-sheet/ (basic)
https://jupyter-notebook.readthedocs.io/en/stable/examples/Notebook/Typesetting%
20Equations.html (more advanced)
• Dividing the portfolio into suitable sections and subsections (with section
and subsection
numbers and meaningful headings) would make your portfolio easier to follow.
• Avoid having too many small Markdown cells that have only one short sentence.
In
addition to Markdown cells, some short comments can be put alongside the
 Python code.
• Use meaningful variable names.
• When printing out your results, provide some textual description so that the
 output is
meaningful.
"""
