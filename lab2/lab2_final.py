"""lab_2"""

import typing
from typing import Dict, List  # Types make method defs more useful.
import warnings  # Used to clearup some sklearn output.

from matplotlib import pyplot
import numpy as np
import pandas as pd

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
training = pd.read_csv("training.csv")
testing = pd.read_csv("testing.csv")

# Task 1.2
(training.columns)
#
training.head()
#
testing.head()

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

3.3 Count the number of instance for each class label.
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
                row_titles=None,
                column_titles=None,
                append_row=None,
                append_coolumn=None):
    print("pass")



""" Task 4. Perform appropriate data normalization before performing classification.You can use
MinMaxScaler, StandardardScaler, or any suitable normalization function in the
sklearn.preprocessing package. You can also write your own normalization code if
you prefer. Either way, ensure that you normalize the training data and the test data
consistently.
"""



""" Task 5. Use the stochastic gradient descent classifier to perform one-versus-all binary classification
on the 4 class labels. Show the confusion matrix on the test set. You should try exper-
imenting with some hyperparameters to see if you can improve the performance of the
classification.
"""

""" Task 6. Repeat the above step using the logistic regression classifier for multi-class classification
with the Softmax function."""

""" Task 7. What is your conclusion? Which classifier gave better performance?"""


"""
5
Presentation
A few tips on the presentation of your ipynb files:
• Present your ipynb file as a portfolio, with Markdown cells inserted appropriately to
explain your code. See the following links if you are not familiar with Markdown:
https://www.markdownguide.org/cheat-sheet/ (basic)
https://jupyter-notebook.readthedocs.io/en/stable/examples/Notebook/Typesetting%
20Equations.html (more advanced)
• Dividing the portfolio into suitable sections and subsections (with section and subsection
numbers and meaningful headings) would make your portfolio easier to follow.
• Avoid having too many small Markdown cells that have only one short sentence. In
addition to Markdown cells, some short comments can be put alongside the Python code.
• Use meaningful variable names.
• When printing out your results, provide some textual description so that the output is
meaningful.
"""