import os
import shutil
import tarfile
import urllib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split



# Get the housing data

URL_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
DIR_NAME = "datasets/housing/"

def fetch_housing_data():
    dir_path = DIR_NAME
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
    housing_path = DIR_NAME + "housing.tgz"
    with open(housing_path, 'wb') as archive_file:
        data_url = URL_ROOT + DIR_NAME + "housing.tgz"
        with urllib.request.urlopen(data_url) as response:
            shutil.copyfileobj(response, archive_file)
    with tarfile.open(housing_path) as archive_file:
        archive_file.extractall(dir_path)
#fetch_housing_data()

def load_housing_data():
    csv_path = DIR_NAME + "housing.csv"
    return pd.read_csv(csv_path)
housing = load_housing_data()


# Take a quick look at the data
def explore_housing_data(housing=housing):
    # Example data: first 5 rows
    print(housing.head())
    # Column data types and number of non-null values
    print(housing.info())
    # Take a closer look at the ocean_proximity object column
    # Because it came from a CSV & first 4 rows had same value, looks categorical.
    print(housing["ocean_proximity"].value_counts())
    # Get a summary of numerical attributes using describe.
    print(housing.describe())

    # Plot some histograms of numerical categories.
    housing.hist(bins=50, figsize=(20,15))
    plt.show()
#xplore_housing_data()


# Create a test set

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
