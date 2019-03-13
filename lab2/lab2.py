import pandas as pd

training_csv_path = "ml-labs/datasets/lab2/training.csv"
testing_csv_path = "ml-labs/datasets/lab2/testing.csv"

training = pd.read_csv(training_csv_path)
testing = pd.read_csv(testing_csv_path)

# Remove all columns beginning with "pred_minus_obs"
columns_to_drop = [f"pred_minus_obs_S_b{i}" for i in range(1,10)]
columns_to_drop += [f"pred_minus_obs_H_b{i}" for i in range(1,10)]
training = training.drop(columns=columns_to_drop)
testing = testing.drop(columns=columns_to_drop)
testing.info()
print(training.head())