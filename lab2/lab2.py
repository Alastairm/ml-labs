import matplotlib.pyplot as plt
import pandas as pd

training = pd.read_csv("training.csv")
testing = pd.read_csv("testing.csv")

# Drop "pred_minus_obs_..." columns:
columns_to_drop = [f"pred_minus_obs_H_b{i}" for i in range(1,10)]
columns_to_drop += [f"pred_minus_obs_S_b{i}" for i in range(1,10)]
training = training.drop(columns=columns_to_drop)
testing = testing.drop(columns=columns_to_drop)


def visualise_data():
    training.hist(bins=50, figsize=(20,15))
    plt.show()

    print(training.head())
    print(training.describe())
# visualise_data()


def check_label_balance():
    def count_labels(data, labels):
        """Counts number of instances of each label in data."""
        counts = []
        for label in labels:
            counts.append(sum([x == label for x in data]))
        return counts

    def percents_from_counts(data):
        """Converts label counts to percentages."""
        total = sum(data)
        percents = []
        for x in data:
            percents.append(f"{(x/total*100):4.1f}%")
        return percents

    class_labels = ["d", "h", "o", "s"]
    training_counts = count_labels(training["class"], class_labels)
    testing_counts = count_labels(testing["class"], class_labels)
    print("Total class sizes:")
    print(pd.DataFrame([training_counts, testing_counts],
                ["Training", "Testing"],
                class_labels))

    train_percents = percents_from_counts(training_counts)
    test_percents = percents_from_counts(testing_counts)
    print("\nClass proportions:")
    print(pd.DataFrame([train_percents, test_percents],
                ["Training", "Testing"],
                class_labels))

    print("\nTotal data instances:")
    print(f"Training    {len(training)}")
    print(f"Testing     {len(testing)}")
# check_label_balance()


# Data Normalisation
from sklearn.preprocessing import StandardScaler
train_features = training.drop("class", axis=1)
test_features = testing.drop("class", axis=1)

scaler = StandardScaler().fit(train_features)

train_features = scaler.transform(train_features)
test_features = scaler.transform(test_features)


def create_binary_class(label, data):
    is_label = [label == x for x in data]
    return [x * 1 for x in is_label]

def classification_using_sgdregressor():
    # Stochastic Gradient Descent
    # USE CLASSIFIER NOT REGRESSOR
    # https://scikit-learn.org/stable/modules/sgd.html
    from sklearn.linear_model import SGDRegressor
    train_is_s = create_binary_class('s', training["class"])

    # Create SGD object
    sgd_reg = SGDRegressor()
    # Train the regressor
    sgd_reg.fit(train_features, train_is_s)
    # Run a prediction on the test set
    s_predict = sgd_reg.predict(test_features)
    predict_is_s = [(x > 0.5) * 1 for x in s_predict]
    # Get the actual classes
    test_is_s = create_binary_class('s', testing["class"])
    # Zip these into tuples for our confusion matrix:
    is_s = list(zip(predict_is_s, test_is_s))
    # Calculate confusion matrix values
    sum_predict_s_is_s = sum([x == (1,1) for x in is_s])
    sum_predict_s_not_s = sum([x == (1,0) for x in is_s])
    sum_predict_not_s_is_s = sum([x == (0,1) for x in is_s])
    sum_predict_not_s_not_s = sum([x == (0,0) for x in is_s])
    # Print confusion matrix
    print("              Prediction    ")
    print("            | Is S | Not S")
    print(f"Actual Is S |  {sum_predict_s_is_s}  |  {sum_predict_not_s_is_s}")
    print(f"      Not S |  {sum_predict_s_not_s}  |  {sum_predict_not_s_not_s}")