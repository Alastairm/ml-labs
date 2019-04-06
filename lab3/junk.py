
DAYS_IN_MONTH = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
SEASON = {
    'fall': 80,
    'winter': 172,
    'sprint': 266,
    'summer': 356
}


def get_day_of_year(month: int, day_of_month: int) -> int:
    day_of_year = sum(DAYS_IN_MONTH[:month - 1])
    day_of_year += day_of_month
    return day_of_year


def get_month_and_day(day_of_year: int) -> Tuple[int, int]:
    month = 1
    for i in range(12):
        if day_of_year > DAYS_IN_MONTH[i]:
            month += 1
            day_of_year -= DAYS_IN_MONTH[i]
        else:
            return (month, day_of_year)




SOLAR_COLUMNS_TO_DROP = [
    'Product code',
    'Bureau of Meteorology station number'
]
TEMP_COLUMNS_TO_DROP = [
    'Product code',
    'Bureau of Meteorology station number',
    'Days of accumulation of maximum temperature',
    'Year',
    'Month',
    'Day',
    'Quality'
]

SOLAR = "Daily global solar exposure (MJ/m*m)"
TEMP = "Maximum temperature (Degree C)"
solar_data = solar_data.drop(columns=SOLAR_COLUMNS_TO_DROP)
temp_data = temp_data.drop(columns=TEMP_COLUMNS_TO_DROP)
data = pd.concat(solar_data temp_data, axis=1)


solar = list(solar_data['Daily global solar exposure (MJ/m*m)'])
month = list(solar_data['Month'])
day = list(solar_data['Day'])
temp = list(temp_data['Maximum temperature (Degree C)'])

data = []
for i in range(len(solar)):
    data.append([solar[i], month[i], day[i]])
X = pd.DataFrame(data, columns=['solar', 'month', 'day'])
y = pd.DataFrame(temp, columns=['temp'])
# plt.scatter(data['day'], data['temp'])
# plt.scatter(data['day'], data['solar'])
# plt.show()

data = model_selection.train_test_split(X, y,
                                        test_size=0.2,
                                        train_size=0.8)
for d in data:
    d = d.reset_index()
train_X, test_X, train_y, test_y = data
# scaler = preprocessing.StandardScaler().fit(train_X)


train_y = train['temp']
train_X = train.drop(columns=['temp'])
test_y = test['temp']
test_X = test.drop(columns=['temp'])

svm_regressor = svm.SVR(kernel='linear')

dt_regressor = tree.DecisionTreeRegressor()

svm_pred_y = svm_regressor.fit(train_X, train_y).predict(test_X)
dt_pred_y = dt_regressor.fit(train_X, train_y).predict(test_X)

print(metrics.mean_squared_error(test_y, svm_pred_y))
print(metrics.mean_squared_error(test_y, dt_pred_y))

plt.scatter(test_X['day'], test_y)
plt.scatter(test_X['day'], svm_pred_y)
plt.scatter(test_X['day'], dt_pred_y)
plt.show()