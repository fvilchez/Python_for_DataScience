######################### Algorito de persistencia ###########################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


def parser(x):
    return pd.datetime.strptime('190'+x, '%Y-%m')


series = pd.read_csv('shampoo-sales.csv', header = 0, index_col = 0,
                     parse_dates = True, squeeze = True, date_parser = parser)

values = DataFrame(series.values)
dataframe = concat([values.shift(1), values], axis = 1)
dataframe.columns = ['t', 't+1']

X = dataframe.values
train_size = int(len(X) * 0.66)

train, test = X[1:train_size], X[train_size:]
train_X, train_y = train[:,0], train[:,1]
test_X, test_y = test[:,0], test[:,1]

def model_persistence(x):
    yhat = model_persistence(x)
    predictions.append(yhat)

rmse = sqrt(mean_squared_error(test_y, predictions))
print('Test RMSE: %3.f' % rmse)

plt.plot(train_y)
plt.plot([None for i in train_y] + [x for x in test_y])
plt.plot([None for i in train_y] + [x for x in predictions])
plt.show()
