############################## Moving Average Smoothing ########################


#################################### Rolling #################################
#
#La función rolling() nos permite crear una ventanza deslizante del tamaño que
#le indiquemos como parámetro.

import pandas as pd
import matplotlib.pyplot as plt

series = pd.read_csv('daily-total-female-births.csv', header = 0, index_col = 0,
                     parse_dates = True, squeeze = True)

rolling = series.rolling(window = 3)
rolling_mean = rolling.mean()

#Mostramos la serie original y con rolling
series.plot()
rolling_mean.plot(color = 'red')
plt.show()

series[:100].plot()
rolling_mean[:100].plot(color = 'red')
plt.show()


################################ Moving Average as feature #####################

series = pd.read_csv('daily-total-female-births.csv', header = 0, index_col = 0,
                     parse_dates = True, squeeze = True)

df = pd.DataFrame(series.values)
width = 3
lag1 = df.shift(1)
lag3 = df.shift(width-1)
window = lag3.rolling(window = width)
means = window.mean()
dataframe = pd.concat([means, lag1, df], axis = 1)
dataframe.columns = ['mean', 't', 't+1']



################################# Moving Average Predictions ###################
from sklearn.metrics import mean_squared_error
import numpy as np

series = pd.read_csv('daily-total-female-births.csv', header = 0, index_col = 0,
                  parse_dates = True, squeeze = True)

#Preparamos la situación
X = series.values
window = 3
history = [X[i] for i in range(window)]
test = [X[i] for i in range(window, len(X))]
predictions = list()
#Deslizamiento hacia delante
for t in range(len(test)):
    length = len(history)
    yhat = np.mean([history[i] for i in range(length-window, length)])
    obs = test[t]
    predictions.append(yhat)
    history.append(obs)

rmse = np.sqrt(mean_squared_error(test, predictions))
print('Test RMSE: %.3f' % rmse)

plt.plot(test)
plt.plot(predictions, color = 'red')
plt.show()

                     
