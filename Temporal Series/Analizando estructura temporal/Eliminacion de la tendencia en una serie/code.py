################################### Trend ####################################
#
#Detección y eliminación de la tendencia de nuestra serie temporal
#
##############################################################################

import pandas as pd
import matplotlib.pyplot as plt

def parser(x):
    return pd.datetime.strptime('190' + x, '%Y-%m')

series = pd.read_csv('shampoo-sales.csv', header = 0, index_col = 0,
                     parse_dates = True, squeeze = True, date_parser = parser)

X = series.values
diff = list()
diff = [X[i] - X[i-1] for i in range(1, len(X))]
plt.plot(diff)
plt.show()


###################### Usando modelos para ver tendencias ######################

from sklearn.linear_model import LinearRegression
import numpy as  np

X = [i for i in range(0, len(series))]
X = np.reshape(X, (len(X), 1))
y = series.values

#Fijamos el modelo
model = LinearRegression()
model.fit(X,y)

#Calculamos la tendencia
trend = model.predict(X)
plt.plot(y)
plt.plot(trend)
plt.show()

#detrend
detrended = [y[i] - trend[i] for i in range(0, len(series))]
plt.plot(detrended)
plt.show()
