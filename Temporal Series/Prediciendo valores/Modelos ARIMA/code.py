######################## ARIMA CON  PYTHON ##############################

import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import autocorrelation_plot
from sklearn.metrics import mean_squared_error
from math import sqrt


def parser(x):
    return pd.datetime.strptime('190'+x, '%Y-%m')

series = pd.read_csv('shampoo-sales.csv', header = 0, index_col = 0,
                     parse_dates = True, squeeze = True, date_parser = parser)
series.plot()
plt.show()


#Este conjunto de datos se puede ver como tiene una tendencia bastante evidente.
#Esto sugiere que la serie temporal no es estacionaria y que requiere diferenciación
#al menos de grado 1.

autocorrelation_plot(series)
plt.show()


#En este gráfico se puede observar como existe una correlación positiva entre
#los 10-12 primeros lags, además de existir una correlación significativa entre
#los 5 primeros lags.

#Fijamos el modelo ARIMA
model = ARIMA(series, order = (5,1,0))
model_fit = model.fit(disp = 0)

#Mostramos el summary
print(model_fit.summary())

#Vemos los residuos de nuestro modelo
residuals = pd.DataFrame(model_fit.resid)
residuals.plot()
plt.show()

#Vemos la función de densidad
residuals.plot(kind = 'kde')
plt.show()

#Vemos un resumen estadísticos de los residuos
print(residuals.describe())


########### Rolling Forecast ARIMA Model

X = series.values
size = int(len(X)*0.66)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()

for t in range(len(test)):
    model = ARIMA(history, order = (5,1,0))
    model_fit = model.fit(disp = 0)
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = test[t]
    history.append(obs)
    print('predicted=%f, expected=%f' % (yhat, obs))

#Evaluamos nuestra predicción
rmse = sqrt(mean_squared_error(test, predictions))
print('Test RMSE: %.3f' % rmse)

plt.plot(test)
plt.plot(predictions, color = 'red')
plt.show()
