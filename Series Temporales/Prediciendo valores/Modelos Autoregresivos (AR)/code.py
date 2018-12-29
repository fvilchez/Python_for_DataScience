####################### Modelos autoregresivos ################################


import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import lag_plot
from pandas.plotting import autocorrelation_plot
from sklearn.metrics import mean_squared_error
from math import sqrt
#from statsmodels.tsa.ar_model import AR
#import statsmodels.api as sm
#import statsmodels.api as sm

#Los mod#elos autoregresivos asumen que existe una correlación entre nuestra
#variable de salida y los instantes anteriores. Por lo tanto, antes de aplicar
#un modelo autoregresivo debemos de asegurarnos de que esto se cumple. Pandas
#dispone de la función lag_plot() que nos muestra mediante un scatter la
#relacción existente entre una variable y sus instante anterior.

#Cargamos los datos
series = pd.read_csv('daily-minimum-temperatures.csv', header = 0, index_col = 0,
                     parse_dates = True, squeeze = True)

#Vemos nuestro lag_plot
#lag_plot(series)
#plt.show()


#Otra opción que tenemos es la de calcular directamente la correlación entre una
#variable con sus instantes anteriores.

values = pd.DataFrame(series.values)
dataframe = pd.concat([values.shift(1), values], axis = 1)
dataframe.columns = ['t', 't+1']
result = dataframe.corr()
#print(result)


#Esta forma de chequear la correlación entre variables puede ser bastante
#tediosa, para esto pandas dispone de los gráficos autocorrelation_plot, estos
#gráficos nos muestran el coeficiente de correlación entre una variable y sus
#respectivos lags.

#autocorrelation_plot(series)
#plt.show()


#La librería statsmodels también dispone de la función plot_acf, que nos permite
#visualizar la misma información.

##sm.tsa.graphics.plot_acf(series, lags = 31)
##plt.show()


#El modelo más simple que podemos usar para hacer predicciones es el modelo
#de persistencia. Este modelo nos propocionará una base de rendimiento de
#nuestros modelos de predicción.

##
##series = pd.read_csv('daily-minimum-temperatures.csv', header = 0,
##                     index_col = 0, parse_dates = True, squeeze = True)
##
##values = pd.DataFrame(series.values)
##dataframe = pd.concat([values.shift(1), values], axis = 1)
##dataframe.columns = ['t', 't+1']
##
##X = dataframe.values
##train, test = X[1:len(X)-7], X[len(X)-7:]
##train_X, train_y = train[:,0], train[:,1]
##test_X, test_y = test[:,0], test[:,1]
##
##def model_persistence(x):
##    return x
##
##predictions = [model_persistence(x) for x in test_X]
##rmse = sqrt(mean_squared_error(test_y, predictions))
##print('Test RMSE: %.3f' % rmse)
##
##plt.plot(test_y)
##plt.plot(predictions, color = 'red')
##plt.show()


#La librería statsmodels nos permite hacer uso de modelos autoregresivos, esta
#librería selecciona el mejor lag y mediante esto realiza la regresión lineal.

series = pd.read_csv('daily-minimum-temperatures.csv', header = 0, index_col = 0,
                     parse_dates = True, squeeze = True)

X = series.values
train, test = X[1:len(X)-7], X[len(X)-7:]

model = AR(train)
model_fit = model.fit()
print('Lag: %s' % model_fit.k_ar)
print('Coefficients: %s' % model_fit.params)

predictions = model_fit.predict(start = len(train),
                                end = len(train) + len(test)-1,
                                dynamic = False)

for i in range(len(predictions)):
    print('predicted=%f, expected=%f' % (predictions[i], test[i]))

rmse = sqrt(mean_squared_error(test, predictions))
print('Test RMSE: %.3f' % rmse)

plt.plot(test)
plt.plot(predictions, color = 'red')
plt.show()


#Uno de los problemas es que cada vez que tengamos una observación nueva tendremos
#que reentrenar nuestro modelo, esto puede ser en función del número de datos
#bastante costoso. Otra opción es hacer uso de los coeficientes devueltos y
#predecir, aunque de esto forma mantenemos el lag calculado inicialmente.

series = pd.read_csv('daily-minimum-temperatures.csv', header = 0,
                     index_col = 0, parse_dates = True, squeeze = True)

X = series.values
train, test = X[1:len(X)-7], X[len(X)-7:]

model = AR(train)
model_fit = model.fit()
window = model_fit.k_ar
coef = model_fit.params

history = train[len(train) - window:]
history = [history[i] for i in range(len(history))]
predictions = list()

for t in range(len(test)):
    length = len(history)
    lag = [history[i] for i in range(length - window, length)]
    yhat = coef[0]
    for d in range(window):
        yhat += coef[d+1] * lag[window-d-1]
    obs = test[t]
    predictions.append(yhat)
    history.append(obs)
    print('predicted=%f, expected=%f' % (yhat, obs))

rmse = sqrt(mean_squared_error(test, predictions))
print('Test RMSE: %.3f' % rmse)

plt.plot(test)
plt.plot(predictions, color = 'red')
plt.show()


