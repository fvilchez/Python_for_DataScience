############################# Modelos MA ###################################


####################Modelo de persistencia

#La predicción más simple que podemos hacer a la hora de realizar una predicción
#de series temporales es predecir para el instante t+1 lo que ocurrio en el
#instante t. Este tipo de predicción es conocida como naive forecast. Este
#modelo nos proporcionará predicciones para las cuales podremos calcular la
#serie temporal de sus errores residuales. De forma alternativa podemos
#desarrollar un modelo autoregresivo de la serie temporal y usarlo como modelo.

import pandas as pd
import matplotlib.pyplot as plt

from math import sqrt

from sklearn.metrics import mean_squared_error
#from statsmodels.tsa.ar_model import AR



#Cargamos los datos
series = pd.read_csv('daily-total-female-births.csv', header = 0, index_col = 0,
                     parse_dates = True, squeeze = True)

#Creamos nuestro conjunto de datos lagueado
values = pd.DataFrame(series.values)
dataframe = pd.concat([values.shift(1), values], axis = 1)
dataframe.columns = ['t', 't+1']

#Separamos nuestro conjunto de datos en train y test
X = dataframe.values
train_size = int(len(X) * 0.66)
train, test = X[1:train_size], X[train_size:]
train_X, train_y = train[:,0], train[:,1]
test_X, test_y = test[:,0], test[:,1]

#modelo de persistencia
predictions = [x for x in test_X]

#Vemos el rmse
rmse = sqrt(mean_squared_error(test_y, predictions))

#Calculamos los residuos
residuals = [test_y[i] - predictions[i] for i in range(len(predictions))]
residuals = pd.DataFrame(residuals)
print(residuals.head())


####################### Autoregresión a partir de los errores
#Creamos nuestro conjunto de datos lagueado
values = pd.DataFrame(series.values)
dataframe = pd.concat([values.shift(1), values], axis = 1)
dataframe.columns = ['t', 't+1']

#Separamos nuestro conjunto de datos en train y test
X = dataframe.values
train_size = int(len(X) * 0.66)
train, test = X[1:train_size], X[train_size:]
train_X, train_y = train[:,0], train[:,1]
test_X, test_y = test[:,0], test[:,1]

#modelo de persistencia
predictions = [x for x in test_X]

#Vemos el rmse
rmse = sqrt(mean_squared_error(test_y, predictions))

#Calculamos los residuos
residuals = [test_y[i] - predictions[i] for i in range(len(predictions))]
residuals = pd.DataFrame(residuals)

#Hacemos un modelo de persistencia con el conjunto de entrenamiento
train_pred = [x for x in train_X]

#Calculamos los residuos
train_resid = [train_y[i] - train_pred[i] for i in range(len(train_pred))]

#Aplicamos el modelo AR sobre los residuos
model = AR(train_resid)
model_fit = model.fit()
window = model_fit.k_ar
coef = model_fit.params
print('Lag=%d, Coef=%s' % (window, coef))

# Una vez tenemos fijado nuestro modelo vamos a proceder a aplicarlo en el test
history = train_resid[len(train_resid) - window:]
history = [history[i] for i in range(len(history))]
predictions = list()
expected_error = list()
for t in range(len(test_y)):
    #modelo persistencia
    yhat = test_X[t]
    error = test_y[t] - yhat
    expected_error.append(error)
    #predicción del error
    length = len(history)
    lag = [history[i] for i in range(length-window, length)]
    pred_error = coef[0]
    for d in range(window):
        pred_error += coef[d+1] * lag[window-d-1]
    predictions.append(pred_error)
    history.append(error)
    print('predicted error=%f, expected error = %f' % (pred_error, error))

plt.plot(expected_error)
plt.plot(predictions, color = 'red')
plt.show()


################### Corrigiendo predicciones a partir de los errores

#Podemos hacer uso de las predicciones de nuestros errores para corregir la
#predicción de nuestro modelo.

#Cargamos los datos
series = pd.read_csv('daily-total-female-births.csv', header = 0, index_col = 0,
                     parse_dates = True, squeeze = True)

#Creamos el dataset lagueado
values = pd.DataFrame(series.values)
dataframe = pd.concat([values.shift(1), values], axis = 1)
dataframe.columns = ['t', 't+1']

#Separamos en train y test
X = dataframe.values
train_size = int(len(X) * 0.66)
train, test = X[1:train_size], X[train_size:]
train_X, train_y = train[:,0], train[:,1]
test_X, test_y = test[:,0], test[:,1]

#Hacemos el modelo de persistencia
train_pred = [x for x in train_X]

#Calculamos los residuos
train_resid = [train_y[i] - train_pred[i] for i in range(len(train_pred))]

#Modelamos nuestro modelo AR con los residuos de train
model = AR(train_resid)
model_fit = model.fit()
window = model_fit.k_ar
coef = model_fit.params

#Hacemos predicciones para nuestro conjunto de test agregando el error predicho
#a nuestras predicciones
history = train_resid[len(train_resid)-window:]
history = [history[i] for i in range(len(history))]
predictions = list()
for t in range(len(test_y)):
    #persistencia
    yhat = test_X[t]
    error = test_y[t] - yhat
    #predecimos el error
    length = len(history)
    lag = [history[i] for i in range(length-window, length)]
    pred_error = coef[0]
    for d in range(len(window)):
        pred_error += coef[d+1] * lag[window-d-1]
    #Corregimos nuestra prediccion
    yhat = yhat + pred_error
    predictions.append(yhat)
    history.append(error)
    print('predicted=%f, expected=%f' % (yhat, test_y[t]))

#error
rmse = sqrt(mean_squared_error(test_y, predictions))
print('Test RMSE: %.3f' % rmse)

plt.plot(test_y)
plt.plot(predictions, color = 'red')
plt.show()


