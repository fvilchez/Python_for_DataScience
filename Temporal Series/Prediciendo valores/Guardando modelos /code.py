################## Salvando Modelos y Realizando Predicciones #################

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AR
from sklearn.metrics import mean_squared_error
import numpy as np
from math import sqrt
from statsmodels.tsa.ar_model import ARResults

####### Seleccionando nuestro modelo de predicción

def difference(dataset):
    diff = list()
    for i in range(i, len(dataset)):
        value = dataset[i] - dataset[i-1]
        diff.append(value)
    return np.array(diff)


def predict(coef, history):
    yhat = coef[0]
    for i in range(1, len(coef)):
        yhat += coef[i] * history[-i]
    return yhat


series = pd.read_csv('daily-total-female-births.csv', header = 0, index_col = 0,
                     parse_dates = True, squeeze = True)

#Spliteamos nuestro conjunto de datos
X = difference(series.values)
size = int(len(X)*0.66)
train, test = [0:size], X[size:]

#Entrenamos nuestro modelo autoregresivo
model = AR(train)
model_fit = model.fit(maxlag = 6, disp = False)
window = model_fit.k_ar
coef = model_fit.params

#Hacemos predicciones de forma walk forward
history = [train[i] for i in range(len(train))]
predictions = list()
for t in range(len(test)):
    yhat = predict(coef, history)
    obs = test[t]
    predictions.append(yhat)
    history.append(obs)

rmse = sqrt(mean_squared_error(test, predictions))
print('Test RMSE: %.3f' % rmse)

plt.plot(test)
plt.plot(predictions, color = 'red')
plt.show()


######### Finalizando y salvando nuestro modelo

#Una vez hemos seleccionado nuestro modelo, debemos finalizarlo. Esto significa
#que debemos salvar la información aprendida por el modelo para no tener que
#recrear todo de nuevo. Esto envuelve entrenar nuestro modelo con todos los
#datos disponibles y entonces salvar el modelo en un fichero.

def difference(dataset):
    diff = list()
    for i in range(1, len(dataset)):
        value = dataset[i] - dataset[i-1]
        diff.append(value)
    return np.array(diff)


series = pd.read_csv('daily-total-female-births.csv', header = 0, index_col = 0,
                     parse_dates = True, squeeze = True)

#Diferenciamos los datos
X = difference(series.values)

#Fijamos el modelo
model = AR(X)
model_fit = model.fit(maxlag = 6, disp = False)

#Salvamos el modelo en un fichero
model_fit.save('ar_model.pkl')

#Salvamos el dataset diferenciado
numpy.save('ar_data.npy', X)

#Salvaos la última observación
numpy.save('ar_obs.npy', [series.values[-1]])


#Cargamos los datos de nuestro modelo AR
loaded = ARResults.load('ar_model.pkl')
print(loaded.params)
data = np.load('ar_data.npy')
last_ob = np.load('ar_obs.npy')
print(last_ob)


#Ejemplo
series = pd.read_csv('daily-total-female-births.csv', header = 0, index_col = 0,
                     parse_dates = True, squeeze = True)
#Fijamos el modelo
X = difference(series.values)
window_size = 6
model = AR(X)
model_fit = model.fit(maxlag = window_size, disp = False)

#Salvamos el coeficiente
coef = model_fit.params
np.save('man_model.npy', coef)

#Salvamos el lag
lag = X[-window_size:]
np.save('man_data.npy', lag)

#Salvamos la última observación
np.save('man_obs.npy', [series.values[-1]])


################### Haciendo predicciones series temporales

#Cargamos el modelo
model = ARResults.load('ar_model.pkl')
data = np.load('ar_data.npy')
last_ob = np.load('ar_obs.npy')

#Hacemos predicciones
predictions = model.predict(start = len(data), end = len(data))

#Transformamos las predicciones
yhat = predictions[0] + last_ob[0]
print('Prediction: %f' % yhat)


#Esto también se podría realizar de la siguiente manera
def predict(coef, history):
    yhat = coef[0]
    for i in range(1, len(coef)):
        yhat += coef[i]*history[-1]
    return yhat

#Cargamos el modelo
coef = np.load('man_model.npy')
lag = numpy.load('man_data.npy')
last_ob = np.load('man_obs.npy')

#Hacemos predicciones
prediction = predict(coef, lag)

#Transformamos las predicciones
yhat = prediction + last_ob[0]
print('Prediction %f' % yhat)



############### Actualizando nuestro modelo de predicción

#Obtenemos una observación real
observation = 48

#Cargamos los datos
data = np.load('ar_data.npy')
last_ob = np.load('ar_obs.npy')

#Actualizamos y salvamos la observación diferenciada
diffed = observation - last_ob[0]
data = np.append(data, [diffed], axis = 0)
np.save('ar_data.npy', data)

#Actualizamos y salvamos la observación real
last_ob[0] = observation
np.save('ar_obs.npy', last_ob)


############### Otra opción es la siguiente
observation = 48

lag = np.load('man_data.npy')
last_ob = np.load('man_obs.npy')
diffed = observation - last_ob[0]
lag = np.append(lag[1:], [diffed], axis = 0)
np.save('man_data.npy', lag)

#Actualizamos y salvamos las observaciones reales
last_ob[0] = observation
np.save('man_obs.npy', last_ob)

