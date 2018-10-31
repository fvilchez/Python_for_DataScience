############################# Random Walk #####################################



from random import seed, randrange, random
import matplotlib.pyplot as plt
from pandas.plotting import autocorrelation_plot

#La función randrange nos permite generar números enteros aleatorios entre 0 y
#un límite superior. A continuación vamos a generar un total de 1000 números
#aleatorios entre 0 y 10.

seed(1)
series = [randrange(10) for i in range(1000)]
plt.plot(series)
plt.show()

#Esto no es un random walk, uno de los errores más comunes de la gente es confundir
#un random walk como una lista de números aleatorios y esto no es del todo cierto.


#Creamos un random walk
seed(1)
random_walk = list()
random_walk.append(-1 if random() < 0.5 else 1)
for i in range(1,1000):
    movement = -1 if random() < 0.5 else 1
    value = random_walk[i-1] + movement
    random_walk.append(value)

plt.plot(random_walk)
plt.show()

#Si mostramos el gráfico de autocorrelación esperamos que exista una fuerte
#autocorrelación entre instantes anteriores y una caida lineal con el resto de
#instantes.

autocorrelation_plot(random_walk)
plt.show()


#Podemos hacer que nuestra serie sera estacionario mediante diferenciación

diff = list()
for i in range(1, len(random_walk)):
    value = random_walk[i] - random_walk[i-1]
    diff.append(value)

plt.plot(diff)
plt.show()


#Si ahora vemos la autocorrelación para esta serie

autocorrelation_plot(diff)
plt.show()

#Un serie random_walk no puede ser predicha. Podemos pensar que la mejor
#predicción que podríamos hacer es usar la observación del paso anterior
#como lo que sucederá en el instante futuro, puesto que sabemos que el instante
#futuro es una función del instante anterior.

from sklearn.metrics import mean_squared_error
from math import sqrt

#Preparamos el dataset
train_size = int(len(random_walk) * 0.66)
train, test = random_walk[:train_size], random_walk[train_size:]

#Predecimos
predictions = list()
history = train[-1]

for i in range(len(test)):
    yhat = history
    predictions.append(yhat)
    history = test[i]

rmse = sqrt(mean_squared_error(test, predictions))
print('Persistence RMSE: %.3f' % rmse)

#Otro error es sabiendo que el error es 1 o 1, intentar predecir el siguiente
#valor agregando el valor de -1 o 1 de forma aleatoria al valor anterior.

predictions = list()
history = train[-1]
for i in range(len(test)):
    yhat = history + (-1 if random() < 0.5 else 1)
    predictions.append(yhat)
    history = test[i]

rmse = sqrt(mean_squared_error(test,predictions))





