############################ Transformaciones #################################

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


#Las transformaciones son realizadas con el objetivo de eliminar ruido y mejorar
#la señal en la predicción de nuestra serie temporal.


#Cargamos los datos

series = pd.read_csv('airline-passengers.csv', header = 0, index_col = 0,
                     parse_dates = True, squeeze = True)

plt.figure(1)

#Nos creamos un gráfico de tipo line plot
plt.subplot(211)
plt.plot(series)
#Nos creamos un gráfico de tipo histograma
plt.subplot(212)
plt.hist(series)
plt.show()

#Podemos ver claramente que nuestra serie no es estacionaria, es decir, la media
#y la varianza cambian a lo largo del tiempo. Esto hace complicado modelar
#nuestra serie mediante métodos estadísticos clásicos, como ARIMA. Esto es
#causado por la presencia de tendencia y una componente estacional.
#Además se puede apreciar como la varianza aumenta con el tiempo.

############################# Square Root Transform ###########################
#
#Una serie temporal que tiene una tendencia de crecimiento cuadrática, se puede
#hacer lineal haciendo uso de la raíz cuadrada.
#
################################################################################

#Nos creamos un array que contiene el cuadrado de los números del 1 al 99
lista = [i**2 for i in range(1,100)]
plt.figure(1)
#line plot
plt.subplot(211)
plt.plot(lista)
#histogram
plt.subplot(212)
plt.hist(lista)

plt.show()


#Aplicamos la transformación
lista_transform = np.sqrt(lista)
plt.figure(1)
#line plot
plt.subplot(211)
plt.plot(lista_transform)
#histograma
plt.subplot(212)
plt.hist(lista_transform)
plt.show()

#Podriamos pensar que nuestro conjunto de datos AirPassenger tiene una tendencia
#cuadrática, si es así tras aplicar la transformación nuestra tendencia pasará
#a ser lineal y la distribución de nuestras observaciones quizás sean Gaussianas.

series = pd.read_csv('airline-passengers.csv', header = 0, index_col = 0,
                     parse_dates = True, squeeze = True)

series_transform = np.sqrt(series)

plt.figure(1)
#line plot
plt.subplot(211)
plt.plot(series_transform)
#histograma
plt.subplot(212)
plt.hist(series)

plt.show()

#En nuestro caso la tendencia se ve reducida, pero no llega a ser lineal, esto
#indica que nuestra serie no tiene una tendencia lineal.


########################## Transformación Logarítmica ##########################
#
#Una de las tendencias más extremos es la tendencia exponencial. Las series
#temporales con una distribución exponencial pueden ser convertidas a lineales
#tomando el logaritmo de sus valores.
#
################################################################################

#Nos creamos una lista donde estamos obteniendo valores exponenciales
lista = [np.exp(i) for i in range(1,100)]
plt.figure(1)
#line plot
plt.subplot(211)
plt.plot(lista)
#histograma
plt.subplot(212)
plt.hist(lista)

plt.show()


#Aplicamos la transformación logarítmica
transform_lista = np.log(lista)
plt.figure(1)
#line plot
plt.subplot(211)
plt.plot(transform_lista)
#histograma
plt.subplot(212)
plt.hist(transform_lista)

plt.show()

#Veamos si nuestro conjunto de datos airline-passegers tiene una tendencia expo-
#nencial.

series = pd.read_csv('airline-passengers.csv', header = 0, index_col = 0,
                     parse_dates = True, squeeze = True)

series_transform = np.log(series)

plt.figure(1)
#line plot
plt.subplot(211)
plt.plot(series_transform)
#histograma
plt.subplot(212)
plt.hist(series_transform)

plt.show()

#La transformación logarítmica se trata de una de las transformaciones más
#populares en las series temporales ya que se trata de una transformación
#muy efectiva a la hora de eliminar la varianza exponencial. Es importante
#hacer nota que esta transformación asume valores positivos y distintos de cero.
#Es común transformar los valores, sumando a todos estos valores una constante.
#transform = log(constante + X), donde X es la serie temporal.


###################################### Box Cox #################################
#
#La transformación Box-Cox es una transformación configurable que soporta ambas
#transformaciones: logarítmica y raíz cuadrada. Además se puede configurar de
#forma que un conjunto de transformaciones son evaluadas de forma automática
#y se selecciona el mejor ajuste. La librería scipy nos proporciona una
#implementación de Box-Cox. Esta función toma un argumento llamada lambda , que
#controla el tipo de transformación a realizar.
#
#lambda = -1, transformación recíproca
#
#lambda = -0.5, es una transformación raíz cuadrada recíproca
#
#lambda = 0, transformación logarítmica
#
#lambda = 0.5, transformación raíz cuadrada
#
#lambda = 1, no es transformable
#
################################################################################

#Aplicamos Box-Cox
from scipy.stats import boxcox

series = pd.read_csv('airline-passengers.csv', header = 0, index_col = 0,
                     parse_dates = True, squeeze = True)

series_transform = boxcox(series, lmbda = 0.0)
plt.figure(1)
#line plot
plt.subplot(211)
plt.plot(series_transform)
#histograma
plt.subplot(212)
plt.hist(series_transform)
plt.show()

#Este código aplica una transformación logarítmica.

#Podemos poner el valor de lambda sin valor y este nos calculará de forma
#automática el mejor valor posible y nos devolverá nuestra serie transformada

series_transform, lam = boxcox(series)
print('Lambda: %f' % lam)

plt.figure(1)
#line plot
plt.subplot(211)
plt.plot(series_transform)
#histograma
plt.subplot(212)
plt.hist(series_transform)








