############################# White Noise ##################################
#
#A continuación vamos a proceder a crear una serie gaussiana con media cero y
#desviación 1, con el objetivo de ver las herramientas más útiles a la hora
#de comprobar si una serie temporal es o no ruido blanco.
#
#############################################################################

from random import gauss, seed
import pandas as pd
from pandas.plotting import autocorrelation_plot
import matplotlib.pyplot as plt

#Fijamos semilla
seed(1)

#Creamos nuestra serie de ruido blanco
series = [gauss(0,1) for i in range(1000)]
series = pd.Series(series)

#Vemos el resumen estadístico de nuestra serie
print(series.describe())

#Veamos nuestro gráfico de linea
series.plot()
plt.show()

#Creamos el histograma para confirmar que nuestra serie tiene una distribución
#gaussiana
series.hist()
plt.show()

#Finalmente chequeamos nuestro gráfico de autocorrelación para ver la correlación
#entre el instante t y el t-n
autocorrelation_plot(series)
plt.show()
