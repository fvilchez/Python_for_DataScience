########################## Visualizando los erroes ############################

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.gofplots import qqplot
import numpy as np
from pandas.plotting import autocorrelation_plot

####################### Model de persistencia ############################

series = pd.read_csv('daily-total-female-births.csv', header = 0, index_col = 0,
                    parse_dates = True, squeeze = True)

values = pd.DataFrame(series.values)
dataframe = pd.concat([values.shift(1), values], axis = 1)
dataframe.columns = ['t', 't+1']

## Nos creamos los conjuntos de entrenamiento y de test
X = dataframe.values
train_size = int(len(X) * 0.66)
train, test = X[1:train_size], X[train_size:]
train_X, train_y = train[:,0], train[:,1]
test_X, test_y = test[:,0], test[:,1]

## Aplicamos el modelo de persistencia
predictions = [x for x in test_X]

## Calculamos los errores residuales
residuals = [test_y[i] - predictions[i] for i in range(len(predictions))]
residuals = pd.DataFrame(residuals)

######################### Residual line plot ##################################

#Se espera que el gráfico tipo línea no muestre tendencia o patrones cíclicos.
#Además se espera que tome valores aleatorios alrededor de 0.

##residuals.plot()
##plt.show()


###################### Resumen estadísticos de los errores #####################

#En nuestro resumen estadísticos esperamos que nuestro promedio de errores se
#encuentre cerca de cero, esto significará que no tenemos bias en nuestras
#predicciones. Sin embargo, un valor positivo o negativo puede significar un
#bias positivo o negativo.

##print(residuals.describe())


####################### Histogramas y funciones de densidad ####################

#Otro gráfico que nos puede ayudar a estudiar nuestros errores son los histogramas
#y las funciones de densidad. Se espera que este tipo de gráficos sigan una normal
#centrada entorno a cero. Este tipo de gráficos pueden ayudar a encontrar
#distribuciones con skews.

##residuals.hist()
##plt.show()
##
##residuals.plot(kind = 'kde')
##plt.show()


################################### QQ plot ###################################

#Este tipo de gráfico lo que hace es comparar nuestra distribución con una
#distribución gaussiana ideal y nos retorna lo parecido o lo que difiere de
#dicha distribución ideal. En el eje x se representa los valores de la
#distribución ideal mientras que en el eje y se muestra la distribución de
#nuestro ejemplo.

##qqplot(residuals, line = 'r')
##plt.show()


########################## Gráfico de autocorrelación ##########################

#Cuando graficamos un plot de autocorrelación entre los errores de nuestro
#modelo de serie temporal esperamos que no exista ningún tipo de autocorrelación
#entre dichos errores.

autocorrelation_plot(residuals)
plt.show()

