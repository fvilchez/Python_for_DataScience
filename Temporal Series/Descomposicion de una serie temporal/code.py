####################### Descomposición de Series Temporales ####################


#Python dispone de la función seasonal_decompose que nos permite descomponer
#nuestra serie en sus respectivos componentes.

import statsmodels.api as sm
import pandas as pd
from random import randrange
import matplotlib.pyplot as pl

#Realizamos una descomposición aditiva


series = [i + randrange(10) for i in range(1,100)]
result = sm.tsa.seasonal_decompose(series, model = 'additive', freq = 1)
result.plot()
plt.show()

#Realizamos una descomposición multiplicativa

series = [i**2 + randrange(10) for i in range(1,100)]
result = sm.tsa.seasonal_decompose(series, model = 'additive', freq = 1)
result.plot()
plt.show()


#Ejemplo con un dataset real
series = pd.read_csv('airline-passengers.csv', header = 0, index_col = 0,
                     squeeze = True)
result = sm.tsa.seasonal_decompose(series, model = 'multiplicative')
result.plot()
plt.show()
