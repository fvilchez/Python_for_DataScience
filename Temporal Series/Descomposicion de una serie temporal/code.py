####################### Descomposición de Series Temporales ####################


#Python dispone de la función seasonal_decompose que nos permite descomponer
#nuestra serie en sus respectivos componentes.

from statsmodels import tsa
import pandas as pd

#Realizamos una descomposición aditiva
from random import randrange
import matplotlib.pyplot as plt

series = [i + randrange(10) for i in range(1,100)]
result = tsa.seasonal.seasonal_decompose(series, model = 'additive', freq = 1)
result.plot()
plt.show()

#Realizamos una descomposición multiplicativa

series = [i**2 + randrange(10) for i in range(1,100)]
result = tsa.seasonal.seasonal_decompose(series, model = 'additive', freq = 1)
result.plot()
plt.show()


#Ejemplo con un dataset real
series = pd.read_csv('airline-passengers.csv', header = 0, index_col = 0,
                     squeeze = True)
result = tsa.seasonal.seasonal_decompose(series, model = 'multiplicative')
result.plot()
plt.show()
