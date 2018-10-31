############## Estudio de la estacionariedad en series temporales ##############

import pandas as pd
import matplotlib.pyplot as plt

#################### Ejemplo de serie temporal estacionaria
series_stationary = pd.read_csv('daily-total-female-births.csv', header = 0, index_col = 0,
                     parse_dates = True, squeeze = True)

series_stationary.plot()
plt.show()


###################### Ejemplo de serie temporal no estacionaria
series_nonstationary = pd.read_csv('airline-passengers.csv', header = 0, index_col = 0,
                     parse_dates = True, squeeze = True)

series_nonstationary.plot()
plt.show()



####################### Chequeando si una serie es o no estacionaria
#Una forma de comprobar si una serie temporal es o no estacionaria es
#visualizando estadísticos en una particion (dos o más) de nuestra serie
#y comprobar si se produce un cambio en valores como la media, la varianza etc.


###################### Serie estacionaria
X = series_stationary.values
split = int(len(X) / 2)

X1, X2 = X[0:split], X[split:]

mean1, mean2, var1, var2 = X1.mean(), X2.mean(), X1.var(), X2.var()

print('mean1 = %f, mean2 = %f' % (mean1, mean2))
print('variance1 = %f, variance2 = %f' % (var1,var2))

####################### Serie no estacionaria
X = series_nonstationary.values
split = int(len(X)/2)

X1, X2 = X[0:split], X[split:]

mean1, mean2, var1, var2 = X1.mean(), X2.mean(), X1.var(), X2.var()

print('mean1 = %f, mean2 = %f' % (mean1,mean2))
print('var1 = %f, var2 = %f' % (var1,var2))


####################### Serie no estacionaria con transformación log
X = series_nonstationary.values
X = log(X)
plt.plot(X)
plt.show()


######################### Test de Dickey Fuller serie estacionaria

from statsmodels.tsa.stattools import adfuller

X = series_stationary.values
result = adfuller(X)

print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key,value in result[4].items():
    print('\t%s: %.3f' % (key, value))

########################### Test de Dickey Fuller serie no estacionaria

X = series_nonstationary.values
result = adfuller(X)

print('ADF Statistic: %f' % result[0])
print('p.value: %f', % result[1])
print('Critical Values:')
for key,value in result[4].items():
    print('\t%s: %.3f' % (key,value))

