#Cargamos librerías
import pandas as pd

#Cargamos los datos
series = pd.read_csv('daily-minimum-temperatures.csv', header = 0,
                     index_col = 0, parse_dates = True, squeeze = True)

#Vemos nuestra serie
print(series.head(3))


#Podriamos pensar que quizás podría ser útil montarnos 2 nuevas variables que
#fuesen mes y día y su temperatura asociada, es decir,  estas variables podrían
#aportar información a nuestro algoritmo por ejemplo para intentar encontrar
#algún tipo de estacionalidad en nuestro conjunto de datos

dataframe = pd.DataFrame()

#Nos creamos las columnas día mes y temperatura asociada
dataframe['month'] = [series.index[i].month for i in range(len(series))]
dataframe['day'] = [series.index[i].day for i in range(len(series))]
dataframe['temperature'] = [series[i] for i in range(len(series))]

#Vemos el resultado
print(dataframe.head(4))

#Intentar predecir la temperatura solo con la información del día y del mes,
#no tiene pinta de ser una forma muy sofisticada y seguramente obtengamos un
#modelo bastante pobre. Podríamos pensar en las características asociadas a una
#variable de tipo timestamp y cuales nos podrían ayudar. Por ejemplo:
#
# Hora del día
#
# Minutos transcurridos
#
# Hora comercial o no



# La función shift() nos permite agregar instantes temporales con lags.

#Nos creamos un dataframe que contenga los valores de nuestra serie
temps = pd.DataFrame(series.values)

#Concatenamos los valores con lag y sin lag
temps_shift = pd.concat([temps.shift(1), temps], axis = 1)
temps_shift.columns = ['t', 't+1']
print(temps_shift.head())


#Podemos ver como podemos descartar la primera fila de nuestro conjunto de datos
#a la hora de entrenar nuestro modelo ya que no aporta suficience información
#para poder entrenar nuestro modelo.


#Podemos agregar tantos shifts como queramos

temps = pd.DataFrame(series.values)
temps_shift_3 = pd.concat([temps.shift(3), temps.shift(2), temps.shift(1), temps], axis = 1)
temps_shift_3.columns = ['t-2', 't-1', 't', 't+1']
print(temps_shift_3.head())


######################################
#
#La función rolling nos permite por ejemplo calcular la media de los instantes
#anterioes a aquellos que queremos predecir e intentar predecir nuestro resultado
#a partir de estos valores.
#
#####################################

temps = pd.DataFrame(series.values)
temps_shifted = temps.shift(1)
window = temps_shifted.rolling(window = 2)
means = window.mean()
dataframe = pd.concat([means, temps], axis = 1)
dataframe.columns = ['mean(t-1,t)', 't+1']
print(dataframe.head(5))

#El primer valor NaN es creado debido a que no se pudo realizar el shift, el
#segundo NaN fue creado debido a que no se pudo calcular la media. Finalmente
#el tercer valor muestra la media de los instantes anteriores para intentar predecir
#el siguiente valor (18.8).

temps = pd.DataFrame(series.values)
width = 3
shifted = temps.shift(width -1)
window = shifted.rolling(window = width)
dataframe = pd.concat([window.min(), window.max(), window.mean(), temps], axis = 1)
dataframe.columns = ['min', 'max', 'mean', 't+1']
print(dataframe.head())


###########################################
#
#La funcion expanding() se trata de una función que nos permite seleccioanr todos los valores anteriores
#a nuestra serie.

temps = pd.DataFrame(series.values)
window = temps.expanding()
dataframe = pd.concat([window.mean(), window.max(), window.min(), temps.shift(-1)], axis = 1)
dataframe.columns = ['mean', 'min', 'max', 't+1']
print(dataframe.head())




