#Carga del conjunto de datos

import pandas as pd


series = pd.read_csv('daily-total-female-births.csv', header = 0,
                     parse_dates = True, index_col = 0, squeeze = True)


####################################################################
#
# header = 0: indicamos que la primera fila de nuestro conjunto de datos es usada como cabecera
#
# parse_dates = True: le indicamos que la primera columna de nuestro conjunto de datos es de tipo fecha y debe ser parseada como ta.
#
# index_col = 0: le indicamos que nuestra serie debe ser indexada por la primera columna.
#
# squeeze = True: le indicamos que solo queremos trabajar con una columna de datos, es decir, esto nos retorna un objeto de tipo Serires
#
#######################################################################


#Visualizamos las primeras observaciones de nuestra serie
print(series.head(5))

#Vemos el número de elementos que tiene nuestra serie
print(series.size)

#Seleccionamos determinados espaciones temporales de nuestra serie
print(series['1959-01'])


#Resumen estadístico. La funciçón describe() nos permite obtener un resumen
#estadístico de nuestra serie, nos retorna infor como: media, mediana, valor mínimo,
#valor máximo, desviación estándar, primer, segundo y tercer cuartil, etc.

print(series.describe())
