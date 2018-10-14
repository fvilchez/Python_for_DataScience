#### Cargamos las librerías
import pandas as pd
import matplotlib.pyplot as plt

############################ Line plots ####################################
#
#Se trata de la forma más popular a la hora de visualizar nuestra serie, y es
#uno de los los primeros gráficos que deberíamos de representar. Este tipo de
#gráfico nos permite ver la evolución de nuestra serie a lo largo del tiempo.
#
#############################################################################

series = pd.read_csv('daily-minimum-temperatures.csv', header = 0,
                     index_col = 0, parse_dates = True, squeeze = True)

series.plot()
plt.show()


#En ciertas ocasiones los gráficos de tipo line plot pueden ser densos. En
#determinadas ocasiones cambiar el estilo de nuestro line plot, puede ser de
#gran ayuda.

series.plot(style = 'k.')
plt.show()

#Puede ser de gran utilidad comparar gráficos de línea en un mismo intervalo,
#tales como diarios, mensuales, anuales etc.

series_by_year = series.groupby(pd.Grouper(freq = 'A'))
years = pd.DataFrame()
for name, group in series_by_year:
    years[name.year] = group.values


years.plot(subplots = True, legend = False)
plt.show()


######################### Histogram and Density Plots #########################
#
#Este tipo de gráficos nos van a permitir conocer la distribución de nuestros
#conjuntos de datos. Esto es bastante importante, ya que algunos métodos lineales
#de predicción de series temporales asumen que los datos siguen algún tipo de
#distribución.
#
###############################################################################

#Nos creamos el histograma de nuestra serie
series.hist()
plt.show()

#Nos creamos la función de densidad de nuestra serie
series.plot(kind = 'kde')
plt.show()



########################## Box and Whisker Plots #############################
#
#Este tipo de gráficos pueden ser útiles para visualizar la distribución de
#nuestra serie por espacios temporales. Nos muestra información de los percentiles
#de la mediana, outliers etc.
#
###############################################################################

groups = series.groupby(pd.Grouper(freq = 'A'))
years = pd.DataFrame()
for name,group in groups:
    years[name.year] = group.values

years.boxplot()
plt.show()

#Podríamos estar interesados en ver como se distribuyen nuestros datos por meses
#para un año en concreto

year_1990 = series['1990']
groups = year_1990.groupby(pd.Grouper(freq = 'M'))
months = pd.concat([pd.DataFrame(x[1].values) for x in groups], axis = 1)
months = pd.DataFrame(months)
months.columns = range(1,13)
months.boxplot()
plt.show()


############################## Heat Maps #####################################
#
#Este tipo de gráficos nos permiten comparar datos de una forma sencilla de
#comprender por gente no experta y muy vistosa.
#
##############################################################################

#Representamos la temperatura a nivel anual para cada año, cada row es un año
# y cada columna es un día.
groups = series.groupby(pd.Grouper(freq = 'A'))
years = pd.DataFrame()
for name, group in groups:
    years[name.year] = group.values

years = years.T
plt.matshow(years, interpolation = None, aspect = 'auto')
plt.show()

#También podríamos representar para un año dado la temperatura a nivel de día y
#por cada mes

one_year = series['1990']
groups = one_year.groupby(pd.Grouper(freq = 'M'))
months = pd.concat([pd.DataFrame(x[1].values) for x in groups], axis=1)
months = pd.DataFrame(months)
months.columns = range(1,13)
plt.matshow(months, interpolation = None, aspect = 'auto')
plt.show()


################################### Lag plots #################################
#
#A la hora de modelizar series temporales se asume que existe una relacción entre
#una observación y su lag. Las observaciones previas son conocidas como lags.
#Un gráfico muy útil a la hora de explorar la relación existente entre una
#observación y su lag es el scatter plot. Pandas dispone de un tipo de gráfico
#denominado lag_plot() que por defecto muestra la relacción entre la observación
#en el instante t y el instante t+1.
#
#Si los puntos están concentrados en una línea desde la parte inferior izquierda
#a la parte superior derecha indica que existe una correlación positiva
#
#Si los puntos están concentrados en una línea desde la parte superior izquierda
#a la parte inferior derecha indica que existe una correlación negativa.
#
#Ambas relacciones son adecuadas ya que estas puenden ser modeladas.
#
#Una fuerte concentración de puntos en la línea indican una relación fuerte
#mientras que si están alejados de la línea sugieren una relación debil.
#
#Una pelota en el medio de una propagación sugiere una relación débil o nula.
#
################################################################################

#Nos creamos nuestro gráfico tipo lag, por defecto tomar lag = 1
pd.plotting.lag_plot(series)
plt.show()

#Podemos agregar tantos plots o lags como queramos
values = pd.DataFrame(series.values)
lags = 7
columns = [values]
for i in range(1, (lags + 1)):
    columns.append(values.shift(i))

dataframe = pd.concat(columns, axis = 1)
columns = ['t']
for i in range(1, (lags + 1)):
    columns.append('t-' + str(i))

dataframe.columns = columns

plt.figure(1)
for i in range(1, (lags + 1)):
    ax = plt.subplot(240 + i)
    ax.set_title('t vs t-' + str(i))
    plt.scatter(x = dataframe['t'].values, y = dataframe['t-'+str(i)].values)

plt.show()


############################ Autocorrelation plot ##############################
#
#Este tipo de gráficos nos ayuda a ver la correlación existente entre una serie
#y sus lags.bit_length Este gráfico nos muestra a lo largo del eje x los lags,
#mientras que a lo largo del eje y nos muestra la correlación. Las líneas de tipo
#punto nos indican que cualquier correlación con valores por encima de estas líneas
#son estadísticamente significativos.
#
################################################################################

pd.plotting.autocorrelation_plot(series)
plt.show()
