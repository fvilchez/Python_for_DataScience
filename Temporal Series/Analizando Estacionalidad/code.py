########################## Analizando la Estacionalidad ########################


################ Eliminando estacionaliad por diferenciación ###################
#
#Si somos capaces de detectar la estacionalidad, esta puede ser eliminada
#simplemente mediante diferenciación. Es decir, si tenemos una estacionalidad
#semanal podemos eliminar la estacionalidad de nuestra serie, restando a un
#día actual, el día de la semana anterior.
#
################################################################################

import pandas as pd
import matplotlib.pyplot as plt

series = pd.read_csv('daily-minimum-temperatures.csv', header = 0, index_col = 0,
                     squeeze = True, parse_dates = True)


X = series.values
diff = [X[i] - X[i - 365] for i in range(365, len(X))]

plt.plot(diff)
plt.show()


#Podemos suponer que la temperatura en un periodo dado del año es probablemente
#estable. Quizás en periódos semanales. Podemos pensar y considerar que todas
#las temperaturas de un determinado mes son estables. Y mejorar  el modelo
#restando la temperatura media para cada uno de los meses del año anterior, en
#lugar del mismo día.

resample = series.resample('M').mean()
resample.plot()
plt.show()


#Si ahora lo que hacemos es realizar una diferenciación mensual
resample = series.resample('M')
monthly_mean = resample.mean()
X = series.values
diff = list()
months_in_year = 12
for i in range(months_in_year, len(monthly_mean)):
    value = monthly_mean[i] - monthly_mean[i - months_in_year]
    diff.append(value)

plt.plot(diff)
plt.show()


#Ahora podemos hacer uso de la media mensual para hacer la correción mensual

X = series.values
diff = list()
days_in_year = 365

for i in range(days_in_year, len(X)):
    month_str = str(series.index[i].year-1)+'-'+str(series.index[i].month)
    month_mean_last_year =  series[month_str].mean()
    value = X[i] - month_mean_last_year
    diff.append(value)

plt.plot(diff)
plt.show()


########## Ajustando la componente estacional mediante su modelización #########

from numpy import polyfit

#Fijamos nuestro polinomio
X = [i%365 for i in range(0, len(series))]
y = series.values
degree = 4
coef = polyfit(X, y, degree)
print('Coefficients: %s' % coef)

curve = list()
for i in range(len(X)):
    value = coef[-1]
    for d in range(degree):
        value += X[i]**(degree -d)*coef[d]
    curve.append(value)

plt.plot(series.values)
plt.plot(curve, color = 'red', linewidth = 3)
plt.show()

#La curva parece es adecuada para fijar la estacionalidad de nuestros datos,
#por lo tanto podemos crear nuestra desestacionalizada mediante:

diff = [series.values[i] - curve[i] for i in range(len(series.values))]
plt.plot(diff)
plt.show()




