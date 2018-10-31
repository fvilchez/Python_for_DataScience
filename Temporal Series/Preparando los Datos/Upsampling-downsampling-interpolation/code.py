########################### Upsampling and Interpolation########################
#Cuando aplicamos la técnica de upsampling a una serie temporal lo que hacemos
#es pasar a una frecuencia mayor, es decir, expandimos nuestra serie.
#Por ejemplo, podemos tener una serie temporal a nivel horario y podemos querer
#tenerla a nivel de  minutos.
###############################################################################

import pandas as pd
import matplotlib.pyplot as plt

def parser(X):
    return pd.datetime.strptime('190'+X, '%Y-%m')

series = pd.read_csv('shampoo-sales.csv', header = 0, index_col = 0,
                     parse_dates = True, squeeze = True, date_parser = parser)

print(series.head(3))

#Hacemos un upsampling, es decir pasamos de frecuencia mensual a frecuencia
#diaria

upsampled = series.resample('D').mean()
print(upsampled.head())

#Podemos ver como en los nuevos valores aparecen NaN. Debemos de hacer uso de
#la interpolación para corregir esto. Pandas dispone de la función interpolate()
#que nos permite interpolar valores perdidos, a la hora de interpolar podemos
#hacer uso de funciones sencillas o de funciones más complejas. Tener conocimiento
#del dominio nos ayudará a interpolar los datos. Un buen punto de comienzo es
#hacer uso de una interpolación lineal. Esto pinta una línea recta entre los
#puntos disponibles.

interpolated = upsampled.interpolate(method = 'linear')
print(interpolated.head(32))
interpolated.plot()
plt.show()

#Otro método muy común es hacer uso de un polinomio o un spline para conectar
#los valores. Esto crea curvas y parece más natural en muchos datasets. Hacer
#uso de la interpolación spline requiere especificar el grado del polinomio que
#queremos utilizar.

interpolated = upsampled.interpolate(method = 'spline', order = 2)
interpolated.plot()
plt.show()

#Generalmente la interpolación es una herramienta útil cuando tenemos valores
#perdidos.


################################# Downsampling ################################
#
#Cuando hacemos downsamplin lo que hacemos es disminuir la frecuencia de nuestra
#serie, por ejemplo, tenemos una serie diaria y la queremos pasar a mensual.
#La función resample() de Pandas nos permite realizar esto.
#
###############################################################################

#Hacemos un downsampling de nivel mensual a nivel quartil
resample = series.resample('Q')
quarterly_mean_sales = resample.mean()
quarterly_mean_sales.plot()
plt.show()

#Hacemos un dowsampling a nivel anual
resample = series.resample('A')
yearly_sum_sales = resample.sum()
yearly_sum_sales.plot()
plt.show()




