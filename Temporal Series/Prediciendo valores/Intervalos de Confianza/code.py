############################# Intervalo de confianza ##########################

import pandas as pd
from statsmodels.tsa.arima_model import ARIMA
import matplotlib.pyplot as plt


series = pd.read_csv('daily-total-female-births.csv', header = 0, index_col = 0,
                     parse_dates = True, squeeze = True)

X = series.values
X = X.astype('float32')
size = len(X) - 1
train, test = X[0:size], X[size:]

#Fijamos el modelo
model = ARIMA(train, order = (5,1,1))
model_fit = mode.fit(disp = False)

#Hacemos predicciones
forecast, stderr, conf = model_fit.forecast()

#Vemos los resultados
print('Expected: %.3f' % test[0])
print('Forecast: %.3f' % forecast)
print('Standard Error: %.3f' % stderr)
print('95%% Confidence Interval: %.3f to %.3f' % (conf[0][0], conf[0][1]))


#Podemos el valor del intervalo de confianza de nuestros datos
intervals = [0.2, 0.1, 0.05, 0.01]
for a in intervals:
    forecast, stderr, conf = model_fit.forecast(alpha = a)
    print('%.1f%% Confidence Interval: %.3f between %.3f and %.3f' % ((1-a)*100, forecast, conf[0][0], conf[0][1]))


##### Podemos automáticamente visualizar las prediccines y su intervalo de confianza

model = ARIMA(train, order = (5,1,1))
model_fit = model.fit(disp = False)

model_fit.plot_predict(len(train)-10, len(train)+1)
plt.legend(loc = 'upper left')
plt.show()
