############################### ACF Y PACF ###################################

########### ACF

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf

series = pd.read_csv('daily-minimum-temperatures.csv', header = 0, index_col = 0,
                     parse_dates = True, squeeze = True)

plot_acf(series, lags = 50)
plt.show()

#Por defecto se muestra un gráfico ACF que tiene en cuenta todos los plots esto
#puede ser modificado a partir del parámetro lags, que nos permite seleccionar
#hasta que lag queremos mostrar.



########## PACF
plot_pacf(series, lags = 50)
plt.show()

