####################### Grid Search ARIMA #################################
import pandas as pd
import warnings
import numpy as np
from math import sqrt
from pandas import datetime
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error

warnings.filterwarnings('ignore')

def evaluate_arima_model(X, arima_order):
    #Preparamos nuestro dataset
    train_size = int(len(X) * 0.66)
    train, test = X[0:train_size], X[train_size:]
    history = [x for x in train]
    #Hacemos predicciones
    predictions = list()
    for t in range(len(test)):
        model = ARIMA(history, order = arima_order)
        model_fit = model.fit(disp = 0)
        yhat = model_fit.forecast()[0]
        predictions.append(yhat)
        history.append(test[t])
    #Calculamos el error
    rmse = sqrt(mean_squared_error(test, predictions))
    return rmse


def evaluate_models(dataset, p_values, d_values, q_values):
    dataset = dataset.astype('float32')
    best_score, best_cfg = float('inf'), None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p,d,q)
                try:
                    rmse = evaluate_arima_model(dataset, order)
                    if rmse < best_score:
                        best_score, best_cfg = rmse, order
                    print('ARIMA%s RMSE=%.3f' % (order, rmse))
                except:
                    continue
    print('Best ARIMA%s RMSE=%.3f' % (best_cfg, best_score))


########################## Aplicamos esto sobre shampoo ########################

def parser(x):
    return datetime.strptime('190'+x, '%Y-%m')


series = pd.read_csv('shampoo-sales.csv', header = 0, parse_dates = [0],
                     index_col = 0, squeeze = True, date_parser = parser)

#Elegimos los parámetros a evaluar
p_values = [0,1,2,4,6,8,10]
d_values = range(0,3)
q_values = range(0,3)
evaluate_models(series.values, p_values, d_values, q_values)


    
