#################### Validando modelos de series temporales ####################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit

########## Train-Test Split
series = pd.read_csv('sunspots.csv', header = 0, index_col = 0,
                     parse_dates = True, squeeze = True)

X = series.values

train_size = int(len(X) * 0.66)
train,test = X[0:train_size], X[train_size:len(X)]

print('Observations: %d' % (len(X)))
print('Training Observations: %d' % (len(train)))
print('Testing Observations: %d' % (len(test)))

plt.plot(train)
plt.plot([None for i in train] + [x for x in test])
plt.show()


########## Multiple Train-Test Split

X = series.values
splits = TimeSeriesSplit(n_splits = 3)

plt.figure(1)
index = 1

for train_index, test_index in splits.split(X):

    train = X[train_index]
    test = X[test_index]

    print('Observations: %d' % (len(train) + len(test)))
    print('Training Observations: %d' % (len(train)))
    print('Testing Observations: %d' % (len(test)))

    plt.subplot(310 + index)
    plt.plot(train)
    plt.plot([None for i in train] + [x for x in test])
    index += 1

plt.show()


############# Walk Forward Validation

X = series.values

#FIjamos el número  mínimo de puntos para entrenar el modelo
n_train = 500
n_records = len(X)
for i in range(n_train, n_records):
    train, test = X[0:i], X[i:i+1]
    print('train=%d, test=%d' % (len(train), len(test)))
