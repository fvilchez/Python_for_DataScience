import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

#Vamos a generar un modelo sencillo donde suponenmos que nuestro
#modelo de datos corresponde a una regresion lineal simple:
#y = W*x + b.

#############################################################################################
#                                                                                           #
#                           Generación del Conjunto de datos                                #
#                                                                                           #
#############################################################################################

num_puntos = 1000
conjunto_puntos = []

x_train = [np.random.normal(0.0,0.55) for i in range(num_puntos)]
y_data = [x*0.1 + 0.3 + np.random.normal(0.0,0.03) for x in x_train]

#Visualizamos
plt.plot(x_train, y_data, 'ro')
plt.legend()
plt.show()

#############################################################################################
#                                                                                           #
#                           Generación del Grafo                                            #
#                                                                                           #
#############################################################################################

#Generamos tres variables
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))
y = W*x_train + b

#Generamos la función de coste
loss  = tf.reduce_mean(tf.square(y - y_data))

#Generamos el gradiente descendiente
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)


#############################################################################################
#                                                                                           #
#                  py               Ejecución del grafo                                       #
#                                                                                           #
#############################################################################################

#Creamos una sesión y inicializamos variables
init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

#Generamos el proceso iterativo
for step in range(15):
    sess.run(train)

print(step, sess.run(W), sess.run(b))


#############################################################################################
#                                                                                           #
#                                Visualización de resultados                                #
#                                                                                           #
#############################################################################################

#Vemos los resultados
plt.plot(x_train, y_data, 'ro')
plt.plot(x_train, sess.run(W)*x_train + sess.run(b))
plt.legend()
plt.show()

