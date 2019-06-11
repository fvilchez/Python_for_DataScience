import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import input_data


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
#Carga y tratamiento de los datos
##(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()


#Pasamos a 4D para poder trabajar de forma adecuada
####x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
####x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
####input_shape = (28, 28, 1)
####
####
####x_train = x_train.astype('float32')
####x_test = x_test.astype('float32')
####
####x_train /= 255
####x_test /= 255


######################################################################
#                                                                    #
#                   Generación del Grafo                             #
#                                                                    #
######################################################################



#Generamos dos variables que contengan los pesos W y los sesgos b del
#modelo. Para ello inicializamos con un tensor constante inicializado
# a su vez a ceros.

W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))


#Vamos a crear un tensor de 2 dimensiones para tener la información de
#de los puntos x. Este tensor se usará para guardar imágenes MNIST en
#forma de vector (con el valor de None indicamos que la dimensión puede
#ser de cualquier tamaño; en nuestro caso será igual al número de elementos
#que incluyamos en el proceso de aprendizaje)

x = tf.placeholder('float', [None, 784])



#Una vez tenemos definidos los tensores, podemos implementar nuestro modelo.
#Para ello, TensorFlow provee varias operaciones, siendo una de ellas tf.nn.softmax(logits,name = None),
#que implementan la función softmax.

y = tf.nn.softmax(tf.matmul(x,W) + b)


#Implementamos la función de coste

y_ = tf.placeholder('float', [None, 10])
cross_entropy = -tf.reduce_sum(y_*tf.log(y))


#Implementamos el algoritmo backpropagation

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

######################################################################
#                                                                    #
#                   Ejecución del grafo                              #
#                                                                    #
######################################################################


sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(100):
    print(i)
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
