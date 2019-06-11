import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf




#############################################################################################
#                                                                                           #
#                           Generación del Conjunto de datos                                #
#                                                                                           #
#############################################################################################

#Generamos el conjunto de datos a partir del cual vamos a proceder a reelizar
#el ejemplo.

num_puntos = 2000
conjunto_puntos = []
for i in range(num_puntos):
    if np.random.random() > 0.5:
        conjunto_puntos.append([np.random.normal(0.0, 0.9),
                                np.random.normal(0.0,0.9)])
    else:
        conjunto_puntos.append([np.random.normal(3.0,0.5),
                                np.random.normal(1.0,0.5)])


#Visualizamos el resultado
df = pd.DataFrame({'x' : [v[0] for v in conjunto_puntos],
                   'y' : [v[1] for v in conjunto_puntos]})

sns.lmplot('x', 'y', data = df, fit_reg = False, size = 6)
plt.show()

#############################################################################################
#                                                                                           #
#                           Generación del Grafo                                            #
#                                                                                           #
#############################################################################################


#Agrupamos los grupos anteriores en 4 grupos, es decir, lanzamos un K-means
#con k = 4.

#En primer lugar pasamos todos nuestros datos a estructuras de datos TensorFlow.
vectors = tf.constant(conjunto_puntos)

#Seleccionamos de forma aleatoria los k centroides.
k = 4
centroides = tf.Variable(tf.slice(tf.random_shuffle(vectors), [0,0], [k,-1]))

#Estos k puntos se guardan en un tensor 2D, para ver esto podemos hacer uso de
#de la operación tf.Tensor.get_shape().
print(vectors.get_shape())
print(centroides.get_shape())

#Tras esto debemos de calcular para cada punto su centroide más cercano, para esto
#vamos hacer uso de la distancia euclidea al cuadrado. Pero si intentamos calcular
#esta distancia sin hacer un paso previo, aparece el problema de que los tensores,
#a pesar de ser 2D, tienen diferentes tamaños en una de las dimensiones.
expanded_vectors = tf.expand_dims(vectors, 0)
expanded_centroides = tf.expand_dims(centroides, 1)

#Si ahora printeamos las dimensiones podemos ver que hay dimensiones en cada caso,
#en las que de momento no se han podido determinar los tamaños.
print(expanded_vectors.get_shape())
print(expanded_centroides.get_shape())


#TensorFlow internamente permite hacer broadcasting, y por lo tanto la función
#tf.sub es capaz de descubrir por sí misma la manera de hacer la sustración de
#elementos entre los dos tensores.
diff = tf.subtract(expanded_vectors, expanded_centroides)
sqr = tf.square(diff)
distances = tf.reduce_sum(sqr, 2)
print(distances.get_shape())
assigments = tf.argmin(distances, 0)
#assignments = tf.argmin(tf.reduce_sum(tf.square(tf.sub(expanded_vectors, expanded_centroides))2,),0)


#Cálculo de los nuevos centroides
means = tf.concat([tf.reduce_mean(tf.gather(vectors, tf.reshape(tf.where(tf.equal(assigments,c)),[1,-1])), reduction_indices = [1]) for c in range(k)], 0)

###tf.equal: obtenemos un tensor booleano(Dimension(2000)) que indica con el valor de True, las posiciones
###donde el valor del tensor assigments coíncide con el cluster.
##
###tf.where: se construye un tensor (Dimension(1) x Dimension(2000)) con la posición donde se encuentran los
###valores True en el tensor booleano recibido como parámetro.
##
###tf.reshape: se construye un tensor (Dimension(2000) x Dimension(1)) con los índices de los puntos en el tensor
###vectors que pertenecen al cluster c.
##
###tf.gather: se construye un tensor (Dimension(1) x Dimension(2000) x Dimension(2)) que reúne las coordenadas de
###los puntos que forman el cluster c.
##
###tf.reduce_mean: se construye un tensor (Dimension(1) x Dimension(2)) que contiene el valor medio de todos los
###puntos que pertenecen al cluster c.


#############################################################################################
#                                                                                           #
#                                 Ejecución del grafo                                       #
#                                                                                           #
#############################################################################################


#Finalmente actualizamos al nueva valor de los centroides
update_centroides = tf.assign(centroides, means)

#Creamos un operador que inicializa las variables
init_op = tf.initialize_all_variables()

#Procedemos a ejecutar el grafo
sess = tf.Session()
sess.run(init_op)

for repeticiones in range(100):
    _,centroid_values, assignment_values = sess.run([update_centroides, centroides, assigments])

print(centroid_values)

#############################################################################################
#                                                                                           #
#                                Visualización de resultados                                #
#                                                                                           #
#############################################################################################

data = {"x": [], "y": [], "cluster": []}
for i in range(len(assignment_values)):
  data["x"].append(conjunto_puntos[i][0])
  data["y"].append(conjunto_puntos[i][1])
  data["cluster"].append(assignment_values[i])
df = pd.DataFrame(data)
sns.lmplot("x", "y", data=df, 
           fit_reg=False, size=7, 
           hue="cluster", legend=False)
plt.show()

