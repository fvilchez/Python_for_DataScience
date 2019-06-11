import tensorflow as tf

#Definimos variables simbolicas para poder manipularlas durante la ejecucion
#del programa
a = tf.placeholder("float")
b = tf.placeholder("float")

#Creamos la expresion simbolica que queremos realizar sobre las variables
resultado = tf.multiply(a,b)

#Creamos una sesion para evaluar la expresion simbolica especificada
sess = tf.Session()
print(sess.run(resultado,feed_dict={a:3, b:3}))


      
