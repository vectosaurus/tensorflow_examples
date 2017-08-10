from __future__ import print_function
import tensorflow as tf

a = tf.constant(2)
b = tf.constant(3)
c = tf.multiply(a,b)

with tf.Session() as sess:
    print(sess.run(a))
    print(sess.run(b))
    print(sess.run(c))
