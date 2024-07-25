import tensorflow as tf


def f(x):
    return 1 / x ** 2


x = tf.Variable(2.0)
with tf.GradientTape() as tape:
    y = f(x)
    dydx = tape.gradient(y, x)
    print(dydx)