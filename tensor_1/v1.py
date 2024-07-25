import tensorflow as tf


def formula(x, y, b):
    x = tf.matmul(x, y)
    x = x + b
    return x


function_that_uses_a_graph = tf.function(formula)

x1 = tf.constant([[1.0, 2.0]])
y1 = tf.constant([[2.0], [3.0]])
b1 = tf.constant(4.0)

orig_value = formula(x1, y1, b1).numpy()
tf_function_value = function_that_uses_a_graph(x1, y1, b1).numpy()

assert(orig_value == tf_function_value)

print(x1.shape)