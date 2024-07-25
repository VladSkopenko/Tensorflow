import tensorflow as tf


class SimpleModule(tf.Module):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.w = tf.Variable(5.0)
        self.b = tf.Variable(5.0)

    def __call__(self, x):
        return self.w * x + self.b


simple_module = SimpleModule(name="simple")
simple_module(tf.constant(5.0))
