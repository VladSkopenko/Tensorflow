import tensorflow as tf


class DenseLayer(tf.Module):
    def __init__(self, in_features, out_features, name=None):
        super().__init__(name=name)
        self.w = tf.Variable(
            tf.random.normal([in_features, out_features]), name="w"
        )
        self.b = tf.Variable(tf.zeros([out_features]), name="b")

    def __call__(self, x):
        y = tf.matmul(x, self.w) + self.b
        return tf.nn.relu(y)


class NN(tf.Module):
  def __init__(self, name=None):
    super().__init__(name=name)
    self.layer_1 = DenseLayer(in_features=3, out_features=3)
    self.layer_2 = DenseLayer(in_features=3, out_features=1)

  def __call__(self, x):
    x = self.layer_1(x)
    return self.layer_2(x)

  


nn = NN(name="neural_network")
print("Results:", nn(tf.constant([[2.0, 2.0, 2.0]])))