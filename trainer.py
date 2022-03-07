import tensorflow as tf

class Trainer:
    
    def __init__(self, X: tf.Tensor, Y: tf.Tensor, initial_theta: tf.Variable):
        self.X = X
        self.Y = Y
        self.theta = initial_theta

    def __cost_function(self):
        h_theta = self.__h_theta()
        cost = tf.reduce_sum(((h_theta - self.Y) ** 2))
        return cost
    
    def __h_theta(self):
        return tf.matmul(self.X, self.theta)