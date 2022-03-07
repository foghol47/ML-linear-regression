import tensorflow as tf

class Trainer:
    
    def __init__(self, X: tf.Tensor, Y: tf.Tensor, initial_theta: tf.Variable):
        self.X = X
        self.Y = Y
        self.theta = initial_theta

        
    def train(self, iterate, alpha):
        m = self.X.shape[0]
        for i in range(iterate):
            dervative = tf.matmul(tf.transpose(self.X), self.theta)
            self.theta.assign_sub(alpha * (1/m) * dervative)
            
            
    def __cost_function(self):
        m = self.X.shape[0]
        h_theta = self.__h_theta()
        cost = tf.reduce_sum(((h_theta - self.Y) ** 2))
        return (cost / (2*m))
    
    def __h_theta(self):
        return tf.matmul(self.X, self.theta)