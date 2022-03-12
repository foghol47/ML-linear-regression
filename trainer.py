import tensorflow as tf

class Trainer:
    
    def __init__(self, X: tf.Tensor, Y: tf.Tensor, initial_theta: tf.Variable):
        self.X = X
        self.Y = Y
        self.theta = initial_theta

    @staticmethod
    def normal_equation(X, Y):
        return tf.matmul(tf.matmul(tf.linalg.inv(tf.matmul(tf.transpose(X), X)), tf.transpose(X)), Y)
        
    def train(self, iterate: int, alpha: float):
        m = self.X.shape[0]
        for i in range(iterate):
            h_theta = Trainer.h_theta(self.X, self.theta)
            derivative = (1/m) * tf.matmul(tf.transpose(self.X), (h_theta - self.Y))
            self.theta.assign_sub(alpha * derivative)
            
            yield self.theta
    
    @staticmethod    
    def cost_function(X: tf.Tensor, Y: tf.Tensor, theta: tf.Tensor):
        m = X.shape[0]
        h_theta = Trainer.h_theta(X, theta)
        cost = tf.reduce_sum((h_theta - Y) ** 2)
        return (cost / (2*m))
    
    @staticmethod
    def h_theta(X: tf.Tensor, theta: tf.Tensor):
        return tf.matmul(X, theta)
        