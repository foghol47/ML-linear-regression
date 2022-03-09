import tensorflow as tf
from plot_util import PlotUtil
import matplotlib.pyplot as plt

class Trainer:
    
    def __init__(self, X: tf.Tensor, Y: tf.Tensor, initial_theta: tf.Variable):
        self.X = X
        self.Y = Y
        self.theta = initial_theta
        
    def train(self, iterate, alpha):
        costs = list()
        plot = PlotUtil.get_instance()
        m = self.X.shape[0]
        for i in range(iterate):
            h_theta = Trainer.h_theta(self.X, self.theta)
            derivative = (1/m) * tf.matmul(tf.transpose(self.X), (h_theta - self.Y))
            self.theta.assign_sub(alpha * derivative)
            plot.plot_line(self.theta.numpy())
            cost = Trainer.cost_function(self.X, self.Y, self.theta)
            costs.append(cost)
     
        plot.plot_cost(costs) 
        plt.show()
    
    @staticmethod    
    def cost_function(X, Y, theta):
        m = X.shape[0]
        h_theta = Trainer.h_theta(X, theta)
        cost = tf.reduce_sum(((h_theta - Y) ** 2))
        return (cost / (2*m))
    
    @staticmethod
    def h_theta(X, theta):
        return tf.matmul(X, theta)