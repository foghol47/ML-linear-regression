import tensorflow as tf
from plot_util import PlotUtil
import time

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
            h_theta = self.__h_theta()
            derivative = (1/m) * tf.matmul(tf.transpose(self.X), (h_theta - self.Y))
            self.theta.assign_sub(alpha * derivative)
            plot.plot_line(self.theta.numpy())
            cost = self.__cost_function()
            costs.append(cost)
            tf.print(cost)
            time.sleep(0.01)    
        plot.plot_cost(costs) 
        
    def __cost_function(self):
        m = self.X.shape[0]
        h_theta = self.__h_theta()
        cost = tf.reduce_sum(((h_theta - self.Y) ** 2))
        return (cost / (2*m))
    
    def __h_theta(self):
        return tf.matmul(self.X, self.theta)