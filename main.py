import tensorflow as tf
from dataset_util import DatasetUtil
from plot_util import PlotUtil
from trainer import Trainer
import matplotlib.pyplot as plt

costs = list() 
x_points = list() 

def plot_function(X: tf.Tensor, Y: tf.Tensor, theta: tf.Tensor, iter_number: int):
    plot = PlotUtil.get_instance()
    plot.plot_line(theta.numpy(), True)
    
    cost = Trainer.cost_function(X, Y, theta)
    costs.append(cost)
    x_points.append(iter_number)
    
    if len(costs) == 2:
        plot.plot_cost(x_points, costs)
        del costs[0]
        del x_points[0]
        


def main():
    m = 150
    n = 1
    X, Y = DatasetUtil.create_dataset(m)
    plot = PlotUtil.get_instance()
    plot.plot_data(X, Y)
    
    X, minimum, range_of_data = DatasetUtil.normalize_data(tf.constant(X, shape=[m,n]))
    X = tf.concat([tf.ones([m, 1], dtype='float32'), X], axis=1)
    Y = tf.constant(tf.constant(Y, shape=[m,1]))
    theta = tf.Variable((tf.random.uniform((n+1, 1))))
    trainer = Trainer(X, Y, theta)
    
    iter_number = 1
    for theta in trainer.train(150, 3e-1):
        plot_function(X, Y, theta, iter_number)
        iter_number += 1
    
    x = int(input('enter your x:\n'))
    new_x = DatasetUtil.normalize_with_parameters(x, minimum, range_of_data)
    new_x = tf.concat([tf.ones([1, 1], dtype='float32'), tf.constant(new_x, shape=[1,n])], axis=1)
    predict = trainer.h_theta(new_x, theta)
    print('predict = ', predict.numpy())
    plot.plot_data(x, predict.numpy(), 'g')
    
    plt.show()
        
if __name__ == '__main__':
    main()
    