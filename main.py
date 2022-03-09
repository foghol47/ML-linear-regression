import tensorflow as tf
from dataset_util import DatasetUtil
from plot_util import PlotUtil
from trainer import Trainer
import matplotlib.pyplot as plt

costs = list()
thetas = list()


def add(X: tf.Tensor, Y: tf.Tensor, theta: tf.Tensor):
    thetas.append(theta.numpy())
    
    cost = Trainer.cost_function(X, Y, theta)
    costs.append(cost)


def main():
    m = 150
    n = 1
    X, Y = DatasetUtil.create_dataset(m)
    plot = PlotUtil.get_instance()
    plot.plot_data(X, Y)
    
    X = DatasetUtil.normalize_data(tf.constant(X, shape=[m,n]))
    X = tf.concat([tf.ones([m, 1], dtype='float32'), X], axis=1)
    Y = tf.constant(tf.constant(Y, shape=[m,1]))
    theta = tf.Variable((tf.random.uniform((n+1, 1))))
    trainer = Trainer(X, Y, theta, add)

    trainer.train(150, 3e-1)

    plot.animate_line(thetas, True)
    plot.plot_cost(costs)
    plt.show()
        
if __name__ == '__main__':
    main()
    