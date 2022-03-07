import tensorflow as tf
from dataset_util import DatasetUtil
from plot_util import PlotUtil
from trainer import Trainer

def main():
    m = 150
    n = 1
    dataset = DatasetUtil()
    X, Y = dataset.create_dataset(m)
    
    plot = PlotUtil()
    plot.plot_data(X, Y)
    
    X = tf.constant(tf.concat([tf.ones([m, n], dtype='float32'), tf.constant(X, shape=[m,n])], axis=1))
    Y = tf.constant(tf.constant(Y, shape=[m,1]))
    
    theta = tf.Variable((tf.random.uniform((n+1, 1))))
    
    trainer = Trainer(X, Y, theta)
    


if __name__ == '__main__':
    main()
    