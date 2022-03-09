import random
import tensorflow as tf

class DatasetUtil():

    @staticmethod
    def create_dataset(m: int):
        'create a linear regression dataset near line y = 2x + 50'
        X = []
        Y = []
        for i in range(m):
            x = random.uniform(-100,100)
            mu = (5 * x) + 30
            y = random.gauss(mu, mu / 5)
            X.append(x)
            Y.append(y)
        
        return (X, Y)

    @staticmethod
    def normalize_data(X: tf.Tensor):
        return (X - tf.reduce_min(X, axis=0)) / (tf.reduce_max(X, axis=0) - tf.reduce_min(X, axis=0))
   