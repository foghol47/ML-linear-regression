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
            mu = (3 * x) + 20
            y = random.gauss(mu, mu / 5)
            X.append(x)
            Y.append(y)
        
        return (X, Y)

    @staticmethod
    def normalize_data(X: tf.Tensor):
        maximum = tf.reduce_max(X, axis=0)
        minimum = tf.reduce_min(X, axis=0)
        range_of_data = maximum - minimum
        normalized_X = (X - minimum) / range_of_data
        return (normalized_X , minimum, range_of_data)
        
    @staticmethod    
    def normalize_with_parameters(X: tf.Tensor, minimum: tf.Tensor, range_of_data: tf.Tensor):
        return (X - minimum) / range_of_data
   