import random

class DatasetUtil():

    def create_dataset(self, m: int):
        'create a linear regression dataset near line y = 2x + 3'
        X = []
        Y = []
        for i in range(m):
            x = random.uniform(-100,100)
            mu = (2 * x) + 3
            y = random.gauss(mu, mu / 5)
            X.append(x)
            Y.append(y)
        
        return (X, Y)    