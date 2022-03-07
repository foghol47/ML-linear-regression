import random
def create_dataset(m: int):
    'create a linear regression dataset near line y = 2x + 3'
    X =[]
    Y = []
    for i in range(m):
        x = random.uniform(-50,50)
        mu = (2 * x) + 3
        y = random.gauss(mu, mu / 2)
        X.append(x)
        Y.append(y)
    
    return (X, Y)    