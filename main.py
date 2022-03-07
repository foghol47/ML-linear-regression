import tensorflow as tf
from create_dataset import create_dataset
from plot_util import plot_points

def main():
    m = 150
    X, Y = create_dataset(150)

    plot_points(X, Y)


if __name__ == '__main__':
    main()
    