import matplotlib.pyplot as plt

class PlotUtil():

    @staticmethod
    def plot_data(X, Y):
        plt.scatter(X, Y)
        plt.xlabel('x - axis')
        plt.ylabel('y - axis')
        plt.show()
    
    @staticmethod
    def plot_line(theta):
        pass
    