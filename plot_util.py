import matplotlib.pyplot as plt

class PlotUtil():
    
    def plot_data(self, X, Y):
        plt.scatter(X, Y)
        plt.xlabel('x - axis')
        plt.ylabel('y - axis')
        plt.show()
    
    