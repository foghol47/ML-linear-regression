import matplotlib.pyplot as plt
from dataset_util import DatasetUtil
import tensorflow as tf
import time

class PlotUtil():
    __instance = None

    def __init__(self):
        if PlotUtil.__instance != None:
            raise Exception('This class is a Singleton!')
        self.fig , self.ax = plt.subplots(1, 2, figsize=[10, 4.8]) 
        self.ax[0].set(xlim=(-120, 120), ylim=(-250, 250))
        
        self.ax[1].set_xlabel('iterations')
        self.ax[1].set_ylabel('cost')
        
        self.lines = None    
        self.fig.show()
        PlotUtil.__instance = self
    
    @staticmethod    
    def get_instance():
        if PlotUtil.__instance == None:
            PlotUtil.__instance = PlotUtil()
        return PlotUtil.__instance    
    
    def plot_data(self, X: list, Y: list):
        self.ax[0].scatter(X, Y, s=9,  c='k')
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def plot_line(self, theta: list, normalized: bool):
        X = [-100, 100]
        if normalized:
            Y = [theta[0] + (theta[1] * x) for x in DatasetUtil.normalize_data(X)]
        else:
            Y = [theta[0] + (theta[1] * x) for x in X]
        if self.lines == None:
            self.lines = self.ax[0].plot(X, Y, c='r')
        self.lines[0].set_ydata(Y)   
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        
    def plot_cost(self, x: list, costs: list):
        
        self.ax[1].plot(x, costs, c='b')
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        