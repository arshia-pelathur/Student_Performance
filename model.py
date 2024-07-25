import numpy as np
from utils import compute_weights,predict

class MultipleLinearRegression:
    def __init__(self):
        self.__weights = None
    
    def fit(self,X,y):
        self.__weights = compute_weights(X,y)

    def get_weights(self):
        return self.__weights

    def predict(self,X):
        return predict(X,self.__weights)