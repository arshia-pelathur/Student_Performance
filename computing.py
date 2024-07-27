import numpy as np


def gradient(X, y, w, b):
    '''
    computes gradient of the cost function wrt w and b for multiple linear regression

    Args:
    X : (m,n) Input dataset
    y : (m,) target
    w : weights of features (n,)
    b = model parameter

    Returns:
    djdw : partial derivative of the cost function wrt weights
    djdb : partial derivative of the cost function wrt b
    '''

    m = X.shape[0]   # number of rows to iterate over to sum up the derivative
    n = X.shape[1]-1   # number of features to get the partial derivative for each row
    djdw = np.zeros((n,))
    djdb = 0
    for row in range(m):
        f_wb = np.dot(X[row,:-1] , w) + b   # prediction = w1*x1 + w2*x2 + .......... + b
        error = f_wb - y[row]           # subtracting the predicted value and actual value
        for feature in range(n):
            djdw[feature] = djdw[feature] + error * X[row,feature]
        djdb += error

    djdw = djdw / m
    djdb = djdb / m
    return djdw,djdb

def cost_function(X, y, w, b):
    '''
    Computes the cost function

    Args:
    x : Data having a shape of m(size of training data)
    y : target values
    w,b : model parameters

    Returns:
    total_cost : total cost function for a certain w and b value
    '''
    m = X.shape[0]
    cost = 0
    for row in range(m):
        cost += (np.dot(X[row,:-1],w) + b - y[row])**2
    cost /=(2*m)
    return cost