import numpy as np

def compute_weights(X,y):
    '''
    y = X W
    y - target feature values
    X - Matrix of all independent values and a columns of ones added for the intercept term
    W - is a vector of the regression coefficients, or the Weight matrix having weights for all features
    Multiplying X_transpose to both sides and then moving (X* X_transpose) to under y we can calculate W
    '''
    X_transpose = X.T
    weights = np.linalg.inv(X_transpose @ X) @ X_transpose @ y
    return weights

def predict(X,weights):
    '''
    Uses the weights calculated in the above function and multiplies it with the matrix of X
    '''
    return X @ weights