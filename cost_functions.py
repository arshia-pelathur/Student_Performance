import numpy as np

def compute_cost(preds, y):
    '''
    Calculates the cost functions of the multiple linear regression model built
    Mean squared error is calculated and returned here.
    
    Args:
    preds - predictions of the model
    y - actual target value

    Returns:
    cost function
    '''
    m = len(y)
    cost = np.sum((preds - y)**2)/ (2*m)
    return cost