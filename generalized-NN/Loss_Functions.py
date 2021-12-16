import numpy  as np 

def MSELoss(y_pred, y_truth, derivative = False):
    '''
        Mean Squared Error function with derivative
    '''
    if derivative:
        return -1 * (y_truth - y_pred)
    return np.power((y_truth - y_pred), 2) / 2

def MAELoss(y_pred, y_truth, derivative = False):
    '''
        Mean Absolute Error function with derivative
    '''
    if derivative:
        return np.where(y_pred < y_truth, -1, 1)
    return np.absolute(y_truth - y_pred)