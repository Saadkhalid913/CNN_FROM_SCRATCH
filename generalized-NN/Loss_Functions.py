import numpy  as np 

def MSELoss(y_pred, y_truth, derivative = False):
        if derivative:
            return -1 * (y_truth - y_pred)
        return np.power((y_truth - y_pred), 2) / 2

