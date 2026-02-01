import numpy as np

def rmse(predictions, actuals):
    return np.sqrt(np.mean((np.array(predictions) - np.array(actuals)) ** 2))

def mae(predictions, actuals):
    return np.mean(np.abs(np.array(predictions) - np.array(actuals)))
