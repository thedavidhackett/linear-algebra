import numpy as np

def plr(X : np.ndarray, y : np.ndarray, w : np.ndarray,\
    alpha : float, num_iterations : int) -> np.ndarray:
    '''
    Run the perceptron learning rule and return the learned w
    given the data, learning rate alpha, and number of iterations

    Parameters
    ----------
    X : numpy.ndarray
        Training data
    y : numpy.ndarry
        The classification for each data point, 0 or 1
    w : numpy.ndarray
        A starting w
    alpha : float
        The learning rate
    num_iterations : int
        The number of iterations to run

    Returns
    -------
    numpy.ndarray
        The learned w
    '''
    last_idx : int = len(X) - 1
    for _ in range(num_iterations):
        i : int = np.random.randint(0, last_idx)
        prediction : int = int((np.dot(w, X[i]) >= 0))
        error : int = y[i] - prediction
        w = w + (X[i] * alpha * error)

    return w
