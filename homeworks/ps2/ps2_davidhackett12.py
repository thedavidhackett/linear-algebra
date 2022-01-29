import time
from typing import Tuple
import numpy as np

def eliminate(A : np.ndarray, b : np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Performs gaussian elimination on a linear system

    Parameters
    ----------
    A : numpy.ndarray
        A matrix representing the left side of the system
    b : numpy.ndarray
        A vector representing the right side of the system


    Returns
    -------
    Tuple[np.ndarray, np.ndarrary]
        A tuple with the right side of the system in upper triangular form and
        the solution to that system if any.
    """
    i : int = 0
    while i < len(A):
        j : int = 1
        while j < len(A) - i:
            pivot = A[i + j][i]/A[i][i]
            A[i + j] = A[i + j] - (A[i] * pivot)
            b[i + j] = b[i + j] - (b[i] * pivot)
            j += 1
        i += 1
    x : np.ndarray = np.zeros(len(b))

    for i in range(-1, (len(b) * -1 - 1), -1):
        answer : float = b[i]
        for j in range(-1, i, -1):
            answer = answer - x[j] * A[i][j]
        x[i] = answer / A[i][i]
    return A, x

def test_eliminate(n : int) -> Tuple[float, float]:
    """ Tests the eliminate function vs the linalg.solve function

    Parameters
    ----------
    n : int
        The size of the matrix n x n to be tested

    Returns
    -------
    Tuple[float, float]
        A tuple with two floats the first representing the average time over 10
        runs for eliminate to solve the system and the second the average time
        over 10 runs for linalg.solve to solve the system.
    """
    W : np.ndarray = np.random.randn(n, n)
    b : np.ndarray = np.ones(n)

    start : float  = time.time()
    for _ in range(10):
        eliminate(W, b)
    end : float = time.time()
    eliminate_time : float  = (end - start)/10

    start = time.time()
    for _ in range(10):
        np.linalg.solve(W, b)
    end = time.time()
    solve_time : float  = (end - start)/10

    return eliminate_time, solve_time
