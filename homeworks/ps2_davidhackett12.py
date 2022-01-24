import numpy as np
import time

def eliminate(A, b):
    i = 0
    while i < len(A):
        j = 1
        while j < len(A) - i:
            pivot = A[i + j][i]/A[i][i]
            A[i + j] = A[i + j] - (A[i] * pivot)
            b[i + j] = b[i + j] - (b[i] * pivot)
            j += 1
        i += 1
    x = np.zeros(len(b))
    for i in range(-1, (len(b) * -1 - 1), -1):
        answer = b[i]
        for j in range(-1, i, -1):
            answer = answer - x[j] * A[i][j]
        x[i] = answer / A[i][i]
    return A, x

def test_eliminate(n):
    W = np.random.randn(n, n)
    b = np.ones(n)

    start = time.time()
    for _ in range(10):
        eliminate(W, b)
    end = time.time()
    eliminate_time  = (end - start)/10

    start = time.time()
    for _ in range(10):
        np.linalg.solve(W, b)
    end = time.time()
    solve_time  = (end - start)/10

    return eliminate_time, solve_time
