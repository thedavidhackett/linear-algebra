from typing import List, Optional, Tuple
import numpy as np

def solve_elimination(A : np.ndarray, b_list : List[np.ndarray]) -> List[np.ndarray]:
    solutions = []
    for b in b_list:
        A = np.copy(A)
        b = np.copy(b)
        A, b = forward_elimination(A, b)
        solutions.append(backward_substitution(A, b))
    return solutions

def solve_inverse(A : np.ndarray, b_list : List[np.ndarray]) -> List[np.ndarray]:
    solutions = []
    A = np.copy(A)
    I : np.ndarray = np.identity(len(A))
    I, Ainv = forward_elimination(A, I)
    I, Ainv = backward_elimination(I, Ainv)
    for i in range(len(I)):
        Ainv[i] = Ainv[i]/I[i][i]

    for b in b_list:
        solutions.append(np.dot(Ainv, b))
    return solutions

def solve_lu(A : np.ndarray, b_list : List[np.ndarray]) -> List[np.ndarray]:
    solutions = []
    A = np.copy(A)
    I : np.ndarray = np.identity(len(A))
    U, L = forward_elimination(A, I)
    I, U = forward_elimination(L, I)
    for b in b_list:
        c = forward_substitution(L, b)
        solutions.append(backward_substitution(U, c))
    return solutions, L, c


def forward_elimination(A : np.ndarray, b : np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[float]]:
    i : int = 0
    while i < len(A):
        j : int = 1
        while j < len(A) - i:
            pivot = A[i + j][i]/A[i][i]
            A[i + j] = A[i + j] - (A[i] * pivot)
            b[i + j] = b[i + j] - (b[i] * pivot)
            j += 1
        i += 1

    return A, b

def backward_elimination(A: np.ndarray, b : np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    i : int = len(A) - 1
    while i > 0:
        j : int = 1
        while j < len(A) - i:
            pivot = A[i - j][i]/A[i][i]
            A[i - j] = A[i - j] - (A[i] * pivot)
            b[i - j] = b[i - j] - (b[i] * pivot)
            j += 1
        i -= 1

    return A, b

def backward_substitution(A : np.ndarray, b : np.ndarray) -> np.ndarray:
    x : np.ndarray = np.zeros(len(b))
    for i in range(-1, (len(b) * -1 - 1), -1):
        answer : float = b[i]
        for j in range(-1, i, -1):
            answer = answer - x[j] * A[i][j]
        x[i] = answer / A[i][i]

    return x

def forward_substitution(A : np.ndarray, b : np.ndarray) -> np.ndarray:
    x : np.ndarray = np.zeros(len(b))
    for i in range(len(b) - 1):
        answer : float = b[i]
        for j in range(i):
            answer = answer - x[j] * A[i][j]
        x[i] = answer / A[i][i]

    return x
