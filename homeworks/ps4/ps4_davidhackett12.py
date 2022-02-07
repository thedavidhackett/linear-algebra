from typing import List, Optional, Tuple
import numpy as np

def solve_elimination(A : np.ndarray, b_list : List[np.ndarray]) -> List[np.ndarray]:
    """Solves a linear system Ax = b for a single A nxn matrix and k b vectors
    using elimination.

    Parameters
    ----------
    A : numpy.ndarray
        A nxn matrix, the left side of the equation
    b_list : List[np.ndarray]
        A list of vectors of length n, the right side of the equation

    Returns
    -------
    List[np.ndarray]
        A list of solutions for each b value
    """
    solutions = []
    for b in b_list:
        A_copy = np.copy(A)
        b = np.copy(b)
        A_copy, b = forward_elimination(A_copy, b)
        solutions.append(backward_substitution(A_copy, b))
    return solutions

def solve_inverse(A : np.ndarray, b_list : List[np.ndarray]) -> List[np.ndarray]:
    """Solves a linear system Ax = b for a single A nxn matrix and k b vectors
    by finding the inverse of A and multiplying that by b.

    Parameters
    ----------
    A : numpy.ndarray
        A nxn matrix, the left side of the equation
    b_list : List[np.ndarray]
        A list of vectors of length n, the right side of the equation

    Returns
    -------
    List[np.ndarray]
        A list of solutions for each b value
    """
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
    """Solves a linear system Ax = b for a single A nxn matrix and k b vectors
    using LU Factorization.

    Parameters
    ----------
    A : numpy.ndarray
        A nxn matrix, the left side of the equation
    b_list : List[np.ndarray]
        A list of vectors of length n, the right side of the equation

    Returns
    -------
    List[np.ndarray]
        A list of solutions for each b value
    """
    solutions = []
    A = np.copy(A)
    U, L = forward_elimination_factorize(A)
    for b in b_list:
        c = forward_substitution(L, b)
        solutions.append(backward_substitution(U, c))
    return solutions


def forward_elimination(A : np.ndarray, b : np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[float]]:
    """Performs forward elimination on a matrix while modifying a vector or
    matrix with the row actions.

    Parameters
    ----------
    A : numpy.ndarray
        A nxn matrix to perform forward elimination on
    b : numpy.ndarray
        A n length vector or nxn matrix. For elimination this should be the b
        value in Ax = b, for solving by inverse this can perform the first half
        of the process [A I] = [I Ainv] with the result being passed into backward
        elimination to finish the process.

    Returns
    -------
    Tuple[numpy.ndarray, numpy.ndarray]
        A tuple containing the A in upper triangular form and a matrix or
        vector that received the same row actions
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

    return A, b

def forward_elimination_factorize(A : np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[float]]:
    """Performs forward elimination on a matrix to produce L and U for LU
    factorization

    Parameters
    ----------
    A : numpy.ndarray
        A nxn matrix to perform forward elimination on
    Returns
    -------
    Tuple[numpy.ndarray, numpy.ndarray]
        A tuple containing U (A in upper triangular form) and L the inverse of
        the elimination matrix to use in LU factorization
    """
    i : int = 0
    L = np.identity(len(A))
    while i < len(A):
        j : int = 1
        while j < len(A) - i:
            pivot = A[i + j][i]/A[i][i]
            A[i + j] = A[i + j] - (A[i] * pivot)
            L[i + j][i] = pivot
            j += 1
        i += 1

    return A, L

def backward_elimination(A: np.ndarray, b : np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Performs backward elimination on a matrix while modifying a vector or
    matrix with the row actions.

    Parameters
    ----------
    A : numpy.ndarray
        A nxn matrix to perform backward elimination on
    b : numpy.ndarray
        A n length vector or nxn matrix. For solving by inverse this can
        perform the second half of the process [A I] = [I Ainv] using the
        result from performing forward elimination on [A I].

    Returns
    -------
    Tuple[numpy.ndarray, numpy.ndarray]
        A tuple containing the A in lower triangular form and a matrix or
        vector that received the same row actions
    """
    i : int = len(A) - 1
    while i > 0:
        j : int = 1
        while j <= i:
            pivot = A[i - j][i]/A[i][i]
            A[i - j] = A[i - j] - (A[i] * pivot)
            b[i - j] = b[i - j] - (b[i] * pivot)
            j += 1
        i -= 1

    return A, b

def backward_substitution(A : np.ndarray, b : np.ndarray) -> np.ndarray:
    """Performs backward to solve Ax = b when A is in upper triangular form

    Parameters
    ----------
    A : numpy.ndarray
        A nxn matrix, the left side of the equation in upper triangular form
    b : numpy.ndarray
        A vector of length n, the right side of the equation

    Returns
    -------
    numpy.ndarray
        The solution, x in Ax = b
    """
    x : np.ndarray = np.zeros(len(b))
    for i in range(-1, (len(b) * -1 - 1), -1):
        answer : float = b[i]
        for j in range(-1, i, -1):
            answer = answer - x[j] * A[i][j]
        x[i] = answer / A[i][i]

    return x

def forward_substitution(A : np.ndarray, b : np.ndarray) -> np.ndarray:
    """Performs forward to solve Ax = b when A is in lower triangular form

    Parameters
    ----------
    A : numpy.ndarray
        A nxn matrix, the left side of the equation in lower triangular form
    b : numpy.ndarray
        A vector of length n, the right side of the equation

    Returns
    -------
    numpy.ndarray
        The solution, x in Ax = b
    """
    x : np.ndarray = np.zeros(len(b))
    for i in range(len(b)):
        answer : float = b[i]
        for j in range(i):
            answer = answer - x[j] * A[i][j]
        x[i] = answer / A[i][i]

    return x


def test_solve_linear_system(solve_func : callable, n_list : List[int], k : int):
    """Tests functions that solve linear systems

    Parameters
    ----------
    solve_func : callable
        The function to be tested, must take a nxn matrix and a list of vector
        of length k and return a list of solutions of length k
    n_list : List[int]
        A list of integers indicating the n for the nxn matrix and n length
        vectors to be used in the test
    k : int
        The number of b vectors to be created for the tested function
    """
    for n in n_list:
        A : np.ndarray = np.random.normal(size=(n, n))
        b_list : List[np.ndarray] = []
        for i in range(k):
            b_list.append(np.random.normal(size = n))
        solutions : np.ndarray = solve_func(A, b_list)
        for b, x in zip(b_list, solutions):
            try:
                assert np.allclose(A.dot(x), b)
            except AssertionError:
                print(f"Failed asserting {x} was solution for {A}x = {b}")
                print(A.dot(x))
