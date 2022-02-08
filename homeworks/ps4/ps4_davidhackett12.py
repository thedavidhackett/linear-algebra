import time
from typing import Callable, Dict, List, Tuple
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
        # Unlike other functions A copy needs to be renamed because of the scope
        A_copy = np.copy(A)
        b = np.copy(b)

        # Uses elimination and backward substitution to solve Ax = b for one b
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

    # Passes A and I to forward elimination for the Gauss half of Gauss-Jordan
    I, Ainv = forward_elimination(A, I)
    # Passes results of that to backward elimination to complete Gauss Jordan
    I, Ainv = backward_elimination(I, Ainv)
    # Finishes by diving the Ainv by the pivots of the almost I matrix
    for i in range(len(I)):
        Ainv[i] = Ainv[i]/I[i][i]

    # x = Ainv * b
    for b in b_list:
        b = np.copy(b)
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

    # This forward elimination function returns A as U and L
    U, L = forward_elimination_factorize(A)

    for b in b_list:
        b = np.copy(b)
        # forward substitution to solve Lc = b
        c = forward_substitution(L, b)
        # backward substitution to solve Ux = c
        solutions.append(backward_substitution(U, c))

    return solutions


def forward_elimination(A : np.ndarray, b : np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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
            l = A[i + j][i]/A[i][i]
            # Performs row operation on A
            A[i + j] = A[i + j] - (A[i] * l)
            # For elimination this line solves b
            # For inverse it performs Gauss Elimination
            b[i + j] = b[i + j] - (b[i] * l)
            j += 1

        i += 1

    return A, b

"""
I created a second forward elimination function that would also create the L
matrix. I debated on whether to have a bool variable that would change whats
done to the b parameter but this felt computationally inefficient and passing
a function into this function to act on b felt less readable.
"""
def forward_elimination_factorize(A : np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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
            l = A[i + j][i]/A[i][i]
            # Performs row operation on A
            A[i + j] = A[i + j] - (A[i] * l)
            # creates the L by passing lji into correct spot of an identity matrix
            L[i + j][i] = l
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
            u = A[i - j][i]/A[i][i]
            A[i - j] = A[i - j] - (A[i] * u)
            # Jordan half of gauss jordan on the identity matrix
            b[i - j] = b[i - j] - (b[i] * u)
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


def test_solve_linear_system(solve_func : Callable, n_list : List[int], k : int):
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

    Raises
    ------
    AssertionError
        If the function does not correctly solve the linear system
    """
    for n in n_list:
        A : np.ndarray = np.random.normal(size=(n, n))
        b_list : List[np.ndarray] = [np.random.normal(size = n) for _ in range(k)]

        # Runs function and tests solutions
        solutions : np.ndarray = solve_func(A, b_list)
        for b, x in zip(b_list, solutions):
            try:
                # Checks if x is solution by checking if b = A * x
                assert np.allclose(A.dot(x), b)
            except AssertionError:
                print(f"Failed asserting {x} was solution for {A}x = {b}")
                return

    # Will only print this if no assertion fails
    print("Function passes all tests")


def time_solve_functions() -> None:
    """Times functions that solve linear systems. This function times each
    approach to solving linear systems: elimination, inverse, and LU
    factorization. It takes the average amount of time for each function to
    solve a linear system Ax = b with an nxn matrix (A) and k n length vectors
    (b) over ten runs. It compares the functions for various n and k values.

    Raises
    ------
    AssertionError
        If the functions produce the wrong answer or if the inputs to the
    functions are modified.
    """
    n_list : List[int] = [2, 4, 8, 32, 64]
    k_list : List[int] = [1, 2, 4, 16, 32]
    funcs : Dict[str, Callable] = {
        "Solve Elimination" : solve_elimination,
        "Solve Inverse" : solve_inverse,
        "Solve LU" : solve_lu
    }

    for n in n_list:
        for k in k_list:
            # Generate the same nxn matrix and k length list of n length vectors
            A : np.ndarray = np.random.normal(size=(n, n))
            b_list : List[np.ndarray] = [np.random.normal(size = n) for _ in range(k)]

            # Uses those same inputs for each function
            for f_name, func in funcs.items():
                times : np.ndarray = np.zeros(10)
                for i in range(10):
                    # Times each run using helper function
                    times[i] = time_and_test_solve_function(A, b_list, func)

                # prints average of times
                print(f"{f_name}: averaged {times.mean()} for n = {n} and k = {k}")


# This function was copied from the provided code on Ed Discussion.
# I have made the following modifications: Added type hinting for consistency.
def time_and_test_solve_function(A : np.ndarray, b_list : List[np.ndarray], \
    solve_function : Callable) -> float:
    """
    Solve A x = b_i for each vector b_i in b_list, using the provided
    solve_function. Additionally, measure the amount of time that the
    solve_function took, and test that it solved the system correctly,
    and without modifying any of the inputs to solve_function.
    Since solve_function is a function, time_and_test_solve is a
    higher-order function.

    Inputs:
        A: two-dimensional NumPy array, with shape (n, n)
        b_list: standard Python list with k vectors, each of which is
            represented by a one-dimensional NumPy array of size n.
        solve_function: A function that takes A and b_list as inputs,
            and returns a list x_list of solution vectors. For our
            purposes, solve_function will be solve_elimination,
            solve_with_inverse, or solve_lu.

    Returns: (float) the number of seconds it took to run solve_function.
        If the solve_function returns the wrong output, or if it modifies
        the input, raises an AssertionError.
    """

    backup_of_original_A : np.ndarray = A.copy()
    backup_of_original_b_list : List[np.ndarray] = [b.copy() for b in b_list]

    start_time : float = time.perf_counter()
    x_list : List[np.ndarray] = solve_function(A, b_list)
    end_time : float = time.perf_counter()

    # Check that the original arrays were not modified.
    # Because they were not modified, they should exactly equal the original;
    # there should be no rounding error.
    assert np.array_equal(A, backup_of_original_A)
    for b, b_copy in zip(b_list, backup_of_original_b_list):
        assert np.array_equal(b, b_copy)

    # Check that the computation was correct.
    # Because there may be some rounding error in the course of the
    # computation, we use np.allclose to check if the arrays are equal
    # instead of np.array_equal.
    for b, x in zip(b_list, x_list):
        assert np.allclose(A.dot(x), b)

    return end_time - start_time
