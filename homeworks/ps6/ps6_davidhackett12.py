import numpy as np
from typing import List, Optional, Tuple

def forward_elimination(A : np.ndarray, b : Optional[np.ndarray] = None) -> \
    Tuple[np.ndarray, list, int, Optional[np.ndarray]]:
    """Performs forward elimination on a matrix while modifying a vector or
    matrix with the row actions. Cannot handle row exchanges.

    Parameters
    ----------
    A : numpy.ndarray
        A m x n sized matrix to perform forward elimination on, the left side
        of Ax = b
    b : numpy.ndarray, optional
        A n sized vector representing the right side of Ax = b

    Returns
    -------
    Tuple[numpy.ndarray, list, numpy.ndarray]
        A tuple containing the matrix A in row echelon form, a list of pivots
        with the index representing the row for the pivot and the value
        representing the column of the pivot and the rank of the matrix.
        Optionally a vector b with the same row operations applied to it as on A.
    """
    pivots : list = []
    A = A.copy()
    try:
        b = b.copy()
    except AttributeError:
        pass
    # Using i and j since the col, j, no longer needs to be the same as the row, i
    i : int = 0
    j : int = 0
    while i < len(A) and j < len(A[i]):
        # Check if the pivot is 0 skip to next column if so
        if np.isclose(A[i][j], 0):
            j += 1
            continue
        pivots.append(j)

        k : int = 1
        while k < len(A) - i:
            l = A[i + k][j]/A[i][j]
            # Performs row operation on A
            A[i + k] = A[i + k] - (A[i] * l)
            # Perform the same row operation on b if present
            try:
                b[i + k] = b[i + k] - (b[i] * l)
            except TypeError:
                pass
            k += 1

        i += 1
        j += 1

    return A, pivots, len(pivots), b

def solve_completely(A : np.ndarray, b : np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """ Finds the complete solution xp + xn for Ax + b given a matrix A and a
    vector b using forward elimination and then backwards substitution

    Parameters
    ----------
    A : numpy.ndarray
        A m x n matrix, the left side of the equation Ax = b
    b_list : List[np.ndarray]
        A list of vectors of length n, the right side of the equation Ax = b

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]:
        A tuple with the first array representing a particular solution (xp) to
        Ax = b and the second array being a matrix xn with reach column
        representing a vector in N(A)
    """
    A, pivots, _, b = forward_elimination(A, b)

    # Backward substitution to get particular solution xp
    xp : np.ndarray = np.zeros(len(A[0]))
    i : int = len(pivots) - 1
    while i >= 0:
        answer : float = b[i]

        for j in range(len(xp) - 1, pivots[i], -1):
            answer = answer - xp[j] * A[i][j]

        xp[pivots[i]] = answer / A[i][pivots[i]]
        i -= 1

    # Figure out what the free columns are
    free_cols : list[int] = []
    for i in range(len(A[0])):
        if i not in pivots:
            free_cols.append(i)

    # Create an empty matrix for xn where each row will correspond to a free col
    xn : np.ndarray
    if len(free_cols) > 0 and len(A[0]) >= len(A):
        xn = np.zeros((len(A[0]), len(free_cols)))
        # For each free col find the special solution using backward substitution
        for i, free in enumerate(free_cols):
        # Set the free variable that we are currently on to one
            xn[free][i] = 1
            # Only need to solve for the pivot rows
            for j in range(len(pivots) - 1, -1, -1):
                answer = 0
                for k in range(len(A[0]) - 1, pivots[j], -1):
                    answer = answer - xn[k][i] * A[j][k]

                xn[pivots[j]][i] = answer / A[j][pivots[j]]
    # If there aren't any free cols or row > col make xn only include the 0 vector
    else:
        xn = np.zeros((len(A[0]), 1))

    return xp, xn

def test_solve_completely(A : np.ndarray, b : np.ndarray) -> None:
    """Tests that solve completely works by plugging xp and xn back into the
    equation Ax = b and verifying it produces the correct b

    Parameters
    ----------
    A : numpy.ndarray
        A m x n matrix, the left side of the equation Ax = b
    b_list : List[np.ndarray]
        A list of vectors of length n, the right side of the equation Ax = b

    Raises
    -------
    AssertionError
        If solve_completely does not correctly solve a system Ax = b
    """
    xp, xn = solve_completely(A, b)
    # Generate a random value for the coefficient for xn
    c : np.ndarray = np.random.rand(1)
    # Choose one of the vectors in xn at random
    i : int = np.random.randint(0, len(xn[0]))
    # Check if solving using xp and xn gives correct b
    assert np.allclose(b, A.dot(xp) + A.dot((c * xn[:,i])))


def generate_random_system(shape : Tuple[int, int], pivots : List[int]) -> Tuple[np.ndarray, np.ndarray]:
    """Generates a random system Ax = b with give A matrix shape and pivot cols

    Parameters
    ----------
    shape : Tuple[int, int]
        The shape of the matrix m x n
    pivots: List[int]
        A list of pivot cols with their index corresponding to their row

    Returns
    -------
    Tuple[numpy.ndarray, numpy.ndarray]
        A tuple with matrix A and vector b with at least 1 solution
    """
    A : np.ndarray = np.zeros(shape)
    num_rows : int = len(A)
    num_cols : int = len(A[0])
    # Fill in pivot cols with random floats
    for pivot_col in pivots:
        A[:,pivot_col] = np.random.randn(num_rows)

    # Make free columns into linear combinations of cols that came before
    for i in range(num_cols):
        if i not in pivots:
            if i == 0:
                free_col = np.random.randn(num_rows)
                free_col[0] = 0
            else:
                free_col = np.zeros(num_rows)
                for j in range(i):
                    free_col += np.random.rand(1) * A[:, j]
            A[:,i] = free_col

    # Create a b given a solution x
    x = np.zeros(num_cols)
    # Ensure zeros in free columns
    for pivot_col in pivots:
        x[pivot_col] = np.random.randn(1)[0]

    b = A.dot(x)

    return A, b

def test_solve_completely_on_random() -> None:
    """Tests solve completely on random matrices of various shapes and with various
    pivot columns. Will print passes and failures after the test.
    """
    # Generate a bunch of shapes and pivot columns
    shapes = [(1,2), (2, 2), (3, 2), (4, 2), (2,3), (3,3), (4,3), (3, 4),\
         (4,4), (4,5), (5,4), (5,5)]
    pivot_sets = [[0], [0,1], [1], [0], [0, 2], [0, 1], [0, 1, 2], [0, 3],\
         [0, 1, 2], [0, 1, 2, 3], [2, 3], [0, 1, 2, 3]]

    passes = 0
    failures = 0
    for shape, pivots in zip(shapes, pivot_sets):
        A, b = generate_random_system(shape, pivots)
        try:
            test_solve_completely(A, b)
            passes += 1
        except AssertionError:
            print(f"matrix \n{A} \nwith solution \n{b} \ndid not produce the correct x")
            failures += 1
    print(f"Tests Passed: {passes} \nTests Failed: {failures}")
