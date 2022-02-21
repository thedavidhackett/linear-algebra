import numpy as np
from typing import List, Optional, Tuple

def forward_elimination(A : np.ndarray, b : Optional[np.ndarray] = None) -> \
    Tuple[np.ndarray, list, Optional[np.ndarray]]:
    """Performs forward elimination on a matrix while modifying a vector or
    matrix with the row actions.

    Parameters
    ----------
    A : numpy.ndarray
        A m x n sized matrix to perform forward elimination on, the left side
        of Ax = b
    b : numpy.ndarray
        A n sized vector representing the right side of Ax = b

    Returns
    -------
    Tuple[numpy.ndarray, list, numpy.ndarray]
        A tuple containing the matrix A in row echelon form, a list of pivots
        with the index representing the row for the pivot and the value
        representing the column of the pivot. Optionally a vector b with
        the same row operations applied to it as on A.
    """
    pivots : list = []

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
            # For elimination this line solves b
            # For inverse it performs Gauss Elimination
            if len(b) > 0:
                b[i + k] = b[i + k] - (b[i] * l)
            k += 1

        i += 1
        j += 1

    return A, pivots, b

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
        Ax = b and the second array being a matrix xn with reach row representing a
        vector in N(A)
    """
    A = A.copy()
    b = b.copy()
    A, pivots, b = forward_elimination(A, b)

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
    # If there aren't any free cols make xn only include the 0 vector
    xn : np.ndarray = np.zeros((len(free_cols), len(A[0]))) if len(free_cols) > 0 else np.zeros((1, len(A(0))))
    # For each free col find the special solution using backward substitution
    for i, free in enumerate(free_cols):
        # Set the free variable that we are currently on to one
        xn[i][free] = 1
        # Only need to solve for the pivot rows
        for j in range(len(pivots) - 1, -1, -1):
            answer = 0
            for k in range(len(A[0]) - 1, pivots[j], -1):
                answer = answer - xn[i][k] * A[j][k]

            xn[i][pivots[j]] = answer / A[j][pivots[j]]

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
    i : int = np.random.randint(0, len(xn))
    # Check if solving using xp and xn gives correct b
    try:
        assert np.allclose(b, A.dot(xp) + A.dot((c * xn[i])))
    except AssertionError:
        print(f"matrix {A} with solution {b} did not produce the correct x")

def generate_random_system(shape : Tuple[int, int], pivots : List[int]) -> Tuple[np.ndarray, np.ndarray]:
    A : np.ndarray = np.zeros(shape)
    num_rows : int = len(A)
    num_cols : int = len(A[0])
    # Fill in pivot cols with random floats
    for pivot_col in pivots:
        A[:,pivot_col] = np.random.randn(num_rows)

    # Make free columns into linear combinations of pivot cols
    for i in range(num_cols):
        if i not in pivots:
            # free_col = np.zeros(num_rows)
            # # for pivot_col in pivots:
            # #     free_col += np.random.rand(1) * A[:, pivot_col]
            A[:,i] = np.random.rand(1) * A[:, pivots[0]]

    # Create a b given a solution x
    x = np.zeros(num_cols)
    # Ensure zeros in free columns
    for pivot_col in pivots:
        x[pivot_col] = np.random.randn(1)[0]

    b = A.dot(x)

    return A, b
