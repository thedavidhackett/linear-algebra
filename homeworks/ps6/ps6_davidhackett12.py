from typing import List, Optional, Tuple
import numpy as np

def forward_elimination(A : np.ndarray, b : np.ndarray = None) -> \
    Tuple[np.ndarray, list, int, Optional[np.ndarray]]:
    """Performs forward elimination on a matrix while optionally modifying an
    additional vector or matrix with the row actions. Cannot handle row exchanges.
    Returns the matrix in Row Echelon form, along with the pivot columns, rank
    and optionally the modified b

    Parameters
    ----------
    A : numpy.ndarray
        A m x n sized matrix to perform forward elimination on, the left side
        of Ax = b
    b : numpy.ndarray, default = None
        A n sized vector representing the right side of Ax = b

    Returns
    -------
    Tuple[numpy.ndarray, list, int, numpy.ndarray]
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

def solve_completely(A : np.ndarray, b : np.ndarray) -> Tuple[Optional[np.ndarray], np.ndarray]:
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
    Tuple[Optional[np.ndarray], np.ndarray]:
        A tuple with the first array representing a particular solution (xp) to
        Ax = b and the second array being a matrix xn with reach column
        representing a vector in N(A)
    """
    A, pivots, _, b = forward_elimination(A, b)

    xp : Optional[np.ndarray] = None
    # Check if solvable, non pivot rows should have 0 in b vector
    if np.allclose(0, b[len(pivots):]) or len(pivots) == len(A):
        # Backward substitution to get particular solution xp
        xp = np.zeros(len(A[0]))
        i : int = len(pivots) - 1
        while i >= 0:
            answer : float = b[i]

            for j in range(len(xp) - 1, pivots[i], -1):
                answer = answer - xp[j] * A[i][j]

            xp[pivots[i]] = answer / A[i][pivots[i]]
            i -= 1

    # Figure out what the free columns are
    free_cols : List[int] = []
    for i in range(len(A[0])):
        if i not in pivots:
            free_cols.append(i)

    # Create an empty matrix for xn where each row will correspond to a free col
    xn : np.ndarray
    if len(pivots) < len(A) and len(free_cols) > 0:
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

def test_solve_completely(A : np.ndarray, b : np.ndarray, solvable : bool = True) -> None:
    """Tests that solve completely works by plugging xp and xn back into the
    equation Ax = b and verifying it produces the correct b

    Parameters
    ----------
    A : numpy.ndarray
        A m x n matrix, the left side of the equation Ax = b
    b : np.ndarray
        A n length vector, the right side of the equation Ax = b
    solvable : bool, default=True
        Whether the b for the equation is expected to produce a particular
        solution or not

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
    if solvable:
        assert np.allclose(b, A.dot(xp) + A.dot((c * xn[:,i])))
    else:
        assert xp is None
        assert np.allclose(np.zeros(len(A)), A.dot((c * xn[:,i])))


def test_forward_elimination() -> None:
    # Tests forward elimination on a known A matrix to ensure it works
    A = np.array([
     [1.0, 3.0, 1.0, 2.0],
     [2.0, 6.0, 4.0, 8.0],
     [0.0, 0.0, 2.0, 4.0]
    ])

    R_expected =  np.array([
     [1.0, 3.0, 1.0, 2.0],
     [0.0, 0.0, 2.0, 4.0],
     [0.0, 0.0, 0.0, 0.0]
    ])

    pivots_expected = [0, 2]
    rank_expected = 2

    R, pivots, rank, _ = forward_elimination(A)

    try:
        print(f"Input A:\n{A}")
        assert np.allclose(R, R_expected)
        print(f"\nR expected:\n{R_expected}\nvs R output\n{R}")
        assert pivots == pivots_expected
        assert rank == rank_expected
        print(f"\nExpected Rank and Pivots:\n {rank_expected}, {pivots_expected}")
        print(f"\nOutput Rank and Pivots:\n {rank}, {pivots}")
        print("Test successful")
    except AssertionError:
        print("Test Failed")


def generate_random_system(shape : Tuple[int, int], pivots : List[int], solvable : bool = True) \
    -> Tuple[np.ndarray, np.ndarray]:
    """Generates a random system Ax = b with give A matrix shape and pivot cols

    Parameters
    ----------
    shape : Tuple[int, int]
        The shape of the matrix m x n
    pivots: List[int]
        A list of pivot cols with their index corresponding to their row
    solvable : bool, default=True
        Whether the b for the equation is expected to produce a particular
        solution or not

    Returns
    -------
    Tuple[numpy.ndarray, numpy.ndarray]
        A tuple with matrix A and vector b with at least 1 solution

    Raises
    -----
    PivotException
        If the pivot columns are out of range, out of order, or one column is designated twice
    UnsolvableException
        If the system is designated unsolvable but the rank equals the rows and columns
    """
    # Make sure the pivot list is valid
    for i, pivot in enumerate(pivots):
        if pivot >= shape[1]:
            raise PivotException("The pivot column designated is out of range")
        if not i == 0 and pivot <= pivots[i - 1]:
            raise PivotException("The pivots are out of order or duplicate columns")

    # Ensure matrix is unsolvable if designated
    if not solvable and len(pivots) == shape[0]:
        raise UnsolvableException(f"shape {shape} with pivots {pivots} will always be solvable")

    A : np.ndarray = np.zeros(shape)
    num_rows : int = len(A)
    num_cols : int = len(A[0])
    # Fill in pivot cols with random floats
    for pivot_col in pivots:
        A[:,pivot_col] = np.random.randn(num_rows)

    # Make free columns into linear combinations of cols that came before
    # Start at the first pivot incase its not the first col
    for i in range(pivots[0], num_cols):
        if i not in pivots:
            free_col = np.zeros(num_rows)
            for j in range(i):
                free_col += np.random.rand(1) * A[:, j]
            A[:,i] = free_col
    # Now go back and make any free cols that came before the first pivot
    for i in range(pivots[0]):
        # Make a free col that is still a linear combo but has 0 in the first row
        free_col = A[:,-1] - (A[:,-2] * (A[0,-1]/A[0,-2]))
        A[:,i] = free_col

    multiply_by_random = lambda x : x * np.random.rand(1)

    if solvable:
        # Create a b in the column space of A
        b = np.sum(np.apply_along_axis(multiply_by_random, 0, A), axis=1)
    else:
        # Create a random b, unlikely to be solved
        b = np.random.randn(len(A))

    return A, b

def test_solve_completely_on_random() -> None:
    """Tests solve completely on random matrices of various shapes and with various
    pivot columns. Will print passes and failures after the test.
    """
    # Generate a bunch of shapes and pivot columns
    shapes = [(2, 2), (3, 2), (4, 2), (2,3), (3,3), (4,3), (3, 4),\
         (4,4), (4,5), (5,4), (5,5)]
    pivot_sets = [[0,1], [0, 1], [0], [0, 2], [0, 1], [0, 1, 2], [0, 3],\
         [0, 1, 2], [0, 1, 2, 3], [0, 2, 3], [0, 2, 3]]

    passes = 0
    failures = 0
    for shape, pivots in zip(shapes, pivot_sets):
        try:
            A, b = generate_random_system(shape, pivots)
            test_solve_completely(A, b)
            passes += 1
        except AssertionError:
            print(f"solvable matrix \n{A} \nwith solution \n{b} \ndid not produce the correct x")
            failures += 1
        try:
            A2, b2 = generate_random_system(shape, pivots, False)
            test_solve_completely(A2, b2, False)
            passes += 1
        except UnsolvableException:
            passes += 1
        except AssertionError:
            print(f"unsolvable matrix\n{A2}\nwith solution\n{b2}\ndid not produce the correct x")
            failures += 1


    print(f"Tests Passed: {passes} \nTests Failed: {failures}")


class PivotException(Exception):
    # Raises if a pivot col is not possible
    pass

class UnsolvableException(Exception):
    # Raises if a system that always has a solution is designated unsolvable
    pass
