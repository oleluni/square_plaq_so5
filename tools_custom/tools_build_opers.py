import numpy as np
import scipy.sparse as sparse
from scipy.sparse import issparse
from scipy.special import mathieu_sem
import constants as const

def get_states(q, order='mL_down_mR_down') -> dict:
    """
    Fill the dictionary states with the Hilbert space states (j, mL, mR)
    up to a given j = q.

    The counter i is for the checkerboard index.

    Args:
        q (int or half-int): max value for j

    Returns:
        states (dictionary): keys are flat indices, values are numpy arrays
        with given labels (j, mL, mR)
    """
    states = {}
    i = 0

    for j in [k/2 for k in range(0, int(2*q)+1)]:

        if order == 'mL_down_mR_down':
            range_mR = range(0, int(2 * j) + 1)
            range_mL = range(0, int(2 * j) + 1)
        elif order == 'mL_up_mR_down':
            range_mR = range(0, int(2 * j) + 1)
            range_mL = range(int(2 * j), -1, -1)
        elif order == 'mL_down_mR_up':
            range_mR = range(int(2 * j), -1, -1)
            range_mL = range(0, int(2 * j) + 1)
        elif order == 'mL_up_mR_up':
            range_mR = range(int(2 * j), -1, -1)
            range_mL = range(int(2 * j), -1, -1)
        else:
            raise ValueError("Incorrect input for the order argument.")


        # for mR in [j - k for k in range_mR]:
        #     for mL in [j - k for k in range_mL]:
        #         states[i] = np.array([j, mL, mR])
        #         i += 1

        for mL in [j - k for k in range_mL]:
            for mR in [j - k for k in range_mR]:
                states[i] = np.array([j, mL, mR])
                i += 1

    return states


def get_Nq(q) -> int:
    """

    Calculate number of eigenstates with j<=q

    N_q = \sum_{j=0}^{q} (2j + 1)^2

    Args:
      q : maximal value of j

    Returns:
      Nq (int): number of eigenstates with j<=q
    """
    return int((q + 1) * (2*q + 1) * (4*q + 3) / 3)


def get_states_plaquette(q):
    Nq = get_Nq(q=q)
    states = np.zeros((Nq, 3))

    i = 0
    # site_1
    for j in [k / 2 for k in range(0, int(2 * q) + 1)]:
        for mR in [j - k for k in range(0, int(2 * j) + 1)]:
            for mL in [j - k for k in range(0, int(2 * j) + 1)]:
                states[i] = np.array([j, mL, mR])
                i += 1

    return states


def merge_rows_outer(array1, array2=None):
    if array2 == None:
        array2 = array1
    num_rows_1 = array1.shape[0]
    num_cols_1 = array1.shape[1]

    num_rows_2 = array2.shape[0]
    num_cols_2 = array2.shape[1]

    result_array = np.zeros((num_rows_1*num_rows_2, num_cols_1 + num_cols_2))

    for i in range(num_rows_1):
        for j in range(num_rows_2):
            result_array[i * num_rows_2 + j, :] = np.concatenate((array1[i], array2[j]))

    return result_array

# TODO: finish imlementing commutator for sparse matrices
def commutator(A, B, tol=1e-10):
    # check for wrong input
    if (A.shape != B.shape):
        raise "Shapes of input matrices do not match."

    C = (A @ B) - (B @ A)
    if np.all(np.abs(C) < 1e-8):
        return 0
    else:
        return C


def is_diagonal(matrix):
    return np.all(matrix == np.diagflat(np.diag(matrix)))


def tensor_prod(*args):
    # for the case of several entries
    # args in this case is a tuple args = (args[0], args[1], ... , args[n-1])
    is_sparse = all(sparse.issparse(arg) for arg in args)

    # args in this case is an iterable args = (args,)
    # for the case of a list of sparse matrices
    if len(args) == 1 and isinstance(args[0], list) and all(sparse.issparse(arg) for arg in args[0]):
        arg_copy = args[0].copy()
        prod = arg_copy[0]
        for i in range(len(arg_copy)-1):
            prod = sparse.kron(prod, arg_copy[i+1], format='csr')
        return prod
    elif len(args) == 1 and isinstance(args[0], (list, np.ndarray)):
        arg_copy = args[0].copy()
        prod = arg_copy[0]
        for i in range(len(arg_copy)-1):
            prod = np.kron(prod, arg_copy[i+1])
        return prod
    # case of a tuple of dense matrices
    elif len(args) == 1 and isinstance(args[0], tuple):
        prod = args[0][0]
        for i in range(len(args[0]) - 1):
            prod = np.kron(prod, args[0][i + 1])
        return prod

    # for the case of several entries
    # account for sparse matrices
    elif len(args) > 1 and is_sparse:
        prod = args[0]
        for i in range(len(args)-1):
            prod = sparse.kron(prod, args[i+1])
        return prod
    elif len(args) > 1:
        # in this case args is a tuple of args, so it's immutable
        # and doesn't need to be copied, since tuples cannot be changed
        prod = args[0]
        for i in range(len(args)-1):
            prod = np.kron(prod, args[i+1])
        return prod
    else:
        raise ValueError("Invalid Input")


def is_real_within_tolerance(arr: np.ndarray, tolerance=1e-10) -> bool:
    return np.isclose(arr, arr.real, atol=tolerance).all()


def gram_schmidt(V):
    """
    Gram-Schmidt ortho-normalization procedure for a set of vectors.

    Parameters:
    V : array-like, shape (n, m)
        Matrix containing m linearly independent vectors of dimension n.

    Returns:
    Q : array-like, shape (n, m)
        Orthonormal basis for the space spanned by the input vectors.
    """
    V = np.array(V, dtype=complex)
    m, n = V.shape
    Q = np.zeros((m, n), dtype=complex)

    for i in range(m):
        Q[i, :] = V[i, :]
        for j in range(i):
            Q[i, :] -= np.dot(V[i, :], np.conj(Q[j, :])) * V[j, :] / (np.linalg.norm(V[j, :]) ** 2)

        Q[i, :] /= np.linalg.norm(Q[i, :])

    return Q

def get_analytic_ham_eigs(g=const.G_COUPLING):
    q = -8 / (g ** 2)
    omega = 0.21 # why? How I differentiate between the eigenvalues? How do I alter truncation?
    y = mathieu_sem(2, q, omega)[0]
    return y


def initialize_sparse_vectors(indices, dimension) -> list[sparse.csr_matrix]:
    """
    Initialize sparse vectors with a 1 at the specified index.

    Parameters:
        indices (list): List of indices where each vector will have a 1.
        dimension (int): Dimension of each vector.

    Returns:
        list of csr_matrix: List of initialized sparse vectors.
    """
    sparse_vectors = []
    for index in indices:
        # Create the data, row, and column arrays for the COO format
        data = [1]  # value at the specified index
        col = [index]  # one row per index
        row = [0]  # only one column

        # Construct the list of csr_matrix matrices
        sparse_vector = sparse.coo_matrix((data, (row, col)), shape=(1, dimension))
        sparse_vectors.append(sparse_vector.tocsr())
        # each vector is a column
    return sparse_vectors

