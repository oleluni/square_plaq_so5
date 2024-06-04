import numpy as np
from scipy.sparse import issparse
import scipy.sparse as sparse


def gram_schmidt_complex(V):
    """
    Gram-Schmidt orthogonalization for complex-valued vectors.

    Parameters:
        V (list of ndarrays): List of column vectors stored as ndarrays.

    Returns:
        Q (list of ndarrays): List of orthogonalized column vectors.
    """
    m = len(V)
    Q = [np.zeros_like(v, dtype=np.complex128) for v in V]

    for i in range(m):
        v = V[i]
        q = v.copy()
        for j in range(i):
            q -= np.vdot(Q[j], v) / np.vdot(Q[j], Q[j]) * Q[j]
        Q[i] = q / np.linalg.norm(q)

    return Q


def gram_schmidt_complex_ndarray(V) -> np.ndarray:
    """
    Gram-Schmidt orthogonalization for complex-valued vectors.

    Parameters:
        V (ndarray): 2D ndarray where each column represents a complex-valued vector.

    Returns:
        Q (ndarray): Orthogonalized complex-valued vectors as columns of the resulting ndarray.
    """

    m, n = V.shape
    Q = np.zeros_like(V, dtype=np.complex128)

    for i in range(n):
        v = V[:, i]
        q = v.copy()
        for j in range(i):
            q -= (np.vdot(Q[:, j], v) / np.vdot(Q[:, j], Q[:, j]) ) * Q[:, j]
        Q[:, i] = q / np.linalg.norm(q)

    return Q




def ortho_norm_vecs(vec_list: list) -> list:
    ortho_norm_list = []

    if not issparse(vec_list[0]):

        for u in vec_list:
            # v_copy = v.copy()
            # u = v_copy
            if ortho_norm_list:
                for p in ortho_norm_list:
                    norm_p_2 = np.dot(p.T.conj(), p)
                    # u -= np.dot(v_copy, np.conj(p)) / (np.linalg.norm(p) ** 2) * p
                    u -= p * (np.dot(u.T.conj(), p) / (norm_p_2))
            # norm_u = np.linalg.norm(u)
            norm_u = np.sqrt(np.dot(u.T.conj(), u))
            if np.abs(norm_u)>1e-10:
                # u /= np.linalg.norm(u)
                u /= norm_u
            ortho_norm_list.append(u)

    else:

        for v in vec_list:
            v_copy = v.copy()
            u = v_copy
            if ortho_norm_list:
                for p in ortho_norm_list:
                    u -= v_copy.dot(p.H) / (sparse.linalg.norm(p) ** 2) * p
            norm_u = sparse.linalg.norm(u)
            if norm_u != 0:
                u /= sparse.linalg.norm(u)
            ortho_norm_list.append(u)

    return ortho_norm_list

# TEST

# the vector dimension has to be n, where n is the numbers of vectors, otherwise, end up with zero-vectors
# due to them not being linearly-independent

# 3 dims
# v1 = np.array([1., 0., 0.j], dtype=complex)
# v2 = np.array([0., 178., 1+0.5j], dtype=complex)
# v3 = np.array([0., 0., 1.5j], dtype=complex)
# vec_list = [v1, v2, v3]

# 4 dims
# v1 = np.array([12., 0., 0.j, 2], dtype=complex)
# v2 = np.array([0., 13., 0.j, 2], dtype=complex)
# v3 = np.array([0., 0., 1.j, 2], dtype=complex)
# v4 = np.array([1., 1., 1.j, 2], dtype=complex)
# vec_list = [v1, v2, v3, v4]

# 10 dims
# v1 = np.array([1., 1.j, 2., 0., 0., 0., 0., 0., 0., 0.], dtype=complex)
# v2 = np.array([0., 1., 1.j, 2., 0., 0., 0., 0., 0., 0.], dtype=complex)
# v3 = np.array([0., 0., 1., 1.j, 2., 0., 0., 0., 0., 0.], dtype=complex)
# v4 = np.array([0., 0., 0., 1., 1.j, 2., 0., 0., 0., 0.], dtype=complex)
# v5 = np.array([0., 0., 0., 0., 1., 1.j, 2., 0., 0., 0.], dtype=complex)
# v6 = np.array([0., 0., 0., 0., 0., 1., 1.j, 2., 0., 0.], dtype=complex)
# v7 = np.array([0., 0., 0., 0., 0., 0., 1., 1.j, 2., 0.], dtype=complex)
# v8 = np.array([0., 0., 0., 0., 0., 0., 0., 1., 1.j, 2.], dtype=complex)
# v9 = np.array([2., 0., 0., 0., 0., 0., 0., 0., 1., 1.j], dtype=complex)
# v10 = np.array([1.j, 2., 0., 0., 0., 0., 0., 0., 0., 1.], dtype=complex)
#
# vec_list = [v1, v2, v3, v4, v5, v6, v7, v8, v9, v10]
#
# u = ortho_norm_vecs(vec_list=vec_list)
#


# TEST IMPORTANT
#
# v1 = np.array([1, 0, 0, 3j, 0, 0, 0, 0, 0, 0], dtype=complex)
# v2 = np.array([0, 1, 0, 0, 0, 24, 0, 0, 0, 0], dtype=complex)
# v3 = np.array([0, 0, 4.5, 0, 0, 1-1.j, 0, 0, 0, 0], dtype=complex)
#
#
# vec_list = [v1, v2, v3]
#
# u = ortho_norm_vecs(vec_list=vec_list)
# from hamiltonian_opers import gram_schmidt as gs
# vec_dict = {idx: v for idx, v in enumerate(vec_list)}
# u = gs(vec_dict)
#
# [[print(f"u{idx_i} * u{idx_j}", np.dot(ui, np.conj(uj))) for idx_i, ui in enumerate(u)] for idx_j, uj in enumerate(u)]
# [print(ui, ui.shape) for ui in u] # result is 3 3-dim vectors for gs from ham
