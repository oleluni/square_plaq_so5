import numpy as np
from scipy.sparse import csr_matrix


def get_tau_i(i: int) -> np.ndarray:
    tau = np.array([[[1, 0], [0, 1]], [[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]])
    if i != 0:
        tau = 0.5 * tau
    return tau[i]



def get_L_i(i:int) -> csr_matrix:
    tau_i = get_tau_i(i=i)
    upper_row = np.concatenate((tau_i, np.zeros_like(tau_i)), axis=1)
    lower_row = np.zeros_like(upper_row)

    L_i_matrix = np.concatenate((upper_row, lower_row), axis=0)
    return csr_matrix(L_i_matrix)


def get_R_i(i: int) -> csr_matrix:
    tau_i = get_tau_i(i=i)
    lower_row = np.concatenate((np.zeros_like(tau_i), tau_i), axis=1)
    upper_row = np.zeros_like(lower_row)

    R_i_matrix = np.concatenate((upper_row, lower_row), axis=0)
    return csr_matrix(R_i_matrix)


def get_L_squared() -> csr_matrix:
    L_squared = csr_matrix((4, 4))
    for i in range(1, 4):
        L_squared += get_L_i(i=i) @ get_L_i(i=i)
    return L_squared


def get_R_squared() -> csr_matrix:
    R_squared = csr_matrix((4, 4))
    for i in range(1, 4):
        R_squared += get_R_i(i=i) @ get_R_i(i=i)
    return R_squared


