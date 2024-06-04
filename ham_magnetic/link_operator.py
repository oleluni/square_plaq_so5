import numpy as np
from scipy.sparse import csr_matrix


def get_tau_i(i: int) -> np.ndarray:
    tau = np.array([[[1, 0], [0, 1]], [[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]])
    return tau[i]


def get_u_i(i: int) -> np.ndarray:
    if i not in [0, 1, 2, 3]:
        raise ValueError("Input 'i' must be one of 0, 1, 2, or 3.")

    tau_i = get_tau_i(i=i)
    upper_row = np.concatenate((np.zeros_like(tau_i), tau_i), axis=1)
    lower_row = np.concatenate((tau_i, np.zeros_like(tau_i)), axis=1)

    u_i_matrix = np.concatenate((upper_row, lower_row), axis=0)

    if i!=0:
        u_i_matrix[:2, 2:] = u_i_matrix[2:, :2] * (-1j)
        u_i_matrix[2:, :2] = u_i_matrix[2:, :2] * (+1j)
    return u_i_matrix


def get_u_matrix() -> np.ndarray:
    u_i_list = [np.kron(get_tau_i(i=i), get_u_i(i=i)) * (1j if i in (1, 2, 3) else 1) for i in range(0, 4)]
    u_matrix = sum(u_i_list)

    return u_matrix


def get_u_ab(a: int, b: int) -> csr_matrix:
    # print(a, b)
    if (a!=0 and a!=1) or (b!=0 and b!=1):
        raise ValueError("Input indices `a` and `b` should take values 0 or 1 only.")
    u_matrix = get_u_matrix()

    N = int(u_matrix.shape[0] / 2)

    start_row = a * N
    end_row = (a + 1) * N
    start_col = b * N
    end_col = (b + 1) * N

    u_ab_matrix = u_matrix[start_row:end_row, start_col:end_col]
    return csr_matrix(u_ab_matrix)


def get_u_dag_matrix() -> np.ndarray:
    u_i_list = [np.kron(get_tau_i(i=i), get_u_i(i=i)) * (-1j if i in (1, 2, 3) else 1) for i in range(0, 4)]
    u_matrix = sum(u_i_list)

    return u_matrix


def get_u_dag_ab(a: int, b: int) -> csr_matrix:
    if (a!=0 and a!=1) or (b!=0 and b!=1):
        raise ValueError("Input indices `a` and `b` should take values 0 or 1 only.")
    u_dag_matrix = get_u_dag_matrix()

    N = int(u_dag_matrix.shape[0] / 2)

    start_row = a * N
    end_row = (a + 1) * N
    start_col = b * N
    end_col = (b + 1) * N

    u_ab_matrix = u_dag_matrix[start_row:end_row, start_col:end_col]
    return csr_matrix(u_ab_matrix)

# u_00  =get_u_ab(0, 0)
# u_01  =get_u_ab(0, 1)
# u_10  =get_u_ab(1, 0)
# u_11  =get_u_ab(1, 1)
#
# u_dag_00  =get_u_dag_ab(0, 0)
# u_dag_01  =get_u_dag_ab(0, 1)
# u_dag_10  =get_u_dag_ab(1, 0)
# u_dag_11  =get_u_dag_ab(1, 1)
#
# print(u_01-u_dag_10.H)



