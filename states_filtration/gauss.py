import numpy as np
from ham_electric import momenta

import scipy.sparse as sparse
from tools_custom import tools_build_opers as tools
import constants as const

N = const.STATES_NUMBER
n_sites = const.PLAQUETTE_SITES_NUMBER


def gauss_a_at_site(a: int, s_num: int) -> sparse.csr_matrix:
    if a not in (1, 2, 3):
        raise ValueError("Argument 'a' can only take values from 1 to 3.")

    L_a = momenta.get_L_i(i=a)
    R_a = momenta.get_R_i(i=a)

    unity = sparse.eye(N, format='csr')

    if s_num == 1:
        list_1 = [L_a, unity, unity, unity]
        list_2 = [unity, unity, unity, L_a]
    elif s_num == 2:
        list_1 = [R_a, unity, unity, unity]
        list_2 = [unity, L_a, unity, unity]
    elif s_num == 3:
        list_1 = [unity, R_a, unity, unity]
        list_2 = [unity, unity, R_a, unity]
    elif s_num == 4:
        list_1 = [unity, unity, L_a, unity]
        list_2 = [unity, unity, unity, R_a]
    else:
        raise ValueError("Argument 's_num' can only take values from 1 to 4.")

    result_sum = sparse.csr_matrix((N ** n_sites, N ** n_sites), dtype=np.float64)

    result_sum += tools.tensor_prod(list_1) + tools.tensor_prod(list_2)

    return result_sum


def get_g_squared() -> sparse.csr_matrix:
    g_2 = sparse.csr_matrix((N ** n_sites, N ** n_sites), dtype=np.float64)
    for s_num in range(1, n_sites+1):
        for a in range(1, 4):
            g_2 += gauss_a_at_site(a=a, s_num=s_num) @ gauss_a_at_site(a=a, s_num=s_num)

    return g_2


