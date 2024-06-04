import numpy as np
import scipy.sparse as sparse

from ham_electric.momenta import get_L_squared as L_sq
from ham_electric.momenta import get_R_squared as R_sq

def ham_el_plaq(g: float) -> sparse.csr_matrix:
    L_squared = L_sq()
    R_squared = R_sq()

    N = 4
    n_sites = 4

    input_arr = [sparse.diags([1], [0], shape=(N, N), format='csr', dtype=np.float16) for _ in range(n_sites)]
    result_sum = sparse.csr_matrix((N**n_sites, N**n_sites), dtype=np.float16)

    for i in range(len(input_arr)):
        # store original value at the i-th position
        original_value = input_arr[i].copy()
        # set L^2 as i-th element of the array
        input_arr[i] = L_squared + R_squared

        prod = input_arr[0]
        for j in range(1, len(input_arr)):
            prod = sparse.kron(prod, input_arr[j])

        # add to result_sum
        result_sum += prod
        # restore the original value at the i-th position
        input_arr[i] = original_value

    return 0.25 * (g**2) * result_sum


