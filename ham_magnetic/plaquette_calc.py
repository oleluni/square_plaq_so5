import numpy as np

import scipy.sparse as sparse
from scipy.sparse import kron
from scipy.sparse import csr_matrix

from ham_magnetic.link_operator import get_u_ab as u_ab


def get_plaquette_ab(a, b) -> csr_matrix:
    n_sites = 4
    n_col = 2
    N = 4

    plaq: csr_matrix = sparse.csr_matrix((N**n_sites, N**n_sites), dtype=np.complex128)
    for a1 in range(n_col):
        for a2 in range(n_col):
            for a3 in range(n_col):
                plaq += kron(u_ab(a, a1), kron(u_ab(a1, a2), kron(u_ab(a3, a2).H, u_ab(b, a3).H)))
    return plaq


def trace_plaq_color() -> csr_matrix:
    trace = get_plaquette_ab(a=0, b=0) + get_plaquette_ab(a=1, b=1)
    return trace

