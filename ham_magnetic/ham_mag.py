from scipy.sparse import csr_matrix
from ham_magnetic.plaquette_calc import trace_plaq_color as trace


def ham_mag_plaq(g: float) -> csr_matrix:
    return (- 2.0 / (g**2)) * trace() # for multiplier 2.0 is kinda ok


