import numpy as np
import scipy.sparse as sparse
from scipy.sparse import csr_matrix
import scipy

import constants as const
from tools_custom import tools_build_opers as tools
# from tools_custom.gram_schmidt import ortho_norm_vecs as gs
from tools_custom.gram_schmidt import gram_schmidt_complex as gs_complex
from tools_custom.gram_schmidt import gram_schmidt_complex_ndarray as gs_arr

from ham_electric.ham_el import ham_el_plaq
from ham_magnetic.ham_mag import ham_mag_plaq

from states_filtration import gauss
from states_filtration.gauge_fixing import get_phys_states
from states_filtration.gauge_fixing import gz_filtered_states
from states_filtration.gauge_fixing import projector
from states_filtration.gauge_fixing import projector_phys

import os

def get_eigvals(g: float) -> np.ndarray:
    ham = ham_el_plaq(g=g) + ham_mag_plaq(g=g)
    # ham = ham_el_plaq(g=g)
    # first projection
    proj = projector()
    ham_subspace = proj.dot(ham.dot(proj.H))

    # second projection
    proj_phys = projector_phys()
    ham_phys = proj_phys.dot(ham_subspace.dot(proj_phys.T.conj()))

    eigenvalues = np.linalg.eig(ham_phys)[0]

    return np.sort(np.real(eigenvalues))


def get_eigvals_mag(g: float) -> np.ndarray:
    ham = ham_mag_plaq(g=g) #+ ham_el_plaq(g=g)

    # first projection
    proj = projector()
    ham_subspace = proj.dot(ham.dot(proj.H))

    # second projection
    proj_phys = projector_phys()
    ham_phys = proj_phys.dot(ham_subspace.dot(proj_phys.T.conj()))

    eigenvalues = np.linalg.eig(ham_phys)[0]

    return np.sort(np.real(eigenvalues))
