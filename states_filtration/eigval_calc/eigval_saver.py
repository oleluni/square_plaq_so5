import numpy as np

from states_filtration.eigval_calc.eigen_solver import get_eigvals1


def save_eigvals_g(q, g_values, num_points) -> None:

    phys_states = np.load(f'phys_states/phys_states_cache/phys_states_q{q}.npy')

    Ei = np.array([[sorted(get_eigvals1(g=g, q=q, phys_states=phys_states))[i] for g in g_values] for i in range(int(2*q+1))])
    Ei = np.real(Ei)

    np.save(f'states_filtration/eigval_calc/eigvals_cache/eigvals{num_points}_q{q}.npy', Ei)

#TODO: fix the arguments
def save_eigvals_b_inv(q, num_b_inv_points=20, b_inv_start=0.1, b_inv_end=1.0) -> None:
    beta_inv = np.linspace(b_inv_start, b_inv_end, num_b_inv_points)
    g_values = 2 * np.sqrt(beta_inv)

    phys_states = np.load(f'phys_states_cache/phys_states_q{q}.npy')

    Ei = np.array([[sorted(get_eigvals1(g=g, q=q, phys_states=phys_states))[i] for g in g_values] for i in range(int(2*q+1))])
    Ei = np.real(Ei)

    np.save(f'vals/b_inv_vals/b_inv_eigs{num_b_inv_points}_q{q}.npy', Ei)

# num_points = 20
# q = 0.5
#
# b_inv_values = np.linspace(0.1, 1.0, num_points)
# g_values = 2 * (np.sqrt(b_inv_values))
#
# save_eigvals_g(q=0.5)
