import matplotlib
import numpy as np
from matplotlib import pyplot as plt
# matplotlib.use('pdf')
import constants as const

from analytic.eigs_analytic import lambda_n_analytic, asymptote_energy
from states_filtration.eigval_calc.eigen_solver import get_eigvals1_arr
from states_filtration.eigval_calc.eigen_solver import get_eigvals1


def plot_analytic_bauer_mass_gap(q, b_inv_values) -> None:
    n = int(4*q+2)

    # PLOT ANALYTIC
    eig_vals = lambda_n_analytic(n=n, g0=2*np.sqrt(b_inv_values)) - lambda_n_analytic(n=2, g0=2*np.sqrt(b_inv_values))
    # plt.plot(g_values, eig_vals, color="green")
    plt.plot(b_inv_values, eig_vals, color="black")

    # add labels
    plt.plot([], [], color="black", label="analytic")


def plot_asymptote_bauer(q, b_inv_values) -> None:
    # PLOT ASYMPTOTES
    asympt_vals = asymptote_energy(j=q, g0=2 * np.sqrt(b_inv_values))
    plt.plot(b_inv_values, asympt_vals, color="gray", linestyle='--')

    # add labels
    plt.plot([], [], color="black", linestyle='--', label='free energy j(j+1)')


def plot_mass_gap(q, b_inv_values, is_for_b_inv=True) -> None:
    q_range = [k / 2 + 0.5 for k in range(0, int(2 * q))]
    colors = ['orange', 'red', 'blue', 'magenta', 'pink', 'brown']
    markers = ['^', '+', 's', 'o']
    for idx, _q in enumerate(q_range):

        Ei = get_eigvals1_arr(g_values=2*np.sqrt(b_inv_values), q=_q, is_for_b_inv=is_for_b_inv)
        Ei = np.real(Ei) #* (2 ** 0.5)
        mass_gap = Ei[1, :] - Ei[0, :]

        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]

        plt.plot(b_inv_values, mass_gap, marker=marker, color=color, linestyle='-', fillstyle='none', label=f'q={_q}')

def make_mass_gap_plot(q, b_inv_values, is_all_plotted=False, is_for_b_inv=True) -> None:
    plot_analytic_bauer_mass_gap(q=0.5, b_inv_values=b_inv_values)
    plot_asymptote_bauer(q=0.5, b_inv_values=b_inv_values)

    plot_mass_gap(q=q, b_inv_values=b_inv_values, is_for_b_inv=is_for_b_inv)

    # plt.xticks([0.2, 0.4, 0.6, 0.8])

    plt.xlabel(r'$1 / \beta$', fontsize=14)
    plt.ylabel('$E_1-E_0$', fontsize=14)

    plt.title(fr"Mass gap $E_1-E_0$ vs. $1/\beta$ (up to q={q})", fontfamily='serif')
    plt.grid(True)
    plt.legend()




