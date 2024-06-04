import numpy as np
from matplotlib import pyplot as plt
# matplotlib.use('pdf')

from analytic.eigs_analytic import lambda_n_analytic, asymptote_energy
from states_filtration.eigval_calc.eigen_solver import get_eigvals1_arr


def plot_analytic_bauer_vacuum(b_inv_values) -> None:
    # PLOT ANALYTIC
    eig_vals = - lambda_n_analytic(n=2, g0=2*np.sqrt(b_inv_values))
    plt.plot(b_inv_values, eig_vals, color="black")

    # add labels
    plt.plot([], [], color="black", label="analytic")


def plot_asymptote_bauer_vacuum(b_inv_values) -> None:
    # PLOT ASYMPTOTES
    asympt_vals = asymptote_energy(j=0.0, g0=2*np.sqrt(b_inv_values))
    plt.plot(b_inv_values, asympt_vals, color="gray", linestyle='--')

    # add labels
    plt.plot([], [], color="black", linestyle='--', label='free energy j(j+1)')


def plot_vacuum_energy(q, b_inv_values, is_all_plotted=True, is_for_b_inv=True) -> None:
    if is_all_plotted:
        range_plot = enumerate([k / 2 + 0.5 for k in range(0, int(2 * q))])
    else:
        range_plot = enumerate([q])
    # PLOT NUMERIC
    colors = ['orange', 'red', 'blue', 'magenta', 'pink', 'brown']
    markers = ['^', '+', 's', 'o']
    for i, _q in range_plot:
        Ei = get_eigvals1_arr(g_values=2*np.sqrt(b_inv_values), q=_q, is_for_b_inv=is_for_b_inv)
        Ei = np.real(Ei) #* (2**0.5)

        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]

        plt.plot(b_inv_values, - Ei[0, :], label=f"$-E_0$ (numeric q={_q})", color=color, marker=marker, fillstyle="none")


def make_vacuum_plot(q, b_inv_values, is_all_plotted=True, is_for_b_inv=True) -> None:
    plot_analytic_bauer_vacuum(b_inv_values=b_inv_values)
    plot_asymptote_bauer_vacuum(b_inv_values=b_inv_values)

    plot_vacuum_energy(q=q, b_inv_values=b_inv_values, is_all_plotted=is_all_plotted, is_for_b_inv=is_for_b_inv)

    # plt.xticks([0.2, 0.4, 0.6, 0.8])

    plt.xlabel(r'$1 / \beta$', fontsize=14)
    plt.ylabel('$-E_0$', fontsize=14)

    plt.title(fr"Vacuum energy $-E_0$ vs. $1/\beta$ (up to q={q})", fontfamily='serif')
    plt.grid(True)
    plt.legend()


