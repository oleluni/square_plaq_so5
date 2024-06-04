import numpy as np
from matplotlib import pyplot as plt
# matplotlib.use('pdf')

from analytic.eigs_analytic import lambda_n_analytic, asymptote_energy
from states_filtration.eigval_calc.eigen_solver import get_eigvals1_arr
from states_filtration.eigval_calc.eigen_solver import get_eigvals1


def plot_analytic_bauer(q, b_inv_values, is_all_plotted=True) -> None:
    if is_all_plotted:
        range_plot = range(2, int(4*q+2)+2, 2)
    else:
        range_plot = [int(4*q+2)]
    # PLOT ANALYTIC
    for n in range_plot:
        eig_vals = lambda_n_analytic(n=n, g0=2*np.sqrt(b_inv_values)) - lambda_n_analytic(n=2, g0=2*np.sqrt(b_inv_values))
        # plt.plot(g_values, eig_vals, color="green")
        plt.plot(b_inv_values, eig_vals, color="black")

    # add labels
    plt.plot([], [], color="black", label="analytic")


def plot_asymptote_bauer(q, b_inv_values) -> None:
    # PLOT ASYMPTOTES
    for j in [k / 2 for k in range(int(2 * q) + 1)]:
        asympt_vals = asymptote_energy(j=j, g0=2*np.sqrt(b_inv_values))
        # plt.plot(g_values, asympt_vals, color="black", linestyle='--')
        plt.plot(b_inv_values, asympt_vals, color="gray", linestyle='--')
    # add labels
    plt.plot([], [], color="black", linestyle='--', label='free energy j(j+1)')


def plot_numeric_vs_g(q, b_inv_values, is_all_plotted=False, is_for_b_inv=False) -> None:
    if is_all_plotted:
        range_plot = enumerate([k / 2 + 0.5 for k in range(0, int(2 * q))])
    else:
        range_plot = enumerate([q])

    # PLOT NUMERIC
    colors = ['orange', 'blue', 'cyan', 'magenta', 'pink', 'brown']

    for i, _q in range_plot:
        Ei = get_eigvals1_arr(g_values=2*np.sqrt(b_inv_values), q=_q, is_for_b_inv=is_for_b_inv)
        Ei = np.real(Ei)# * (2**0.5)

        # choose color iteratively
        color = colors[i % len(colors)]
        # print(color, _q)

        # plot vacuum

        # plt.scatter(g_values, - Ei[0, :], label=f"-E0 (numeric q={_q})", color=color)
        plt.scatter(b_inv_values, - Ei[0, :], label=f"-E0 (numeric q={_q})", color=color, marker='o', facecolors='none')
        for row in range(1, int(2*_q) + 1):
            # print(row)
            plt.scatter(b_inv_values, Ei[row, :] - Ei[0, :], label=f"E{row}-E0 (numeric q={_q})", color=color, marker='o', facecolors='none')
            # plt.scatter(g_values, Ei[row, :] - Ei[0, :], label=f"E{row}-E0 (numeric q={_q})", color=color)


def plot_energy_error(q, g) -> None:
    # TODO: now for one g only, later enlarge for g_values
    q_range = [k / 2 + 0.5 for k in range(0, int(2 * q))]
    for _q in q_range:
        phys_states = np.load(f'phys_states/phys_states_cache/phys_states_q{_q}.npy')
        E_num = np.array([sorted(get_eigvals1(g=g, q=_q, phys_states=phys_states))[i] for i in range(int(2*_q)+1)])
        E_num = np.real(E_num) * np.sqrt(2)

        E_an = np.array([lambda_n_analytic(n=n, g0=g) for n in range(2, int(4 * _q + 2) + 2, 2)])

        E_err = np.abs((E_num - E_an) / E_an) * 100 # in percents

        _q_range = np.full(int(2*_q)+1, _q)
        plt.scatter(_q_range, E_err, marker='o')


def plot_energy_error_mass_gap(q, b_inv_values) -> None:
    # TODO: now for one g only, later enlarge for g_values
    q_range = [k / 2 + 0.5 for k in range(0, int(2 * q))]
    for b_inv in b_inv_values:
        # phys_states = np.load(f'phys_states/phys_states_cache/phys_states_q{_q}.npy')

        E_num1 = np.array([get_eigvals1(g=(2*(b_inv**0.5)), q=_q,
                              phys_states=np.load(f'phys_states/phys_states_cache/phys_states_q{_q}.npy'))[1]
                 for _q in q_range])
        E_num0 = np.array([get_eigvals1(g=(2*(b_inv**0.5)), q=_q,
                              phys_states=np.load(f'phys_states/phys_states_cache/phys_states_q{_q}.npy'))[0]
                 for _q in q_range])
        E_num = E_num1 - E_num0
        E_num = np.real(E_num) * np.sqrt(2)

        E_an = np.full(int(2*q), lambda_n_analytic(n=4, g0=(2*(b_inv**0.5)) ))

        E_err = np.abs((E_num - E_an) / E_an) * 100  # in percents

        plt.plot(q_range, E_err, marker='o', fillstyle='none', label=f'b_inv={b_inv}')









