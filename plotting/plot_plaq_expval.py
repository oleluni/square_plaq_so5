from plaq_expval.plaq_expval import plaq_vev
from matplotlib import pyplot as plt

import numpy as np

def plot_plaq_expval(q, b_inv_values, is_all_plotted=True) -> None:
    if is_all_plotted:
        range_plot = enumerate([k / 2 + 0.5 for k in range(0, int(2 * q))])
    else:
        range_plot = enumerate([q])
    # PLOT NUMERIC
    colors = ['orange', 'red', 'blue', 'magenta', 'pink', 'brown']
    markers = ['^', '+', 's', 'o']
    for i, _q in range_plot:
        plaq_vals = np.array([plaq_vev(_q, b_inv) for b_inv in b_inv_values])

        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]

        plt.plot(b_inv_values, plaq_vals, label=fr"$\left<P\right>$ (numeric q={_q})", color=color, marker=marker, fillstyle="none")


def make_plaq_expval_plot(q, b_inv_values, is_all_plotted=True) -> None:
    plot_plaq_expval(q=q, b_inv_values=b_inv_values, is_all_plotted=is_all_plotted)

    plt.xlabel(r'$1 / \beta$', fontsize=14)
    plt.ylabel(fr"$\left<P\right>$", fontsize=14)

    plt.title(fr"Plaquette expval vs. $1/\beta$ (up to q={q})", fontfamily='serif')
    plt.grid(True)
    plt.legend()
