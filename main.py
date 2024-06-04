import time
import constants as const

import numpy as np
import scipy.sparse as sparse

from ham_electric.ham_el import ham_el_plaq
from ham_magnetic.ham_mag import ham_mag_plaq

import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt


from states_filtration.eigval_calc.eigen_solver import get_eigvals
from states_filtration.eigval_calc.eigen_solver import get_eigvals_mag

if __name__ == "__main__":
    start_time = time.time()


    # STAGE 1 - DEFINITIONS
    # q = 0.5
    num_points = 20

    # g2_values = np.linspace(0.25, 0.8, num_points)
    # g_values = np.sqrt(g2_values)

    b_inv_values = np.linspace(0.1, 0.8, num_points)
    g0 = 2 * np.sqrt(b_inv_values)

    E0 = np.array([get_eigvals(g=2 * np.sqrt(b_inv))[0] for b_inv in b_inv_values])
    E1 = np.array([get_eigvals(g=2 * np.sqrt(b_inv))[1] for b_inv in b_inv_values])

    E0_mag = np.array([get_eigvals_mag(g=2 * np.sqrt(b_inv))[0] for b_inv in b_inv_values])
    E1_mag = np.array([get_eigvals_mag(g=2 * np.sqrt(b_inv))[1] for b_inv in b_inv_values])

    plt.scatter(b_inv_values, -E0_mag - 3.0*b_inv_values , label=r"$-E0_{\mathrm{mag}}+c\cdot \beta^{-1}$", color='red', marker='^', facecolors='none')
    plt.scatter(b_inv_values, E1_mag-E0_mag, label=f"E1_mag-E0_mag", color='green', marker='o', facecolors='none')

    plt.scatter(b_inv_values, -E0, label=f"-E0", color='blue', marker='^', facecolors='none')
    plt.scatter(b_inv_values, E1-E0, label=f"E1-E0", color='orange', marker='o', facecolors='none')

    plt.xlabel(r'1 / $\beta$')
    plt.ylabel('E')
    plt.legend()
    plt.title(r'E vs. $1 / \beta$ for SO(5) plaquette')


    # plt.show()
    # TODO: 1) compare with analytic, 2) asymptotes, 3) with SU(2)
    plt.savefig(f"E_vs_b_inv_so5_const_shift_proof.pdf")


    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time is {elapsed_time} s")

