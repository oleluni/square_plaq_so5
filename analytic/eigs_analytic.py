from scipy.special import mathieu_b


def lambda_n_analytic(n: int, g0):
    g = g0/((2)**(1/4))
    q = -8/(g**4)
    E = (g**2)/8 * (mathieu_b(n, q) - 4)# + 2 / (g**2)
    return E * (2**(0.5))

def asymptote_energy(j, g0) -> float:
    g = g0 / ((2) ** (1 / 4))
    E_asym = (j * (j + 1) * (g ** 2) * 2)
    return E_asym * (2**(0.5))

