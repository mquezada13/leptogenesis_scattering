"""
Scattering functions following Pedestrians paper (hep-ph/0401240).
Eqs. (73), (84), (221), (222), (223).

These are called by the 1DME_sct and 1BE1F_sct models.
"""
import numpy as np
from scipy.integrate import quad
from scipy.special import kn as sp_kn


def my_kn1(x):
    return sp_kn(1, np.real(x))

def my_kn2(x):
    return sp_kn(2, np.real(x))


def scat_Ss(k1, z):
    """S_{phi,s} from Eqs. (73) and (222-223)."""
    def func_s(x):
        return ((x - 1) / x) ** 2

    def integrand_s(x, z):
        return func_s(x / z**2) * np.sqrt(x) * my_kn1(np.sqrt(x))

    def f_s(z):
        ans, _ = quad(integrand_s, z**2, np.inf, args=(z,))
        return ans / z**2 / my_kn2(z)

    return 0.1 * k1 / 6 * f_s(z)


def scat_St(k1, z, M1, MH):
    """S_{phi,t} from Eqs. (73) and (221, 223)."""
    a_h = (MH / M1) ** 2

    def func_t(x):
        return (x - 1) / x * (
            (x - 2 + 2 * a_h) / (x - 1 + a_h)
            + (1 - 2 * a_h) / (x - 1) * np.log((x - 1 + a_h) / a_h)
        )

    def integrand_t(x, z):
        return func_t(x / z**2) * np.sqrt(x) * my_kn1(np.sqrt(x))

    def f_t(z):
        ans, _ = quad(integrand_t, z**2, np.inf, args=(z,))
        return ans / z**2 / my_kn2(z)

    return 0.1 * k1 / 3 * f_t(z)


def D_plus_S(k1, z, M1, MH):
    """D + S_s + S_t (source term for N1 equation)."""
    from ulysses.ulsbase import ULSBase
    d = k1 * z * my_kn1(z) / my_kn2(z)
    Ss = scat_Ss(k1, z)
    St = scat_St(k1, z, M1, MH)
    return d + Ss + St


def washout_sct(w1, d, N1, n1eq, Ss, St):
    """
    Washout with scattering correction: Eq. (84).
    W_sct = W1 * (1 + 1/D * (2*N1/N1eq * Ss + 4*St))
    """
    if d > 0 and n1eq > 0:
        return w1 * (1 + (1 / d) * (2 * N1 / n1eq * Ss + 4 * St))
    else:
        return w1