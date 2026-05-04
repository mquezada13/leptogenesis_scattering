"""
Scattering functions for leptogenesis Boltzmann equations.
==========================================================
Implements ΔL=1 scattering corrections following the "Pedestrians" paper:
    Buchmüller, Di Bari, Plümacher (2005), hep-ph/0401240.

These functions are used by the 1DME_sct and 1BE1F_sct solvers.

Physical picture
-----------------
In the early universe, the right-handed neutrino N1 can interact via:
  - Decays/inverse decays:  N1 <-> l + Φ     (captured by D and W1)
  - Scatterings:            N1 + X <-> Y + Z  (captured by S_s, S_t)

The scattering corrections enter the Boltzmann equations in two places:
  1. N1 production:  dN1/dz = -(D + S)(N1 - N1_eq)     [Eq. (9)]
  2. Washout:        W -> W * (1 + 1/D * (2N1/N1eq * Ss + 4St))  [Eq. (84)]

Note: the scattering does NOT enter the source term of the asymmetry
equation — that still uses D only, not D+S. See Eq. (10).

Functions
---------
  scat_Ss  :  S-channel Higgs scattering  (Eq. 222-223)
  scat_St  :  T-channel Higgs scattering  (Eq. 221, 223)
  washout_sct : Modified washout with scattering (Eq. 84)
"""
import numpy as np
from scipy.integrate import quad
from scipy.special import kn as sp_kn


def my_kn1(x):
    """Modified Bessel function K_1(x)."""
    return sp_kn(1, np.real(x))


def my_kn2(x):
    """Modified Bessel function K_2(x)."""
    return sp_kn(2, np.real(x))


def scat_Ss(k1, z):
    """
    S-channel Higgs scattering rate S_{phi,s}.

    Computed from Eqs. (73), (222), and (223) of hep-ph/0401240.
    The prefactor 0.1 comes from K_s ≈ 0.1 * K  [Eq. (72)].

    Parameters
    ----------
    k1 : float
        Decay parameter K = Gamma1 / H(T=M1).
    z : float
        Dimensionless inverse temperature z = M1 / T.

    Returns
    -------
    float
        S-channel scattering rate at temperature T = M1/z.
    """
    # Eq. (222): f_{phi,s}(x) = ((x-1)/x)^2
    def func_s(x):
        return ((x - 1) / x) ** 2

    # Integrand of Eq. (223): f_{phi,s}(ψ/z²) * sqrt(ψ) * K1(sqrt(ψ))
    def integrand_s(psi, z):
        return func_s(psi / z**2) * np.sqrt(psi) * my_kn1(np.sqrt(psi))

    # Eq. (223): thermally averaged cross section
    def f_s(z):
        ans, _ = quad(integrand_s, z**2, np.inf, args=(z,))
        return ans / z**2 / my_kn2(z)

    # At large z, Bessel functions decay exponentially -> return 0
    if z > 500:
        return 0.0

    # Eq. (73): S_s = K_s/6 * f_s(z), with K_s ≈ 0.1*K
    return 0.1 * k1 / 6 * f_s(z)


def scat_St(k1, z, M1, MH):
    """
    T-channel Higgs scattering rate S_{phi,t}.

    Computed from Eqs. (73), (221), and (223) of hep-ph/0401240.

    Parameters
    ----------
    k1 : float
        Decay parameter K = Gamma1 / H(T=M1).
    z : float
        Dimensionless inverse temperature z = M1 / T.
    M1 : float
        Mass of lightest right-handed neutrino [GeV].
    MH : float
        Higgs mass [GeV].

    Returns
    -------
    float
        T-channel scattering rate at temperature T = M1/z.
    """
    a_h = (MH / M1) ** 2  # Higgs-to-N1 mass ratio squared

    # Eq. (221): f_{phi,t}(x)
    def func_t(x):
        return (x - 1) / x * (
            (x - 2 + 2 * a_h) / (x - 1 + a_h)
            + (1 - 2 * a_h) / (x - 1) * np.log((x - 1 + a_h) / a_h)
        )

    # Integrand of Eq. (223)
    def integrand_t(psi, z):
        return func_t(psi / z**2) * np.sqrt(psi) * my_kn1(np.sqrt(psi))

    # Eq. (223): thermally averaged cross section
    def f_t(z):
        ans, _ = quad(integrand_t, z**2, np.inf, args=(z,))
        return ans / z**2 / my_kn2(z)

    # At large z, Bessel functions decay exponentially -> return 0
    if z > 500:
        return 0.0

    # Eq. (73): S_t = K_s/3 * f_t(z), with K_s ≈ 0.1*K
    return 0.1 * k1 / 3 * f_t(z)


def washout_sct(w1, d, N1, n1eq, Ss, St):
    """
    Washout rate with ΔL=1 scattering correction.

    Implements Eq. (84) of hep-ph/0401240:
        W_sct = W_ID * (1 + 1/D * (2*N1/N1eq * S_s + 4*S_t))

    where W_ID is the inverse-decay washout. The scattering adds
    two contributions:
      - S_s term: proportional to N1 abundance (vanishes in equilibrium)
      - S_t term: always present (does not depend on N1)

    Parameters
    ----------
    w1 : float
        Inverse-decay washout rate W_ID.
    d : float
        Decay rate D.
    N1 : complex
        Current N1 abundance.
    n1eq : float
        Equilibrium N1 abundance.
    Ss : float
        S-channel scattering rate.
    St : float
        T-channel scattering rate.

    Returns
    -------
    float
        Total washout rate including scattering.
    """
    if d > 0 and n1eq > 0:
        return np.real(w1 * (1 + (1 / d) * (2 * N1 / n1eq * Ss + 4 * St)))
    else:
        return w1