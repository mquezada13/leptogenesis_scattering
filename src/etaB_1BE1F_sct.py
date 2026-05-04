"""
1BE1F + scattering solver.
==========================
Boltzmann equation with 1 RHN, no flavour effects, plus ΔL=1 scattering.
Follows TY's implementation based on hep-ph/0401240.

Same scattering prescription as 1DME_sct:
  - N1 equation uses D+S
  - Asymmetry source uses D only
  - Washout uses Eq. (84)

Without flavour effects, the asymmetry is a single number N_{B-L}
instead of a 3x3 density matrix.

Also contains EtaB_1BE1F_custom: same as ULYSSES' 1BE1F but with
adjustable tolerances for numerical stability checks.
"""
import ulysses
import numpy as np
from odeintw import odeintw
from ulysses.etab1BE1F import fast_RHS
from src.scattering import scat_Ss, scat_St, washout_sct


class EtaB_1BE1F_sct(ulysses.ULSBase):
    def shortname(self): return "1BE1F_sct"
    def flavourindices(self): return [1]
    def flavourlabels(self): return ["$NBL$"]

    def RHS(self, y0, z, epstt, epsmm, epsee, k):
        """
        Right-hand side of the 1BE1F+scattering ODE system.

        State vector y0 = [N1, N_{B-L}]
        Only 2 equations (no flavour structure).
        """
        if z != self._currz or z == self.zmin:
            self._n1eq = self.N1Eq(z)
            self._d    = np.real(self.D1(k, z))
            self._w1   = np.real(self.W1(k, z))
            M1         = np.real(self.DM[0, 0])
            self._Ss   = np.real(scat_Ss(k, z))
            self._St   = np.real(scat_St(k, z, M1, self.MH))
            self._ds   = self._d + self._Ss + self._St
            self._currz = z

        N1, NBL = y0
        n1eq = self._n1eq
        d, ds = self._d, self._ds

        # Total CP asymmetry (sum over flavours)
        eps = epstt + epsmm + epsee

        # Washout with scattering correction (Eq. 84)
        w_sct = washout_sct(self._w1, d, N1, n1eq, self._Ss, self._St)

        # N1 abundance: uses D+S
        rhs1 = -ds * (N1 - n1eq)

        # B-L asymmetry: source = eps*D, washout = W_sct
        rhs2 = eps * d * (N1 - n1eq) - w_sct * NBL

        return [rhs1, rhs2]

    @property
    def EtaB(self):
        """Solve the ODE system and return the final baryon asymmetry η_B."""
        epstt = np.real(self.epsilon1ab(2, 2))
        epsmm = np.real(self.epsilon1ab(1, 1))
        epsee = np.real(self.epsilon1ab(0, 0))

        k  = np.real(self.k1)
        y0 = np.array([0+0j, 0+0j], dtype=np.complex128)
        params = np.array([epstt, epsmm, epsee, k], dtype=np.complex128)

        ys = odeintw(self.RHS, y0, self.zs, args=tuple(params), atol=1e-14, rtol=1e-10)
        self.setEvolData(ys)
        return self.ys[-1][-1]


class EtaB_1BE1F_custom(ulysses.ULSBase):
    """
    Same as ULYSSES' 1BE1F but with adjustable tolerances.
    Useful for numerical stability checks.
    """
    def shortname(self): return "1BE1F"
    def flavourindices(self): return [1]
    def flavourlabels(self): return ["$NBL$"]

    def RHS(self, y0, z, epstt, epsmm, epsee, k):
        if z != self._currz or z == self.zmin:
            self._d    = np.real(self.D1(k, z))
            self._w1   = np.real(self.W1(k, z))
            self._n1eq = self.N1Eq(z)
            self._currz = z
        return fast_RHS(y0, self._d, self._w1, self._n1eq, epstt, epsmm, epsee)

    @property
    def EtaB(self):
        epstt = np.real(self.epsilon1ab(2, 2))
        epsmm = np.real(self.epsilon1ab(1, 1))
        epsee = np.real(self.epsilon1ab(0, 0))

        k  = np.real(self.k1)
        y0 = np.array([0+0j, 0+0j], dtype=np.complex128)
        params = np.array([epstt, epsmm, epsee, k], dtype=np.complex128)

        ys = odeintw(self.RHS, y0, self.zs, args=tuple(params), atol=1e-10, rtol=1e-10)
        self.setEvolData(ys)
        return self.ys[-1][-1]

# ------------------------------------------------------------------------------------------------------------
# The following class is a variant of the above (without scattering) using even tighter tolerances for testing
# ------------------------------------------------------------------------------------------------------------
class EtaB_1BE1F_custom(ulysses.ULSBase):
    """
    Same as 1BE1F but with tighter tolerances.
    """
    def shortname(self): return "1BE1F"
    def flavourindices(self): return [1]
    def flavourlabels(self): return ["$NBL$"]

    def RHS(self, y0, z, epstt, epsmm, epsee, k):
        if z != self._currz or z == self.zmin:
            self._d    = np.real(self.D1(k, z))
            self._w1   = np.real(self.W1(k, z))
            self._n1eq = self.N1Eq(z)
            self._currz = z
        return fast_RHS(y0, self._d, self._w1, self._n1eq, epstt, epsmm, epsee)

    @property
    def EtaB(self):
        epstt = np.real(self.epsilon1ab(2, 2))
        epsmm = np.real(self.epsilon1ab(1, 1))
        epsee = np.real(self.epsilon1ab(0, 0))

        k  = np.real(self.k1)
        y0 = np.array([0+0j, 0+0j], dtype=np.complex128)

        params = np.array([epstt, epsmm, epsee, k], dtype=np.complex128)

        ys = odeintw(self.RHS, y0, self.zs, args=tuple(params), atol=1e-10, rtol=1e-10)
        self.setEvolData(ys)
        return self.ys[-1][-1]