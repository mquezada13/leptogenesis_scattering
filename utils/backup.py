"""
Backup: original scattering models using ULYSSES built-in DS/scat functions.
============================================================================
These were our first implementation before TY's feedback.
They use ULYSSES' approximations (self.DS and self.scat) instead of
the numerical integrals from the Pedestrians paper.

Differences from TY's implementation:
  - DS is used for BOTH the N1 equation AND the asymmetry source
    (TY uses D+S for N1 but only D for asymmetry source)
  - scat(z)*W1 is used for washout
    (TY uses the full Eq. 84: W1*(1 + 1/D*(2N1/N1eq*Ss + 4St)))

Keep these for comparison / debugging.
"""
import ulysses
import numpy as np
from odeintw import odeintw
from ulysses.etab1DME import fast_RHS as fast_RHS_DME
from ulysses.etab1BE1F import fast_RHS as fast_RHS_BE


class EtaB_1DME_sct_backup(ulysses.ULSBase):
    """1DME + scattering using ULYSSES built-in DS and scat."""

    def shortname(self): return "1DME_sct_backup"
    def flavourindices(self): return [1, 2, 3]
    def flavourlabels(self): return ["$N_{\\tau\\tau}$", "$N_{\\mu\\mu}$", "$N_{ee}$"]

    def RHS(self, y0, z, epstt, epsmm, epsee, epstm, epste, epsme, c1t, c1m, c1e, k):
        if z != self._currz or z == self.zmin:
            self._d       = np.real(self.DS(k, z))
            self._w1      = self.scat(z) * np.real(self.W1(k, z))
            self._n1eq    = self.N1Eq(z)
            self._currz   = z

        widtht = 485e-10 * self.MP / self.M1
        widthm = 1.7e-10 * self.MP / self.M1

        return fast_RHS_DME(y0, self._d, self._w1, self._n1eq,
                        epstt, epsmm, epsee, epstm, epste, epsme,
                        c1t, c1m, c1e, widtht, widthm)

    @property
    def EtaB(self):
        epstt = np.real(self.epsilon1ab(2, 2))
        epsmm = np.real(self.epsilon1ab(1, 1))
        epsee = np.real(self.epsilon1ab(0, 0))
        epstm =         self.epsilon1ab(2, 1)
        epste =         self.epsilon1ab(2, 0)
        epsme =         self.epsilon1ab(1, 0)

        c1t, c1m, c1e = self.c1a(2), self.c1a(1), self.c1a(0)
        k  = np.real(self.k1)
        y0 = np.array([0+0j]*7, dtype=np.complex128)
        params = np.array([epstt, epsmm, epsee, epstm, epste, epsme, c1t, c1m, c1e, k], dtype=np.complex128)

        ys, _ = odeintw(self.RHS, y0, self.zs, args=tuple(params), full_output=True)
        self.setEvolData(ys)
        return self.ys[-1][-1]


class EtaB_1BE1F_sct_backup(ulysses.ULSBase):
    """1BE1F + scattering using ULYSSES built-in DS and scat."""

    def shortname(self): return "1BE1F_sct_backup"
    def flavourindices(self): return [1]
    def flavourlabels(self): return ["$NBL$"]

    def RHS(self, y0, z, epstt, epsmm, epsee, k):
        if z != self._currz or z == self.zmin:
            self._d    = np.real(self.DS(k, z))
            self._w1   = self.scat(z) * np.real(self.W1(k, z))
            self._n1eq = self.N1Eq(z)
            self._currz = z
        return fast_RHS_BE(y0, self._d, self._w1, self._n1eq, epstt, epsmm, epsee)

    @property
    def EtaB(self):
        epstt = np.real(self.epsilon1ab(2, 2))
        epsmm = np.real(self.epsilon1ab(1, 1))
        epsee = np.real(self.epsilon1ab(0, 0))

        k  = np.real(self.k1)
        y0 = np.array([0+0j, 0+0j], dtype=np.complex128)
        params = np.array([epstt, epsmm, epsee, k], dtype=np.complex128)

        ys = odeintw(self.RHS, y0, self.zs, args=tuple(params), atol=1e-14, rtol=1e-10)
        self.setEvolData(ys)
        return self.ys[-1][-1]