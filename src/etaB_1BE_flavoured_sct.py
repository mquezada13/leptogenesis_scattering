"""Two- and three-flavoured Boltzmann equations with Delta-L=1 scattering.

The prescription matches the repository's 1DME and 1BE1F extensions:
``D+S`` equilibrates N1, the CP-odd source contains ``D`` only, and inverse
decay washout is replaced by the scattering-corrected washout.
"""

import numpy as np
import ulysses
from odeintw import odeintw

from src.scattering import scat_Ss, scat_St, washout_sct


class _FlavouredScatteringBase(ulysses.ULSBase):
    def _rates(self, z, k, n1):
        if z != self._currz or z == self.zmin:
            self._n1eq = self.N1Eq(z)
            self._decay = np.real(self.D1(k, z))
            self._washout = np.real(self.W1(k, z))
            mass1 = np.real(self.DM[0, 0])
            self._ss = np.real(scat_Ss(k, z))
            self._st = np.real(scat_St(k, z, mass1, self.MH))
            self._total_decay = self._decay + self._ss + self._st
            self._currz = z
        total_washout = washout_sct(
            self._washout, self._decay, n1, self._n1eq, self._ss, self._st
        )
        return self._n1eq, self._decay, self._total_decay, total_washout


class EtaB_1BE2F_sct(_FlavouredScatteringBase):
    """Two-flavoured BE (tau, tau-perp) including scattering."""

    def shortname(self): return "1BE2F_sct"
    def flavourindices(self): return [1, 2]
    def flavourlabels(self): return [r"$N_{\tau\tau}$", r"$N_{\tau^\perp\tau^\perp}$"]

    def RHS(self, y0, z, epstt, epsmm, epsee, c1t, c1m, c1e, k):
        n1, ntt, nbb = y0
        n1eq, decay, total_decay, total_washout = self._rates(z, k, n1)
        p_tau = np.abs(c1t) ** 2
        p_perp = np.abs(c1e) ** 2 + np.abs(c1m) ** 2
        departure = n1 - n1eq
        return [
            -total_decay * departure,
            epstt * decay * departure - p_tau * total_washout * ntt,
            (epsee + epsmm) * decay * departure - p_perp * total_washout * nbb,
        ]

    @property
    def EtaB(self):
        epsilon = [np.real(self.epsilon1ab(a, a)) for a in range(3)]
        projectors = [self.c1a(a) for a in range(3)]
        params = (epsilon[2], epsilon[1], epsilon[0],
                  projectors[2], projectors[1], projectors[0], np.real(self.k1))
        initial = np.zeros(3, dtype=np.complex128)
        solution = odeintw(self.RHS, initial, self.zs, args=params,
                           atol=1e-10, rtol=1e-10)
        self.setEvolData(solution)
        return self.ys[-1, -1]


class EtaB_1BE3F_sct(_FlavouredScatteringBase):
    """Three-flavoured BE (tau, mu, e) including scattering."""

    def shortname(self): return "1BE3F_sct"
    def flavourindices(self): return [1, 2, 3]
    def flavourlabels(self): return [r"$N_{\tau\tau}$", r"$N_{\mu\mu}$", r"$N_{ee}$"]

    def RHS(self, y0, z, epstt, epsmm, epsee, c1t, c1m, c1e, k):
        n1, ntt, nmm, nee = y0
        n1eq, decay, total_decay, total_washout = self._rates(z, k, n1)
        departure = n1 - n1eq
        return [
            -total_decay * departure,
            epstt * decay * departure - np.abs(c1t) ** 2 * total_washout * ntt,
            epsmm * decay * departure - np.abs(c1m) ** 2 * total_washout * nmm,
            epsee * decay * departure - np.abs(c1e) ** 2 * total_washout * nee,
        ]

    @property
    def EtaB(self):
        epsilon = [np.real(self.epsilon1ab(a, a)) for a in range(3)]
        projectors = [self.c1a(a) for a in range(3)]
        params = (epsilon[2], epsilon[1], epsilon[0],
                  projectors[2], projectors[1], projectors[0], np.real(self.k1))
        initial = np.zeros(4, dtype=np.complex128)
        solution = odeintw(self.RHS, initial, self.zs, args=params,
                           atol=1e-10, rtol=1e-10)
        self.setEvolData(solution)
        return self.ys[-1, -1]
