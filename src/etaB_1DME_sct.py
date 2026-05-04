"""
etaB_1DME_sct — density-matrix equation solver with scattering (3-flavour)
===========================================================================
Extends the ULYSSES ``EtaB_1DME`` model by incorporating scattering
corrections to both the production and washout terms:

    D  ->  DS(k, z)           (Eq. 73  of hep-ph/0401240)
    W1 ->  scat(z) * W1(k, z) (Eq. 223 of hep-ph/0401240)

The density-matrix (DME) formulation tracks the full 3×3 lepton asymmetry
matrix N_ab, capturing quantum coherence between flavours.  The ODE system
evolves 7 variables as a function of z = M1/T:

    y0[0]     : N1 abundance
    y0[1..3]  : diagonal entries N_tt, N_mm, N_ee of the lepton asymmetry
    y0[4..6]  : off-diagonal entries N_tm, N_te, N_me (complex)

eta_B is computed from N_BL = N_tt + N_mm + N_ee at z = zmax.

Dependencies
------------
- ulysses            : base class ULSBase and helpers (DS, W1, scat, N1Eq, …)
- odeintw            : ODE integrator for complex-valued systems
- ulysses.etab1DME.fast_RHS : compiled right-hand side for the DME system
"""

import ulysses
import numpy as np
from odeintw import odeintw
from ulysses.etab1DME import fast_RHS


class EtaB_1DME_sct(ulysses.ULSBase):
    """Three-flavour density-matrix equation solver with scattering corrections.

    Inherits all parameter handling, epsilon/c1a computation, and ODE
    bookkeeping from ``ulysses.ULSBase``.  Compared to the vanilla
    ``EtaB_1DME`` model, the two changes are:

    * The source uses ``DS(k, z)`` instead of ``D(k, z)``.
    * The washout is multiplied by the scattering suppression ``scat(z)``.

    ULYSSES registry
    ----------------
    shortname      : "1DME_sct"
    flavour indices: [1, 2, 3]   (tau-tau, mu-mu, ee diagonal components)
    flavour labels : [r"$N_{\\tau\\tau}$", r"$N_{\\mu\\mu}$", r"$N_{ee}$"]
    """

    def shortname(self): return "1DME_sct"
    def flavourindices(self): return [1, 2, 3]
    def flavourlabels(self): return ["$N_{\\tau\\tau}$", "$N_{\\mu\\mu}$", "$N_{ee}$"]

    def RHS(self, y0, z, epstt, epsmm, epsee, epstm, epste, epsme, c1t, c1m, c1e, k):
        """Compute the right-hand side of the 1DME ODEs at a single z step.

        Caches intermediate quantities for the current z to avoid redundant
        evaluation when the integrator calls this method multiple times at
        the same step.

        Parameters
        ----------
        y0 : array-like, shape (7,), complex
            Current state vector:
            [N1, N_tt, N_mm, N_ee, N_tm, N_te, N_me].
        z : float
            Integration variable z = M1/T.
        epstt, epsmm, epsee : float
            Diagonal CP-asymmetry parameters (tau, mu, e).
        epstm, epste, epsme : complex
            Off-diagonal CP-asymmetry parameters.
        c1t, c1m, c1e : complex
            Flavour projectors of the N1 Yukawa coupling onto tau, mu, e.
        k : float
            Decay parameter k = Gamma1 / H(T=M1).

        Returns
        -------
        array, shape (7,), complex
            Time derivatives of the state vector.
        """
        if z != self._currz or z == self.zmin:
            self._d       = np.real(self.DS(k, z))
            self._w1      = self.scat(z) * np.real(self.W1(k, z))
            self._n1eq    = self.N1Eq(z)
            self._currz   = z

        widtht = 485e-10 * self.MP / self.M1
        widthm = 1.7e-10 * self.MP / self.M1

        from ulysses.etab1DME import fast_RHS
        return fast_RHS(y0, self._d, self._w1, self._n1eq,
                        epstt, epsmm, epsee, epstm, epste, epsme,
                        c1t, c1m, c1e, widtht, widthm)

    @property
    def EtaB(self):
        """Solve the DME system and return the final baryon-to-photon ratio.

        Reads model parameters from the ULSBase instance, computes all
        CP asymmetries, flavour projectors, and the decay parameter, then
        integrates the 7-component ODE system over ``self.zs``.  The full
        trajectory is stored via ``setEvolData`` and the final N_BL entry
        is returned as eta_B.

        Returns
        -------
        float
            Baryon-to-photon ratio eta_B evaluated at z = zmax.

        Notes
        -----
        ``odeintw`` is called with ``full_output=True``; the returned
        info dict is discarded but can be captured here for diagnostics
        if needed.
        """
        epstt = np.real(self.epsilon1ab(2, 2))
        epsmm = np.real(self.epsilon1ab(1, 1))
        epsee = np.real(self.epsilon1ab(0, 0))
        epstm =         self.epsilon1ab(2, 1)
        epste =         self.epsilon1ab(2, 0)
        epsme =         self.epsilon1ab(1, 0)

        c1t = self.c1a(2)
        c1m = self.c1a(1)
        c1e = self.c1a(0)

        k   = np.real(self.k1)
        y0  = np.array([0+0j]*7, dtype=np.complex128)

        params = np.array([epstt, epsmm, epsee, epstm, epste, epsme,
                           c1t, c1m, c1e, k], dtype=np.complex128)

        ys, _ = odeintw(self.RHS, y0, self.zs, args=tuple(params), full_output=True)
        self.setEvolData(ys)
        return self.ys[-1][-1]