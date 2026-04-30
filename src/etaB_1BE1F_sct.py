"""
etaB_1BE1F_sct — one-flavour Boltzmann equation solver with scattering
=======================================================================
Extends the ULYSSES ``EtaB_1BE1F`` model by adding scattering corrections
to both the decay/production source term and the washout rate:

    D  ->  DS(k, z)          (Eq. 73 of hep-ph/0401240)
    W1 ->  scat(z) * W1(k, z) (Eq. 223 of hep-ph/0401240)

The ODE system tracks two variables as a function of z = M1/T:
    y0[0] : abundance of the heavy right-handed neutrino N1
    y0[1] : total B-L asymmetry N_BL

The final baryon-to-photon ratio eta_B is extracted from N_BL at the end
of the integration range and returned by the ``EtaB`` property.

Dependencies
------------
- ulysses   : base class ULSBase and helper methods (DS, W1, scat, N1Eq, …)
- odeintw   : ODE integrator that handles complex-valued arrays
- ulysses.etab1BE1F.fast_RHS : Cython/numba-compiled right-hand side
"""

import ulysses
import numpy as np
from odeintw import odeintw
from ulysses.etab1BE1F import fast_RHS


class EtaB_1BE1F_sct(ulysses.ULSBase):
    """One-flavour Boltzmann equation solver with scattering corrections.

    Inherits all parameter handling, epsilon computation, and ODE bookkeeping
    from ``ulysses.ULSBase``.  The only modifications with respect to the
    vanilla ``EtaB_1BE1F`` model are:

    * The source term uses ``DS(k, z)`` instead of ``D(k, z)``.
    * The washout term is multiplied by the scattering factor ``scat(z)``.

    ULYSSES registry
    ----------------
    shortname      : "1BE1F_sct"
    flavour indices: [1]        (total B-L, single-flavour approximation)
    flavour labels : ["$NBL$"]
    """

    def shortname(self): return "1BE1F_sct"
    def flavourindices(self): return [1]
    def flavourlabels(self): return ["$NBL$"]

    def RHS(self, y0, z, epstt, epsmm, epsee, k):
        """Compute the right-hand side of the 1BE1F ODEs at a single z step.

        Caches the intermediate quantities (d, w1, n1eq) for the current z
        value to avoid redundant evaluation when the integrator calls this
        method multiple times at the same z.

        Parameters
        ----------
        y0 : array-like, shape (2,), complex
            Current state vector [N1_abundance, N_BL_asymmetry].
        z : float
            Integration variable z = M1/T.
        epstt, epsmm, epsee : float
            CP-asymmetry parameters for tau-tau, mu-mu, and ee flavour
            projections of the N1 decay.
        k : float
            Decay parameter k = Gamma1 / H(T=M1).

        Returns
        -------
        array, shape (2,), complex
            Time derivatives [dN1/dz, dN_BL/dz].
        """
        if z != self._currz or z == self.zmin:
            self._d    = np.real(self.DS(k, z))
            self._w1   = self.scat(z) * np.real(self.W1(k, z))
            self._n1eq = self.N1Eq(z)
            self._currz = z
        return fast_RHS(y0, self._d, self._w1, self._n1eq, epstt, epsmm, epsee)

    @property
    def EtaB(self):
        """Solve the ODEs and return the final baryon-to-photon ratio eta_B.

        Reads model parameters from the ULSBase instance (M1, Yukawa matrix,
        etc.), computes the CP asymmetries and decay parameter, integrates the
        system over ``self.zs``, stores the full evolution via
        ``setEvolData``, and returns the B-L asymmetry at the final z step
        rescaled to eta_B.

        Returns
        -------
        float
            Baryon-to-photon ratio eta_B evaluated at z = zmax.
        """
        epstt = np.real(self.epsilon1ab(2, 2))
        epsmm = np.real(self.epsilon1ab(1, 1))
        epsee = np.real(self.epsilon1ab(0, 0))

        k  = np.real(self.k1)
        y0 = np.array([0+0j, 0+0j], dtype=np.complex128)

        params = np.array([epstt, epsmm, epsee, k], dtype=np.complex128)

        ys = odeintw(self.RHS, y0, self.zs, args=tuple(params), atol=1e-14, rtol=1e-10)
        self.setEvolData(ys)
        return self.ys[-1][-1]



# checks 
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