"""
1DME + scattering solver.
=========================
Density matrix equation with 1 RHN and ΔL=1 scattering corrections.
Follows TY's implementation based on hep-ph/0401240.

Key difference from ULYSSES' built-in DS/scat functions:
  - N1 equation uses D+S (scattering accelerates N1 equilibration)
  - Asymmetry SOURCE uses D only (scattering does not produce asymmetry)
  - Asymmetry WASHOUT uses Eq. (84): W*(1 + 1/D*(2N1/N1eq*Ss + 4St))

The RHS equations (rhs1-rhs7) are the same as in ULYSSES' fast_RHS
(from arXiv:1112.4528), but we cannot call fast_RHS directly because
it uses the same 'd' for rhs1 and rhs2-7, whereas we need 'ds' for
rhs1 and 'd' for rhs2-7.
"""
import ulysses
import numpy as np
from odeintw import odeintw
from src.scattering import scat_Ss, scat_St, washout_sct


class EtaB_1DME_sct(ulysses.ULSBase):
    def shortname(self): return "1DME_sct"
    def flavourindices(self): return [1, 2, 3]
    def flavourlabels(self): return ["$N_{\\tau\\tau}$", "$N_{\\mu\\mu}$", "$N_{ee}$"]

    def RHS(self, y0, z, epstt, epsmm, epsee, epstm, epste, epsme, c1t, c1m, c1e, k):
        """
        Right-hand side of the 1DME+scattering ODE system.

        State vector y0 = [N1, N_tt, N_mm, N_ee, N_tm, N_te, N_me]
        where N_ab are elements of the lepton asymmetry density matrix.
        """
        # Cache scattering rates (expensive integrals) for each z step
        if z != self._currz or z == self.zmin:
            self._n1eq = self.N1Eq(z)
            self._d    = np.real(self.D1(k, z))          # Decay rate
            self._w1   = np.real(self.W1(k, z))          # Inverse decay washout
            M1         = np.real(self.DM[0, 0])
            self._Ss   = np.real(scat_Ss(k, z))          # S-channel scattering
            self._St   = np.real(scat_St(k, z, M1, self.MH))  # T-channel scattering
            self._ds   = self._d + self._Ss + self._St   # D + S (total source for N1)
            self._currz = z

        # Unpack state vector
        N1, Ntt, Nmm, Nee, Ntm, Nte, Nme = y0

        # Cached rates
        n1eq, d, ds = self._n1eq, self._d, self._ds

        # Washout with scattering correction (Eq. 84)
        w_sct = washout_sct(self._w1, d, N1, n1eq, self._Ss, self._St)

        # Conjugates of flavour projectors
        c1tc, c1mc, c1ec = c1t.conjugate(), c1m.conjugate(), c1e.conjugate()

        # Thermal widths (decoherence rates for off-diagonal elements)
        widtht = 485e-10 * self.MP / self.M1
        widthm = (1.7e-10 * self.MP / self.M1
                  if getattr(self, "include_muon_decoherence", True) else 0.0)

        # --- Boltzmann equations ---

        # N1 abundance: uses D+S (Eq. 9 of Pedestrians paper)
        rhs1 = -ds * (N1 - n1eq)

        # Diagonal asymmetries (N_tt, N_mm, N_ee): source = eps*D, washout = W_sct
        # These are Eqs. from arXiv:1112.4528 with W1 -> W_sct
        rhs2 = epstt*d*(N1-n1eq) - 0.5*w_sct*(2*c1t*c1tc*Ntt + c1m*c1tc*Ntm + c1e*c1tc*Nte + (c1m*c1tc*Ntm + c1e*c1tc*Nte).conjugate())
        rhs3 = epsmm*d*(N1-n1eq) - 0.5*w_sct*(2*c1m*c1mc*Nmm + c1m*c1tc*Ntm + c1e*c1mc*Nme + (c1m*c1tc*Ntm + c1e*c1mc*Nme).conjugate())
        rhs4 = epsee*d*(N1-n1eq) - 0.5*w_sct*(2*c1e*c1ec*Nee + c1e*c1mc*Nme + c1e*c1tc*Nte + (c1e*c1mc*Nme + c1e*c1tc*Nte).conjugate())

        # Off-diagonal asymmetries (N_tm, N_te, N_me): include decoherence widths
        rhs5 = epstm*d*(N1-n1eq) - 0.5*w_sct*(c1t*c1mc*Nmm + c1e*c1mc*Nte + c1m*c1mc*Ntm + c1mc*c1t*Ntt + c1t*c1tc*Ntm + c1t*c1ec*(Nme.conjugate())) - widtht*Ntm - widthm*Ntm
        rhs6 = epste*d*(N1-n1eq) - 0.5*w_sct*(c1t*c1ec*Nee + c1e*c1ec*Nte + c1m*c1ec*Ntm + c1t*c1ec*Ntt + c1t*c1mc*Nme + c1t*c1tc*Nte) - widtht*Nte
        rhs7 = epsme*d*(N1-n1eq) - 0.5*w_sct*(c1m*c1ec*Nee + c1e*c1ec*Nme + c1m*c1ec*Nmm + c1t*c1ec*(Ntm.conjugate()) + c1m*c1mc*Nme + c1m*c1tc*Nte) - widthm*Nme

        return [rhs1, rhs2, rhs3, rhs4, rhs5, rhs6, rhs7]

    @property
    def EtaB(self):
        """Solve the ODE system and return the final baryon asymmetry η_B."""
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

        ys, _ = odeintw(self.RHS, y0, self.zs, args=tuple(params), full_output=True, atol=1e-10, rtol=1e-10)
        self.setEvolData(ys)
        return self.ys[-1][-1]
