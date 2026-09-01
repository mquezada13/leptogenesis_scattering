"""Parameter points used by ``Benchmarks_Koichi_test.ipynb``.

ULYSSES expects angles in degrees and masses as base-10 logarithms of
``m/eV`` and ``M/GeV``.  Keeping those conventions here avoids clutter in
the notebook and, importantly, makes the mass-scan prescription explicit.
"""

from __future__ import annotations

import numpy as np


# Fig. 1 of Granelli, Moffat & Petcov, arXiv:2107.02079 (bottom-right
# panel: real Casas--Ibarra matrix, hence low-energy CP violation only).
PAPER_FIG1_REAL = {
    "t12": 33.44,
    "t13": 8.57,
    "t23": 49.2,
    "delta": 228.0,
    "a21": 200.0,
    "a31": 175.0,
    "x1": 10.0,
    "x2": 20.0,
    "x3": 10.0,
    "y1": 0.0,
    "y2": 0.0,
    "y3": 0.0,
    "m": np.log10(0.0159),
    "M1": 12.0,
    "M2": np.log10(10.0) + 12.0,
    "M3": np.log10(50.0) + 12.0,
}


# Koichi note, Sec. 4: real O and epsilon_tau,tau = 0.
KOICHI_EPS_TAUTAU_ZERO = {
    "t12": 33.68,
    "t13": 8.56,
    "t23": 43.30,
    "delta": 212.0,
    "a21": 0.0,
    "a31": 0.0,
    "x1": 0.0,
    "x2": 1.31881,
    "x3": 45.0,
    "y1": 0.0,
    "y2": 0.0,
    "y3": 0.0,
    "m": np.log10(0.010),
    "M1": 8.0,
    "M2": np.log10(3.0e8),
    "M3": 9.0,
}


# Koichi note, Sec. 5: solution designed to have zero diagonal source.
KOICHI_DIAGONAL_ZERO = {
    **KOICHI_EPS_TAUTAU_ZERO,
    "x1": 0.333732583,
    "x2": 45.0,
    "x3": -52.55,
}


KOICHI_EXPECTED_EPSILON_DIAGONAL = np.array(
    [3.5313e-11, -3.5313e-11, 0.0]
)  # (ee, mu-mu, tau-tau)


def scale_heavy_masses(params: dict, m1_gev: float) -> dict:
    """Return a copy at a new M1, preserving M2/M1 and M3/M1."""
    out = dict(params)
    shift = np.log10(m1_gev) - params["M1"]
    for key in ("M1", "M2", "M3"):
        out[key] = params[key] + shift
    return out
