"""
utils/koichi_benchmarks.py — Parameter dictionaries from real_O_vanishing_tau_asymmetry.pdf
==========================================================================================

Defines parameter dictionaries for benchmark points from Koichi's notes
(Granelli, Hamaguchi, Ramirez-Quezada, Shimada, Wada, Yokoyama).

These represent solutions with real Casas–Ibarra matrices exhibiting
specific CP-asymmetry tensor properties.

Convention
----------
* All mixing angles (t12, t13, t23, delta, x1–x3, y1–y3, a21, a31)
  are in **degrees**.
* Heavy-neutrino masses (M1, M2, M3) are stored as **log10(M/GeV)**.
* The lightest active-neutrino mass m is stored as **log10(m/eV)**.
* PMNS angles follow NuFit 6.0 convention.

Available dictionaries
----------------------
koichi_pmns          : PMNS parameters (shared by all Koichi benchmark points)
koichi_section4      : Section 4 solution with ε_ττ = 0
                       Real O matrix, diagonal epsilon (ee, μμ only)
                       x₁=0, x₂≈1.32°, x₃=45°
koichi_section5      : Section 5 solution with all diagonals = 0
                       Purely off-diagonal coherent source
                       x₁≈0.334°, x₂=45°, x₃≈-52.55°

Physical interpretation
-----------------------
Section 4: Flavoured CP-asymmetry cancellations
  - Demonstrates exact cancellation of ε_ττ via real O-matrix structure
  - Results in electron-muon asymmetry: ε_ee = -ε_μμ ≠ 0
  - Trace vanishes by construction (real O property)
  - BAU generated primarily by off-diagonal coherent terms

Section 5: Purely off-diagonal source
  - All diagonal CP-asymmetry elements vanish identically
  - Only off-diagonal components (ε_eτ, ε_μτ) remain non-zero
  - Non-trivial solution requires w₂ ≠ w₃ and discriminant condition
  - Demonstrates possibility of generating BAU from coherent source alone

Neutrino mass parameters (both sections)
-----------------------------------------
m₁ = 0.010 eV (log10(m/eV) ≈ -2.0)
Δm²₂₁ = 7.49 × 10⁻⁵ eV²
Δm²₃₁ = 2.513 × 10⁻³ eV²

Heavy neutrino mass hierarchy (both sections)
----------------------------------------------
M₁ = 10⁸ GeV (reference)
M₂ = 3 × 10⁸ GeV (M₂/M₁ = 3)
M₃ = 10⁹ GeV (M₃/M₁ = 10)

Usage
-----
    from utils.koichi_benchmarks import koichi_section4, koichi_section5
    from src.etaB_1DME_sct import EtaB_1DME_sct
    
    model = EtaB_1DME_sct(**koichi_section4)
    print(f"BAU = {model.EtaB:.3e}")
"""

import numpy as np

# PMNS angles (shared by all Koichi benchmark points, from PDF Section 4)
koichi_pmns = {
    "t12": 33.68,
    "t13": 8.56,
    "t23": 43.30,
    "delta": 212,
}

# Neutrino mass parameters (computed from values in PDF Section 4)
m1_eV = 0.010
delta_m2_21 = 7.49e-5
delta_m2_31 = 2.513e-3

m1 = m1_eV
m2 = np.sqrt(m1**2 + delta_m2_21)
m3 = np.sqrt(m1**2 + delta_m2_31)

# Heavy neutrino masses (PDF Section 4)
M1_ref = 1e8  # GeV
M2_ref = 3e8
M3_ref = 1e9

# ============================================================================
# Section 4: Real O matrix with ε_ττ = 0
# ============================================================================
# From PDF Eq. (33)-(34): x₁=0, x₃=π/4, x₂ determined by Eq. (29)
# Expected results (PDF Eq. 36):
#   ε_ee = 3.5313 × 10⁻¹¹
#   ε_μμ = -3.5313 × 10⁻¹¹
#   ε_ττ = O(10⁻²⁶) ≈ 0
#   Tr(ε) = 0 ✓

koichi_section4 = {
    "x1": 0.0,          # x₁ = 0
    "x2": 1.31881,      # x₂ = 0.0230176 rad ≈ 1.31881°
    "x3": 45.0,         # x₃ = π/4 rad = 45°
    "y1": 0.0,          # Real O matrix: y₁ = y₂ = y₃ = 0
    "y2": 0.0,
    "y3": 0.0,
    "a21": 0.0,         # Vanishing Majorana phases
    "a31": 0.0,
    "m": np.log10(m1),
    "M1": np.log10(M1_ref),
    "M2": np.log10(M2_ref),
    "M3": np.log10(M3_ref),
    **koichi_pmns,
}

# ============================================================================
# Section 5: Purely off-diagonal coherent source
# ============================================================================
# From PDF Eq. (56) and (59): constructed solution for C₁₂ = 0
# With constraints:
#   tan x₃ = -√(m₁/m₂) cot θ₁₂  [Eq. 56]
#   t₁ satisfies quadratic from Eq. (59)
#   x₂ chosen so discriminant condition (60) holds
#
# Expected results (PDF Eq. 65):
#   ε_ee = 0
#   ε_μμ = 0
#   ε_ττ = 0
#   ε_αβ ≠ 0 for α ≠ β (only off-diagonal terms remain)

koichi_section5 = {
    "x1": 0.333732583,    # From Eq. (59): t₁ = tan x₁
    "x2": 45.0,           # x₂ = π/4 = 45°
    "x3": -52.55,         # x₃ ≈ -0.916897865 rad ≈ -52.55°
    "y1": 0.0,            # Real O matrix
    "y2": 0.0,
    "y3": 0.0,
    "a21": 0.0,           # Vanishing Majorana phases
    "a31": 0.0,
    "m": np.log10(m1),
    "M1": np.log10(M1_ref),
    "M2": np.log10(M2_ref),
    "M3": np.log10(M3_ref),
    **koichi_pmns,
}


# ============================================================================
# Utility functions for M₁ scans
# ============================================================================

def create_m1_variant(base_params, M1_new_gev):
    """
    Create a variant of a Koichi benchmark with different M₁.
    
    The mass hierarchy M₂/M₁ and M₃/M₁ are preserved.
    
    Parameters
    ----------
    base_params : dict
        One of koichi_section4 or koichi_section5
    M1_new_gev : float
        New value of M₁ in GeV
    
    Returns
    -------
    dict
        Modified parameter dictionary with updated masses
    """
    params_new = base_params.copy()
    
    # Determine hierarchy ratios from base parameters
    M1_base = 10.0 ** base_params["M1"]
    M2_base = 10.0 ** base_params["M2"]
    M3_base = 10.0 ** base_params["M3"]
    
    ratio_M2_M1 = M2_base / M1_base
    ratio_M3_M1 = M3_base / M1_base
    
    # Update with new M₁ while preserving hierarchy
    params_new["M1"] = np.log10(M1_new_gev)
    params_new["M2"] = np.log10(ratio_M2_M1 * M1_new_gev)
    params_new["M3"] = np.log10(ratio_M3_M1 * M1_new_gev)
    
    return params_new


def scan_M1(base_params, M1_values_gev):
    """
    Generate a list of parameter dictionaries for a M₁ scan.
    
    Parameters
    ----------
    base_params : dict
        One of koichi_section4 or koichi_section5
    M1_values_gev : array-like
        Array of M₁ values in GeV
    
    Returns
    -------
    list of dict
        Parameter dictionaries for each M₁ value
    """
    return [create_m1_variant(base_params, M1_val) for M1_val in M1_values_gev]
