"""
utils/directories.py — parameter dictionaries for leptogenesis model points
============================================================================
Defines ready-to-use parameter dictionaries that can be passed directly to
ULYSSES solver instances (e.g. ``EtaB_1DME_sct``).

Convention
----------
* All mixing angles (t12, t13, t23, delta, x1–x3, y1–y3, a21, a31)
  are in **degrees**.
* Heavy-neutrino masses (M1, M2, M3) are stored as **log10(M/GeV)**,
  so M1 = 12 means M1 = 10^12 GeV.
* The lightest active-neutrino mass m is also stored as **log10(m/eV)**.
* PMNS angles follow **NuFit 6.0 + Super-Kamiokande** best-fit values.

Available dictionaries
----------------------
_pmns          : shared PMNS block (t12, t13, t23, delta); not intended for
                 direct use — merged into every model-point dict via **_pmns.
dir_params_ref : reference point from Granelli, Moffat & Petcov (2021),
                 with the mass hierarchy M3 = 5·M2 = 50·M1.
dir_params_A   : model point A (source doc p. 4),  M = (1e12, 1e13, 1e14) GeV.
dir_params_B   : model point B (source doc p. 4),  M = (1e13, 3e14, 1e15) GeV.
dir_params_C   : model point C (source doc p. 5),  M = (1e14, 3e14, 1e15) GeV.

Usage
-----
    from utils.directories import dir_params_A
    model = EtaB_1DME_sct(**dir_params_A)
    print(model.EtaB)

Key parameter glossary
----------------------
x1, x2, x3 : real parts of the Casas–Ibarra complex angles (degrees)
y1, y2, y3 : imaginary parts of the Casas–Ibarra complex angles (degrees)
a21, a31   : Majorana CP phases alpha_21, alpha_31 (degrees)
m          : log10 of the lightest neutrino mass in eV
M1, M2, M3 : log10 of the right-handed neutrino masses in GeV
t12, t13, t23 : PMNS mixing angles theta_12, theta_13, theta_23 (degrees)
delta      : Dirac CP phase in the PMNS matrix (degrees)
"""



import numpy as np

# PMNS angles (shared by all points)
_pmns = {
    "t12"   : 33.41,
    "t13"   : 8.56,
    "t23"   : 43.3,
    "delta" : 212,
}

# Reference point from Granelli, Moffat, Petcov (2021)
dir_params_ref = {
    "x1": -10, "x2": -20, "x3": -10,
    "y1": 0,   "y2": 0,   "y3": 0,
    "a21": 200, "a31": 175,
    "m": -10,
    "M1": 12, "M2": 12.699, "M3": 13.699,
    **_pmns,
}

# Model point A — corrected (Majorana phases converted to degrees)
dir_params_A = {
    "x1": 263.5996576367934,
    "x2": 78.06536128821098,
    "x3": 16.090775063535286,
    "y1": 87.08034776599204,
    "y2": 33.74248484411295,
    "y3": 110.98939710457948,
    "a21": 4.205989366306699,   # deg
    "a31": 2.9871664822187105,  # deg
    "m": -1.911493610087044,
    "M1": 12, "M2": 13, "M3": 14,
    **_pmns,
}

# Model point B (page 4) — no bug
dir_params_B = {
    "x1": 344.3019127098013,
    "x2": 274.2650159666047,
    "x3": 80.37014910265017,
    "y1": 7.604581448181876,
    "y2": 29.82785588041296,
    "y3": 23.569554149828075,
    "a21": 182.08348752793933,
    "a31": 330.44716463631573,
    "m": -1.0726315948299519,
    "M1": 13, "M2": 14, "M3": 15,
    **_pmns,
}

# Model point C (page 5) — no bug
dir_params_C = {
    "x1": 0.4441311543362225,
    "x2": 269.9954112688561,
    "x3": 172.51813623717405,
    "y1": 93.23980961354052,
    "y2": 49.90053530824168,
    "y3": 69.23729143217459,
    "a21": 225.5123507432832,
    "a31": 265.32734498226273,
    "m": -9.07163440664015,
    "M1": 14, "M2": 14.477121254719663, "M3": 15,
    **_pmns,
}