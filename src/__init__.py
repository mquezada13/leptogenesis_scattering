"""
lepto_scattering — source package
==================================
Custom extensions to the ULYSSES leptogenesis framework that incorporate
scattering contributions to the right-hand-side of the Boltzmann equations.

Modules
-------
etaB_1BE1F_sct
    One-flavour Boltzmann equation solver (1BE1F) with scattering corrections.
    Replaces the standard decay term D with DS and multiplies the washout W1
    by a scattering factor scat(z).

etaB_1DME_sct
    Density-matrix equation solver (1DME) with scattering corrections.
    Three-flavour extension of the above; tracks the full lepton asymmetry
    matrix N_ab as well as the individual tau, mu, and electron asymmetries.

plot_style
    Publication-quality matplotlib style utilities: house rcParams, axis
    styling helpers, and colour palettes.

Usage
-----
    from src.etaB_1BE1F_sct import EtaB_1BE1F_sct
    from src.etaB_1DME_sct  import EtaB_1DME_sct
    from src.plot_style      import use_house_style, style_axes, clean_legend
"""
