# Leptogenesis with CP-Asymmetry and Scattering

Analysis and numerical tools for studying baryon asymmetry generation in type-I seesaw leptogenesis. The code examines CP-asymmetry in different parameterizations of the Casas-Ibarra matrix and quantifies the impact of Delta-L=1 scattering processes.

## Overview

This project uses ULYSSES, a numerical solver for leptogenesis Boltzmann equations, to compute the baryon asymmetry of the universe (BAU) across different scenarios. The main focus is understanding how:

- Real versus complex Casas-Ibarra matrices affect CP asymmetry
- Scattering corrections modify BAU predictions
- Mass hierarchy of right-handed neutrinos influences final asymmetry

## Getting Started

### Setup

Activate the virtual environment:

```bash
source leptogenesis_env/bin/activate
```

All dependencies are already installed. If needed, reinstall them:

```bash
pip install -r requirements.txt
```

### Quick Run

Open the main analysis notebook:

```bash
jupyter notebook notebooks/koichi_benchmarks.ipynb
```

Or run directly with a specific parameter set:

```python
from utils.koichi_benchmarks import koichi_section4
from src.etaB_1DME_sct import EtaB_1DME_sct

model = EtaB_1DME_sct(**koichi_section4)
print(f"BAU = {model.EtaB:.3e}")
```

## Project Structure

```
src/
  etaB_1DME_sct.py        1D density matrix equation solver with scattering
  etaB_1BE1F_sct.py       1 RHN, 1-flavor solver with scattering
  scattering.py           Delta-L=1 scattering rate functions
  plot_style.py           Utilities for consistent plot formatting

utils/
  koichi_benchmarks.py    Parameter sets and helper functions
  directories.py          Model point definitions

notebooks/
  koichi_benchmarks.ipynb Main analysis with M1 scans and comparisons
  ulysses_test.ipynb      Reference computations

plots/                    Generated figures
data/                     Input/output data files
reports/                  Analysis documents
```

## Key Features

### Benchmark Points

Two main scenarios are analyzed:

**Diagonal Asymmetry (epsilon_tau,tau = 0)**
- Real Casas-Ibarra matrix with x1 = 0
- Electron-muon asymmetry survives
- Tau asymmetry vanishes by construction
- Trace constraint Tr(epsilon) = 0 automatically satisfied

**Off-Diagonal Source (all diagonal elements = 0)**
- All diagonal CP asymmetries vanish exactly
- BAU generated from off-diagonal coherent terms only
- Requires specific loop weight constraints

### Scattering Analysis

Includes Delta-L=1 corrections to washout rates following the pedestrians approach:

- S-channel Higgs scattering
- T-channel Higgs scattering
- Thermal averaging with modified Bessel functions
- Impact on BAU across M1 = 10^8 to 10^13 GeV

### Parameter Scans

Automatic tools for exploring parameter space:

```python
from utils.koichi_benchmarks import create_m1_variant, scan_M1
import numpy as np

# Create single variant
params_new = create_m1_variant(koichi_section4, M1_new=1e10)

# Generate full scan
M1_range = np.logspace(8, 13, 11)
param_list = scan_M1(koichi_section4, M1_range)
```

## Working with Models

### Available Solvers

The codebase includes two main solver classes:

1. **EtaB_1DME_sct**: Density matrix with flavor structure plus scattering
2. **EtaB_1BE1F_sct**: Single RHN, one-flavor plus scattering

Both inherit from ULYSSES base solvers and add scattering corrections.

### Setting Parameters

Parameters are defined as dictionaries with standard keys:

```python
params = {
    "x1": 0.0,          # Casas-Ibarra angle (degrees)
    "x2": 1.32,
    "x3": 45.0,
    "y1": 0.0,          # Imaginary parts (0 for real matrix)
    "y2": 0.0,
    "y3": 0.0,
    "a21": 0.0,         # Majorana phases (degrees)
    "a31": 0.0,
    "m": -2.0,          # log10(m1/eV)
    "M1": 8.0,          # log10(M1/GeV)
    "M2": 8.48,         # log10(M2/GeV)
    "M3": 9.0,          # log10(M3/GeV)
    "t12": 33.68,       # PMNS angles (degrees)
    "t13": 8.56,
    "t23": 43.30,
    "delta": 212,       # Dirac CP phase
}
```

### Accessing Results

After running a solver:

```python
model = EtaB_1DME_sct(**params)

# Final BAU
print(model.EtaB)

# Evolution data: columns are [z, N_tau, N_mu, N_e, eta_tau, eta_mu, eta_e, ...]
z = model.evolData[:, 0]
eta = model.evolData[:, 4]

# Plot evolution
import matplotlib.pyplot as plt
plt.loglog(z, eta)
plt.show()
```

## Physics Background

### Casas-Ibarra Parametrization

The Yukawa coupling of right-handed neutrinos connects to the light neutrino mass matrix through:

Y = i*sqrt(2)/v * U * diag(sqrt(m_light)) * O * sqrt(f^-1(M))

where O is a real or complex orthogonal matrix parametrized by Euler angles. The CP-asymmetry depends on the imaginary parts of this coupling.

### Real Matrix Constraint

Setting all imaginary parts y_i = 0 (real Casas-Ibarra matrix) imposes a trace constraint on the CP-asymmetry tensor. This can be exploited to construct solutions with specific cancellations.

### Scattering Corrections

Following Buchmüller, Di Bari, Plümacher (2005), the washout rate receives corrections from tree-level scattering processes. These enter the Boltzmann equations through a multiplicative factor involving the I2 function:

W_total = W_decay * (1 + 1/D * (2*N1/N1_eq * S_s + 4*S_t))

where S_s and S_t are the s-channel and t-channel scattering rates respectively.

## Output and Analysis

### Generated Plots

The main notebook produces three key plots:

- M1_scan_section4.png: BAU dependence on M1 with/without scattering
- I2_function_section4.png: Washout correction showing sign behavior
- section4_vs_section5_complete.png: Comprehensive scenario comparison

### Data Files

Computed results can be saved for further analysis:

```python
import numpy as np

# Save evolution data
np.savetxt("data/evolution.txt", model.evolData)

# Save final results
with open("data/results.txt", "w") as f:
    f.write(f"BAU = {model.EtaB:.3e}\n")
```

## References

[1] A. Granelli, K. Hamaguchi, M. E. Ramirez-Quezada, K. Shimada, J. Wada, T. Yokoyama.
    Flavoured CP-asymmetry cancellations with a real Casas-Ibarra matrix.
    arXiv:2502.10093 [hep-ph] (2025).

[2] W. Buchmüller, P. Di Bari, M. Plümacher.
    Leptogenesis for Pedestrians.
    Nucl. Phys. B 665 (2003) 445-469. hep-ph/0401240.

[3] S. Davidson, E. Nardi, Y. Nir.
    Leptogenesis.
    Phys. Rept. 466 (2008) 105-177. arXiv:0802.2962 [hep-ph].

## Notes

The current implementation focuses on 1DME (one right-handed neutrino, density matrix) and 1BE1F (single RHN, one-flavor) scenarios. Extension to full three-flavor density matrix evolution would capture additional flavor coherence effects.

The benchmark points represent specific parameter choices satisfying particular constraints. Other regions of parameter space may show qualitatively different scattering effects, particularly regarding coherence preservation at different scales.
