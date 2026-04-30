# ULYSSES Quick Manual

## 1. Import

```python
import ulysses
```

## 2. Available models

```python
ulysses.EtaB_1DME()       # Density Matrix, 1 RHN, with flavor effects
ulysses.EtaB_1BE1F()      # Boltzmann, 1 RHN, no flavor effects
ulysses.EtaB_1DME_sct()   # 1DME + scattering
```

## 3. Set parameters and run

```python
uls = ulysses.EtaB_1DME()
uls.setParams(params)      # pass a dict, solver runs automatically
```

Parameters: angles in degrees, masses in log10.

## 4. Access results

```python
uls.EtaB                  # final η (float, not a function)
uls.evolData              # full evolution array: rows = time steps, columns = variables
uls.evolData.shape        # check dimensions
uls.flavourlabels()       # column labels (method, needs parentheses)
```

### evolData columns

- **1DME**: `[z, N_ττ, N_μμ, N_ee, η]` → 5 columns
- **1BE1F**: `[z, N_{B-L}, η]` → 3 columns

## 5. Inspect model

```python
dir(uls)                  # all attributes/methods
uls.isPerturbative()      # perturbativity check
uls.printParams()         # print parameters
uls.k1, uls.k2, uls.k3   # washout parameters
uls.epsilon1ab(i, j)      # CP asymmetry element (i,j = 0,1,2 for τ,μ,e)
```