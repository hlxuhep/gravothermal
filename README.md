# Gravothermal Evolution of Self-Interacting Dark Matter Halos with Static Central Baryonic Potentials

This repository provides two programs that work together to create and evolve spherically symmetric, non-rotating, gravothermally conducting fluid halos. If this code contributes to your research, please cite [Feng, Yu, & Zhong (2025)](https://arxiv.org/abs/2506.17641) [![](https://img.shields.io/badge/arXiv-2506.17641-red)](https://arxiv.org/abs/2506.17641). For questions about running the code, please [contact me](mailto:yiming.zhong@cityu.edu.hk).

| File            | Language | Purpose                                                                                                   |
|-----------------|----------|-----------------------------------------------------------------------------------------------------------|
| `initialize.py` | Python   | Generates initial NFW self-interacting dark matter (SIDM) profiles under a Plummer baryonic potential and exports radius, density, mass, energy, and luminosity data as plain text files. |
| `evolution.cpp` | C++      | Evolves SIDM profiles forward in time by solving conduction and hydrostatic relaxation equations under the static baryonic potential. |

---

## 1. Quick Start

### 1.1 Prerequisites
* **Python ≥ 3.8** with `mpmath`, `numpy`, and `matplotlib`
* **C++ 17** compiler with `Eigen3` headers

### 1.2 Generating Initial Conditions

Configure the following parameters in `initialize.py`:
```python
# base_path            Output directory for profiles
# mass_norm            M_b / (4 π ρ_s r_s³)
# scale_norm           Plummer scale radius a / r_s
# sigma                (σ/m) ρ_s r_s
# tag                  Label for this realization
```

Then generate the SIDM profiles:
```bash
python initialize.py
```

All output profiles will be saved to `base_path/initial/<tag>/`, including a PDF visualization of the profiles.

### 1.3 Building and Running the Evolution Solver

First, update the following parameters in `evolution.cpp` to match your initial conditions:
- `inputDir` and `tag` to match the Python output path
- `sigma`, `mass_norm`, and `scale_norm` to match the initial profile parameters

Compile the C++ code:
```bash
# Using GCC
g++ -O3 -std=c++17 -I/path/to/eigen evolution.cpp -o evolve

# Or using Intel compiler
icc -Ofast -std=c++17 -I/path/to/eigen evolution.cpp -o evolve
```

Run the evolution:
```bash
./evolve
```

The program will write time series data to `output/result_<tag>.txt`.

---

## 2. Directory Structure
```
initial/
  <tag>/            
    RList-<tag>.txt      # Radii of the Lagrangian zones
    MList-<tag>.txt      # Enclosed masses
    RhoList-<tag>.txt    # Densities
    uList-<tag>.txt      # Specific kinetic energies
    LList-<tag>.txt      # Luminosities
    Basic-<tag>.txt      # Basic parameters
    profile.pdf          # Visualization of profiles
output/
  result_<tag>.txt       # Complete time evolution data
```

The output file contains:
1. **Header**: All simulation parameters and setup information,
2. **Body**: Time series data including evolution time, step number, and the radii, densities, enclosed masses, specific kinetic energies, and luminosities of all SIDM Lagrangian zones at each time step.

---

## 3. Configuration Options

* **Enable verbose diagnostics**: Set `Logger logger(true)` in `main()`
* **Increase snapshot frequency**: Decrease `DEFAULT_SAVE_STEPS` (note: this will increase output file size)
* **Use alternative baryon models**: Replace the default Plummer model with `MbaryonH` (Hernquist) or `MbaryonSPL` (power-law) in the `updateProfiles()` function

---

## 4. References

For implementation details, please refer to Appendix B of [Zhong, Yang, & Yu (2023)](https://arxiv.org/abs/2306.08028). The implementation is based on [Pollack (2012)](https://inspirehep.net/files/f80416c2eaf8c69c788a20d4c24a5554).


g++ -O3 -std=c++17 -I/opt/homebrew/Cellar/eigen/3.4.0_1/include/eigen3 evolution.cpp -o evolve