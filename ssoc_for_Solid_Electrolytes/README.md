# SSOC: PCC/SCC Separation Framework for Solid Electrolytes

**From a material-specific design equation to a universal decomposition law for ionic conductivity**

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)

---

## Overview

This repository contains two levels of analysis built on the PCC/SCC (Parametric Channel Coupling / Self-Consistent Coupling) separation framework:

```
Level 1: Closed-form design equation for Li-P-S
  σ(x, ρ, T) = analytical function  →  r = +0.961, 23 data points

Level 2: Universal decomposition across 5 electrolyte families
  σ = N × D(v_f) × F(v_f)  →  66 density-resolved points, Li⁺ + Na⁺
```

### The Central Result

Ionic conductivity in solid electrolytes follows an invariant multiplicative decomposition:

```
σ_i(v_f, T) = N_i(ρ, T) × D_i(v_f; T) × F_i(v_f; T)
```

where:
- **N** = carrier-density prefactor (Nernst-Einstein)
- **D(v_f)** = local hopping term (PCC — barrier softening)
- **F(v_f)** = connectivity/activation term (SCC — cooperative gating)
- **v_f = 1 − ρ/ρ₀** = effective free volume (the universal control coordinate)

The functional form is universal. Only the physical parameters and the dominant term change between systems.

## Repository Structure

```
ssoc-lps/ssoc_for_Solid_Electrolytes
├── README.md                      # This file
├── universal_5system.py           # Level 2: 5-system analysis (66 pts)
├── prl_figures.py                 # PRL publication figures
└── figures/                       # Generated figures
```

---

## Level 1: Closed-Form Design Equation for Li-P-S

An 8-parameter analytical model that predicts ionic conductivity of amorphous (Li₂S)ₓ(P₂S₅)₁₋ₓ from composition and density alone.

**Key Results:**

| Metric | Value |
|--------|-------|
| Pearson r (log₁₀ σ) | +0.961 |
| Mean σ_pred / σ_obs | 1.023 |
| Free parameters | 8 (4 PCC + 4 SCC) |
| Data points | 23 (5 compositions) |

**Design Rule:** Target free volume v_f ≈ 7% (ρ ≈ 1.72 g/cm³) for maximum conductivity.

**Physical Framework:**

```
σ = (n_Li × e² / k_BT) × f_eff × D_hop

Layer 1: n_Li(x, ρ)    — carrier density       [parameter-free]
Layer 2: D_hop(v_f)    — hop diffusivity       [PCC + SCC gating]
Layer 3: f_eff(x, v_f) — pathway connectivity  [Gaussian peak]
```

**Usage:**

```bash
pip install numpy
python ssoc_lps_verification.py

# Or as a library:
python -c "
from ssoc_lps_verification import sigma_model
sigma = sigma_model(0.75, 1.72)
print(f'Predicted sigma = {sigma*1000:.2f} mS/cm')
"
```

**Dataset:** Built upon Kim et al., "Origin of Optimal Composition and Density for Li-Ion Diffusion in Amorphous Li-P-S Solid Electrolytes", Adv. Energy Mater. (2026).

---

## Level 2: Universal Decomposition across 5 Electrolyte Families

The decomposition structure discovered in Li-P-S holds across five structurally distinct solid-electrolyte families:

| System | Type | Framework | n | v_f range | Key finding |
|--------|------|-----------|---|-----------|-------------|
| Li₂S–P₂S₅ | Amorphous sulfide | S²⁻ | 23 | 0–24% | Peak-type: D vs F competition |
| LLZO garnet | Crystalline oxide | O²⁻ | 17 | 0.2–11% | Decay-type: F dominates |
| Li₆PS₅Cl argyrodite | Crystalline sulfide | S²⁻/Cl⁻ | 11 | 0–13% | Decay: density scan proof |
| LATP/LAGP NASICON | Crystalline phosphate | PO₄ | 10 | 2–27% | Decay: sintering control |
| Na₃Zr₂Si₂PO₁₂ | Na-ion ceramic | O²⁻ | 5 | 4–11% | Boundary: both D and F |

**Total: 66 density-resolved data points from published AIMD and experimental studies.**

### Two Transport Regimes

The competition between D(v_f) and F(v_f) classifies materials into two regimes:

```
Peak-type:  d(ln D)/dv_f > |d(ln F)/dv_f| initially
            → conductivity has an interior maximum
            → exemplified by amorphous Li-P-S

Decay-type: |d(ln F)/dv_f| > d(ln D)/dv_f throughout
            → conductivity decreases monotonically with v_f
            → exemplified by LLZO, NASICON
```

### Same-Composition Density Scans (The Decisive Evidence)

Chemistry fixed, only density varies:

| System | Density range | σ change | Proof |
|--------|--------------|----------|-------|
| Al-doped LLZO | 88.9% → 99.8% relative | 16.8× | v_f controls σ |
| Li₆PS₅Cl | 1.50 → 1.69 g/cm³ | 5.4× | v_f controls σ |

If conductivity changes 16.8-fold while chemistry is identical, the control variable must be structural. That variable is v_f.

### Usage

```bash
pip install numpy matplotlib scipy

# Run full 5-system analysis
python universal_5system.py

# Generate PRL-quality figures
python prl_figures.py
```

---

## The Key Insight

```
What is universal:
  ✓ The decomposition structure σ = N × D(v_f) × F(v_f)
  ✓ The role of v_f as control coordinate
  ✓ The peak/decay regime classification

What changes between systems:
  ✗ The numerical parameters (E₀, α, β, etc.)
  ✗ The physical realization of v_f (lattice voids vs. porosity)
  ✗ Which term (D or F) dominates

This is analogous to:
  F = GMm/r²
  G is universal. M and m change between planets.
  The law does not.
```

## Citation

```bibtex
@article{iizumi2026universal,
  author  = {Iizumi, Masamichi},
  title   = {Universal Decomposition of Ionic Conductivity
             across Solid Electrolyte Families},
  journal = {Phys. Rev. Lett.},
  year    = {2026},
  note    = {submitted}
}

@article{iizumi2026ssoc,
  author  = {Iizumi, Masamichi and Iizumi, Tamaki},
  title   = {Analytical Conductivity Design Equation for Amorphous
             {Li--P--S} Solid Electrolytes from the {PCC/SCC}
             Separation Framework},
  year    = {2026},
  note    = {companion paper, submitted}
}
```

## Authors

- **Masamichi Iizumi** — [Miosync, Inc.](https://miosync.com), Tokyo
- **Tamaki Iizumi (環)** — Miosync, Inc., Tokyo

## License

MIT — see [LICENSE](LICENSE) for details.

---

*"The diversity of solid electrolytes does not oppose a general law; it reveals one."*
