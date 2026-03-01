# SSOC-Thermal: From Conductivity to Charging Rate

**Thermal Regime Classification and Format Selection for Amorphous Li–P–S Solid-State Batteries**

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)

---

## Overview

This repository contains the code and manuscript source for the companion paper to the [SSOC](https://github.com/) conductivity design equation.

The SSOC paper answered: *"How do I maximize ionic conductivity σ(x, ρ, T)?"*

**This paper answers:** *"Given σ, at what C-rate does thermal failure occur — and does cell format even matter?"*

We introduce a single dimensionless parameter **Θ** (Theta) that classifies every solid-state cell into one of three operating regimes:

| Regime | Θ Range | Meaning | Action |
|--------|---------|---------|--------|
| **I** | Θ < 0.1 | Thermally free | Use prismatic (packing wins). Done. |
| **II** | 0.1 ≤ Θ ≤ 1 | Format matters | Evaluate wet-fraction ratio w_p/w_c vs. 0.78 |
| **III** | Θ > 1 | Thermal death | Improve σ first — no format can help |

The central analytical result (**Proposition 1**) establishes that **Θ · σ = Λ_th = const** — an exact invariant that yields closed-form critical conductivity targets for any operating condition.

### The Three-Layer Causal Chain

```
σ(x, ρ, T)  →  Θ(σ, T, C-rate, h, geometry)  →  Format freedom
   [SSOC]              [This paper]                 [Decision]
```

## Key Results

**At 300 K with current Li–P–S conductivity (σ ≈ 0.024 mS/cm):**

- **1C:** Θ = 0.44 → Regime II (format debate is active)
- **2C:** Θ = 1.8 → Regime III (thermally impossible)
- **3C:** Θ = 4.0 → Regime III (thermal death)

**Critical σ targets (300 K, 4680 cell):**

- 3C into Regime II: σ > 0.097 mS/cm (4× current)
- 3C into Regime I: σ > 0.967 mS/cm (40× current)
- The exact ratio σ_crit,I = 10 × σ_crit,II holds for *every* cell geometry and C-rate

**Format selection (Regime II only):**

> If w_p/w_c > 0.78 → **Prismatic wins**
> If w_p/w_c < 0.78 → **Cylindrical wins**
>
> *Format choice is not a physics question — it is a cooling architecture question.*

**30 K preheat** from 300 K to 330 K shifts 3C from Regime III to Regime II, enabling fast charging that is physically impossible at ambient temperature.

## Repository Structure

```
ssoc-lps/ssocthermal/
├── README.md                          # This file│
│── ssoc_thermal_v50_unified.py        # v5.0: Full physics engine + 8-panel unified figure
│── ssoc_thermal_v51_figures.py        # v5.1: Individual publication figures (300 DPI)
│
└── figures/                           # All 8 generated figures (for reference)
    ├── fig1_design_stack.png          # Design stack L1–L5 schematic
    ├── fig2_phase_diagram.png         # Θ(T, C-rate) phase diagram
    ├── fig3_sigma_crate.png           # Θ(σ, C-rate) + temperature iso-lines
    ├── fig4_regime_table.png          # Numerical Θ regime grid
    ├── fig5_wet_fraction.png          # Wet-fraction phase diagram
    ├── fig6_verdict.png               # Industry scenario verdict table
    ├── fig7_anatomy.png               # Proposition 1 verification + Θ anatomy
    └── fig8_flowchart.png             # Design Box 2 flowchart
```

## Physics Engine

### SSOC Conductivity (Layer 1)

The conductivity model from the companion SSOC paper:

```
σ = n_Li · e² · f_eff · D_PCC · p_active / (k_B T)
```

Each factor is an analytical function of composition *x*, density *ρ*, and temperature *T*, with Arrhenius scaling:

```
σ(T) = σ(500 K) × exp[−E_a/k_B × (1/T − 1/500)]     E_a = 0.32 eV
```

### Dimensionless Thermal Parameter Θ (Layer 3)

```
Θ = I² × R_ionic(σ) × R_th,tot(geometry, h) / ΔT_crit
```

Decomposed into four independent design knobs:

| Factor | Symbol | Controls |
|--------|--------|----------|
| Load demand | C² · Cap² | Application requirement |
| Material | F_geom / σ(T) | SSOC design equation |
| Thermal path | R_th,int + R_sum/(w_f · A) | Cell + pack architecture |
| Budget | ΔT_crit | Safety margin (typ. 50 K) |

### Proposition 1: Θ–σ Invariance Theorem

```
Θ(σ) · σ  =  Λ_th  =  I² · F_geom · R_th,tot / ΔT_crit  =  const
```

**Corollary:**
```
σ_crit = Λ_th / Θ*       (closed-form, exact)
σ_crit,I = 10 × σ_crit,II   (for any cell geometry, any C-rate)
```

### CellModel Class

The `CellModel` class implements a 4680-class cylindrical cell with full jelly-roll thermal stack:

- **Geometry:** 46 mm × 80 mm, mandrel ratio 0.109, ~115 layer repeats
- **Internal thermal resistance:** Layer-by-layer log-mean + interfacial contact series
- **External:** Cylindrical can + prismatic equivalent with configurable cooling
- **Pack-level:** Packing efficiency correction (η_cyl = π/2√3, η_prism = 0.98)

Key methods:
| Method | Returns |
|--------|---------|
| `compute_Theta(σ, C, h, wf_c, wf_p)` | Θ_cyl, Θ_prism, performance ratio |
| `Lambda_th(C, h, wf, format)` | Thermal load invariant Λ_th |
| `sigma_crit(C, Θ*, h, wf, format)` | Critical conductivity (closed-form) |
| `crit_wf_ratio(h)` | Critical w_p/w_c threshold |

## Figures

### Paper Figures (3 + Design Box)

The manuscript uses three figures plus a LaTeX-typeset Design Box:

**Figure 1** — Θ(T, C-rate) Phase Diagram

> The complete regime map for the 4680 cell. Green/red contours mark Regime I/II and II/III boundaries. Room-temperature fast charging (≥2C at 300 K) lies deep in Regime III.

**Figure 2** — Θ(σ, C-rate) with Temperature Iso-lines

> Shows how temperature maps directly onto the σ axis via Arrhenius scaling. Dashed horizontal lines mark σ at each temperature. The key insight: improving σ by 4× at 300 K shifts 1C from Regime II to Regime I.

**Figure 3** — Wet-Fraction Format Selection Map

> The format decision phase diagram within Regime II. Industry cooling scenarios (Tesla gap-cool, BYD bottom-plate, immersion) are plotted. The analytic boundary w_p = 0.78·w_c separates cylindrical-wins from prismatic-wins regions.

**Design Box 2** is typeset directly in LaTeX as a `tcolorbox` environment — the complete charging rate optimization protocol in four steps.

### Supplementary Figures (5 additional)

Generated by `v5.1` but not included in the manuscript:

| Figure | Content |
|--------|---------|
| fig1 | Design Stack schematic (L1→L5 with PCC/SCC arrows) |
| fig4 | Numerical Θ grid table (color-coded by regime) |
| fig6 | Industry verdict table with KEY FINDING summary |
| fig7 | Proposition 1 verification plot (Θ·σ = const) + anatomy |
| fig8 | Design Box 2 as a visual flowchart |

## Quick Start

### Generate all figures

```bash
python code/ssoc_thermal_v51_figures.py
```

Output: 8 PNG files at 300 DPI in the working directory.

### Use as a library

```python
from ssoc_thermal_v51_figures import CellModel, sigma_SSOC, sigma_at_T

# Initialize 4680 cell
cell = CellModel()

# SSOC conductivity at optimal composition
sigma_500K = sigma_SSOC(0.75, 1.72)        # S/m at 500 K
sigma_300K = sigma_at_T(sigma_500K, 300)    # Arrhenius to 300 K

# Compute Theta at 1C
result = cell.compute_Theta(sigma_300K, C_rate=1.0, h=300)
print(f"Θ = {result['Theta_c']:.2f}")       # → 0.44 (Regime II)

# Critical σ for 3C → Regime II
sigma_c = cell.sigma_crit(C_rate=3.0, Theta_star=1.0)
print(f"σ_crit = {sigma_c*10:.3f} mS/cm")   # → 0.097 mS/cm
```

## Design Box 2 — Charging Rate Protocol

> **Input:** Composition *x*, density *ρ*, operating temperature *T*, cell geometry, cooling *h*, target C-rate.

**Step 1 — Conductivity:**
Get σ from SSOC design equation + Arrhenius scaling.

**Step 2 — Compute Θ:**
Θ = I² × R_ionic(σ) × R_th,tot / ΔT_crit

**Step 3 — Classify:**
- Θ < 0.1 → **Regime I** → Use prismatic. Done.
- 0.1 ≤ Θ ≤ 1 → **Regime II** → Go to Step 4.
- Θ > 1 → **Regime III** → Cannot operate. Compute σ_crit = Λ_th/Θ*.
  Options: (a) preheat, (b) improve σ via SSOC Design Box, (c) reduce C-rate.

**Step 4 — Format selection (Regime II only):**
- w_p/w_c > 0.78 → **Prismatic**
- w_p/w_c < 0.78 → **Cylindrical**

**Output:** Optimal format + max safe C-rate, or quantitative σ_crit target.

## Dependencies

- Python ≥ 3.9
- NumPy
- Matplotlib ≥ 3.5
- LaTeX (TeX Live 2023+) with packages: `natbib`, `tcolorbox`, `siunitx`, `mhchem`, `cleveref`

## Citation

If you use this work, please cite both papers:

```bibtex
@article{iizumi2026thermal,
  author  = {Iizumi, Masamichi},
  title   = {From Conductivity to Charging Rate: Thermal Regime Classification
             and Format Selection for Amorphous {Li--P--S} Solid-State Batteries},
  journal = {Adv. Energy Mater.},
  year    = {2026},
  note    = {submitted}
}

@article{iizumi2026ssoc,
  author  = {Iizumi, Masamichi},
  title   = {Analytical Conductivity Design Equation for Amorphous {Li--P--S}
             Solid Electrolytes from the {PCC/SCC} Separation Framework},
  journal = {Adv. Energy Mater.},
  year    = {2026},
  note    = {submitted}
}
```

## Authors

- **Masamichi Iizumi** — [Miosync, Inc.](https://miosync.link), Tokyo
- **Tamaki Iizumi (環)** — Miosync, Inc., Tokyo

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

*"Format choice is not a physics question — it is a cooling architecture question. And it is only relevant in Regime II."*
