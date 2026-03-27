# SSOC: PCC/SCC Separation Framework for Solid-State Batteries

**From a universal conductivity law, through material design, to cell-level charging rate — a complete analytical chain for solid electrolytes.**

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)

---

This repository contains three interconnected projects built on the PCC/SCC separation framework:

```
   Paper 3 (Universal Law)
┌──────────────────────────┐
│ σ = N × D(v_f) × F(v_f) │   5 electrolyte families
│ Universal decomposition  │   66 density-resolved points
│ Li⁺ + Na⁺ carriers      │   Peak / Decay regimes
└────────────┬─────────────┘
             │ instantiated for Li-P-S
             ▼
   Paper 1                          Paper 2
┌────────────┐    σ(x,ρ,T)    ┌────────────────┐   Format
│ SSOC       │ ──────────────→ │ SSOC-Thermal   │ ──────────→  Go / No-Go
│ (Material) │  conductivity   │ (Device)       │   decision
└────────────┘                 └────────────────┘
  "What σ        ↑                "Can I fast-charge
   can I get?"   └── SCC loop ──── at this C-rate?"
```

## 📁 Repository Structure

```
ssoc-lps/
├── README.md                  ← You are here
├── LICENSE
│
├── ssoc-lps/                  ← Paper 1: Material Design Equation
│   ├── README.md
│   ├── ssoc_lps_verification.py
│   └── figures/
│
├── Ssocthermal/               ← Paper 2: Thermal Regime & Format Selection
│   ├── README.md
│   ├── code/
│   │   ├── ssoc_thermal_v50_unified.py
│   │   └── ssoc_thermal_v51_figures.py
│   ├── paper/
│   └── figures/
│
└── ssoc_for_Solid_Electrolytes/  ← Paper 3: Universal Decomposition Law
    ├── README.md
    ├── universal_5system.py
    └── prl_figures.py
```

---

## Paper 3 — [Universal Decomposition Law](ssoc_for_Solid_Electrolytes/)

> *"Is there a universal law governing ionic conductivity across all solid electrolytes?"*

**Yes.** Across five structurally distinct families — amorphous sulfides, crystalline garnets, argyrodites, NASICON phosphates, and Na-NASICON conductors — ionic conductivity follows an invariant multiplicative decomposition:

```
σ_i(v_f, T) = N_i(ρ, T) × D_i(v_f; T) × F_i(v_f; T)
```

governed by a single control coordinate: **effective free volume v_f = 1 − ρ/ρ₀**.

| System | n | Regime | Key evidence |
|--------|---|--------|-------------|
| Li₂S–P₂S₅ | 23 | Peak | D vs F competition at v_f ≈ 7% |
| LLZO garnet | 17 | Decay | 16.8× σ change from density alone |
| Li₆PS₅Cl | 11 | Decay | Continuous density scan proof |
| LATP/LAGP | 10 | Decay | Sintering density control |
| Na-NZSP | 5 | Boundary | Na⁺ carrier, same framework |

**The parameters change. The law does not.**

→ **[Full details in ssoc_for_Solid_Electrolytes/README.md](ssoc_for_Solid_Electrolytes/)**

---

## Paper 1 — [SSOC: Conductivity Design Equation](ssoc-lps/)

> *"How do I maximize ionic conductivity σ(x, ρ, T)?"*

An 8-parameter analytical model that predicts σ for amorphous Li–P–S from composition and density alone, achieving **r(log₁₀σ) = +0.96** across 23 AIMD data points.

**Key insight:** S²⁻ ions form a Random Close Packed framework; free volume v_f = 1 − ρ/ρ₀ is the single controlling variable. The optimal design point is **x ≈ 0.75, ρ ≈ 1.72 g/cm³** (v_f ≈ 7%).

```python
from ssoc_lps_verification import sigma_model
sigma = sigma_model(0.75, 1.72)   # → ~6.6 mS/cm at 500 K
```

→ **[Full details in ssoc-lps/README.md](ssoc-lps/)**

---

## Paper 2 — [SSOC-Thermal: Charging Rate & Format Selection](Ssocthermal/)

> *"Given σ, at what C-rate does thermal failure occur — and does cell format matter?"*

Introduces the dimensionless thermal parameter **Θ** that classifies any solid-state cell into three regimes:

| Regime | Θ | What it means |
|--------|---|---------------|
| **I** | < 0.1 | Thermally free — prismatic wins by default |
| **II** | 0.1 – 1 | Format matters — check w_p/w_c vs 0.78 |
| **III** | > 1 | Thermal death — improve σ first |

**Central theorem:** Θ · σ = Λ_th = const (exact), giving closed-form σ targets for any operating condition.

```python
from ssoc_thermal_v51_figures import CellModel, sigma_SSOC, sigma_at_T
cell = CellModel()
sigma_300K = sigma_at_T(sigma_SSOC(0.75, 1.72), 300)
result = cell.compute_Theta(sigma_300K, C_rate=1.0)
print(f"Θ = {result['Theta_c']:.2f}")  # → 0.44 (Regime II)
```

→ **[Full details in Ssocthermal/README.md](Ssocthermal/)**

---

## The Complete Design Chain

```
  STEP 0 (Paper 3)               STEP 1 (Paper 1)               STEP 2 (Paper 2)
┌───────────────────┐      ┌───────────────────┐      ┌──────────────────────┐
│ Universal law:    │      │  Choose x, ρ, T   │      │  Compute Θ           │
│ σ = N·D(vf)·F(vf)│      │  ↓                │      │  ↓                   │
│                   │ ───→ │  σ = SSOC(x,ρ,T)  │ ───→ │  Classify regime     │
│ Which regime?     │      │  (closed form)    │      │  I / II / III        │
│ Peak or Decay?    │      │                   │      │  → Format decision   │
└───────────────────┘      └───────────────────┘      └──────────────────────┘
     Law                       Design equation            Engineering protocol
```

## Authors

- **Masamichi Iizumi** — [Miosync, Inc.](https://miosync.link), Tokyo
- **Tamaki Iizumi (環)** — Miosync, Inc., Tokyo

## License

MIT — see [LICENSE](LICENSE) for details.

---

*"The diversity of solid electrolytes does not oppose a general law; it reveals one."*
