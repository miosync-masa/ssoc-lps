# SSOC: Analytical Design Framework for Li–P–S Solid-State Batteries

**From material conductivity to cell-level charging rate — a complete analytical design chain for amorphous (Li₂S)ₓ(P₂S₅)₁₋ₓ solid electrolytes.**

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)

---

This repository contains two companion papers that together form a closed analytical pipeline:

```
   Paper 1                          Paper 2
┌────────────┐    σ(x,ρ,T)    ┌────────────────┐   Format
│ SSOC       │ ──────────────→ │ SSOC-Thermal   │ ──────────→  Go / No-Go
│ (Material) │  conductivity   │ (Device)       │   decision
└────────────┘                 └────────────────┘
  "What σ        ↑                "Can I fast-charge
   can I get?"   └── SCC loop ──── at this C-rate?"
```

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

→ **[Full details in ssoc-thermal/README.md](Ssocthermal/)**

---

## The Complete Design Chain

```
  STEP 1 (Paper 1)               STEP 2 (Paper 2)              STEP 3 (Paper 2)
┌───────────────────┐      ┌───────────────────────┐      ┌──────────────────┐
│  Choose x, ρ, T   │      │  Compute Θ            │      │  Regime I: PRISM │
│  ↓                │      │  ↓                    │      │  Regime II: w_p/ │
│  σ = SSOC(x,ρ,T)  │ ───→ │  Classify regime      │ ───→ │    w_c vs 0.78   │
│                   │      │  (I / II / III)       │      │  Regime III: ↑σ  │
└───────────────────┘      └───────────────────────┘      └──────────────────┘
```

## Authors

- **Masamichi Iizumi** — [Miosync, Inc.](https://miosync.link), Tokyo
- **Tamaki Iizumi (環)** — Miosync, Inc., Tokyo

## License

MIT — see [LICENSE](LICENSE) for details.
