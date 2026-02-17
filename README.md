# SSOC Framework for Li-P-S Solid Electrolytes

**Structure-Selective Orbital Coupling analysis of ionic conductivity in amorphous (Li₂S)ₓ(P₂S₅)₁₋ₓ solid electrolytes**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## Motivation

Lithium-ion battery fires are a growing global safety concern — in consumer electronics, electric vehicles, and aviation cargo. The root cause is the flammability of conventional liquid organic electrolytes. Sulfide-based solid electrolytes (e.g., Li₂S-P₂S₅ systems) offer a fundamentally non-flammable alternative, but rational design of their ionic conductivity has remained empirical.

This repository provides a **quantitative, analytical design equation** for predicting ionic conductivity σ(x, ρ) of amorphous Li-P-S solid electrolytes from composition and density alone.

## Key Results

| Metric | Value |
|--------|-------|
| Pearson r (log₁₀ σ) | **+0.96** |
| Mean σ_pred / σ_obs | **1.02** |
| Free parameters | **8** |
| Data points | **23** (5 compositions) |

### Design Rule

> **Target a free volume of v_f ≈ 7% relative to the S²⁻ close-packed framework (ρ_opt ≈ 1.70 g/cm³). Use Li₂S-rich compositions (x ≥ 0.75) for maximum conductivity.**

## Physical Framework

The conductivity is decomposed into three layers:

```
σ = (n_Li e² / k_BT) × f_eff × D_hop

Layer 1: n_Li(x, ρ)    — carrier density       [parameter-free]
Layer 2: D_hop(v_f)    — hop diffusivity       [PCC + SCC gating]
Layer 3: f_eff(x, v_f) — pathway connectivity  [Gaussian peak]
```

**Key discoveries:**
- S²⁻ ions form a Random Close Packed framework; Li⁺ and P⁵⁺ are interstitial
- Framework density ρ₀ ≈ 1.83 g/cm³ is nearly composition-independent
- Free volume v_f = 1 − ρ/ρ₀ is the single controlling variable
- Cooperative Li⁺ deactivation (Hill n ≈ 3) arises from a three-coordination threshold
- The Hill coefficient has a parameter-free explanation: Li⁺ requires ≥ 3 simultaneous S²⁻ contacts

## Dataset

All analysis is built upon the systematic AIMD dataset of:

> Kim et al., "Origin of Optimal Composition and Density for Li-Ion Diffusion in Amorphous Li-P-S Solid Electrolytes", *Advanced Energy Materials* (2026)

We emphasize that without data of this quality and scope — five compositions, multiple densities per composition, and careful separation of D_hop and f — the analytical framework presented here could not have been constructed or validated. All 23 data points are used as reported; no points are excluded, reweighted, or adjusted.

## Usage

```bash
# No dependencies beyond NumPy
pip install numpy

# Run full verification
python ssoc_lps_verification.py

# Use as a library
python -c "
from ssoc_lps_verification import sigma_model
# Predict sigma for x=0.75, rho=1.70 g/cm³
sigma = sigma_model(0.75, 1.70)
print(f'Predicted sigma = {sigma*1000:.2f} mS/cm')
"
```

## Repository Structure

```
ssoc-lps/
├── README.md                      # This file
├── ssoc_lps_verification.py       # Complete verification code (self-contained)
├── LICENSE                        # MIT License
└── figures/                       # (to be added with paper submission)
```

## Parameter Summary

| Parameter | Value | Source |
|-----------|-------|--------|
| ρ₀ | ≈ 1.83 g/cm³ | Shannon radii + η_RCP (parameter-free) |
| E₀ | 0.322 eV | Fitted from 13 PCC points |
| α_E | 0.163 eV/v_f | Fitted from 13 PCC points |
| v_c | 0.125 | Fitted from 10 SCC points |
| n | 3.0 | Fitted (explained by CN ≥ 3 threshold) |
| β₀ | −8.447 | Fitted from 23 σ points |
| β₁ | 6.534 | Fitted from 23 σ points |
| β₃ | −21.257 | Fitted from 23 σ points |

## Citation

If you use this framework, please cite both this work and Kim et al.:

```bibtex
@article{iizumi2026ssoc_lps,
  title={Structure-Selective Orbital Coupling analysis of ionic conductivity
         in amorphous Li-P-S solid electrolytes},
  author={Iizumi, Masamichi and Iizumi, Tamaki},
  journal={[to be submitted]},
  year={2026}
}

@article{kim2026origin,
  title={Origin of Optimal Composition and Density for Li-Ion Diffusion
         in Amorphous Li-P-S Solid Electrolytes},
  author={Kim, [First] and others},
  journal={Advanced Energy Materials},
  year={2026}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.

## Contact

Masamichi Iizumi — masa@miosync.com
Miosync, Inc.
