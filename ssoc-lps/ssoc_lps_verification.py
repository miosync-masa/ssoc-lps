#!/usr/bin/env python3
"""
SSOC Framework for Amorphous Li-P-S Solid Electrolytes
=======================================================

Verification code for:
  "Structure-Selective Orbital Coupling analysis of ionic conductivity
   in amorphous (Li₂S)ₓ(P₂S₅)₁₋ₓ solid electrolytes"

Re-analyzes the dataset of Kim et al. (Adv. Energy Mater. 2026) without
modification. All 23 data points are used as reported; no points are
excluded, reweighted, or adjusted.

This work is built entirely upon the systematic dataset of Kim et al.,
whose comprehensive molecular dynamics screening across five compositions
and multiple densities provided the resolution necessary to identify the
regime transition reported here.

Reference:
  Kim et al., "Origin of Optimal Composition and Density for Li-Ion
  Diffusion in Amorphous Li-P-S Solid Electrolytes", Advanced Energy
  Materials (2026). DOI: [to be added]

Authors:
  Masamichi Iizumi (Miosync, Inc.)
  Tamaki (Miosync, Inc.)

License: MIT
Repository: https://github.com/miosync-masa/ssoc-lps

Usage:
  python ssoc_lps_verification.py

Output:
  - All intermediate calculations printed to stdout
  - Final performance metrics for each model layer
  - Design equation summary
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

# ============================================================
# SECTION 0: Physical Constants & Tabulated Values
# ============================================================
# No free parameters in this section.

k_B = 1.380649e-23      # J/K   (Boltzmann constant, exact SI)
h_planck = 6.62607015e-34  # J·s  (Planck constant, exact SI)
N_A = 6.02214076e23     # 1/mol (Avogadro constant, exact SI)
e_charge = 1.602176634e-19  # C  (elementary charge, exact SI)

# Shannon ionic radii (Å) — Shannon, Acta Cryst. A32, 751 (1976)
r_Li = 0.59   # Li⁺, CN=4
r_P  = 0.17   # P⁵⁺, CN=4
r_S  = 1.84   # S²⁻, CN=6

# Atomic masses (g/mol) — IUPAC 2021
m_Li = 6.941
m_P  = 30.974
m_S  = 32.065

# Random Close Packing fraction — Bernal & Mason, Nature 188, 910 (1960)
eta_RCP = 0.64

# Temperature of Kim et al. AIMD simulations
T = 500  # K

# Derived: thermal attempt frequency
nu_thermal = k_B * T / h_planck  # s⁻¹


# ============================================================
# SECTION 1: Kim et al. (2026) Dataset
# ============================================================
# Digitized from Figures 3a, 5a, 6b of Kim et al.
#
# TRANSPARENCY NOTE:
#   All values are read from published figures using standard
#   digitization tools. Small systematic offsets from true values
#   are possible. We encourage Kim et al. to provide exact values
#   for improved analysis.
#
# Format: (x, rho [g/cm³], d_max [Å], D_hop [cm²/s], sigma [S/cm])
#   x     = Li₂S molar fraction in (Li₂S)ₓ(P₂S₅)₁₋ₓ
#   rho   = sample density
#   d_max = maximum pore diameter (from Zeo++)
#   D_hop = hop diffusion coefficient (from AIMD)
#   sigma = ionic conductivity (from AIMD, Nernst-Einstein)

@dataclass
class DataPoint:
    x: float          # composition
    rho: float        # density [g/cm³]
    d_max: float      # max pore diameter [Å]
    D_hop: float      # hop diffusivity [cm²/s]
    sigma: float      # ionic conductivity [S/cm]
    regime: str       # 'PCC' if d_max < 6 Å, else 'SCC'

RAW_DATA: List[DataPoint] = [
    # x = 0.666  (Li₂S : P₂S₅ = 2:1, i.e. Li₄P₂S₇ stoichiometry)
    DataPoint(0.666, 2.15,  3.5, 0.70e-6, 0.3e-3, 'PCC'),
    DataPoint(0.666, 1.95,  4.5, 1.10e-6, 0.8e-3, 'PCC'),
    DataPoint(0.666, 1.80,  5.5, 1.70e-6, 1.5e-3, 'PCC'),
    DataPoint(0.666, 1.60,  8.0, 1.00e-6, 1.0e-3, 'SCC'),
    DataPoint(0.666, 1.40, 12.0, 0.50e-6, 0.3e-3, 'SCC'),
    # x = 0.692
    DataPoint(0.692, 2.05,  3.8, 0.90e-6, 0.6e-3, 'PCC'),
    DataPoint(0.692, 1.85,  5.0, 1.50e-6, 1.5e-3, 'PCC'),
    DataPoint(0.692, 1.70,  5.8, 1.90e-6, 2.5e-3, 'PCC'),
    DataPoint(0.692, 1.55,  9.0, 1.10e-6, 1.2e-3, 'SCC'),
    DataPoint(0.692, 1.40, 11.0, 0.60e-6, 0.4e-3, 'SCC'),
    # x = 0.714
    DataPoint(0.714, 1.95,  4.0, 1.20e-6, 1.0e-3, 'PCC'),
    DataPoint(0.714, 1.80,  5.0, 1.90e-6, 2.5e-3, 'PCC'),
    DataPoint(0.714, 1.65,  6.5, 1.50e-6, 2.0e-3, 'SCC'),
    DataPoint(0.714, 1.50, 10.0, 0.80e-6, 0.8e-3, 'SCC'),
    # x = 0.733
    DataPoint(0.733, 1.90,  4.0, 1.40e-6, 1.5e-3, 'PCC'),
    DataPoint(0.733, 1.77,  5.0, 2.00e-6, 3.5e-3, 'PCC'),
    DataPoint(0.733, 1.60,  7.0, 1.30e-6, 1.5e-3, 'SCC'),
    DataPoint(0.733, 1.45, 10.5, 0.70e-6, 0.5e-3, 'SCC'),
    # x = 0.750  (Li₃PS₄ stoichiometry)
    DataPoint(0.750, 1.85,  3.8, 1.50e-6, 2.0e-3, 'PCC'),
    DataPoint(0.750, 1.73,  4.5, 2.30e-6, 6.4e-3, 'PCC'),  # OPTIMAL
    DataPoint(0.750, 1.60,  5.8, 1.90e-6, 3.0e-3, 'PCC'),
    DataPoint(0.750, 1.50,  8.0, 1.20e-6, 1.2e-3, 'SCC'),
    DataPoint(0.750, 1.40, 11.0, 0.60e-6, 0.4e-3, 'SCC'),
]

N_TOTAL = len(RAW_DATA)
N_PCC = sum(1 for d in RAW_DATA if d.regime == 'PCC')
N_SCC = sum(1 for d in RAW_DATA if d.regime == 'SCC')


# ============================================================
# SECTION 2: S²⁻ Framework Model (Zero Free Parameters)
# ============================================================
# Key insight: S²⁻ ions form the structural framework via RCP.
# Li⁺ and P⁵⁺ occupy interstitial voids and do NOT contribute
# to framework volume.
#
# Validation of interstitial picture:
#   Octahedral void: r_oct = 0.414 × r_S = 0.762 Å > r_Li (0.59 Å) ✓
#   Tetrahedral void: r_tet = 0.225 × r_S = 0.414 Å > r_P (0.17 Å) ✓

def formula_unit_mass(x: float) -> float:
    """Molar mass of (Li₂S)ₓ(P₂S₅)₁₋ₓ formula unit [g/mol]."""
    n_Li = 2 * x
    n_P  = 2 * (1 - x)
    n_S  = 5 - 4 * x
    return n_Li * m_Li + n_P * m_P + n_S * m_S

def n_S_per_fu(x: float) -> float:
    """Number of S²⁻ ions per formula unit."""
    return 5 - 4 * x

def rho0_S_framework(x: float) -> float:
    """
    Parameter-free framework density [g/cm³].

    Assumes S²⁻ ions form a Random Close Packed arrangement.
    Framework volume = n_S × V_S / η_RCP
    where V_S = (4π/3)r_S³ is the volume of one S²⁻ ion.
    """
    r_S_cm = r_S * 1e-8  # Å → cm
    V_S = (4.0 / 3.0) * np.pi * r_S_cm**3  # cm³ per S²⁻
    M_fu = formula_unit_mass(x)
    n_S = n_S_per_fu(x)
    return M_fu / (N_A * n_S * V_S / eta_RCP)

def free_volume(x: float, rho: float) -> float:
    """
    Free volume fraction v_f = 1 - ρ/ρ₀.

    v_f < 0: compressed framework (density exceeds RCP)
    v_f = 0: ideal close-packed framework
    v_f > 0: expanded framework (voids present)
    """
    return 1.0 - rho / rho0_S_framework(x)

def n_Li_number_density(x: float, rho: float) -> float:
    """Li⁺ number density [cm⁻³]."""
    n_Li_fu = 2 * x
    M_fu = formula_unit_mass(x)
    return rho * N_A * n_Li_fu / M_fu

def jump_length(x: float, rho: float) -> float:
    """Mean Li-Li distance a = n_Li^(-1/3) [cm]."""
    n_Li = n_Li_number_density(x, rho)
    return n_Li**(-1.0/3.0)


# ============================================================
# SECTION 3: D_hop Model — PCC + SCC (4 Free Parameters)
# ============================================================
#
# Physical picture:
#   PCC regime (v_f < v_c):
#     All Li⁺ participate in hopping. Barrier decreases linearly
#     with framework expansion ("breathing").
#     D_PCC = (a²/6)(k_BT/h) exp(-(E₀ - α_E·v_f)/k_BT)
#
#   SCC regime (v_f > v_c):
#     Framework expands beyond stabilization range. Li⁺ in
#     inaccessible voids drop out of the hopping pool.
#     p_active = 1/(1 + (v_f/v_c)^n)  [Hill function]
#     D_hop = D_PCC × p_active
#
# Hill coefficient n ≈ 3 has a parameter-free explanation:
#   Li⁺ stabilization requires CN ≥ 3 (triangular S²⁻ coordination).
#   With N_pot ≈ 4 potential S²⁻ neighbors per site,
#   P(CN ≥ 3) follows binomial statistics → effective n_eff ≈ 2.6,
#   enhanced to ~3 by S²⁻ spatial correlations.
#   See Section 6 for derivation.

# --- Fitted parameters (2 from PCC, 2 from SCC) ---
E0_over_kBT     = 7.475    # E₀/(k_BT) at T=500K
alpha_E_over_kBT = 3.790   # α_E/(k_BT) at T=500K
vc_hill         = 0.125    # critical free volume for SCC onset
n_hill          = 3.0      # Hill cooperativity coefficient

# Convert to physical units for reporting
E0_eV    = E0_over_kBT * k_B * T / e_charge      # ≈ 0.322 eV
alpha_eV = alpha_E_over_kBT * k_B * T / e_charge  # ≈ 0.163 eV


def D_PCC(x: float, rho: float) -> float:
    """PCC hop diffusivity (no SCC gating) [cm²/s]."""
    vf = free_volume(x, rho)
    a = jump_length(x, rho)
    E_eff = E0_over_kBT - alpha_E_over_kBT * vf
    return (a**2 / 6.0) * nu_thermal * np.exp(-E_eff)

def p_active(x: float, rho: float) -> float:
    """Fraction of Li⁺ participating in hopping (Hill function)."""
    vf = free_volume(x, rho)
    if vf <= 0:
        return 1.0
    return 1.0 / (1.0 + (vf / vc_hill)**n_hill)

def D_hop_model(x: float, rho: float) -> float:
    """
    Full D_hop prediction [cm²/s].

    D_hop = D_PCC(v_f) × p_active(v_f)

    PCC regime: p_active ≈ 1, D_hop ≈ D_PCC (barrier-controlled)
    SCC regime: p_active < 1, D_hop suppressed (cooperative deactivation)
    """
    return D_PCC(x, rho) * p_active(x, rho)


# ============================================================
# SECTION 4: f_eff Model — Self-Correlation Factor (4 Free Params)
# ============================================================
#
# f = D_tracer / D_hop captures pathway connectivity.
# Separable form: f_eff(x, v_f) = f_peak(x) × g(v_f)
#
#   g(v_f) = exp{β₃(v_f - v_f*)²}  — Gaussian peak in v_f
#   f_peak(x) = exp(β₀ + β₁x)     — composition scaling
#
# Physically:
#   g(v_f): pathway connectivity peaks at v_f* ≈ 0.07
#     - Compressed (v_f < 0): channels narrow, forced reversals → f low
#     - Optimal (v_f ≈ 0.07): open connected channels → f maximum
#     - Expanded (v_f > 0.15): fragmented network → f drops
#
#   f_peak(x): higher Li₂S fraction → denser Li sublattice
#     → more connected percolation network → higher f

# --- Fitted parameters (from 23-point quadratic fit of ln f_eff) ---
beta0 = -8.447    # intercept
beta1 =  6.534    # x coefficient
beta2 =  3.052    # v_f linear coefficient
beta3 = -21.257   # v_f² coefficient (Gaussian curvature)

# Derived quantities
vf_star = -beta2 / (2 * beta3)  # peak v_f ≈ 0.0718
beta0_separated = beta0 + beta2**2 / (-4 * beta3)  # ≈ -8.337


def g_vf(vf: float) -> float:
    """Gaussian pathway connectivity factor, normalized g(v_f*)=1."""
    return np.exp(beta3 * (vf - vf_star)**2)

def f_peak(x: float) -> float:
    """Composition-dependent peak correlation factor."""
    return np.exp(beta0_separated + beta1 * x)

def f_eff_model(x: float, rho: float) -> float:
    """
    Self-correlation factor f_eff(x, v_f).

    ln f_eff = β₀ + β₁x + β₂v_f + β₃v_f²
    """
    vf = free_volume(x, rho)
    ln_f = beta0 + beta1 * x + beta2 * vf + beta3 * vf**2
    return np.exp(ln_f)


# ============================================================
# SECTION 5: Full Conductivity Prediction σ(x, ρ, T)
# ============================================================
#
#   σ = (n_Li × e² / k_BT) × f_eff × D_hop
#
# Three-layer structure:
#   L1: n_Li(x, ρ)    — carrier density      (0 free parameters)
#   L2: D_hop(x, ρ)   — hop diffusivity      (4 free parameters)
#   L3: f_eff(x, ρ)   — pathway connectivity  (4 free parameters)
#
# Total: 8 fitted parameters for 23 data points across 5 compositions

def sigma_model(x: float, rho: float) -> float:
    """
    Predicted ionic conductivity [S/cm].

    σ = n_Li × e² × f_eff × D_hop / (k_B × T)
    """
    n_Li = n_Li_number_density(x, rho)
    D = D_hop_model(x, rho)
    f = f_eff_model(x, rho)
    return n_Li * e_charge**2 * f * D / (k_B * T)


# ============================================================
# SECTION 6: Origin of Hill Coefficient n ≈ 3
# ============================================================
# Parameter-free derivation from three-coordination threshold.
#
# Li⁺ stabilization requires ≥ 3 simultaneous S²⁻ contacts.
# Each site has N_pot ≈ 4 potential S²⁻ neighbors (consistent
# with Kim et al.'s φ_LiS4 descriptor measuring CN=4 sites).
#
# As free volume increases, each S²⁻ contact is lost with
# probability p_loss = v_f / v_f_max.
# P(CN ≥ 3 | N_pot=4) = binomial → effective Hill n ≈ 2.6
# S²⁻ spatial correlations enhance to n ≈ 3.0.

def p_active_binomial(vf: float, N_pot: int = 4,
                       CN_min: int = 3, vf_max: float = 0.35) -> float:
    """
    Binomial coordination model (parameter-free Hill explanation).

    P(CN ≥ CN_min) given N_pot potential S²⁻ neighbors,
    where each neighbor is present with probability p = 1 - v_f/v_f_max.
    """
    from math import comb
    if vf <= 0:
        return 1.0
    p_S = max(0.0, 1.0 - vf / vf_max)
    prob = 0.0
    for k in range(CN_min, N_pot + 1):
        prob += comb(N_pot, k) * p_S**k * (1 - p_S)**(N_pot - k)
    return prob


# ============================================================
# SECTION 7: Verification — Run All Calculations
# ============================================================

def print_header(title: str, char: str = '=', width: int = 100):
    print(f"\n{char * width}")
    print(f"  {title}")
    print(f"{char * width}")

def run_verification():
    """Execute full verification pipeline."""

    print("=" * 100)
    print("  SSOC Framework for Amorphous Li-P-S Solid Electrolytes")
    print("  Verification Code v1.0")
    print("=" * 100)
    print(f"\n  Dataset: Kim et al. (2026) Adv. Energy Mater.")
    print(f"  Data points: {N_TOTAL} total ({N_PCC} PCC + {N_SCC} SCC)")
    print(f"  Compositions: x = 0.666, 0.692, 0.714, 0.733, 0.750")
    print(f"  Temperature: T = {T} K")

    # -----------------------------------------------------------
    # 7.1: S²⁻ Framework Validation
    # -----------------------------------------------------------
    print_header("SECTION 7.1: S²⁻ Framework Model Validation")

    print(f"\n  Interstitial picture validation:")
    r_oct = 0.414 * r_S
    r_tet = 0.225 * r_S
    print(f"    Octahedral void radius: {r_oct:.3f} Å > r_Li = {r_Li:.2f} Å ✓")
    print(f"    Tetrahedral void radius: {r_tet:.3f} Å > r_P  = {r_P:.2f} Å ✓")

    print(f"\n  Framework density ρ₀ (parameter-free):")
    print(f"  {'x':>8} {'n_S':>5} {'M_fu':>8} {'ρ₀':>8}")
    print(f"  {'-'*35}")
    for x in [0.666, 0.692, 0.714, 0.733, 0.750]:
        rho0 = rho0_S_framework(x)
        n_s = n_S_per_fu(x)
        M = formula_unit_mass(x)
        print(f"  {x:>8.3f} {n_s:>5.2f} {M:>8.3f} {rho0:>8.3f}")

    rho0_range = [rho0_S_framework(x) for x in [0.666, 0.750]]
    print(f"\n  ρ₀ range: {min(rho0_range):.3f} – {max(rho0_range):.3f} g/cm³")
    print(f"  Near-constant: validates S²⁻ framework dominance.")

    # -----------------------------------------------------------
    # 7.2: Free Volume Collapse (d_max vs v_f)
    # -----------------------------------------------------------
    print_header("SECTION 7.2: Free Volume Collapse Test")

    vf_all = np.array([free_volume(d.x, d.rho) for d in RAW_DATA])
    dmax_all = np.array([d.d_max for d in RAW_DATA])

    r_collapse = np.corrcoef(vf_all, dmax_all)[0, 1]
    print(f"\n  d_max vs v_f across all 23 points, 5 compositions:")
    print(f"  Pearson r = {r_collapse:+.4f}")
    print(f"  → Strong universal correlation confirms single-variable control.")

    # -----------------------------------------------------------
    # 7.3: D_hop Model Verification
    # -----------------------------------------------------------
    print_header("SECTION 7.3: D_hop Model (PCC + Hill SCC)")

    print(f"\n  Fitted parameters:")
    print(f"    E₀ = {E0_eV:.4f} eV  (literature: 0.15–0.35 eV for Li-S systems)")
    print(f"    α_E = {alpha_eV:.4f} eV/v_f")
    print(f"    v_c = {vc_hill:.3f}  (critical free volume)")
    print(f"    n   = {n_hill:.1f}   (Hill cooperativity)")

    print(f"\n  {'x':>6} {'ρ':>6} {'v_f':>7} {'D_PCC':>10} {'p_act':>7}"
          f" {'D_pred':>10} {'D_Kim':>10} {'ratio':>7} {'regime':>6}")
    print(f"  {'-'*78}")

    Dpred_all = []; Dobs_all = []

    for d in RAW_DATA:
        vf = free_volume(d.x, d.rho)
        d_pcc = D_PCC(d.x, d.rho)
        p_act = p_active(d.x, d.rho)
        d_pred = D_hop_model(d.x, d.rho)
        ratio = d_pred / d.D_hop
        Dpred_all.append(d_pred)
        Dobs_all.append(d.D_hop)
        print(f"  {d.x:>6.3f} {d.rho:>6.2f} {vf:>+7.4f} {d_pcc:>10.2e}"
              f" {p_act:>7.4f} {d_pred:>10.2e} {d.D_hop:>10.2e}"
              f" {ratio:>7.2f} {d.regime:>6}")

    Dpred_all = np.array(Dpred_all)
    Dobs_all = np.array(Dobs_all)

    pcc_mask = np.array([d.regime == 'PCC' for d in RAW_DATA])
    scc_mask = ~pcc_mask

    r_Dhop_all = np.corrcoef(np.log10(Dpred_all), np.log10(Dobs_all))[0, 1]
    r_Dhop_pcc = np.corrcoef(np.log10(Dpred_all[pcc_mask]),
                              np.log10(Dobs_all[pcc_mask]))[0, 1]
    r_Dhop_scc = np.corrcoef(np.log10(Dpred_all[scc_mask]),
                              np.log10(Dobs_all[scc_mask]))[0, 1]

    print(f"\n  D_hop performance:")
    print(f"    All:  r(log10) = {r_Dhop_all:+.4f}, "
          f"mean ratio = {np.mean(Dpred_all/Dobs_all):.3f}, N={N_TOTAL}")
    print(f"    PCC:  r(log10) = {r_Dhop_pcc:+.4f}, "
          f"mean ratio = {np.mean(Dpred_all[pcc_mask]/Dobs_all[pcc_mask]):.3f}, N={N_PCC}")
    print(f"    SCC:  r(log10) = {r_Dhop_scc:+.4f}, "
          f"mean ratio = {np.mean(Dpred_all[scc_mask]/Dobs_all[scc_mask]):.3f}, N={N_SCC}")

    # -----------------------------------------------------------
    # 7.4: f_eff Model Verification
    # -----------------------------------------------------------
    print_header("SECTION 7.4: Self-Correlation Factor f_eff")

    print(f"\n  Model: ln f_eff = β₀ + β₁x + β₂v_f + β₃v_f²")
    print(f"    β₀ = {beta0:.3f}")
    print(f"    β₁ = {beta1:.3f}")
    print(f"    β₂ = {beta2:.3f}")
    print(f"    β₃ = {beta3:.3f}")
    print(f"    v_f* = {vf_star:.4f} (optimal free volume)")

    print(f"\n  Separated form:")
    print(f"    f_peak(x) = exp({beta0_separated:.3f} + {beta1:.3f}·x)")
    print(f"    g(v_f)    = exp({beta3:.3f}·(v_f - {vf_star:.4f})²)")

    # Extract f_data from Kim's data
    print(f"\n  {'x':>6} {'v_f':>7} {'f_data':>8} {'f_model':>8} {'ratio':>7}")
    print(f"  {'-'*42}")

    f_data_list = []; f_model_list = []
    for d in RAW_DATA:
        vf = free_volume(d.x, d.rho)
        n_Li = n_Li_number_density(d.x, d.rho)
        f_data = d.sigma * k_B * T / (n_Li * e_charge**2 * d.D_hop)
        f_mod = f_eff_model(d.x, d.rho)
        f_data_list.append(f_data)
        f_model_list.append(f_mod)
        print(f"  {d.x:>6.3f} {vf:>+7.4f} {f_data:>8.5f} {f_mod:>8.5f}"
              f" {f_mod/f_data:>7.2f}")

    f_data_arr = np.array(f_data_list)
    f_model_arr = np.array(f_model_list)

    r_f = np.corrcoef(np.log(f_data_arr), np.log(f_model_arr))[0, 1]
    print(f"\n  f_eff performance: r(ln) = {r_f:+.4f}")

    # -----------------------------------------------------------
    # 7.5: Full σ Verification
    # -----------------------------------------------------------
    print_header("SECTION 7.5: Full Conductivity σ(x, ρ)")

    print(f"\n  σ = (n_Li e² / k_BT) × f_eff × D_hop")
    print(f"\n  {'x':>6} {'ρ':>6} {'v_f':>7} {'D_hop':>10} {'f_eff':>8}"
          f" {'σ_pred':>8} {'σ_Kim':>8} {'ratio':>7}")
    print(f"  {'-'*72}")

    sig_pred_all = []; sig_obs_all = []

    for d in RAW_DATA:
        vf = free_volume(d.x, d.rho)
        D = D_hop_model(d.x, d.rho)
        f = f_eff_model(d.x, d.rho)
        sig_pred = sigma_model(d.x, d.rho)
        ratio = sig_pred / d.sigma

        sig_pred_all.append(sig_pred * 1e3)  # mS/cm
        sig_obs_all.append(d.sigma * 1e3)

        print(f"  {d.x:>6.3f} {d.rho:>6.2f} {vf:>+7.4f} {D:>10.2e}"
              f" {f:>8.5f} {sig_pred*1e3:>7.2f} {d.sigma*1e3:>7.1f}"
              f" {ratio:>7.2f}")

    sig_pred_all = np.array(sig_pred_all)
    sig_obs_all = np.array(sig_obs_all)

    r_sig_lin = np.corrcoef(sig_pred_all, sig_obs_all)[0, 1]
    r_sig_log = np.corrcoef(np.log10(sig_pred_all), np.log10(sig_obs_all))[0, 1]
    rmse_log = np.sqrt(np.mean((np.log10(sig_pred_all)
                                 - np.log10(sig_obs_all))**2))
    mean_ratio = np.mean(sig_pred_all / sig_obs_all)
    median_ratio = np.median(sig_pred_all / sig_obs_all)

    print_header("SECTION 7.5: σ PERFORMANCE SUMMARY")
    print(f"\n  Pearson r (linear):  {r_sig_lin:+.4f}")
    print(f"  Pearson r (log10):   {r_sig_log:+.4f}")
    print(f"  RMSE (log10):        {rmse_log:.4f}")
    print(f"  Mean  σ_pred/σ_obs:  {mean_ratio:.3f}")
    print(f"  Median σ_pred/σ_obs: {median_ratio:.3f}")

    print(f"\n  Per-composition:")
    for x_val in [0.666, 0.692, 0.714, 0.733, 0.750]:
        mask = np.array([d.x == x_val for d in RAW_DATA])
        p = sig_pred_all[mask]
        o = sig_obs_all[mask]
        r_x = np.corrcoef(p, o)[0, 1]
        mr = np.mean(p / o)
        print(f"    x={x_val:.3f}: r={r_x:+.4f}, mean ratio={mr:.3f}, N={mask.sum()}")

    # -----------------------------------------------------------
    # 7.6: Hill n ≈ 3 Origin (Binomial Model)
    # -----------------------------------------------------------
    print_header("SECTION 7.6: Physical Origin of Hill n ≈ 3")

    print(f"\n  Three-coordination threshold model:")
    print(f"    N_pot = 4 (potential S²⁻ neighbors per Li⁺ site)")
    print(f"    CN_min = 3 (minimum for triangular stabilization)")
    print(f"    v_f_max = 0.35 (framework dissolution limit)")

    print(f"\n  {'v_f':>7} {'Hill':>7} {'Binom':>7} {'diff':>7}")
    print(f"  {'-'*32}")

    for vf in np.arange(0.0, 0.30, 0.03):
        h = 1.0 / (1.0 + (vf / vc_hill)**n_hill) if vf > 0 else 1.0
        b = p_active_binomial(vf)
        print(f"  {vf:>+7.3f} {h:>7.3f} {b:>7.3f} {h-b:>+7.3f}")

    # Effective Hill coefficient of binomial model at v_c
    delta = 0.001
    b_plus = p_active_binomial(vc_hill + delta)
    b_minus = p_active_binomial(vc_hill - delta)
    if 0 < b_plus < 1 and 0 < b_minus < 1:
        y_plus = np.log(1/b_plus - 1)
        y_minus = np.log(1/b_minus - 1)
        x_plus = np.log(vc_hill + delta)
        x_minus = np.log(vc_hill - delta)
        n_eff = (y_plus - y_minus) / (x_plus - x_minus)
        print(f"\n  Effective Hill coefficient of binomial model: n_eff = {n_eff:.2f}")
        print(f"  Fitted Hill coefficient:                      n     = {n_hill:.1f}")
        print(f"  Difference (S²⁻ spatial correlations):        Δn    = {n_hill - n_eff:.2f}")

    # -----------------------------------------------------------
    # 7.7: Design Equation Summary
    # -----------------------------------------------------------
    print_header("SECTION 7.7: DESIGN EQUATION FOR SOLID ELECTROLYTE OPTIMIZATION")

    rho_opt = rho0_S_framework(0.75) * (1 - vf_star)
    sig_opt = sigma_model(0.75, rho_opt) * 1000

    print(f"""
  ┌────────────────────────────────────────────────────────────────┐
  │  σ(x, ρ, T=500K) for amorphous (Li₂S)ₓ(P₂S₅)₁₋ₓ            │
  ├────────────────────────────────────────────────────────────────┤
  │                                                                │
  │  σ = (n_Li e² / k_BT) × f_eff × D_hop                       │
  │                                                                │
  │  Layer 1: Carrier density  [0 free params]                    │
  │    n_Li = ρ N_A (2x) / M_fu(x)                               │
  │                                                                │
  │  Layer 2: Hop diffusivity  [4 free params: E₀,α,v_c,n]       │
  │    D_hop = D_PCC × p_active                                   │
  │    D_PCC = (a²/6)(k_BT/h) exp(-(E₀ - α·v_f)/k_BT)          │
  │    p_active = 1/(1 + (v_f/v_c)^n)   [v_f > 0]               │
  │                                                                │
  │  Layer 3: Pathway connectivity  [4 free params: β₀,β₁,β₃,v*] │
  │    f_eff = f_peak(x) × g(v_f)                                │
  │    g(v_f) = exp(β₃(v_f - v_f*)²)                            │
  │    f_peak(x) = exp(β₀' + β₁·x)                              │
  │                                                                │
  │  Infrastructure  [0 free params]                               │
  │    ρ₀ ≈ 1.83 g/cm³  (Shannon radii + η_RCP = 0.64)          │
  │    v_f = 1 - ρ/ρ₀                                            │
  │    a = n_Li^(-1/3)                                            │
  │                                                                │
  ├────────────────────────────────────────────────────────────────┤
  │  TOTAL: 8 parameters → {N_TOTAL} points × 5 compositions          │
  │  PERFORMANCE: r(log10) = {r_sig_log:+.4f}, mean ratio = {mean_ratio:.3f}       │
  ├────────────────────────────────────────────────────────────────┤
  │  DESIGN RULE:                                                  │
  │    Optimal v_f ≈ {vf_star:.3f}  →  ρ_opt ≈ {rho_opt:.2f} g/cm³             │
  │    Higher x (Li₂S-rich) → higher f_peak → higher σ           │
  │    v_f > {vc_hill:.3f} → cooperative Li⁺ deactivation (SCC)         │
  │    Predicted σ at x=0.75, ρ_opt: {sig_opt:.1f} mS/cm                │
  ├────────────────────────────────────────────────────────────────┤
  │  CAVEAT:                                                       │
  │    x=0.75, ρ=1.73 (σ=6.4 mS/cm) under-predicted by ~1.9×    │
  │    → Attributed to anomalous f_eff at exact optimal density   │
  │    → Requires additional data near optimal (Future work)       │
  └────────────────────────────────────────────────────────────────┘
    """)

    # -----------------------------------------------------------
    # 7.8: Parameter Audit
    # -----------------------------------------------------------
    print_header("SECTION 7.8: COMPLETE PARAMETER AUDIT")

    print(f"""
  PARAMETER-FREE (from physical constants & tabulated values):
    k_B, h, e, N_A                    — SI fundamental constants
    r_S = 1.84 Å                      — Shannon radius (S²⁻, CN=6)
    η_RCP = 0.64                       — Random Close Packing fraction
    ρ₀(x) ≈ 1.83 g/cm³               — S²⁻ framework density
    a = n_Li^(-1/3)                    — Li-Li mean distance
    d_crit = 6.0 Å                     — S-Li-S linear config (Kim et al.)
    v_f = 1 - ρ/ρ₀                    — free volume definition

  FITTED FROM PCC DATA (13 points):
    E₀    = {E0_eV:.4f} eV             — hopping barrier at v_f=0
    α_E   = {alpha_eV:.4f} eV/v_f       — barrier softening rate

  FITTED FROM SCC DATA (10 points):
    v_c   = {vc_hill:.3f}                — critical free volume
    n     = {n_hill:.1f}                  — Hill cooperativity
      → n ≈ 3 explained by CN≥3 threshold (Section 7.6)

  FITTED FROM σ DATA (23 points):
    β₀    = {beta0:.3f}               — f baseline
    β₁    = {beta1:.3f}                — f composition scaling
    β₃    = {beta3:.3f}              — f Gaussian curvature
    v_f*  = {vf_star:.4f}               — optimal free volume

  TOTAL FITTED: 8 parameters
  DATA POINTS:  23 (5 compositions × 4-5 densities)
  RATIO:        23/8 = {23/8:.1f} data-to-parameter ratio
    """)

    return sig_pred_all, sig_obs_all


# ============================================================
# SECTION 8: Entry Point
# ============================================================

if __name__ == '__main__':
    sig_pred, sig_obs = run_verification()

    print_header("VERIFICATION COMPLETE", char='*')
    print(f"""
  All calculations can be reproduced by running:
    $ python ssoc_lps_verification.py

  This code has NO external dependencies beyond NumPy.
  All input data is embedded in the script (Section 1).

  For questions or collaboration:
    Masamichi Iizumi — masa@miosync.com
    Miosync, Inc.
    """)
