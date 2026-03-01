"""
SSOC Thermal v5.0 — Publication Unified Figures
================================================
Merges v4.5c (wet-fraction format selection) + v4.5d-rev (Theta regime map)
into 8 publication-quality figures for the AEM sequel paper.

Figure 1: Design Stack Schematic (L1→L5 with SCC loop)
Figure 2: Theta(T, C-rate) Phase Diagram — 4680
Figure 3: sigma × C-rate → Theta with temperature iso-lines
Figure 4: Regime Map Table (numerical Theta grid)
Figure 5: Wet-Fraction Phase Diagram (Regime II detail)
Figure 6: Industry Scenario Verdict Map
Figure 7: Theta Anatomy & Design Knobs
Figure 8: Design Box 2 Flowchart

Authors: Masamichi Iizumi, Tamaki Iizumi (環)
Date: 2026-02-28
"""

import numpy as np
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import TwoSlopeNorm, BoundaryNorm, LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import matplotlib.patheffects as pe

# ============================================================
# PHYSICS ENGINE (unified from v4.5c + v4.5d-rev)
# ============================================================
k_B = 1.380649e-23; h_pl = 6.62607015e-34; e_q = 1.602176634e-19; N_A = 6.02214076e23
kB_eV = 8.617333e-5  # eV/K
eta_RCP = 0.64; m_Li = 6.941; m_P = 30.974; m_S = 32.065
r_S = 1.84e-10; V_S = (4/3)*math.pi*r_S**3

# SSOC parameters
E0_kBT = 7.475; aE_kBT = 3.790; v_c = 0.125; n_Hill = 3.0
f_peak_a = -8.337; f_peak_b = 6.534; vf_star = 0.0718; beta3 = -21.257


def M_fu(x):
    return 2*x*m_Li + 2*(1-x)*m_P + (5-4*x)*m_S

def rho_0(x):
    return M_fu(x)*eta_RCP / ((5-4*x)*(V_S*1e6)*N_A)

def sigma_SSOC(x, rho, T=500):
    """Full SSOC conductivity at temperature T (default 500K)."""
    nLi = (rho*1e6)*N_A*2*x/M_fu(x)
    a = nLi**(-1/3)
    vf = 1 - rho/rho_0(x)
    E0 = E0_kBT*k_B*500; aE = aE_kBT*k_B*500
    D_pcc = (a**2/6)*(k_B*T/h_pl)*math.exp(-(E0-aE*vf)/(k_B*T))
    p_act = 1.0/(1.0+(max(vf,0)/v_c)**n_Hill)
    f_eff = math.exp(f_peak_a+f_peak_b*x)*math.exp(beta3*(vf-vf_star)**2)
    return nLi*e_q**2*f_eff*D_pcc*p_act/(k_B*T)

def sigma_at_T(sigma_500K, T, Ea=0.32):
    """Arrhenius scaling from 500K reference."""
    return sigma_500K * math.exp(-Ea/kB_eV * (1.0/T - 1.0/500.0))


# ============================================================
# CELL MODEL (unified)
# ============================================================
class CellModel:
    """4680-class cell with full thermal stack."""

    def __init__(self, r_can=23e-3, L=80e-3,
                 t_elec=30e-6, t_cath=70e-6, t_an=80e-6, t_cc=15e-6,
                 kappa_elec=0.50, kappa_electrode=1.5, kappa_cc=238.0,
                 Rpp_ssb=2e-5, Rpp_pack=2e-4,
                 t_can_c=0.3e-3, t_can_p=0.5e-3, k_can=16.0,
                 t_plate=2e-3, k_plate=200.0,
                 energy_density=400e3, dT_crit=50):
        self.r_can = r_can
        self.L = L
        self.r_mandrel = r_can * 0.109
        self.V_cell = math.pi * r_can**2 * L
        self.dT_crit = dT_crit
        self.Rpp_pack = Rpp_pack
        self.energy_density = energy_density

        # Layer thicknesses
        self.layers = [(t_cc, kappa_cc), (t_cath, kappa_electrode),
                       (t_elec, kappa_elec), (t_an, kappa_electrode)]
        self.t_elec = t_elec
        self.t_unit = sum(t for t, _ in self.layers)
        self.n_rep = max(1, int((r_can - self.r_mandrel) / self.t_unit))

        # Build jelly-roll thermal resistance
        Rm = 0.0; Rc = 0.0; r = self.r_mandrel
        r_interfaces = []
        for rep in range(self.n_rep):
            for i, (t_i, k_i) in enumerate(self.layers):
                r_next = r + t_i
                if r_next > r and r > 0:
                    Rm += math.log(r_next/r) / (2*math.pi*k_i*L)
                is_last = (rep == self.n_rep-1) and (i == len(self.layers)-1)
                if not is_last and r_next > 0:
                    Rc += Rpp_ssb / (2*math.pi*r_next*L)
                    r_interfaces.append(r_next)
                r = r_next
        self.Rth_int = Rm + Rc
        self.Rm = Rm; self.Rc = Rc
        self.r_interfaces = np.array(r_interfaces)

        # Ionic area
        self.A_ion = sum(2*math.pi*(self.r_mandrel + (i+0.5)*self.t_unit)*L
                         for i in range(self.n_rep))

        # External areas
        self.A_cyl = 2*math.pi*r_can*L
        W = 46e-3; D = 26e-3; H_p = self.V_cell/(W*D)  # Realistic prismatic dims (v4.5c)
        self.W = W; self.D = D; self.H_p = H_p
        self.A_prism_bottom = W*D
        self.A_prism_side = 2*(W+D)*H_p
        self.A_prism_both = self.A_prism_bottom + self.A_prism_side

        # Packing efficiencies
        self.eta_cyl = math.pi/(2*math.sqrt(3))
        self.eta_prism = 0.98
        self.inv_eta2 = (self.eta_prism/self.eta_cyl)**2

        # Pack thermal resistance components
        self.R_sum_c_base = t_can_c/k_can + Rpp_pack  # + 1/h added per call
        self.R_sum_p_base = t_can_p/k_can + Rpp_pack + t_plate/k_plate

        # Capacity
        self.capacity_Ah = self.V_cell * energy_density / 3.7

    def R_ionic(self, sigma):
        if sigma <= 0 or self.A_ion <= 0:
            return 1e10
        return self.F_geom / sigma

    @property
    def F_geom(self):
        """Geometric ionic resistance factor: R_ionic = F_geom / sigma"""
        return self.t_elec / self.A_ion

    def Lambda_th(self, C_rate, h=300, wf_c=0.75, format='cyl'):
        """Thermal load invariant: Lambda_th = Theta * sigma = const.
        From Proposition 1 (Theta-sigma invariance theorem)."""
        I = C_rate * self.capacity_Ah
        if format == 'cyl':
            Rth_tot = self.Rth_int + self.R_pack_cyl(h, wf_c)
        else:
            Rth_tot = self.Rth_int + self.R_pack_prism(h, wf_c)
        return I**2 * self.F_geom * Rth_tot / self.dT_crit

    def sigma_crit(self, C_rate, Theta_star, h=300, wf_c=0.75, format='cyl'):
        """Closed-form critical conductivity (Corollary of Proposition 1).
        sigma_crit = Lambda_th / Theta_star"""
        return self.Lambda_th(C_rate, h, wf_c, format) / Theta_star

    def R_pack_cyl(self, h, wf=0.75):
        A_eff = wf * self.A_cyl
        if A_eff <= 0: return 1e10
        return (self.R_sum_c_base + 1.0/h) / A_eff

    def R_pack_prism(self, h, wf=0.70, mode='both'):
        if mode == 'both':
            A_eff = wf * self.A_prism_both
        else:
            A_eff = wf * self.A_prism_bottom
        if A_eff <= 0: return 1e10
        return (self.R_sum_p_base + 1.0/h) / A_eff

    def compute_Theta(self, sigma, C_rate, h=300, wf_c=0.75, wf_p=0.70, mode_p='both'):
        """Compute Theta for both formats and the performance ratio."""
        R_ion = self.R_ionic(sigma)
        I = C_rate * self.capacity_Ah  # Amperes (CORRECTED: Ah not As)
        Q_dot = I**2 * R_ion

        Rth_c = self.Rth_int + self.R_pack_cyl(h, wf_c)
        Rth_p = self.Rth_int + self.R_pack_prism(h, wf_p, mode_p)

        Theta_c = Q_dot * Rth_c / self.dT_crit
        Theta_p = Q_dot * Rth_p / self.dT_crit

        I_max_c = math.sqrt(self.dT_crit / (R_ion * Rth_c)) if R_ion > 0 and Rth_c > 0 else 0
        I_max_p = math.sqrt(self.dT_crit / (R_ion * Rth_p)) if R_ion > 0 and Rth_p > 0 else 0
        ratio = (I_max_c*math.sqrt(self.eta_cyl))/(I_max_p*math.sqrt(self.eta_prism)) if I_max_p > 0 else 1

        return {
            "Theta_c": Theta_c, "Theta_p": Theta_p,
            "ratio": ratio, "Q_dot": Q_dot, "I": I,
            "R_ionic": R_ion, "Rth_c": Rth_c, "Rth_p": Rth_p,
        }

    def crit_wf_ratio(self, h):
        """Critical wp/wc for format boundary (from v4.5c)."""
        Rs_c = self.R_sum_c_base + 1.0/h
        Rs_p = self.R_sum_p_base + 1.0/h
        return self.inv_eta2 * (self.A_cyl / self.A_prism_both) * (Rs_c / Rs_p)


# ============================================================
# MAIN: Generate all 8 figures
# ============================================================
def main():
    print("="*90)
    print("  SSOC Thermal v5.0 — Publication Unified Figures")
    print("  Integrating v4.5c (format selection) + v4.5d-rev (Theta regimes)")
    print("="*90)

    cell = CellModel()
    sigma_500K = sigma_SSOC(0.75, 1.72)  # S/m at 500K

    print(f"\n  Cell: 4680 ({cell.r_can*2e3:.0f}mm × {cell.L*1e3:.0f}mm)")
    print(f"  V = {cell.V_cell*1e6:.1f} cm^3, Cap = {cell.capacity_Ah:.2f} Ah")
    print(f"  n_layers = {cell.n_rep}, R_th,int = {cell.Rth_int:.3f} K/W")
    print(f"  sigma(500K) = {sigma_500K*10:.2f} mS/cm")
    for T in [273, 300, 330, 350, 400, 500]:
        s = sigma_at_T(sigma_500K, T)
        print(f"    sigma({T}K) = {s*10:.4f} mS/cm  (×{s/sigma_500K:.5f})")

    # ============================================================
    # COMPUTE DATA
    # ============================================================

    # --- Data for Fig 2: T × C-rate → Theta ---
    T_range = np.linspace(250, 520, 140)
    C_range = np.logspace(-0.5, 1.2, 120)  # 0.3C to 15C
    Theta_TC = np.zeros((len(T_range), len(C_range)))
    ratio_TC = np.zeros((len(T_range), len(C_range)))
    for i, T in enumerate(T_range):
        sig = sigma_at_T(sigma_500K, T)
        for j, C in enumerate(C_range):
            res = cell.compute_Theta(sig, C)
            Theta_TC[i, j] = res["Theta_c"]
            ratio_TC[i, j] = res["ratio"]

    # --- Data for Fig 3: sigma × C-rate → Theta ---
    sigma_range = np.logspace(-3.5, 0.3, 160)  # 0.0003 to 2 S/m
    Theta_sC = np.zeros((len(sigma_range), len(C_range)))
    for i, sig in enumerate(sigma_range):
        for j, C in enumerate(C_range):
            res = cell.compute_Theta(sig, C)
            Theta_sC[i, j] = res["Theta_c"]

    # --- Data for Fig 5: wet-fraction sweep (h=300) ---
    wf_range = np.linspace(0.25, 1.0, 120)
    sig_300 = sigma_at_T(sigma_500K, 300)

    # At h=300, compute format ratio for each (wf_c, wf_p) pair
    ratio_wf = np.zeros((len(wf_range), len(wf_range)))
    for i, wfc in enumerate(wf_range):
        for j, wfp in enumerate(wf_range):
            res = cell.compute_Theta(sig_300, 1.0, h=300, wf_c=wfc, wf_p=wfp)
            ratio_wf[i, j] = res["ratio"]

    # --- Data for Fig 5 panel 2: critical ratio vs h ---
    h_range_fine = np.logspace(1, 3.5, 500)
    crit_ratio_vs_h = np.array([cell.crit_wf_ratio(h) for h in h_range_fine])

    # --- Data for Fig 6: Industry scenarios ---
    scenarios = [
        ("Tesla 4680\n(gap cool)", 0.75, 0.30, "bottom", 300, "tab:blue"),
        ("BYD Blade\n(bottom plate)", 0.75, 0.35, "both", 100, "tab:green"),
        ("Fair test\n(both-side, equal wf)", 0.70, 0.70, "both", 300, "tab:gray"),
        ("Prism best\n(immersion, both)", 0.50, 0.90, "both", 300, "tab:red"),
        ("Next-gen\n(full immersion)", 0.85, 0.85, "both", 1000, "tab:orange"),
    ]

    # ============================================================
    # FIGURE LAYOUT: 4 rows × 2 cols
    # ============================================================
    fig = plt.figure(figsize=(22, 32))
    gs = GridSpec(4, 2, figure=fig, hspace=0.32, wspace=0.28,
                  left=0.06, right=0.96, top=0.96, bottom=0.02)

    panel_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)']

    # ==========================================================
    # Figure 1 (a): Design Stack Schematic
    # ==========================================================
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_xlim(0, 10); ax1.set_ylim(0, 10)
    ax1.axis('off')

    # Draw boxes
    box_specs = [
        (1.0, 8.2, 3.8, 1.2, '#4472C4', 'L1: σ(x,ρ,T)\nSSOC Design Eq.'),
        (1.0, 6.2, 3.8, 1.2, '#ED7D31', 'L2: Q̇ = I²/σ\nJoule Heating'),
        (1.0, 4.2, 3.8, 1.2, '#A5A5A5', 'L3: Θ(σ,T,C,h)\nRegime Class.'),
        (1.0, 2.2, 3.8, 1.2, '#70AD47', 'L4: wp/wc > 0.78\nFormat Select.'),
        (1.0, 0.2, 3.8, 1.2, '#FFC000', 'L5: G_total\nPack Integration'),
    ]
    for x, y, w, h, color, text in box_specs:
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                             facecolor=color, edgecolor='black', lw=2, alpha=0.85)
        ax1.add_patch(box)
        ax1.text(x + w/2, y + h/2, text, ha='center', va='center',
                fontsize=10, fontweight='bold', color='white',
                path_effects=[pe.withStroke(linewidth=2, foreground='black')])

    # Forward arrows (PCC)
    for y_start in [8.2, 6.2, 4.2, 2.2]:
        ax1.annotate('', xy=(2.9, y_start), xytext=(2.9, y_start + 0.6),
                    arrowprops=dict(arrowstyle='->', lw=2.5, color='black'))

    # PCC label
    ax1.text(5.2, 7.0, 'PCC\n(forward)', fontsize=11, ha='center',
            fontweight='bold', color='#4472C4',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # SCC backward arrow: from Regime III back to L1
    ax1.annotate('', xy=(0.5, 8.8), xytext=(0.5, 4.8),
                arrowprops=dict(arrowstyle='->', lw=3, color='#C00000',
                               connectionstyle='arc3,rad=0.4'))
    ax1.text(-0.2, 6.8, 'SCC\n(Regime III\n→ improve σ)', fontsize=9,
            ha='center', fontweight='bold', color='#C00000',
            bbox=dict(boxstyle='round', facecolor='#FFE0E0', alpha=0.9))

    # Regime labels on right
    ax1.text(6.5, 8.8, 'Material\n(SSOC paper [1])', fontsize=9, color='#4472C4',
            bbox=dict(boxstyle='round', facecolor='#D6E4F0', alpha=0.8))
    ax1.text(6.5, 5.8, 'Θ < 0.1 → Regime I (skip L4)\n'
             '0.1 < Θ < 1 → Regime II (use L4)\n'
             'Θ > 1 → Regime III (SCC to L1)',
             fontsize=8.5, color='#333333', va='center',
             bbox=dict(boxstyle='round', facecolor='#F0F0F0', alpha=0.9))
    ax1.text(6.5, 3.0, 'Only active in\nRegime II', fontsize=9, color='#70AD47',
            bbox=dict(boxstyle='round', facecolor='#E2EFDA', alpha=0.8))

    ax1.set_title("(a) Design Stack: σ → Θ → Format Freedom",
                  fontweight='bold', fontsize=13, pad=10)

    # ==========================================================
    # Figure 2 (b): Theta(T, C-rate) Phase Diagram
    # ==========================================================
    ax2 = fig.add_subplot(gs[0, 1])

    # Use log-scaled Theta for better color resolution
    log_Theta = np.log10(np.clip(Theta_TC, 1e-4, 1e4))
    cs2 = ax2.pcolormesh(C_range, T_range, log_Theta,
                          cmap='RdYlGn_r', vmin=-3, vmax=2.5, shading='auto')
    cbar2 = plt.colorbar(cs2, ax=ax2, label="log₁₀(Θ)", shrink=0.9)
    cbar2.set_ticks([-3, -2, -1, 0, 1, 2])
    cbar2.set_ticklabels(['0.001', '0.01', '0.1', '1', '10', '100'])

    # Regime boundaries
    c01 = ax2.contour(C_range, T_range, Theta_TC, levels=[0.1],
                       colors='#00AA00', linewidths=3.5, linestyles='-')
    c10 = ax2.contour(C_range, T_range, Theta_TC, levels=[1.0],
                       colors='#CC0000', linewidths=3.5, linestyles='-')

    # Regime labels
    ax2.text(0.4, 480, "Regime I\nΘ < 0.1", color='#006600',
             fontweight='bold', fontsize=11, ha='center',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    ax2.text(1.5, 330, "II", color='black', fontweight='bold', fontsize=16,
             bbox=dict(boxstyle='round', facecolor='#FFFFCC', alpha=0.7))
    ax2.text(5, 290, "Regime III\nΘ > 1", color='#990000',
             fontweight='bold', fontsize=11, ha='center',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    # Mark key operating points
    key_points = [(1.0, 300, '1C@300K'), (3.0, 300, '3C@300K'),
                  (3.0, 350, '3C@350K'), (1.0, 400, '1C@400K')]
    for cx, cy, label in key_points:
        ax2.plot(cx, cy, 'ko', ms=8, zorder=5)
        ax2.annotate(label, xy=(cx, cy), xytext=(cx*1.3, cy+12),
                    fontsize=8, fontweight='bold',
                    arrowprops=dict(arrowstyle='->', lw=1, color='black'),
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax2.set_xscale('log')
    ax2.set_xlabel("C-rate", fontsize=12)
    ax2.set_ylabel("Temperature [K]", fontsize=12)
    ax2.set_title("(b) Θ Phase Diagram: 4680 cell, h = 300 W/m²K",
                  fontweight='bold', fontsize=13, pad=10)
    ax2.set_ylim(260, 510)

    # ==========================================================
    # Figure 3 (c): sigma × C-rate → Theta
    # ==========================================================
    ax3 = fig.add_subplot(gs[1, 0])

    log_Theta_sC = np.log10(np.clip(Theta_sC, 1e-4, 1e4))
    cs3 = ax3.pcolormesh(C_range, sigma_range*10, log_Theta_sC,
                          cmap='RdYlGn_r', vmin=-3, vmax=2.5, shading='auto')
    cbar3 = plt.colorbar(cs3, ax=ax3, label="log₁₀(Θ)", shrink=0.9)
    cbar3.set_ticks([-3, -2, -1, 0, 1, 2])
    cbar3.set_ticklabels(['0.001', '0.01', '0.1', '1', '10', '100'])

    ax3.contour(C_range, sigma_range*10, Theta_sC, levels=[0.1],
                colors='#00AA00', linewidths=3.5)
    ax3.contour(C_range, sigma_range*10, Theta_sC, levels=[1.0],
                colors='#CC0000', linewidths=3.5)

    # Temperature iso-lines (horizontal)
    temp_marks = [(273, '#0066CC', '273K (0°C)'),
                  (300, '#0099FF', '300K (27°C)'),
                  (330, '#FF9900', '330K (57°C)'),
                  (350, '#FF6600', '350K (77°C)'),
                  (400, '#CC3300', '400K (127°C)'),
                  (500, '#990000', '500K (227°C)')]
    for T, color, label in temp_marks:
        sig_T = sigma_at_T(sigma_500K, T) * 10  # mS/cm
        ax3.axhline(sig_T, color=color, ls='--', lw=1.8, alpha=0.8)
        ax3.text(C_range[-1]*0.85, sig_T*1.15, label, fontsize=7.5,
                color=color, fontweight='bold', ha='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, pad=0.1))

    ax3.set_xscale('log'); ax3.set_yscale('log')
    ax3.set_xlabel("C-rate", fontsize=12)
    ax3.set_ylabel("σ [mS/cm]", fontsize=12)
    ax3.set_title("(c) Θ(σ, C-rate) — Temperature maps onto σ axis",
                  fontweight='bold', fontsize=13, pad=10)
    ax3.set_ylim(0.003, 20)

    # ==========================================================
    # Figure 4 (d): Regime Map Table
    # ==========================================================
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')

    # Build the table
    T_grid = [273, 300, 330, 350, 400, 500]
    C_grid = [0.5, 1.0, 2.0, 3.0, 5.0]

    # Table header
    col_labels = ['T [K]', 'σ [mS/cm]'] + [f'{C}C' for C in C_grid]
    n_cols = len(col_labels)
    n_rows = len(T_grid)

    cell_text = []
    cell_colors = []
    regime_I_color = '#C6EFCE'    # green
    regime_II_color = '#FFEB9C'   # yellow
    regime_III_color = '#FFC7CE'  # red

    for T in T_grid:
        sig = sigma_at_T(sigma_500K, T)
        row_text = [f'{T}', f'{sig*10:.3f}']
        row_colors = ['#E8E8E8', '#E8E8E8']
        for C in C_grid:
            res = cell.compute_Theta(sig, C)
            Th = res["Theta_c"]
            if Th < 0.1:
                tag = 'I'
                color = regime_I_color
            elif Th < 1.0:
                tag = 'II'
                color = regime_II_color
            else:
                tag = 'III'
                color = regime_III_color
            row_text.append(f'{Th:.2f}\n({tag})')
            row_colors.append(color)
        cell_text.append(row_text)
        cell_colors.append(row_colors)

    table = ax4.table(cellText=cell_text, colLabels=col_labels,
                      cellColours=cell_colors,
                      colColours=['#4472C4']*n_cols,
                      loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 2.2)

    # Header text color
    for j in range(n_cols):
        table[0, j].set_text_props(color='white', fontweight='bold')

    ax4.set_title("(d) Θ Regime Map — 4680, h = 300 W/m²K, ΔT_crit = 50 K",
                  fontweight='bold', fontsize=13, pad=15)

    # Subtitle
    ax4.text(0.5, 0.08,
             "Green = Regime I (Θ<0.1, thermal-free) | "
             "Yellow = Regime II (0.1<Θ<1, format matters) | "
             "Red = Regime III (Θ>1, thermal death)",
             transform=ax4.transAxes, ha='center', fontsize=9.5, style='italic')

    # ==========================================================
    # Figure 5 (e): Wet-Fraction Phase Diagram (Regime II)
    # ==========================================================
    ax5 = fig.add_subplot(gs[2, 0])

    norm5 = TwoSlopeNorm(vmin=0.85, vcenter=1.0, vmax=1.35)
    cs5 = ax5.contourf(wf_range, wf_range, ratio_wf.T,
                        levels=np.linspace(0.85, 1.35, 26),
                        cmap='RdYlBu', norm=norm5)
    plt.colorbar(cs5, ax=ax5, label="Pack Perf. Ratio (Cyl/Prism)", shrink=0.9)

    # 1.0 boundary
    ax5.contour(wf_range, wf_range, ratio_wf.T, levels=[1.0],
                colors='black', linewidths=3.5)

    # Analytic boundary line: wp = 0.78 * wc
    cr_300 = cell.crit_wf_ratio(300)
    wc_line = np.linspace(0.25, 1.0, 100)
    wp_line = cr_300 * wc_line
    valid = wp_line <= 1.0
    ax5.plot(wc_line[valid], wp_line[valid], 'k--', lw=2.5, alpha=0.7,
            label=f'Analytic: wp = {cr_300:.3f}·wc')

    # Industry scenario points
    scenario_points = [
        (0.75, 0.30, 'Tesla', 'tab:blue', 's'),
        (0.75, 0.35, 'BYD', 'tab:green', '^'),
        (0.70, 0.70, 'Fair', 'tab:gray', 'D'),
        (0.85, 0.85, 'Next-gen', 'tab:orange', 'o'),
        (0.50, 0.90, 'Prism-best', 'tab:red', 'v'),
    ]
    for wc, wp, label, color, marker in scenario_points:
        ax5.plot(wc, wp, marker=marker, ms=14, color=color, markeredgecolor='black',
                markeredgewidth=1.5, zorder=10)
        ax5.annotate(label, xy=(wc, wp), xytext=(wc+0.03, wp+0.04),
                    fontsize=9, fontweight='bold', color=color)

    # Zone labels
    ax5.text(0.35, 0.85, "PRISM\nwins", fontsize=14, fontweight='bold',
            color='#CC0000', alpha=0.5, ha='center')
    ax5.text(0.85, 0.35, "CYL\nwins", fontsize=14, fontweight='bold',
            color='#0000CC', alpha=0.5, ha='center')

    ax5.set_xlabel("Wet fraction — Cylindrical (wf_c)", fontsize=12)
    ax5.set_ylabel("Wet fraction — Prismatic (wf_p)", fontsize=12)
    ax5.set_title("(e) Format Selection in Regime II (1C, 300K, h=300)",
                  fontweight='bold', fontsize=13, pad=10)
    ax5.legend(fontsize=9, loc='upper left')
    ax5.set_xlim(0.25, 1.0); ax5.set_ylim(0.25, 1.0)

    # ==========================================================
    # Figure 6 (f): Unified Verdict Table
    # ==========================================================
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.axis('off')

    verdict_header = ['Scenario', 'wf_c', 'wf_p', 'h', 'Θ(1C)', 'Regime', 'wp/wc', 'Winner']
    verdict_rows = []
    verdict_colors = []

    for name, wfc, wfp, mode, h_val, _ in scenarios:
        clean_name = name.replace('\n', ' ')
        sig_300K = sigma_at_T(sigma_500K, 300)
        res = cell.compute_Theta(sig_300K, 1.0, h=h_val, wf_c=wfc, wf_p=wfp, mode_p=mode)
        Th = res["Theta_c"]
        regime = "I" if Th < 0.1 else "II" if Th < 1.0 else "III"
        ratio_val = wfp / wfc
        if regime == 'III':
            winner = "N/A (↑σ)"
            wc = regime_III_color
        elif regime == 'I':
            winner = "PRISM"
            wc = regime_I_color
        else:
            winner = "CYL" if ratio_val < 0.78 else "PRISM"
            wc = regime_II_color

        verdict_rows.append([clean_name, f'{wfc:.2f}', f'{wfp:.2f}',
                            f'{h_val}', f'{Th:.2f}', regime, f'{ratio_val:.2f}', winner])
        verdict_colors.append([wc]*8)

    vtable = ax6.table(cellText=verdict_rows, colLabels=verdict_header,
                       cellColours=verdict_colors,
                       colColours=['#4472C4']*8,
                       loc='upper center', cellLoc='center')
    vtable.auto_set_font_size(False)
    vtable.set_fontsize(9.5)
    vtable.scale(1.0, 2.5)
    for j in range(8):
        vtable[0, j].set_text_props(color='white', fontweight='bold')

    # Add critical insight text below
    insight = (
        "━━━ KEY FINDING ━━━\n"
        "• At equal wet fraction (wf_p = wf_c): PRISM always wins (packing advantage)\n"
        "• CYL wins ONLY when natural coolant channels give wf_c >> wf_p\n"
        "• Threshold: wp/wc > 0.78 (weakly h-dependent: 0.75–0.78)\n\n"
        "FORMAT CHOICE IS NOT A PHYSICS QUESTION.\n"
        "IT IS A COOLING ARCHITECTURE QUESTION.\n"
        "(And only relevant in Regime II.)"
    )
    ax6.text(0.5, 0.06, insight, transform=ax6.transAxes,
             ha='center', va='bottom', fontsize=10.5,
             fontfamily='monospace', fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='#FFFFF0', alpha=0.95, pad=0.5))

    ax6.set_title("(f) Unified Industry Verdict (1C at 300K)",
                  fontweight='bold', fontsize=13, pad=15)

    # ==========================================================
    # Figure 7 (g): Theta Anatomy & Design Knobs
    # ==========================================================
    ax7 = fig.add_subplot(gs[3, 0])
    ax7.axis('off')

    anatomy = """
 Θ ANATOMY & PROPOSITION 1
 ═════════════════════════════════════════════

 Θ = I^2 * (F_geom/sigma) * R_th,tot / dT_crit

    [C^2 * Cap^2]   [F_geom/sigma]
    ─────────────    ──────────────
    LOAD demand      MATERIAL (SSOC)

    [R_th,int + R_sum/(wf*A)]    [dT_crit]
    ─────────────────────────    ─────────
    THERMAL PATH                 BUDGET

 ┌──────────────────────────────────────────────┐
 │  PROPOSITION 1 (Theta-sigma Invariance)      │
 │                                              │
 │  Theta * sigma = Lambda_th = const (EXACT)   │
 │                                              │
 │  Lambda_th = I^2 * F_geom * R_th,tot / dT   │
 │  (all non-material params in one number)     │
 │                                              │
 │  COROLLARY: sigma_crit = Lambda_th / Theta*  │
 │                                              │
 │  sigma_crit,I = 10 * sigma_crit,II  (exact) │
 │  Improvement = Theta_now / Theta*  (1 line!) │
 └──────────────────────────────────────────────┘

 DESIGN KNOBS:
  [1] sigma -> sigma_now * (Theta_now/Theta*)
  [2] Raise T -> Preheat (Arrhenius gain)
  [3] Reduce C-rate -> Application limit
  [4] Improve cooling -> h, wf, A_ext
  [5] Reduce cell V -> More cells

 sigma (SSOC) -> Theta (regime) -> Format freedom
 ^                                      |
 └── SCC: Regime III -> improve sigma ──┘
"""
    ax7.text(0.02, 0.98, anatomy, transform=ax7.transAxes,
             fontsize=9.8, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='#FFFFF0', alpha=0.95, pad=0.5))

    ax7.set_title("(g) Θ Anatomy: Decomposition & Design Knobs",
                  fontweight='bold', fontsize=13, pad=10)

    # ==========================================================
    # Figure 8 (h): Design Box 2 Flowchart
    # ==========================================================
    ax8 = fig.add_subplot(gs[3, 1])
    ax8.axis('off')

    flowchart = """
 ╔═══════════════════════════════════════════════════╗
 ║        DESIGN BOX 2 — CHARGING RATE PROTOCOL     ║
 ╚═══════════════════════════════════════════════════╝

 INPUT: x, ρ, T_op, cell geometry, h, target C-rate

 STEP 1 — Conductivity (from SSOC [1]):
 ┌─────────────────────────────────────────────────┐
 │ σ = σ(x,ρ,500K) × exp[−Ea/kB(1/T − 1/500)]    │
 └─────────────────────────────────────────────────┘
           │
 STEP 2 — Compute Θ:
 ┌─────────────────────────────────────────────────┐
 │ Θ = I² × R_ionic(σ) × R_th,tot(geom,h) / ΔT   │
 └─────────────────────────────────────────────────┘
           │
 STEP 3 — Classify:
 ┌─────────────────────────────────────────────────┐
 │                                                  │
 │  Θ < 0.1 ──→ REGIME I                           │
 │               Use PRISMATIC. Done. ✓             │
 │                                                  │
 │  0.1 ≤ Θ ≤ 1 ──→ REGIME II                      │
 │                    Go to STEP 4                  │
 │                                                  │
 │  Θ > 1 ──→ REGIME III                            │
 │             ✗ Cannot operate at this C-rate.     │
 │             OPTIONS:                             │
 │               a) Increase T (preheat)            │
 │               b) Improve σ (→ SSOC Design Box)   │
 │               c) Reduce C-rate                   │
 └─────────────────────────────────────────────────┘
           │
 STEP 4 — Format Selection (Regime II only):
 ┌─────────────────────────────────────────────────┐
 │  Compute wp/wc for your cooling config.          │
 │                                                  │
 │  wp/wc > 0.78 → PRISMATIC  ✓                    │
 │  wp/wc < 0.78 → CYLINDRICAL ✓                   │
 │                                                  │
 │  (threshold weakly h-dependent: 0.75–0.78)       │
 └─────────────────────────────────────────────────┘

 OUTPUT: Optimal format + max safe C-rate
         OR "improve σ" with quantitative target
"""
    ax8.text(0.02, 0.98, flowchart, transform=ax8.transAxes,
             fontsize=9.5, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='#F0F8FF', alpha=0.95, pad=0.5))

    ax8.set_title("(h) Design Box 2: Charging Rate Optimization Protocol",
                  fontweight='bold', fontsize=13, pad=10)

    # ==========================================================
    # SUPER TITLE
    # ==========================================================
    fig.suptitle("SSOC Thermal v5.0: σ → Θ → Format Freedom\n"
                 "From Conductivity to Charging Rate — Unified Framework",
                 fontsize=18, fontweight='bold', y=0.985)

    # Footer
    fig.text(0.5, 0.005,
             f"4680 cell (∅46×80mm) | V={cell.V_cell*1e6:.1f}cm³ | "
             f"σ(500K)={sigma_500K*10:.1f}mS/cm | "
             f"Ea=0.32eV | ΔT_crit=50K | "
             f"η_cyl={cell.eta_cyl:.3f}, η_prism={cell.eta_prism:.2f}",
             ha='center', fontsize=10, style='italic', color='gray')

    # ==========================================================
    # SAVE
    # ==========================================================
    out_png = "/mnt/user-data/outputs/ssoc_thermal_v50_unified.png"
    plt.savefig(out_png, dpi=160, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"\n  ✓ Saved: {out_png}")

    # ==========================================================
    # PRINT SUMMARY
    # ==========================================================
    print(f"\n{'='*80}")
    print("  v5.0 UNIFIED FRAMEWORK — SUMMARY")
    print(f"{'='*80}")
    print(f"\n  THREE-LAYER CAUSAL CHAIN:")
    print(f"    σ(x,ρ,T) → Θ(σ,T,C,h) → Format freedom")
    print(f"\n  AT 300K (room temperature):")
    for C in [0.5, 1.0, 2.0, 3.0, 5.0]:
        res = cell.compute_Theta(sig_300, C)
        Th = res["Theta_c"]
        tag = "I" if Th < 0.1 else "II" if Th < 1.0 else "III"
        print(f"    {C:.1f}C: Θ={Th:.2f} → Regime {tag}")
    print(f"\n  σ REQUIREMENTS FOR REGIME TRANSITIONS (300K, 4680):")
    print(f"  ── Proposition 1: σ_crit = Λ_th / Θ*  (closed-form, exact) ──")
    for target_C, target_regime, target_Theta in [(1.0, 'I', 0.1), (3.0, 'II', 1.0), (3.0, 'I', 0.1)]:
        sc = cell.sigma_crit(target_C, target_Theta)
        factor = sc / sig_300
        print(f"    {target_C}C → Regime {target_regime}: σ > {sc*10:.4f} mS/cm ({factor:.1f}× current)")

    # Verify Theta-sigma invariance
    print(f"\n  PROPOSITION 1 VERIFICATION (Θ·σ = Λ_th = const):")
    Lambda_th_3C = cell.Lambda_th(3.0)
    print(f"    Λ_th(3C, h=300) = {Lambda_th_3C:.6e} S/m")
    print(f"    {'T [K]':>8} {'σ [mS/cm]':>12} {'Θ':>10} {'Θ·σ':>14} {'Ratio':>8}")
    ref_prod = None
    for T in [273, 300, 330, 350, 400, 500]:
        sig = sigma_at_T(sigma_500K, T)
        res = cell.compute_Theta(sig, 3.0)
        prod = sig * res["Theta_c"]
        if ref_prod is None: ref_prod = prod
        print(f"    {T:>6}K {sig*10:>12.4f} {res['Theta_c']:>10.4f} {prod:>14.6e} {prod/ref_prod:>8.4f}")
    print(f"    σ_crit,I = 10 × σ_crit,II : {cell.sigma_crit(3.0, 0.1)*10:.4f} / {cell.sigma_crit(3.0, 1.0)*10:.4f} = {cell.sigma_crit(3.0, 0.1)/cell.sigma_crit(3.0, 1.0):.1f}×  ✓")

    print(f"\n  WET-FRACTION THRESHOLD:")
    print(f"    wp/wc > {cr_300:.3f} (at h=300 W/m²K)")
    print(f"    Range: {crit_ratio_vs_h[-1]:.3f} (h→∞) to {crit_ratio_vs_h[0]:.3f} (h→0)")

    print(f"\n  FORMAT VERDICT EXAMPLES:")
    for name, wfc, wfp, mode, h_val, _ in scenarios:
        clean = name.replace('\n', ' ')
        res = cell.compute_Theta(sig_300, 1.0, h=h_val, wf_c=wfc, wf_p=wfp, mode_p=mode)
        Th = res["Theta_c"]
        regime = "I" if Th < 0.1 else "II" if Th < 1.0 else "III"
        r = wfp/wfc
        if regime == 'III':
            winner = "N/A"
        elif regime == 'I':
            winner = "PRISM"
        else:
            winner = "CYL" if r < 0.78 else "PRISM"
        print(f"    {clean:<35} Θ={Th:.2f} ({regime}) wp/wc={r:.2f} → {winner}")

    print(f"\n{'='*80}")
    print("  v5.0 COMPLETE. 8 panels generated.")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
