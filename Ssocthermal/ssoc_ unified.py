"""
SSOC Thermal v5.1 — Individual Publication Figures
===================================================
Based on v5.0 unified code. Generates 8 separate publication-quality
figures for LaTeX \includegraphics insertion.

Figure 1: Design Stack Schematic (L1→L5 with SCC loop)
Figure 2: Theta(T, C-rate) Phase Diagram — 4680
Figure 3: sigma × C-rate → Theta with temperature iso-lines
Figure 4: Regime Map Table (numerical Theta grid)
Figure 5: Wet-Fraction Phase Diagram (Regime II detail)
Figure 6: Industry Scenario Verdict Map
Figure 7: Theta Anatomy & Proposition 1 Verification
Figure 8: Design Box 2 Flowchart

Authors: Masamichi Iizumi, Tamaki Iizumi (環)
Date: 2026-03-01
"""

import numpy as np
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import FancyBboxPatch
import matplotlib.patheffects as pe
import os

# ============================================================
# PHYSICS ENGINE (identical to v5.0)
# ============================================================
k_B = 1.380649e-23; h_pl = 6.62607015e-34; e_q = 1.602176634e-19; N_A = 6.02214076e23
kB_eV = 8.617333e-5
eta_RCP = 0.64; m_Li = 6.941; m_P = 30.974; m_S = 32.065
r_S = 1.84e-10; V_S = (4/3)*math.pi*r_S**3

E0_kBT = 7.475; aE_kBT = 3.790; v_c = 0.125; n_Hill = 3.0
f_peak_a = -8.337; f_peak_b = 6.534; vf_star = 0.0718; beta3 = -21.257

def M_fu(x):
    return 2*x*m_Li + 2*(1-x)*m_P + (5-4*x)*m_S

def rho_0(x):
    return M_fu(x)*eta_RCP / ((5-4*x)*(V_S*1e6)*N_A)

def sigma_SSOC(x, rho, T=500):
    nLi = (rho*1e6)*N_A*2*x/M_fu(x)
    a = nLi**(-1/3)
    vf = 1 - rho/rho_0(x)
    E0 = E0_kBT*k_B*500; aE = aE_kBT*k_B*500
    D_pcc = (a**2/6)*(k_B*T/h_pl)*math.exp(-(E0-aE*vf)/(k_B*T))
    p_act = 1.0/(1.0+(max(vf,0)/v_c)**n_Hill)
    f_eff = math.exp(f_peak_a+f_peak_b*x)*math.exp(beta3*(vf-vf_star)**2)
    return nLi*e_q**2*f_eff*D_pcc*p_act/(k_B*T)

def sigma_at_T(sigma_500K, T, Ea=0.32):
    return sigma_500K * math.exp(-Ea/kB_eV * (1.0/T - 1.0/500.0))


class CellModel:
    """4680-class cell with full thermal stack (identical to v5.0)."""
    def __init__(self, r_can=23e-3, L=80e-3,
                 t_elec=30e-6, t_cath=70e-6, t_an=80e-6, t_cc=15e-6,
                 kappa_elec=0.50, kappa_electrode=1.5, kappa_cc=238.0,
                 Rpp_ssb=2e-5, Rpp_pack=2e-4,
                 t_can_c=0.3e-3, t_can_p=0.5e-3, k_can=16.0,
                 t_plate=2e-3, k_plate=200.0,
                 energy_density=400e3, dT_crit=50):
        self.r_can = r_can; self.L = L
        self.r_mandrel = r_can * 0.109
        self.V_cell = math.pi * r_can**2 * L
        self.dT_crit = dT_crit; self.Rpp_pack = Rpp_pack
        self.energy_density = energy_density

        self.layers = [(t_cc, kappa_cc), (t_cath, kappa_electrode),
                       (t_elec, kappa_elec), (t_an, kappa_electrode)]
        self.t_elec = t_elec
        self.t_unit = sum(t for t, _ in self.layers)
        self.n_rep = max(1, int((r_can - self.r_mandrel) / self.t_unit))

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
        self.Rth_int = Rm + Rc; self.Rm = Rm; self.Rc = Rc

        self.A_ion = sum(2*math.pi*(self.r_mandrel + (i+0.5)*self.t_unit)*L
                         for i in range(self.n_rep))
        self.A_cyl = 2*math.pi*r_can*L
        W = 46e-3; D = 26e-3; H_p = self.V_cell/(W*D)
        self.W = W; self.D = D; self.H_p = H_p
        self.A_prism_bottom = W*D
        self.A_prism_side = 2*(W+D)*H_p
        self.A_prism_both = self.A_prism_bottom + self.A_prism_side
        self.eta_cyl = math.pi/(2*math.sqrt(3))
        self.eta_prism = 0.98
        self.inv_eta2 = (self.eta_prism/self.eta_cyl)**2
        self.R_sum_c_base = t_can_c/k_can + Rpp_pack
        self.R_sum_p_base = t_can_p/k_can + Rpp_pack + t_plate/k_plate
        self.capacity_Ah = self.V_cell * energy_density / 3.7

    @property
    def F_geom(self):
        return self.t_elec / self.A_ion

    def R_ionic(self, sigma):
        if sigma <= 0 or self.A_ion <= 0: return 1e10
        return self.F_geom / sigma

    def Lambda_th(self, C_rate, h=300, wf_c=0.75, fmt='cyl'):
        I = C_rate * self.capacity_Ah
        Rth = self.Rth_int + (self.R_pack_cyl(h, wf_c) if fmt == 'cyl'
                               else self.R_pack_prism(h, wf_c))
        return I**2 * self.F_geom * Rth / self.dT_crit

    def sigma_crit(self, C_rate, Theta_star, h=300, wf_c=0.75, fmt='cyl'):
        return self.Lambda_th(C_rate, h, wf_c, fmt) / Theta_star

    def R_pack_cyl(self, h, wf=0.75):
        A_eff = wf * self.A_cyl
        return (self.R_sum_c_base + 1.0/h) / A_eff if A_eff > 0 else 1e10

    def R_pack_prism(self, h, wf=0.70, mode='both'):
        A_eff = wf * (self.A_prism_both if mode == 'both' else self.A_prism_bottom)
        return (self.R_sum_p_base + 1.0/h) / A_eff if A_eff > 0 else 1e10

    def compute_Theta(self, sigma, C_rate, h=300, wf_c=0.75, wf_p=0.70, mode_p='both'):
        R_ion = self.R_ionic(sigma)
        I = C_rate * self.capacity_Ah
        Q_dot = I**2 * R_ion
        Rth_c = self.Rth_int + self.R_pack_cyl(h, wf_c)
        Rth_p = self.Rth_int + self.R_pack_prism(h, wf_p, mode_p)
        Theta_c = Q_dot * Rth_c / self.dT_crit
        Theta_p = Q_dot * Rth_p / self.dT_crit
        I_max_c = math.sqrt(self.dT_crit / (R_ion * Rth_c)) if R_ion > 0 and Rth_c > 0 else 0
        I_max_p = math.sqrt(self.dT_crit / (R_ion * Rth_p)) if R_ion > 0 and Rth_p > 0 else 0
        ratio = (I_max_c*math.sqrt(self.eta_cyl))/(I_max_p*math.sqrt(self.eta_prism)) if I_max_p > 0 else 1
        return {"Theta_c": Theta_c, "Theta_p": Theta_p, "ratio": ratio,
                "Q_dot": Q_dot, "I": I, "R_ionic": R_ion, "Rth_c": Rth_c, "Rth_p": Rth_p}

    def crit_wf_ratio(self, h):
        Rs_c = self.R_sum_c_base + 1.0/h
        Rs_p = self.R_sum_p_base + 1.0/h
        return self.inv_eta2 * (self.A_cyl / self.A_prism_both) * (Rs_c / Rs_p)


# ============================================================
# GLOBAL SETUP
# ============================================================
OUT = "/mnt/user-data/outputs"
DPI = 300

# AEM-style formatting
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
    'figure.dpi': DPI,
    'savefig.dpi': DPI,
    'savefig.bbox': 'tight',
    'savefig.facecolor': 'white',
})

cell = CellModel()
sigma_500K = sigma_SSOC(0.75, 1.72)
sig_300 = sigma_at_T(sigma_500K, 300)

# Shared data grids
T_range = np.linspace(250, 520, 140)
C_range = np.logspace(-0.5, 1.2, 120)
sigma_range = np.logspace(-3.5, 0.3, 160)

SCENARIOS = [
    ("Tesla 4680 (gap cool)",       0.75, 0.30, "bottom", 300, "tab:blue"),
    ("BYD Blade (bottom plate)",    0.75, 0.35, "both",   100, "tab:green"),
    ("Fair test (both-side, eq.)",  0.70, 0.70, "both",   300, "tab:gray"),
    ("Prism best (immersion)",      0.50, 0.90, "both",   300, "tab:red"),
    ("Next-gen (full immersion)",   0.85, 0.85, "both",  1000, "tab:orange"),
]


def footer_text():
    return (f"4680 cell ({cell.r_can*2e3:.0f}mm \u00d7 {cell.L*1e3:.0f}mm)  |  "
            f"\u03c3(500 K) = {sigma_500K*10:.1f} mS/cm  |  "
            f"Ea = 0.32 eV  |  \u0394Tcrit = {cell.dT_crit} K")


# ============================================================
# FIGURE 1: Design Stack Schematic
# ============================================================
def fig1_design_stack():
    fig, ax = plt.subplots(figsize=(8, 9))
    ax.set_xlim(-1, 11); ax.set_ylim(-0.5, 11)
    ax.axis('off')

    boxes = [
        (2.0, 9.0, 5.0, 1.3, '#4472C4', 'L1: \u03c3(x,\u03c1,T)\nSSOC Design Equation'),
        (2.0, 7.0, 5.0, 1.3, '#ED7D31', 'L2: Q\u0307 = I\u00b2/\u03c3\nJoule Heating'),
        (2.0, 5.0, 5.0, 1.3, '#808080', 'L3: \u0398(\u03c3,T,C,h)\nRegime Classification'),
        (2.0, 3.0, 5.0, 1.3, '#70AD47', 'L4: wp/wc > 0.78\nFormat Selection'),
        (2.0, 1.0, 5.0, 1.3, '#FFC000', 'L5: Gtotal\nPack Integration'),
    ]
    for bx, by, bw, bh, col, txt in boxes:
        p = FancyBboxPatch((bx, by), bw, bh, boxstyle="round,pad=0.18",
                           facecolor=col, edgecolor='black', lw=2.2, alpha=0.88)
        ax.add_patch(p)
        ax.text(bx+bw/2, by+bh/2, txt, ha='center', va='center',
                fontsize=12, fontweight='bold', color='white',
                path_effects=[pe.withStroke(linewidth=2.5, foreground='black')])

    # PCC forward arrows
    for y0 in [9.0, 7.0, 5.0, 3.0]:
        ax.annotate('', xy=(4.5, y0), xytext=(4.5, y0+0.8),
                    arrowprops=dict(arrowstyle='->', lw=2.8, color='#333333'))

    ax.text(7.8, 8.0, 'PCC\n(forward)', fontsize=12, ha='center',
            fontweight='bold', color='#4472C4',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#4472C4', alpha=0.9))

    # SCC backward arrow
    ax.annotate('', xy=(1.2, 9.65), xytext=(1.2, 5.65),
                arrowprops=dict(arrowstyle='->', lw=3.5, color='#C00000',
                               connectionstyle='arc3,rad=0.5'))
    ax.text(-0.3, 7.6, 'SCC\n(Regime III\n\u2192 improve \u03c3)', fontsize=10,
            ha='center', fontweight='bold', color='#C00000',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFE0E0', edgecolor='#C00000', alpha=0.95))

    # Right-side annotations
    ax.text(8.2, 9.6, 'Material\n(SSOC paper [1])', fontsize=10, color='#4472C4',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#D6E4F0', alpha=0.9))
    ax.text(8.2, 5.6, '\u0398 < 0.1 \u2192 Regime I (skip L4)\n'
            '0.1 < \u0398 < 1 \u2192 Regime II (use L4)\n'
            '\u0398 > 1 \u2192 Regime III (SCC to L1)',
            fontsize=9.5, color='#333333', va='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#F0F0F0', alpha=0.95))
    ax.text(8.2, 3.6, 'Only active in\nRegime II', fontsize=10, color='#70AD47',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#E2EFDA', alpha=0.9))

    fig.text(0.5, 0.01, footer_text(), ha='center', fontsize=8.5, color='gray', style='italic')
    ax.set_title("Figure 1.  Design Stack: \u03c3 \u2192 \u0398 \u2192 Format Freedom",
                 fontweight='bold', fontsize=14, pad=12)

    fig.savefig(f"{OUT}/fig1_design_stack.png")
    plt.close(fig)
    print("  \u2713 Fig 1 saved")


# ============================================================
# FIGURE 2: Theta(T, C-rate) Phase Diagram
# ============================================================
def fig2_phase_diagram():
    Theta_TC = np.zeros((len(T_range), len(C_range)))
    for i, T in enumerate(T_range):
        sig = sigma_at_T(sigma_500K, T)
        for j, C in enumerate(C_range):
            Theta_TC[i, j] = cell.compute_Theta(sig, C)["Theta_c"]

    fig, ax = plt.subplots(figsize=(9, 6.5))
    log_Th = np.log10(np.clip(Theta_TC, 1e-4, 1e4))
    cs = ax.pcolormesh(C_range, T_range, log_Th,
                       cmap='RdYlGn_r', vmin=-3, vmax=2.5, shading='auto')
    cbar = plt.colorbar(cs, ax=ax, label=r"$\log_{10}\Theta$", shrink=0.92)
    cbar.set_ticks([-3, -2, -1, 0, 1, 2])
    cbar.set_ticklabels(['0.001', '0.01', '0.1', '1', '10', '100'])

    ax.contour(C_range, T_range, Theta_TC, levels=[0.1],
               colors='#00AA00', linewidths=3.0, linestyles='-')
    ax.contour(C_range, T_range, Theta_TC, levels=[1.0],
               colors='#CC0000', linewidths=3.0, linestyles='-')

    ax.text(0.4, 478, "Regime I\n$\\Theta < 0.1$", color='#006600',
            fontweight='bold', fontsize=12, ha='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    ax.text(1.5, 335, "II", color='black', fontweight='bold', fontsize=18,
            bbox=dict(boxstyle='round,pad=0.2', facecolor='#FFFFCC', alpha=0.8))
    ax.text(5, 285, "Regime III\n$\\Theta > 1$", color='#990000',
            fontweight='bold', fontsize=12, ha='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    pts = [(1.0, 300, '1C @ 300 K'), (3.0, 300, '3C @ 300 K'),
           (3.0, 350, '3C @ 350 K'), (1.0, 400, '1C @ 400 K')]
    for cx, cy, lab in pts:
        ax.plot(cx, cy, 'ko', ms=7, zorder=5)
        ax.annotate(lab, xy=(cx, cy), xytext=(cx*1.35, cy+14),
                    fontsize=9, fontweight='bold',
                    arrowprops=dict(arrowstyle='->', lw=1.2, color='black'),
                    bbox=dict(boxstyle='round,pad=0.15', facecolor='white', alpha=0.85))

    ax.set_xscale('log')
    ax.set_xlabel("C-rate")
    ax.set_ylabel("Temperature [K]")
    ax.set_ylim(260, 510)
    ax.set_title(r"Figure 2.  $\Theta$ Phase Diagram: 4680 cell, $h = 300$ W m$^{-2}$ K$^{-1}$",
                 fontweight='bold', pad=10)
    fig.text(0.5, 0.01, footer_text(), ha='center', fontsize=8.5, color='gray', style='italic')
    fig.savefig(f"{OUT}/fig2_phase_diagram.png")
    plt.close(fig)
    print("  \u2713 Fig 2 saved")


# ============================================================
# FIGURE 3: sigma x C-rate → Theta with T iso-lines
# ============================================================
def fig3_sigma_crate():
    Theta_sC = np.zeros((len(sigma_range), len(C_range)))
    for i, sig in enumerate(sigma_range):
        for j, C in enumerate(C_range):
            Theta_sC[i, j] = cell.compute_Theta(sig, C)["Theta_c"]

    fig, ax = plt.subplots(figsize=(9, 6.5))
    log_Th = np.log10(np.clip(Theta_sC, 1e-4, 1e4))
    cs = ax.pcolormesh(C_range, sigma_range*10, log_Th,
                       cmap='RdYlGn_r', vmin=-3, vmax=2.5, shading='auto')
    cbar = plt.colorbar(cs, ax=ax, label=r"$\log_{10}\Theta$", shrink=0.92)
    cbar.set_ticks([-3, -2, -1, 0, 1, 2])
    cbar.set_ticklabels(['0.001', '0.01', '0.1', '1', '10', '100'])

    ax.contour(C_range, sigma_range*10, Theta_sC, levels=[0.1],
               colors='#00AA00', linewidths=3.0)
    ax.contour(C_range, sigma_range*10, Theta_sC, levels=[1.0],
               colors='#CC0000', linewidths=3.0)

    temp_marks = [(273, '#0055BB', '273 K (0 \u00b0C)'),
                  (300, '#0088EE', '300 K (27 \u00b0C)'),
                  (330, '#DD8800', '330 K (57 \u00b0C)'),
                  (350, '#DD5500', '350 K (77 \u00b0C)'),
                  (400, '#BB2200', '400 K (127 \u00b0C)'),
                  (500, '#880000', '500 K (227 \u00b0C)')]
    for T, col, lab in temp_marks:
        sig_T = sigma_at_T(sigma_500K, T) * 10
        ax.axhline(sig_T, color=col, ls='--', lw=1.8, alpha=0.85)
        ax.text(C_range[-1]*0.8, sig_T*1.18, lab, fontsize=8, color=col,
                fontweight='bold', ha='right',
                bbox=dict(boxstyle='round,pad=0.1', facecolor='white', alpha=0.85))

    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel("C-rate")
    ax.set_ylabel(r"$\sigma$ [mS cm$^{-1}$]")
    ax.set_ylim(0.003, 20)
    ax.set_title(r"Figure 3.  $\Theta(\sigma, C\text{-rate})$ — Temperature maps onto $\sigma$ axis",
                 fontweight='bold', pad=10)
    fig.text(0.5, 0.01, footer_text(), ha='center', fontsize=8.5, color='gray', style='italic')
    fig.savefig(f"{OUT}/fig3_sigma_crate.png")
    plt.close(fig)
    print("  \u2713 Fig 3 saved")


# ============================================================
# FIGURE 4: Regime Map Table
# ============================================================
def fig4_regime_table():
    T_grid = [273, 300, 330, 350, 400, 500]
    C_grid = [0.5, 1.0, 2.0, 3.0, 5.0]

    col_labels = ['T [K]', r'$\sigma$ [mS/cm]'] + [f'{C}C' for C in C_grid]
    n_cols = len(col_labels)

    RI = '#C6EFCE'; RII = '#FFEB9C'; RIII = '#FFC7CE'
    cell_text = []; cell_colors = []
    for T in T_grid:
        sig = sigma_at_T(sigma_500K, T)
        row_t = [f'{T}', f'{sig*10:.3f}']
        row_c = ['#E8E8E8', '#E8E8E8']
        for C in C_grid:
            Th = cell.compute_Theta(sig, C)["Theta_c"]
            if Th < 0.1:   tag, color = 'I', RI
            elif Th < 1.0: tag, color = 'II', RII
            else:          tag, color = 'III', RIII
            row_t.append(f'{Th:.2f}\n({tag})')
            row_c.append(color)
        cell_text.append(row_t)
        cell_colors.append(row_c)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis('off')
    tbl = ax.table(cellText=cell_text, colLabels=col_labels,
                   cellColours=cell_colors,
                   colColours=['#4472C4']*n_cols,
                   loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1.0, 2.5)
    for j in range(n_cols):
        tbl[0, j].set_text_props(color='white', fontweight='bold')

    ax.text(0.5, 0.06,
            "Green = Regime I (\u0398 < 0.1, thermal-free)  |  "
            "Yellow = Regime II (0.1 < \u0398 < 1, format matters)  |  "
            "Red = Regime III (\u0398 > 1, thermal death)",
            transform=ax.transAxes, ha='center', fontsize=10, style='italic')
    ax.set_title(r"Figure 4.  $\Theta$ Regime Map — 4680, $h = 300$ W m$^{-2}$ K$^{-1}$, $\Delta T_{\rm crit} = 50$ K",
                 fontweight='bold', fontsize=13, pad=15)
    fig.text(0.5, 0.01, footer_text(), ha='center', fontsize=8.5, color='gray', style='italic')
    fig.savefig(f"{OUT}/fig4_regime_table.png")
    plt.close(fig)
    print("  \u2713 Fig 4 saved")


# ============================================================
# FIGURE 5: Wet-Fraction Phase Diagram (Regime II)
# ============================================================
def fig5_wet_fraction():
    wf_range = np.linspace(0.25, 1.0, 120)
    ratio_wf = np.zeros((len(wf_range), len(wf_range)))
    for i, wfc in enumerate(wf_range):
        for j, wfp in enumerate(wf_range):
            ratio_wf[i, j] = cell.compute_Theta(sig_300, 1.0, h=300, wf_c=wfc, wf_p=wfp)["ratio"]

    cr_300 = cell.crit_wf_ratio(300)

    fig, ax = plt.subplots(figsize=(8, 7))
    norm = TwoSlopeNorm(vmin=0.85, vcenter=1.0, vmax=1.35)
    cs = ax.contourf(wf_range, wf_range, ratio_wf.T,
                     levels=np.linspace(0.85, 1.35, 26),
                     cmap='RdYlBu', norm=norm)
    plt.colorbar(cs, ax=ax, label="Pack Performance Ratio (Cyl / Prism)", shrink=0.92)

    ax.contour(wf_range, wf_range, ratio_wf.T, levels=[1.0],
               colors='black', linewidths=3.0)

    wc_line = np.linspace(0.25, 1.0, 100)
    wp_line = cr_300 * wc_line
    valid = wp_line <= 1.0
    ax.plot(wc_line[valid], wp_line[valid], 'k--', lw=2.2, alpha=0.7,
            label=f'Analytic: $w_p = {cr_300:.3f} \\cdot w_c$')

    pts = [(0.75, 0.30, 'Tesla', 'tab:blue', 's'),
           (0.75, 0.35, 'BYD', 'tab:green', '^'),
           (0.70, 0.70, 'Fair', 'tab:gray', 'D'),
           (0.85, 0.85, 'Next-gen', 'tab:orange', 'o'),
           (0.50, 0.90, 'Prism-best', 'tab:red', 'v')]
    for wc, wp, lab, col, mk in pts:
        ax.plot(wc, wp, marker=mk, ms=13, color=col, markeredgecolor='black',
                markeredgewidth=1.5, zorder=10)
        ax.annotate(lab, xy=(wc, wp), xytext=(wc+0.03, wp+0.04),
                    fontsize=10, fontweight='bold', color=col)

    ax.text(0.35, 0.85, "PRISM\nwins", fontsize=16, fontweight='bold',
            color='#CC0000', alpha=0.4, ha='center')
    ax.text(0.85, 0.35, "CYL\nwins", fontsize=16, fontweight='bold',
            color='#0000CC', alpha=0.4, ha='center')

    ax.set_xlabel(r"Wet fraction — Cylindrical ($w_c$)")
    ax.set_ylabel(r"Wet fraction — Prismatic ($w_p$)")
    ax.set_xlim(0.25, 1.0); ax.set_ylim(0.25, 1.0)
    ax.legend(fontsize=10, loc='upper left')
    ax.set_title("Figure 5.  Format Selection in Regime II (1C, 300 K, $h = 300$)",
                 fontweight='bold', pad=10)
    fig.text(0.5, 0.01, footer_text(), ha='center', fontsize=8.5, color='gray', style='italic')
    fig.savefig(f"{OUT}/fig5_wet_fraction.png")
    plt.close(fig)
    print("  \u2713 Fig 5 saved")


# ============================================================
# FIGURE 6: Industry Verdict Table
# ============================================================
def fig6_verdict():
    RI = '#C6EFCE'; RII = '#FFEB9C'; RIII = '#FFC7CE'
    header = ['Scenario', '$w_c$', '$w_p$', '$h$', r'$\Theta$(1C)', 'Regime', '$w_p/w_c$', 'Winner']

    rows = []; colors = []
    for name, wfc, wfp, mode, h_val, _ in SCENARIOS:
        res = cell.compute_Theta(sig_300, 1.0, h=h_val, wf_c=wfc, wf_p=wfp, mode_p=mode)
        Th = res["Theta_c"]
        regime = "I" if Th < 0.1 else "II" if Th < 1.0 else "III"
        ratio = wfp / wfc
        if regime == 'III':    winner, c = "N/A (\u2191\u03c3)", RIII
        elif regime == 'I':    winner, c = "PRISM", RI
        else:                  winner, c = ("CYL" if ratio < 0.78 else "PRISM"), RII
        rows.append([name, f'{wfc:.2f}', f'{wfp:.2f}', f'{h_val}',
                     f'{Th:.2f}', regime, f'{ratio:.2f}', winner])
        colors.append([c]*8)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.axis('off')
    tbl = ax.table(cellText=rows, colLabels=header,
                   cellColours=colors, colColours=['#4472C4']*8,
                   loc='upper center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.0, 2.5)
    for j in range(8):
        tbl[0, j].set_text_props(color='white', fontweight='bold')

    insight = (
        "\u2501\u2501\u2501 KEY FINDING \u2501\u2501\u2501\n"
        "\u2022 At equal wet fraction (wp = wc): PRISM always wins (packing advantage)\n"
        "\u2022 CYL wins ONLY when natural coolant channels give wc >> wp\n"
        "\u2022 Threshold: wp/wc > 0.78 (weakly h-dependent: 0.75\u20130.78)\n\n"
        "FORMAT CHOICE IS NOT A PHYSICS QUESTION.\n"
        "IT IS A COOLING ARCHITECTURE QUESTION.\n"
        "(And only relevant in Regime II.)")
    ax.text(0.5, 0.04, insight, transform=ax.transAxes,
            ha='center', va='bottom', fontsize=10.5, fontfamily='monospace',
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFFFF0', alpha=0.95))

    ax.set_title("Figure 6.  Unified Industry Verdict (1C at 300 K)",
                 fontweight='bold', fontsize=13, pad=15)
    fig.text(0.5, 0.01, footer_text(), ha='center', fontsize=8.5, color='gray', style='italic')
    fig.savefig(f"{OUT}/fig6_verdict.png")
    plt.close(fig)
    print("  \u2713 Fig 6 saved")


# ============================================================
# FIGURE 7: Theta Anatomy + Proposition 1 Verification
# ============================================================
def fig7_anatomy():
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(9, 9),
                                          gridspec_kw={'height_ratios': [1, 1.1]})

    # --- Top: Prop 1 verification plot ---
    T_list = np.linspace(273, 500, 80)
    sig_list = [sigma_at_T(sigma_500K, T) for T in T_list]
    Th_list = [cell.compute_Theta(s, 3.0)["Theta_c"] for s in sig_list]
    prod_list = [s * th for s, th in zip(sig_list, Th_list)]

    ax_top.plot(T_list, [p*1e3 for p in prod_list], 'b-', lw=2.5,
                label=r'$\Theta \cdot \sigma$ (3C, $h=300$)')
    ax_top.set_ylabel(r'$\Theta \cdot \sigma$ [$\times 10^{-3}$ S m$^{-1}$]')
    ax_top.set_xlabel('Temperature [K]')
    Lambda_val = cell.Lambda_th(3.0)
    ax_top.axhline(Lambda_val*1e3, color='red', ls='--', lw=2,
                   label=f'$\\Lambda_{{th}}$ = {Lambda_val:.4e} S/m')
    ax_top.set_ylim(Lambda_val*1e3*0.95, Lambda_val*1e3*1.05)
    ax_top.legend(fontsize=11)
    ax_top.set_title("Figure 7a.  Proposition 1 Verification: \u0398\u00b7\u03c3 = \u039b_th = const",
                     fontweight='bold', pad=10)

    # Inset table
    ins = ax_top.inset_axes([0.55, 0.12, 0.42, 0.55])
    ins.axis('off')
    T_check = [273, 300, 330, 350, 400, 500]
    rows = []
    for T in T_check:
        s = sigma_at_T(sigma_500K, T)
        th = cell.compute_Theta(s, 3.0)["Theta_c"]
        rows.append([f'{T}', f'{s*10:.4f}', f'{th:.2f}', f'{s*th:.4e}'])
    t = ins.table(cellText=rows,
                  colLabels=['T [K]', '\u03c3 [mS/cm]', '\u0398', '\u0398\u00b7\u03c3 [S/m]'],
                  loc='center', cellLoc='center')
    t.auto_set_font_size(False); t.set_fontsize(9); t.scale(1.0, 1.6)
    for j in range(4):
        t[0, j].set_text_props(fontweight='bold')

    # --- Bottom: Anatomy text (plain text to avoid matplotlib mathtext issues) ---
    ax_bot.axis('off')
    anatomy = (
        "\u0398 ANATOMY & DESIGN KNOBS\n"
        "\u2501"*42 + "\n\n"
        "  \u0398 = I\u00b2 \u00b7 (F_geom / \u03c3) \u00b7 R_th,tot / \u0394T_crit\n\n"
        "  [C\u00b2 \u00b7 Cap\u00b2]    [F_geom/\u03c3]    [R_th,int + R_sum/(wf\u00b7A)]    [\u0394T_crit]\n"
        "  \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500    \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500    \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500    \u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
        "    LOAD         MATERIAL        THERMAL PATH             BUDGET\n"
        "                 (SSOC)\n\n"
        "\u2501\u2501\u2501 PROPOSITION 1 (Exact) \u2501\u2501\u2501\n"
        "  \u0398 \u00b7 \u03c3 = \u039b_th = const        (EXACT IDENTITY)\n"
        "  \u03c3_crit = \u039b_th / \u0398*\n"
        "  \u03c3_crit,I = 10 \u00d7 \u03c3_crit,II    (exact, parameter-free)\n"
        "  Improvement = \u0398_now / \u0398*     (one division!)\n\n"
        "\u2501\u2501\u2501 DESIGN KNOBS \u2501\u2501\u2501\n"
        "  [1] \u03c3 :   Improve via SSOC Design Box\n"
        "  [2] T :   Preheat (Arrhenius gain)\n"
        "  [3] C :   Reduce C-rate (application limit)\n"
        "  [4] Cool: Improve h, wf, A_ext\n"
        "  [5] Size: Reduce cell V (more cells)\n\n"
        "  \u03c3 (SSOC) \u2192 \u0398 (regime) \u2192 Format freedom\n"
        "  ^                                     |\n"
        "  \u2514\u2500\u2500 SCC: Regime III \u2192 improve \u03c3 \u2500\u2500\u2518"
    )

    ax_bot.text(0.05, 0.95, anatomy, transform=ax_bot.transAxes,
                fontsize=11, va='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFFFF0', alpha=0.95))

    fig.text(0.5, 0.01, footer_text(), ha='center', fontsize=8.5, color='gray', style='italic')
    fig.subplots_adjust(hspace=0.35, bottom=0.04)
    fig.savefig(f"{OUT}/fig7_anatomy.png")
    plt.close(fig)
    print("  \u2713 Fig 7 saved")


# ============================================================
# FIGURE 8: Design Box 2 Flowchart
# ============================================================
def fig8_flowchart():
    fig, ax = plt.subplots(figsize=(9, 10))
    ax.axis('off')

    flowchart = """
 ╔════════════════════════════════════════════════════════════╗
 ║           DESIGN BOX 2 — CHARGING RATE PROTOCOL           ║
 ╚════════════════════════════════════════════════════════════╝

 INPUT: x, ρ, T_op, cell geometry, h, target C-rate

 STEP 1 — Conductivity (from SSOC [1]):
 ┌──────────────────────────────────────────────────────────┐
 │ σ = σ(x,ρ,500K) × exp[−Ea/kB (1/T − 1/500)]            │
 └──────────────────────────────────────────────────────────┘
              │
 STEP 2 — Compute Θ:
 ┌──────────────────────────────────────────────────────────┐
 │ Θ = I² × R_ionic(σ) × R_th,tot(geom,h) / ΔTcrit        │
 └──────────────────────────────────────────────────────────┘
              │
 STEP 3 — Classify:
 ┌──────────────────────────────────────────────────────────┐
 │                                                          │
 │  Θ < 0.1 ──→ REGIME I                                   │
 │               Use PRISMATIC. Done. ✓                     │
 │                                                          │
 │  0.1 ≤ Θ ≤ 1 ──→ REGIME II                              │
 │                    Go to STEP 4                          │
 │                                                          │
 │  Θ > 1 ──→ REGIME III                                    │
 │             ✗ Cannot operate at this C-rate.              │
 │             Compute σ_crit = Λ_th / Θ*                   │
 │             OPTIONS:                                     │
 │               a) Increase T (preheat)                    │
 │               b) Improve σ (→ SSOC Design Box [1])       │
 │               c) Reduce C-rate                           │
 └──────────────────────────────────────────────────────────┘
              │
 STEP 4 — Format Selection (Regime II only):
 ┌──────────────────────────────────────────────────────────┐
 │  Compute wp/wc for your cooling config.                  │
 │                                                          │
 │  wp/wc > 0.78 → PRISMATIC  ✓                            │
 │  wp/wc < 0.78 → CYLINDRICAL ✓                           │
 │                                                          │
 │  (threshold weakly h-dependent: 0.75–0.78)              │
 └──────────────────────────────────────────────────────────┘

 OUTPUT: Optimal format + max safe C-rate
         OR "improve σ" with quantitative target
            (improvement factor = Θ_now / Θ*)
"""

    ax.text(0.02, 0.97, flowchart, transform=ax.transAxes,
            fontsize=10.5, va='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#F0F8FF', alpha=0.95))
    ax.set_title("Figure 8.  Design Box 2: Charging Rate Optimization Protocol",
                 fontweight='bold', fontsize=13, pad=12)
    fig.text(0.5, 0.01, footer_text(), ha='center', fontsize=8.5, color='gray', style='italic')
    fig.savefig(f"{OUT}/fig8_flowchart.png")
    plt.close(fig)
    print("  \u2713 Fig 8 saved")


# ============================================================
# MAIN
# ============================================================
def main():
    os.makedirs(OUT, exist_ok=True)

    print("="*70)
    print("  SSOC Thermal v5.1 — Individual Publication Figures")
    print("="*70)
    print(f"  Cell: 4680 ({cell.r_can*2e3:.0f} mm × {cell.L*1e3:.0f} mm)")
    print(f"  σ(500K) = {sigma_500K*10:.2f} mS/cm")
    print(f"  σ(300K) = {sig_300*10:.4f} mS/cm")
    print(f"  Output: {OUT}/fig[1-8]_*.png @ {DPI} DPI\n")

    fig1_design_stack()
    fig2_phase_diagram()
    fig3_sigma_crate()
    fig4_regime_table()
    fig5_wet_fraction()
    fig6_verdict()
    fig7_anatomy()
    fig8_flowchart()

    print(f"\n{'='*70}")
    print("  v5.1 COMPLETE — 8 individual figures generated.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
