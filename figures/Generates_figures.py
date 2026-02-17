#!/usr/bin/env python3
"""
SSOC-LPS Paper Figures
======================
Generates all figures for the manuscript.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle, FancyArrowPatch
import matplotlib.patches as mpatches
from math import comb
import os

# ============================================================
# SETUP
# ============================================================
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'legend.fontsize': 8,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.linewidth': 0.8,
    'lines.linewidth': 1.5,
    'lines.markersize': 6,
})

OUTDIR = '/figures'
os.makedirs(OUTDIR, exist_ok=True)

# Color scheme by composition
COLORS = {
    0.666: '#1f77b4',  # blue
    0.692: '#ff7f0e',  # orange
    0.714: '#2ca02c',  # green
    0.733: '#d62728',  # red
    0.750: '#9467bd',  # purple
}
LABELS = {
    0.666: 'x = 0.667',
    0.692: 'x = 0.692',
    0.714: 'x = 0.714',
    0.733: 'x = 0.733',
    0.750: 'x = 0.750',
}

# ============================================================
# CONSTANTS & MODEL (from verification code)
# ============================================================
k_B = 1.380649e-23; h_planck = 6.62607015e-34
N_A = 6.02214076e23; e_charge = 1.602176634e-19
r_S = 1.84; m_Li = 6.941; m_P = 30.974; m_S = 32.065
eta_RCP = 0.64; T = 500
nu = k_B * T / h_planck

E0_kBT = 7.475; alpha_kBT = 3.790
vc = 0.125; n_hill = 3.0
beta0 = -8.447; beta1 = 6.534; beta2 = 3.052; beta3 = -21.257
vf_star = -beta2 / (2*beta3)

def M_fu(x): return 2*x*m_Li + 2*(1-x)*m_P + (5-4*x)*m_S
def rho0(x):
    r_cm = r_S*1e-8
    return M_fu(x) / (N_A * (5-4*x) * (4/3)*np.pi*r_cm**3 / eta_RCP)
def vf(x, rho): return 1 - rho/rho0(x)
def n_Li(x, rho): return rho * N_A * 2*x / M_fu(x)
def a_jump(x, rho): return n_Li(x, rho)**(-1/3)

def D_PCC(x, rho):
    v = vf(x, rho); a = a_jump(x, rho)
    return (a**2/6)*nu*np.exp(-(E0_kBT - alpha_kBT*v))

def p_active(vf_val):
    if vf_val <= 0: return 1.0
    return 1.0/(1.0 + (vf_val/vc)**n_hill)

def D_hop(x, rho):
    return D_PCC(x, rho) * p_active(vf(x, rho))

def f_eff(x, rho):
    v = vf(x, rho)
    return np.exp(beta0 + beta1*x + beta2*v + beta3*v**2)

def sigma_model(x, rho):
    return n_Li(x, rho) * e_charge**2 * f_eff(x, rho) * D_hop(x, rho) / (k_B*T)

def p_binom(vf_val, N=4, CN_min=3, vfm=0.35):
    if vf_val <= 0: return 1.0
    p = max(0, 1-vf_val/vfm)
    return sum(comb(N,k)*p**k*(1-p)**(N-k) for k in range(CN_min, N+1))

# ============================================================
# DATA
# ============================================================
DATA = [
    (0.666,2.15,3.5,0.70e-6,0.3e-3,'PCC'),(0.666,1.95,4.5,1.10e-6,0.8e-3,'PCC'),
    (0.666,1.80,5.5,1.70e-6,1.5e-3,'PCC'),(0.666,1.60,8.0,1.00e-6,1.0e-3,'SCC'),
    (0.666,1.40,12.0,0.50e-6,0.3e-3,'SCC'),
    (0.692,2.05,3.8,0.90e-6,0.6e-3,'PCC'),(0.692,1.85,5.0,1.50e-6,1.5e-3,'PCC'),
    (0.692,1.70,5.8,1.90e-6,2.5e-3,'PCC'),(0.692,1.55,9.0,1.10e-6,1.2e-3,'SCC'),
    (0.692,1.40,11.0,0.60e-6,0.4e-3,'SCC'),
    (0.714,1.95,4.0,1.20e-6,1.0e-3,'PCC'),(0.714,1.80,5.0,1.90e-6,2.5e-3,'PCC'),
    (0.714,1.65,6.5,1.50e-6,2.0e-3,'SCC'),(0.714,1.50,10.0,0.80e-6,0.8e-3,'SCC'),
    (0.733,1.90,4.0,1.40e-6,1.5e-3,'PCC'),(0.733,1.77,5.0,2.00e-6,3.5e-3,'PCC'),
    (0.733,1.60,7.0,1.30e-6,1.5e-3,'SCC'),(0.733,1.45,10.5,0.70e-6,0.5e-3,'SCC'),
    (0.750,1.85,3.8,1.50e-6,2.0e-3,'PCC'),(0.750,1.73,4.5,2.30e-6,6.4e-3,'PCC'),
    (0.750,1.60,5.8,1.90e-6,3.0e-3,'PCC'),(0.750,1.50,8.0,1.20e-6,1.2e-3,'SCC'),
    (0.750,1.40,11.0,0.60e-6,0.4e-3,'SCC'),
]

xs = [d[0] for d in DATA]
rhos = [d[1] for d in DATA]
dmaxs = [d[2] for d in DATA]
Dhops = [d[3] for d in DATA]
sigmas = [d[4] for d in DATA]
regimes = [d[5] for d in DATA]
vfs = [vf(d[0],d[1]) for d in DATA]


# ============================================================
# FIG 2: Reparameterization ρ → v_f
# ============================================================
def make_fig2():
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))

    # Panel A: d_max vs v_f (collapse test)
    ax = axes[0]
    for x_val in [0.666, 0.692, 0.714, 0.733, 0.750]:
        mask = [i for i,d in enumerate(DATA) if d[0]==x_val]
        vf_x = [vfs[i] for i in mask]
        dm_x = [dmaxs[i] for i in mask]
        reg_x = [regimes[i] for i in mask]
        for v, dm, r in zip(vf_x, dm_x, reg_x):
            mk = 'o' if r=='PCC' else '^'
            ax.scatter(v, dm, c=COLORS[x_val], marker=mk, s=50,
                      edgecolors='k', linewidth=0.5, zorder=5)

    # Fit line
    vf_arr = np.array(vfs); dm_arr = np.array(dmaxs)
    slope, intercept = np.polyfit(vf_arr, dm_arr, 1)
    vf_line = np.linspace(-0.20, 0.26, 100)
    ax.plot(vf_line, slope*vf_line + intercept, 'k--', lw=1, alpha=0.5)
    ax.axhline(6.0, color='gray', ls=':', lw=0.8, alpha=0.7)
    ax.text(0.20, 6.5, r'$d_\mathrm{crit}$ = 6 Å', fontsize=8, color='gray')

    r_val = np.corrcoef(vf_arr, dm_arr)[0,1]
    ax.text(0.05, 0.95, f'r = {r_val:+.3f}\n(all compositions)',
            transform=ax.transAxes, fontsize=8, va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.5))

    ax.set_xlabel(r'Free volume $v_f$')
    ax.set_ylabel(r'$d_\mathrm{max}$ (Å)')
    ax.set_title('(a) Pore size collapse')

    # Panel B: σ vs v_f
    ax = axes[1]
    for x_val in [0.666, 0.692, 0.714, 0.733, 0.750]:
        mask = [i for i,d in enumerate(DATA) if d[0]==x_val]
        vf_x = [vfs[i] for i in mask]
        sig_x = [sigmas[i]*1e3 for i in mask]
        reg_x = [regimes[i] for i in mask]
        for v, s, r in zip(vf_x, sig_x, reg_x):
            mk = 'o' if r=='PCC' else '^'
            ax.scatter(v, s, c=COLORS[x_val], marker=mk, s=50,
                      edgecolors='k', linewidth=0.5, zorder=5)

    ax.axvline(vf_star, color='green', ls='--', lw=1, alpha=0.6)
    ax.text(vf_star+0.01, 5.5, r'$v_f^*$', fontsize=9, color='green')
    ax.axvline(vc, color='red', ls='--', lw=1, alpha=0.6)
    ax.text(vc+0.01, 5.0, r'$v_c$', fontsize=9, color='red')

    ax.set_xlabel(r'Free volume $v_f$')
    ax.set_ylabel(r'$\sigma$ (mS/cm)')
    ax.set_title(r'(b) Conductivity vs $v_f$')
    ax.set_yscale('log')
    ax.set_ylim(0.15, 10)

    # Panel C: D_hop vs v_f
    ax = axes[2]
    for x_val in [0.666, 0.692, 0.714, 0.733, 0.750]:
        mask = [i for i,d in enumerate(DATA) if d[0]==x_val]
        vf_x = [vfs[i] for i in mask]
        dh_x = [Dhops[i]*1e6 for i in mask]
        reg_x = [regimes[i] for i in mask]
        for v, dh, r in zip(vf_x, dh_x, reg_x):
            mk = 'o' if r=='PCC' else '^'
            ax.scatter(v, dh, c=COLORS[x_val], marker=mk, s=50,
                      edgecolors='k', linewidth=0.5, zorder=5)

    ax.axvline(vc, color='red', ls='--', lw=1, alpha=0.6)
    ax.set_xlabel(r'Free volume $v_f$')
    ax.set_ylabel(r'$D_\mathrm{hop}$ ($\times 10^{-6}$ cm²/s)')
    ax.set_title(r'(c) Hop diffusivity vs $v_f$')

    # Legend
    handles = [mpatches.Patch(color=COLORS[x], label=LABELS[x])
               for x in [0.666,0.692,0.714,0.733,0.750]]
    handles.append(plt.Line2D([],[], marker='o', color='gray', ls='',
                              markersize=5, label='PCC'))
    handles.append(plt.Line2D([],[], marker='^', color='gray', ls='',
                              markersize=5, label='SCC'))
    axes[2].legend(handles=handles, loc='upper right', fontsize=7,
                   ncol=1, framealpha=0.8)

    plt.tight_layout()
    fig.savefig(f'{OUTDIR}/fig2_reparameterization.png')
    fig.savefig(f'{OUTDIR}/fig2_reparameterization.pdf')
    plt.close()
    print("  Fig.2 saved.")


# ============================================================
# FIG 3: D_hop decomposition
# ============================================================
def make_fig3():
    fig = plt.figure(figsize=(10, 8))
    gs = GridSpec(2, 2, height_ratios=[1.2, 1], hspace=0.35, wspace=0.35)

    # Panel A (top, spanning): D_hop observed vs model
    ax_main = fig.add_subplot(gs[0, :])

    vf_curve = np.linspace(-0.20, 0.27, 200)

    # Model curve for x=0.714 (middle composition) as representative
    x_rep = 0.714
    rho_curve = rho0(x_rep) * (1 - vf_curve)
    Dhop_curve = np.array([D_hop(x_rep, r)*1e6 for r in rho_curve])
    ax_main.plot(vf_curve, Dhop_curve, 'k-', lw=2, label='Model (x=0.714)', zorder=3)

    # Data points
    for x_val in [0.666, 0.692, 0.714, 0.733, 0.750]:
        mask = [i for i,d in enumerate(DATA) if d[0]==x_val]
        vf_x = [vfs[i] for i in mask]
        dh_x = [Dhops[i]*1e6 for i in mask]
        reg_x = [regimes[i] for i in mask]
        for v, dh, r in zip(vf_x, dh_x, reg_x):
            mk = 'o' if r=='PCC' else '^'
            ax_main.scatter(v, dh, c=COLORS[x_val], marker=mk, s=60,
                           edgecolors='k', linewidth=0.5, zorder=5)

    # Model predictions (connected by thin lines per composition)
    for x_val in [0.666, 0.692, 0.714, 0.733, 0.750]:
        mask = [i for i,d in enumerate(DATA) if d[0]==x_val]
        vf_x = sorted([vfs[i] for i in mask])
        rho_x_curve = rho0(x_val) * (1 - np.array(vf_x))
        Dh_pred = [D_hop(x_val, r)*1e6 for r in rho_x_curve]
        ax_main.plot(vf_x, Dh_pred, '--', color=COLORS[x_val], lw=0.8, alpha=0.6)

    # Regions
    ax_main.axvspan(-0.25, vc, alpha=0.05, color='blue', zorder=0)
    ax_main.axvspan(vc, 0.30, alpha=0.05, color='red', zorder=0)
    ax_main.text(-0.05, 0.3*1e0, 'PCC', fontsize=12, color='blue',
                fontweight='bold', alpha=0.4)
    ax_main.text(0.18, 0.3*1e0, 'SCC', fontsize=12, color='red',
                fontweight='bold', alpha=0.4)
    ax_main.axvline(vc, color='red', ls=':', lw=1)

    ax_main.set_xlabel(r'Free volume $v_f$')
    ax_main.set_ylabel(r'$D_\mathrm{hop}$ ($\times 10^{-6}$ cm²/s)')
    ax_main.set_title(r'(a) $D_\mathrm{hop}$ = $D_\mathrm{PCC}$ × $p_\mathrm{active}$  —  23 points, 5 compositions, 4 parameters')

    handles = [mpatches.Patch(color=COLORS[x], label=LABELS[x])
               for x in [0.666,0.692,0.714,0.733,0.750]]
    ax_main.legend(handles=handles, loc='upper left', fontsize=7, ncol=2)

    # Panel B (bottom-left): D_PCC component
    ax_pcc = fig.add_subplot(gs[1, 0])
    rho_c = rho0(0.714) * (1 - vf_curve)
    Dpcc_curve = np.array([D_PCC(0.714, r)*1e6 for r in rho_c])
    ax_pcc.plot(vf_curve, Dpcc_curve, 'b-', lw=2)
    ax_pcc.set_xlabel(r'$v_f$')
    ax_pcc.set_ylabel(r'$D_\mathrm{PCC}$ ($\times 10^{-6}$ cm²/s)')
    ax_pcc.set_title(r'(b) $D_\mathrm{PCC}(v_f)$: barrier-controlled')

    # Annotate barrier
    ax_pcc.annotate(r'$E(v_f) = E_0 - \alpha_E v_f$',
                    xy=(0.05, 1.5), fontsize=9, style='italic',
                    bbox=dict(boxstyle='round', fc='lightyellow', alpha=0.8))

    # Panel C (bottom-right): p_active component
    ax_pa = fig.add_subplot(gs[1, 1])
    vf_pos = np.linspace(0, 0.30, 200)
    pa_curve = [p_active(v) for v in vf_pos]
    ax_pa.plot(vf_pos, pa_curve, 'r-', lw=2, label=f'Hill (n={n_hill:.0f}, $v_c$={vc})')
    ax_pa.axvline(vc, color='gray', ls=':', lw=0.8)
    ax_pa.axhline(0.5, color='gray', ls=':', lw=0.8)
    ax_pa.plot(vc, 0.5, 'ko', ms=6)
    ax_pa.text(vc+0.01, 0.53, r'$v_c$ = 0.125', fontsize=8)

    ax_pa.set_xlabel(r'$v_f$')
    ax_pa.set_ylabel(r'$p_\mathrm{active}$')
    ax_pa.set_title(r'(c) $p_\mathrm{active}(v_f)$: cooperative deactivation')
    ax_pa.legend(fontsize=8)
    ax_pa.set_xlim(-0.01, 0.30)

    fig.savefig(f'{OUTDIR}/fig3_Dhop_decomposition.png')
    fig.savefig(f'{OUTDIR}/fig3_Dhop_decomposition.pdf')
    plt.close()
    print("  Fig.3 saved.")


# ============================================================
# FIG 4: Origin of n ≈ 3 (CN ≥ 3 threshold)
# ============================================================
def make_fig4():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # Panel A: Schematic of coordination numbers
    ax = axes[0]
    ax.set_xlim(-1, 11); ax.set_ylim(-1, 9)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('(a) Li⁺ coordination by S²⁻ framework', fontsize=10)

    def draw_config(ax, cx, cy, label, n_S, stable, sublabel):
        # Li center
        ax.add_patch(Circle((cx, cy), 0.25, fc='#ff6b6b', ec='k', lw=1, zorder=5))
        ax.text(cx, cy, 'Li⁺', fontsize=6, ha='center', va='center', zorder=6)

        # S neighbors
        angles = np.linspace(0, 2*np.pi, n_S, endpoint=False)
        for a in angles:
            sx = cx + 1.1*np.cos(a)
            sy = cy + 1.1*np.sin(a)
            ax.add_patch(Circle((sx, sy), 0.35, fc='#4ecdc4', ec='k', lw=0.8,
                               alpha=0.7, zorder=4))
            ax.text(sx, sy, 'S²⁻', fontsize=5, ha='center', va='center', zorder=5)

        color = '#2ecc71' if stable else '#e74c3c'
        ax.text(cx, cy-1.8, f'CN = {n_S}', fontsize=9, ha='center',
               fontweight='bold', color=color)
        ax.text(cx, cy-2.3, sublabel, fontsize=7, ha='center',
               color=color, style='italic')
        ax.text(cx, cy+2.0, label, fontsize=9, ha='center', fontweight='bold')

    draw_config(ax, 1.5, 5.5, 'Octahedral', 4, True, 'Stable ✓')
    draw_config(ax, 5.5, 5.5, 'Triangular', 3, True, 'Threshold ✓')
    draw_config(ax, 9.0, 5.5, 'Linear', 2, False, 'Unstable ✗')

    # Arrow showing v_f increase
    ax.annotate('', xy=(9.5, 2.0), xytext=(1.0, 2.0),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))
    ax.text(5.2, 1.4, r'increasing $v_f$ → loss of S²⁻ contacts',
            fontsize=8, ha='center', color='gray')

    # Panel B: Hill vs Binomial comparison
    ax = axes[1]
    vf_range = np.linspace(0.001, 0.30, 200)
    hill_curve = [p_active(v) for v in vf_range]
    binom_curve = [p_binom(v) for v in vf_range]

    ax.plot(vf_range, hill_curve, 'r-', lw=2.5,
            label=f'Hill fit (n={n_hill:.0f}, $v_c$={vc})')
    ax.plot(vf_range, binom_curve, 'b--', lw=2,
            label=r'Binomial ($N_\mathrm{pot}$=4, CN$\geq$3)')

    # SCC data points (p_active from D_obs/D_PCC_pred)
    for i, d in enumerate(DATA):
        x_val, rho_val = d[0], d[1]
        v = vfs[i]
        if v > 0:
            D_pcc_val = D_PCC(x_val, rho_val)
            p_data = d[3] / D_pcc_val
            mk = '^' if d[5]=='SCC' else 'o'
            ax.scatter(v, min(p_data, 1.3), c=COLORS[x_val], marker=mk,
                      s=40, edgecolors='k', linewidth=0.5, zorder=5)

    ax.axvline(vc, color='gray', ls=':', lw=0.8)
    ax.axhline(0.5, color='gray', ls=':', lw=0.8)

    # Effective n annotation
    ax.annotate(r'$n_\mathrm{eff}^{\mathrm{binom}}$ = 2.56' + '\n' +
                r'$n_\mathrm{fit}$ = 3.0' + '\n' +
                r'$\Delta n$ = 0.44 (S²⁻ corr.)',
                xy=(0.20, 0.65), fontsize=8,
                bbox=dict(boxstyle='round', fc='lightyellow', alpha=0.8))

    ax.set_xlabel(r'Free volume $v_f$')
    ax.set_ylabel(r'$p_\mathrm{active}$')
    ax.set_title(r'(b) Hill $n \approx 3$: statistical threshold')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_xlim(-0.01, 0.30)
    ax.set_ylim(-0.05, 1.35)

    plt.tight_layout()
    fig.savefig(f'{OUTDIR}/fig4_hill_origin.png')
    fig.savefig(f'{OUTDIR}/fig4_hill_origin.pdf')
    plt.close()
    print("  Fig.4 saved.")


# ============================================================
# FIG 5: Final σ prediction (parity + design map)
# ============================================================
def make_fig5():
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    # Panel A: Parity plot
    ax = axes[0]
    sig_pred = [sigma_model(d[0], d[1])*1e3 for d in DATA]
    sig_obs = [d[4]*1e3 for d in DATA]

    for i, d in enumerate(DATA):
        mk = 'o' if d[5]=='PCC' else '^'
        ax.scatter(sig_obs[i], sig_pred[i], c=COLORS[d[0]], marker=mk,
                  s=60, edgecolors='k', linewidth=0.5, zorder=5)

    # Perfect line
    lims = [0.15, 10]
    ax.plot(lims, lims, 'k-', lw=0.8, alpha=0.5)
    ax.plot(lims, [l*2 for l in lims], 'k:', lw=0.5, alpha=0.3)
    ax.plot(lims, [l*0.5 for l in lims], 'k:', lw=0.5, alpha=0.3)

    r_log = np.corrcoef(np.log10(sig_pred), np.log10(sig_obs))[0,1]
    mean_r = np.mean(np.array(sig_pred)/np.array(sig_obs))

    ax.text(0.05, 0.92, f'r(log₁₀) = {r_log:+.3f}\nmean ratio = {mean_r:.3f}',
            transform=ax.transAxes, fontsize=9, va='top',
            bbox=dict(boxstyle='round', fc='wheat', alpha=0.6))

    # Mark the anomalous point
    ax.annotate('x=0.75\nρ=1.73', xy=(6.4, 3.44), fontsize=7,
                arrowprops=dict(arrowstyle='->', color='gray'),
                xytext=(4.5, 1.5))

    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel(r'$\sigma_\mathrm{Kim}$ (mS/cm)')
    ax.set_ylabel(r'$\sigma_\mathrm{pred}$ (mS/cm)')
    ax.set_title(r'(a) Parity: $\sigma_\mathrm{pred}$ vs $\sigma_\mathrm{obs}$')
    ax.set_aspect('equal')

    handles = [mpatches.Patch(color=COLORS[x], label=LABELS[x])
               for x in [0.666,0.692,0.714,0.733,0.750]]
    ax.legend(handles=handles, loc='lower right', fontsize=7)

    # Panel B: Design map (heatmap)
    ax = axes[1]
    x_range = np.linspace(0.65, 0.76, 100)
    vf_range = np.linspace(-0.20, 0.28, 120)
    X, VF = np.meshgrid(x_range, vf_range)
    SIG = np.zeros_like(X)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            x_val = X[i,j]; vf_val = VF[i,j]
            rho_val = rho0(x_val) * (1 - vf_val)
            if rho_val > 0.5:
                SIG[i,j] = sigma_model(x_val, rho_val) * 1e3
            else:
                SIG[i,j] = np.nan

    # Clip for display
    SIG = np.clip(SIG, 0.1, 8)

    pcm = ax.pcolormesh(X, VF, SIG, cmap='hot_r', shading='auto',
                        vmin=0, vmax=6)
    cbar = fig.colorbar(pcm, ax=ax, label=r'$\sigma_\mathrm{pred}$ (mS/cm)')

    # Optimal ridge
    ax.axhline(vf_star, color='lime', ls='--', lw=1.5, alpha=0.8)
    ax.text(0.755, vf_star+0.01, r'$v_f^*$ = 0.072', fontsize=8,
            color='lime', fontweight='bold')

    # SCC boundary
    ax.axhline(vc, color='cyan', ls=':', lw=1.5, alpha=0.8)
    ax.text(0.755, vc+0.01, r'$v_c$ = 0.125 (SCC)', fontsize=8,
            color='cyan')

    # Data points
    for i, d in enumerate(DATA):
        mk = 'o' if d[5]=='PCC' else '^'
        ax.scatter(d[0], vfs[i], c='white', marker=mk, s=30,
                  edgecolors='k', linewidth=0.8, zorder=5)

    ax.set_xlabel(r'Composition $x$')
    ax.set_ylabel(r'Free volume $v_f$')
    ax.set_title(r'(b) Design map: $\sigma(x, v_f)$')

    plt.tight_layout()
    fig.savefig(f'{OUTDIR}/fig5_sigma_prediction.png')
    fig.savefig(f'{OUTDIR}/fig5_sigma_prediction.pdf')
    plt.close()
    print("  Fig.5 saved.")


# ============================================================
# SI-1: f_eff analysis
# ============================================================
def make_si1():
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    # Extract f_data
    f_data = []
    for d in DATA:
        nli = n_Li(d[0], d[1])
        f_val = d[4] * k_B * T / (nli * e_charge**2 * d[3])
        f_data.append(f_val)

    # Panel A: f_data vs v_f (color by x)
    ax = axes[0]
    for i, d in enumerate(DATA):
        mk = 'o' if d[5]=='PCC' else '^'
        ax.scatter(vfs[i], f_data[i], c=COLORS[d[0]], marker=mk,
                  s=50, edgecolors='k', linewidth=0.5)

    # Model curve for each x
    vf_range = np.linspace(-0.20, 0.27, 100)
    for x_val in [0.666, 0.714, 0.750]:
        f_curve = [np.exp(beta0 + beta1*x_val + beta2*v + beta3*v**2) for v in vf_range]
        ax.plot(vf_range, f_curve, '--', color=COLORS[x_val], lw=1, alpha=0.6)

    ax.axvline(vf_star, color='green', ls=':', lw=0.8)
    ax.set_xlabel(r'$v_f$')
    ax.set_ylabel(r'$f_\mathrm{data}$')
    ax.set_title(r'(a) $f$ vs $v_f$ (Gaussian peak at $v_f^*$)')

    # Panel B: f_avg vs x
    ax = axes[1]
    x_unique = [0.666, 0.692, 0.714, 0.733, 0.750]
    f_avg = []
    for x_val in x_unique:
        mask = [i for i,d in enumerate(DATA) if d[0]==x_val]
        f_avg.append(np.mean([f_data[i] for i in mask]))

    ax.scatter(x_unique, f_avg, c=[COLORS[x] for x in x_unique],
              s=80, edgecolors='k', linewidth=1, zorder=5)

    # Linear fit
    slope, intercept = np.polyfit(x_unique, f_avg, 1)
    x_line = np.linspace(0.66, 0.76, 50)
    ax.plot(x_line, slope*x_line + intercept, 'k--', lw=1)

    r_fx = np.corrcoef(x_unique, f_avg)[0,1]
    ax.text(0.05, 0.92, f'r = {r_fx:.3f}', transform=ax.transAxes, fontsize=9,
            va='top', bbox=dict(boxstyle='round', fc='wheat', alpha=0.5))

    ax.set_xlabel(r'Composition $x$')
    ax.set_ylabel(r'$\bar{f}(x)$')
    ax.set_title(r'(b) Mean $f$ vs composition')

    # Panel C: g(v_f) shape
    ax = axes[2]
    vf_range = np.linspace(-0.20, 0.30, 200)
    g_curve = [np.exp(beta3*(v - vf_star)**2) for v in vf_range]
    ax.plot(vf_range, g_curve, 'k-', lw=2)
    ax.axvline(vf_star, color='green', ls='--', lw=1)
    ax.axhline(1.0, color='gray', ls=':', lw=0.5)
    ax.fill_between(vf_range, g_curve, alpha=0.1, color='green')

    ax.text(vf_star+0.02, 0.95, r'$v_f^*$ = ' + f'{vf_star:.3f}', fontsize=9, color='green')
    ax.set_xlabel(r'$v_f$')
    ax.set_ylabel(r'$g(v_f)$')
    ax.set_title(r'(c) Pathway connectivity $g(v_f)$')

    plt.tight_layout()
    fig.savefig(f'{OUTDIR}/si1_f_analysis.png')
    fig.savefig(f'{OUTDIR}/si1_f_analysis.pdf')
    plt.close()
    print("  SI-1 saved.")


# ============================================================
# SI-2: ρ₀(x) and v_f definition
# ============================================================
def make_si2():
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    # Panel A: ρ₀(x)
    ax = axes[0]
    x_range = np.linspace(0.60, 0.80, 100)
    rho0_curve = [rho0(x) for x in x_range]
    ax.plot(x_range, rho0_curve, 'k-', lw=2)

    for x_val in [0.666, 0.692, 0.714, 0.733, 0.750]:
        ax.scatter(x_val, rho0(x_val), c=COLORS[x_val], s=80,
                  edgecolors='k', linewidth=1, zorder=5)

    ax.set_xlabel(r'Composition $x$')
    ax.set_ylabel(r'$\rho_0$ (g/cm³)')
    ax.set_title(r'(a) S²⁻ framework density $\rho_0(x)$')
    ax.set_ylim(1.82, 1.84)
    ax.text(0.05, 0.15, 'Near-constant:\nvalidates S²⁻\nframework model',
            transform=ax.transAxes, fontsize=8, style='italic',
            bbox=dict(boxstyle='round', fc='lightyellow', alpha=0.8))

    # Panel B: v_f definition visual
    ax = axes[1]
    rho_range = np.linspace(1.2, 2.3, 100)
    for x_val in [0.666, 0.714, 0.750]:
        vf_curve = [1 - r/rho0(x_val) for r in rho_range]
        ax.plot(rho_range, vf_curve, '-', color=COLORS[x_val], lw=1.5,
               label=LABELS[x_val])

    ax.axhline(0, color='k', ls='-', lw=0.5)
    ax.axhline(vf_star, color='green', ls='--', lw=1, alpha=0.6,
              label=r'$v_f^*$ = 0.072')
    ax.axhline(vc, color='red', ls='--', lw=1, alpha=0.6,
              label=r'$v_c$ = 0.125')

    # Data points
    for i, d in enumerate(DATA):
        mk = 'o' if d[5]=='PCC' else '^'
        ax.scatter(d[1], vfs[i], c=COLORS[d[0]], marker=mk, s=30,
                  edgecolors='k', linewidth=0.5, zorder=5)

    ax.set_xlabel(r'Density $\rho$ (g/cm³)')
    ax.set_ylabel(r'Free volume $v_f$')
    ax.set_title(r'(b) $v_f = 1 - \rho/\rho_0$')
    ax.legend(fontsize=7, loc='upper right')

    plt.tight_layout()
    fig.savefig(f'{OUTDIR}/si2_rho0_vf.png')
    fig.savefig(f'{OUTDIR}/si2_rho0_vf.pdf')
    plt.close()
    print("  SI-2 saved.")


# ============================================================
# SI-3: Robustness (leave-one-out + parameter sensitivity)
# ============================================================
def make_si3():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # Panel A: Leave-one-composition-out for D_hop
    ax = axes[0]

    x_unique = [0.666, 0.692, 0.714, 0.733, 0.750]

    for x_out in x_unique:
        # Refit E0 and alpha from PCC points EXCLUDING x_out
        vf_train = []; ln_Dred_train = []
        for d in DATA:
            if d[0] == x_out: continue
            if d[5] != 'PCC': continue
            v = vf(d[0], d[1])
            a_val = a_jump(d[0], d[1])
            D_reduced = d[3] / ((a_val**2/6)*nu)
            vf_train.append(v)
            ln_Dred_train.append(np.log(D_reduced))

        if len(vf_train) < 3: continue
        # Linear fit: ln(D_red) = -(E0 - alpha*vf)/kBT = -E0/kBT + alpha/kBT * vf
        slope, intercept = np.polyfit(vf_train, ln_Dred_train, 1)
        E0_loo = -intercept
        alpha_loo = slope

        # Predict D_hop for held-out composition
        held_out = [(i,d) for i,d in enumerate(DATA) if d[0]==x_out]
        for idx, d in held_out:
            v = vf(d[0], d[1])
            a_val = a_jump(d[0], d[1])
            D_pcc_loo = (a_val**2/6)*nu*np.exp(-(E0_loo - alpha_loo*v))
            pa = p_active(v)
            D_pred_loo = D_pcc_loo * pa
            ratio = D_pred_loo / d[3]
            mk = 'o' if d[5]=='PCC' else '^'
            ax.scatter(d[3]*1e6, D_pred_loo*1e6, c=COLORS[d[0]],
                      marker=mk, s=50, edgecolors='k', linewidth=0.5)

    lims = [0.3, 5]
    ax.plot(lims, lims, 'k-', lw=0.8, alpha=0.5)
    ax.plot(lims, [l*2 for l in lims], 'k:', lw=0.5, alpha=0.3)
    ax.plot(lims, [l*0.5 for l in lims], 'k:', lw=0.5, alpha=0.3)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel(r'$D_\mathrm{hop}^\mathrm{Kim}$ ($\times 10^{-6}$ cm²/s)')
    ax.set_ylabel(r'$D_\mathrm{hop}^\mathrm{LOO}$ ($\times 10^{-6}$ cm²/s)')
    ax.set_title('(a) Leave-one-composition-out')
    ax.set_aspect('equal')

    # Panel B: Parameter sensitivity for v_c and n
    ax = axes[1]

    # Base model RMSE
    base_rmse = np.sqrt(np.mean([(np.log10(D_hop(d[0],d[1])) -
                                   np.log10(d[3]))**2 for d in DATA]))

    # Scan v_c
    vc_range = np.linspace(0.06, 0.20, 50)
    rmse_vc = []
    for vc_try in vc_range:
        errs = []
        for d in DATA:
            v = vf(d[0], d[1])
            D_pcc_val = D_PCC(d[0], d[1])
            pa = 1/(1+(v/vc_try)**n_hill) if v > 0 else 1.0
            D_pred = D_pcc_val * pa
            errs.append((np.log10(D_pred) - np.log10(d[3]))**2)
        rmse_vc.append(np.sqrt(np.mean(errs)))

    ax.plot(vc_range, rmse_vc, 'r-', lw=2, label=r'Scan $v_c$ (n=3 fixed)')

    # Scan n
    n_range = np.linspace(1.0, 6.0, 50)
    rmse_n = []
    for n_try in n_range:
        errs = []
        for d in DATA:
            v = vf(d[0], d[1])
            D_pcc_val = D_PCC(d[0], d[1])
            pa = 1/(1+(v/vc)**n_try) if v > 0 else 1.0
            D_pred = D_pcc_val * pa
            errs.append((np.log10(D_pred) - np.log10(d[3]))**2)
        rmse_n.append(np.sqrt(np.mean(errs)))

    ax2 = ax.twiny()
    ax2.plot(n_range, rmse_n, 'b--', lw=2, label=r'Scan $n$ ($v_c$=0.125 fixed)')
    ax2.set_xlabel(r'Hill coefficient $n$', color='blue')
    ax2.tick_params(axis='x', labelcolor='blue')

    ax.axvline(vc, color='red', ls=':', lw=0.8, alpha=0.5)
    ax.scatter([vc], [base_rmse], c='red', s=80, marker='*', zorder=5)

    ax.set_xlabel(r'$v_c$', color='red')
    ax.set_ylabel(r'RMSE($\log_{10} D_\mathrm{hop}$)')
    ax.set_title('(b) Parameter sensitivity')

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1+lines2, labels1+labels2, fontsize=8, loc='upper right')

    plt.tight_layout()
    fig.savefig(f'{OUTDIR}/si3_robustness.png')
    fig.savefig(f'{OUTDIR}/si3_robustness.pdf')
    plt.close()
    print("  SI-3 saved.")


# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    print("Generating SSOC-LPS paper figures...")
    print(f"Output directory: {OUTDIR}")
    make_fig2()
    make_fig3()
    make_fig4()
    make_fig5()
    make_si1()
    make_si2()
    make_si3()
    print("\nAll figures generated successfully!")
    print(f"Files in {OUTDIR}/:")
    for f in sorted(os.listdir(OUTDIR)):
        print(f"  {f}")
