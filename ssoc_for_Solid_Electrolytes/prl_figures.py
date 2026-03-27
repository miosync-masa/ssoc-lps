"""
PRL Figures: Universal PCC/SCC Decomposition
=============================================
3 figures for Physical Review Letters submission
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from scipy.stats import pearsonr
import math

# ============================================================
# DATA (identical to universal_5system.py)
# ============================================================
k_B = 1.380649e-23; h_pl = 6.62607015e-34; e_q = 1.602176634e-19; N_A = 6.02214076e23
eta_RCP = 0.64; m_Li = 6.941; m_P = 30.974; m_S = 32.065
r_S = 1.84e-10; V_S = (4/3)*math.pi*r_S**3
def M_fu(x): return 2*x*m_Li + 2*(1-x)*m_P + (5-4*x)*m_S
def rho_0(x): return M_fu(x)*eta_RCP / ((5-4*x)*(V_S*1e6)*N_A)

kim_data = [
    (0.667,1.40,1.98e-4),(0.667,1.50,4.45e-4),(0.667,1.60,5.98e-4),
    (0.667,1.70,6.51e-4),(0.667,1.80,3.37e-4),
    (0.700,1.40,4.14e-4),(0.700,1.50,9.46e-4),(0.700,1.60,1.91e-3),
    (0.700,1.70,2.76e-3),(0.700,1.80,8.29e-4),
    (0.714,1.50,8.47e-4),(0.714,1.60,1.74e-3),(0.714,1.70,3.16e-3),
    (0.714,1.80,1.48e-3),
    (0.750,1.40,5.30e-4),(0.750,1.50,1.74e-3),(0.750,1.60,4.80e-3),
    (0.750,1.70,6.63e-3),(0.750,1.72,6.58e-3),(0.750,1.80,2.57e-3),
    (0.750,1.85,7.72e-4),(0.750,1.90,1.29e-4),(0.750,1.50,1.66e-3),
]
lps = [{'s':'LPS','vf':1-r/rho_0(x),'sig':s} for x,r,s in kim_data]

llzo_raw = [
    (99.8,5.10,5.70e-4),(96.5,5.10,3.32e-4),(96.0,5.10,3.30e-4),(88.9,5.10,3.40e-5),
    (96.0,5.20,1.96e-4),
    (92.1,5.10,1.09e-4),(92.4,5.10,1.43e-4),(93.5,5.10,2.37e-4),(93.5,5.10,2.49e-4),
    (93.2,5.10,1.56e-4),(93.6,5.10,3.56e-4),(96.1,5.10,3.86e-4),(94.6,5.10,2.62e-4),
    (92.8,5.10,1.49e-4),(93.2,5.10,1.92e-4),(95.8,5.10,2.36e-4),(94.4,5.10,2.42e-4),
]
llzo = [{'s':'LLZO','vf':(100-rr)/100,'sig':s} for rr,rt,s in llzo_raw]

argy_raw = [
    (1.53,1.64,1.92e-3),(1.64,1.64,2.07e-3),(1.64,1.64,2.05e-3),
    (1.42,1.64,1.92e-3),(1.64,1.64,2.62e-3),(1.66,1.64,2.93e-3),
    (1.50,1.64,6.60e-4),(1.56,1.64,1.61e-3),(1.60,1.64,2.67e-3),
    (1.66,1.64,2.98e-3),(1.69,1.64,3.56e-3),
]
argy = [{'s':'Argyrodite','vf':1-r/rt,'sig':s} for r,rt,s in argy_raw]

nas_raw = [
    (2.79,95.4,2.92,3.80e-4),(3.08,95.4,3.23,6.50e-4),
    (None,73.0,2.92,3.18e-7),(None,75.0,2.92,1.98e-7),
    (None,94.0,2.92,1.26e-5),(None,94.0,2.92,8.20e-6),
    (3.477,97.6,3.5615,3.29e-4),
    (3.33,93.5,3.56,1.64e-4),(3.18,89.3,3.56,3.40e-4),(3.02,84.8,3.56,2.30e-4),
]
nasicon = []
for rho,rr,rt,s in nas_raw:
    if rr is not None:
        vf = (100-rr)/100
    else:
        vf = 1 - rho/rt
    nasicon.append({'s':'NASICON','vf':vf,'sig':s})

nzsp_raw = [(2.89,89.2,3.24,2.84e-4),(3.07,94.8,3.24,9.04e-4),
            (3.10,95.7,3.24,1.27e-3),(3.08,95.1,3.24,9.92e-4),(3.02,93.2,3.24,6.92e-4)]
nzsp = [{'s':'Na-NZSP','vf':(100-rr)/100,'sig':s} for _,rr,_,s in nzsp_raw]

all_data = lps + llzo + argy + nasicon + nzsp

systems = {}
for p in all_data:
    systems.setdefault(p['s'],[]).append(p)

style = {
    'LPS':        {'c':'#d62728','m':'o','ms':5,'label':r'Li$_2$S–P$_2$S$_5$ (500 K)'},
    'LLZO':       {'c':'#1f77b4','m':'s','ms':5,'label':r'LLZO garnet (298 K)'},
    'Argyrodite': {'c':'#2ca02c','m':'D','ms':5,'label':r'Li$_6$PS$_5$Cl (298 K)'},
    'NASICON':    {'c':'#9467bd','m':'^','ms':6,'label':'NASICON (298 K)'},
    'Na-NZSP':    {'c':'#ff7f0e','m':'v','ms':6,'label':r'Na$_3$Zr$_2$Si$_2$PO$_{12}$ (298 K)'},
}

# ============================================================
# FIGURE 1: 4-panel overview (double-column, ~3.375 in wide each)
# ============================================================
fig1, axes = plt.subplots(2, 2, figsize=(7.0, 6.5))
plt.rcParams.update({'font.size': 8, 'font.family': 'serif'})

# (a) All points
ax = axes[0,0]
for s in ['LPS','LLZO','Argyrodite','NASICON','Na-NZSP']:
    pts = systems[s]
    vf = [p['vf']*100 for p in pts]
    sig = [p['sig'] for p in pts]
    st = style[s]
    ax.scatter(vf, sig, c=st['c'], marker=st['m'], s=st['ms']**2,
               alpha=0.75, edgecolors='k', linewidth=0.3, label=st['label'], zorder=5)
ax.set_yscale('log')
ax.set_xlabel(r'Effective free volume $v_f$ (%)', fontsize=8)
ax.set_ylabel(r'$\sigma$ (S cm$^{-1}$)', fontsize=8)
ax.set_title('(a)', fontsize=9, loc='left', fontweight='bold')
ax.legend(fontsize=5.5, loc='lower left', framealpha=0.8)
ax.grid(True, alpha=0.15)
ax.tick_params(labelsize=7)

# (b) Same-composition density scans
ax = axes[0,1]
# Al-LLZO
ax.plot([0.2,3.5,11.1], [5.70e-4,3.32e-4,3.40e-5], 's-',
        color='#1f77b4', ms=6, mec='k', mew=0.3, lw=1.5, label='LLZO Al-doped')
# Argyrodite T-scan
a_tscan = [(1.50,6.60e-4),(1.56,1.61e-3),(1.60,2.67e-3),(1.66,2.98e-3),(1.69,3.56e-3)]
ax.plot([(1-r/1.64)*100 for r,_ in a_tscan], [s for _,s in a_tscan], 'D-',
        color='#2ca02c', ms=6, mec='k', mew=0.3, lw=1.5, label=r'Li$_6$PS$_5$Cl')
# Na-NZSP
nz_vf = sorted(nzsp, key=lambda p: p['vf'])
ax.plot([p['vf']*100 for p in nz_vf], [p['sig'] for p in nz_vf], 'v-',
        color='#ff7f0e', ms=6, mec='k', mew=0.3, lw=1.5, label='Na-NZSP')
ax.set_yscale('log')
ax.set_xlabel(r'$v_f$ (%)', fontsize=8)
ax.set_ylabel(r'$\sigma$ (S cm$^{-1}$)', fontsize=8)
ax.set_title('(b)', fontsize=9, loc='left', fontweight='bold')
ax.legend(fontsize=6, framealpha=0.8)
ax.grid(True, alpha=0.15)
ax.tick_params(labelsize=7)
ax.text(0.95, 0.05, 'Fixed chemistry\nonly density varies',
        transform=ax.transAxes, fontsize=6, ha='right', va='bottom',
        bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow', alpha=0.7))

# (c) Regime map
ax = axes[1,0]
for s in ['LPS','LLZO','Argyrodite','NASICON','Na-NZSP']:
    pts = systems[s]
    vf = np.array([p['vf'] for p in pts])
    sig = np.array([p['sig'] for p in pts])
    sig_n = sig / sig.max()
    st = style[s]
    ax.scatter(vf*100, sig_n, c=st['c'], marker=st['m'], s=st['ms']**2,
               alpha=0.7, edgecolors='k', linewidth=0.3, zorder=5)
# Regime labels
ax.text(8, 0.85, 'Peak\nregime', fontsize=7, ha='center', color='#d62728',
        fontweight='bold', bbox=dict(boxstyle='round', fc='white', alpha=0.7))
ax.text(9, 0.15, 'Decay\nregime', fontsize=7, ha='center', color='#1f77b4',
        fontweight='bold', bbox=dict(boxstyle='round', fc='white', alpha=0.7))
ax.set_yscale('log')
ax.set_xlabel(r'$v_f$ (%)', fontsize=8)
ax.set_ylabel(r'$\sigma/\sigma_{\mathrm{max}}$', fontsize=8)
ax.set_title('(c)', fontsize=9, loc='left', fontweight='bold')
ax.grid(True, alpha=0.15)
ax.tick_params(labelsize=7)

# (d) Schematic
ax = axes[1,1]
ax.axis('off')
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)

# Equation
ax.text(5, 9.0, r'$\sigma_i(v_f) = N_i \times D_i(v_f) \times F_i(v_f)$',
        fontsize=11, ha='center', va='center', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.4', fc='lightyellow', ec='k', lw=1.2))

# Three boxes
boxes = [
    (1.5, 6.0, r'$N_i$'+'\nCarrier\ndensity', '#ffcccc'),
    (5.0, 6.0, r'$D_i(v_f)$'+'\nLocal\nhopping', '#ccffcc'),
    (8.5, 6.0, r'$F_i(v_f)$'+'\nConnectivity\nactivation', '#ccccff'),
]
for bx, by, txt, fc in boxes:
    ax.add_patch(FancyBboxPatch((bx-1.2, by-1.2), 2.4, 2.4,
                                boxstyle='round,pad=0.15', fc=fc, ec='k', lw=0.8))
    ax.text(bx, by, txt, fontsize=7, ha='center', va='center')

# v_f arrow
ax.annotate('', xy=(8.5, 3.5), xytext=(1.5, 3.5),
            arrowprops=dict(arrowstyle='->', lw=1.5, color='gray'))
ax.text(5.0, 3.1, r'$v_f = 1 - \rho/\rho_0$', fontsize=9, ha='center',
        color='gray', fontstyle='italic')

# Regime text
ax.text(2.5, 1.5, 'Peak type:\n'+r'$D\uparrow$ vs $F\downarrow$',
        fontsize=7, ha='center', color='#d62728',
        bbox=dict(boxstyle='round', fc='#ffeeee', ec='#d62728', lw=0.5))
ax.text(7.5, 1.5, 'Decay type:\n'+r'$F\downarrow$ dominates',
        fontsize=7, ha='center', color='#1f77b4',
        bbox=dict(boxstyle='round', fc='#eeeeff', ec='#1f77b4', lw=0.5))

ax.set_title('(d)', fontsize=9, loc='left', fontweight='bold')

plt.tight_layout()
fig1.savefig('prl_fig1.png', dpi=600, bbox_inches='tight')
fig1.savefig('prl_fig1.eps', bbox_inches='tight')
print("[Saved: prl_fig1.png/eps]")

# ============================================================
# FIGURE 2: Correlation bar chart (single-column)
# ============================================================
fig2, ax2 = plt.subplots(figsize=(3.375, 2.5))

sys_order = ['LPS','LLZO','Argyrodite','NASICON','Na-NZSP']
rs = []
ns = []
cols = []
labels = []
for s in sys_order:
    pts = systems[s]
    vf = np.array([p['vf'] for p in pts])
    sig = np.array([p['sig'] for p in pts])
    r, _ = pearsonr(vf, np.log10(sig))
    rs.append(r)
    ns.append(len(pts))
    cols.append(style[s]['c'])
    labels.append(s)

bars = ax2.barh(labels, rs, color=cols, edgecolor='k', linewidth=0.5, height=0.6)
for bar, r_val, n in zip(bars, rs, ns):
    xpos = r_val + (0.03 if r_val > 0 else -0.03)
    ax2.text(xpos, bar.get_y()+bar.get_height()/2,
             f'{r_val:+.2f} ({n})',
             va='center', ha='left' if r_val > 0 else 'right',
             fontsize=7, fontweight='bold')
ax2.axvline(x=0, color='k', linewidth=0.5)
ax2.set_xlabel(r'$r(v_f,\,\log_{10}\sigma)$', fontsize=8)
ax2.set_xlim(-1.15, 0.25)
ax2.tick_params(labelsize=7)
ax2.grid(True, alpha=0.15, axis='x')
plt.tight_layout()
fig2.savefig('prl_fig2.png', dpi=600, bbox_inches='tight')
fig2.savefig('prl_fig2.eps', bbox_inches='tight')
print("[Saved: prl_fig2.png/eps]")

# ============================================================
# FIGURE 3: Money plot — normalized σ vs v_f
# ============================================================
fig3, ax3 = plt.subplots(figsize=(3.375, 3.0))

for s in ['LPS','LLZO','Argyrodite','NASICON','Na-NZSP']:
    pts = systems[s]
    vf = np.array([p['vf']*100 for p in pts])
    sig = np.array([p['sig'] for p in pts])
    sig_n = sig / sig.max()
    st = style[s]
    ax3.scatter(vf, sig_n, c=st['c'], marker=st['m'], s=st['ms']**2,
                alpha=0.7, edgecolors='k', linewidth=0.3, label=st['label'], zorder=5)

ax3.set_yscale('log')
ax3.set_xlabel(r'Effective free volume $v_f$ (%)', fontsize=8)
ax3.set_ylabel(r'$\sigma / \sigma_{\mathrm{max}}$', fontsize=8)
ax3.legend(fontsize=5.5, loc='lower left', framealpha=0.8)
ax3.grid(True, alpha=0.15)
ax3.tick_params(labelsize=7)

# Regime annotations
ax3.axhline(y=1.0, color='gray', linestyle=':', alpha=0.3)

plt.tight_layout()
fig3.savefig('prl_fig3.png', dpi=600, bbox_inches='tight')
fig3.savefig('prl_fig3.eps', bbox_inches='tight')
print("[Saved: prl_fig3.png/eps]")

print(f"\nTotal data points: {len(all_data)}")
for s in sys_order:
    print(f"  {s}: {len(systems[s])}")
