"""SI figure: vertical cross-sections along the AR through the landslide centroid of each simulated storm, control and warmer runs.
Shading: rain water (g/kg); contours: frozen hydrometeors (snow + graupel + ice, g/kg); thick line: 0 degC isotherm; grey: terrain.
Right column: surface rain and snow rates along the transect (24-h window means) in the control and warmer runs."""
import os as _os
REPO = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
DATA = _os.path.join(REPO, 'data'); FIG = _os.environ.get('FIGOUT', _os.path.join(REPO, 'fig')); CODE = _os.path.join(REPO, 'fig_code')
import os, sys
sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from arcommon import *
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
from matplotlib.lines import Line2D
R = f'{DATA}/proc/cluster'
EVS = [('NOR_2011', 'Western Norway, Dec 2011'), ('BC_2021', 'British Columbia, Nov 2021'), ('ORE_1996', 'Oregon, Nov 1996'), ('LIG_2014', 'Liguria and Piedmont, Nov 2014')]
def _ok(ev):
    try: np.load(f'{R}/WRF_XSEC_{ev}.npz')['ctl_T']; return True
    except Exception: return False
EVS = [e for e in EVS if _ok(e[0])]
BLUE, RED, AMB, SNOWB = '#2166AC', '#B2182B', '#F2A63B', '#8FBBD9'
def field(z, run, var):
    a = z[f'{run}_{var}'].astype(float); a[a == float(z['fill'])] = np.nan; return a * float(z[f'scale_{var}'])
W = 183; n = len(EVS); rh = 30.0; H = 22 + n * (rh + 4.0)
fig = plt.figure(figsize=(W * MM, H * MM))
def ax_mm(l, b, w, h, **k): return fig.add_axes([l / W, b / H, w / W, h / H], **k)
cmap = LinearSegmentedColormap.from_list('rain', ['#FFFFFF', '#FDE6B8', '#F2A63B', '#C65D00', '#7A2E00'])
levels = [0.02, 0.05, 0.1, 0.2, 0.3, 0.45, 0.6, 0.8]; norm = BoundaryNorm(levels, cmap.N, extend='max')
xw, sw, gap = 52.0, 43.0, 4.0; x0s = [10.0, 10.0 + xw + gap, 10.0 + 2 * (xw + gap) + 7.0]
letters = iter('abcdefghijklmnop')
for i, (ev, name) in enumerate(EVS):
    z = np.load(f'{R}/WRF_XSEC_{ev}.npz'); d = z['dist_km']; zz = z['z_m'] / 1000.0; hgt = z['hgt_m'] / 1000.0
    b = H - 6.0 - (i + 1) * rh - i * 4.0
    for j, run in enumerate(('ctl', 'pgw585')):
        ax = ax_mm(x0s[j], b, xw, rh)
        T = field(z, run, 'T'); qr = field(z, run, 'qrain'); qf = field(z, run, 'qfrozen')
        cf = ax.contourf(d, zz, qr, levels=levels, cmap=cmap, norm=norm, extend='max', zorder=1)
        ax.contour(d, zz, qf, levels=[0.1, 0.3, 0.6], colors=['#9ECAE1', '#4292C6', '#08519C'], linewidths=[0.5, 0.7, 0.9], zorder=2)
        ax.contour(d, zz, T, levels=[0], colors=['#111111'], linewidths=1.1, zorder=3)
        ax.fill_between(d, 0, hgt, color='#B5AFA4', lw=0, zorder=4); ax.plot(d, hgt, color='#4D4D4D', lw=0.5, zorder=4.5)
        ax.plot([0, 0], [0, 0.25 + hgt[np.argmin(np.abs(d))]], color=RED, lw=1.0, zorder=5)
        ax.set_xlim(-150, 150); ax.set_ylim(0, 6); ax.set_yticks([0, 2, 4, 6]); ax.set_xticks([-100, 0, 100])
        ax.spines[['top', 'right']].set_visible(False)
        if i == n - 1 and j == 0: fig.text((x0s[0] + xw + gap / 2) / W, (b - 7.0) / H, 'Distance along the AR from the landslides (km)', ha='center', va='top')
        if i != n - 1: ax.set_xticklabels([])
        if j == 0: ax.set_ylabel('Height (km)', labelpad=2)
        else: ax.set_yticklabels([])
        lab = 'Control run' if run == 'ctl' else 'Warmer run'
        ax.text(0.01, 0.97, lab, transform=ax.transAxes, fontsize=5.6, va='top', ha='left', path_effects=HALO, zorder=6)
        fig.text((x0s[j] - 9.5 if j == 0 else x0s[j] - 2.5) / W, (b + rh + 0.3) / H, next(letters), fontsize=9, fontweight='bold', va='bottom')
        if j == 0: fig.text((x0s[0] + 1.0) / W, (b + rh + 0.5) / H, name, fontsize=6.4, va='bottom')
    # surface strip
    ax = ax_mm(x0s[2], b, sw, rh)
    for run, col, al in (('ctl', BLUE, 1.0), ('pgw585', RED, 1.0)):
        r = field(z, run, 'rain_mmh'); s = field(z, run, 'snow_mmh')
        ax.plot(d, r, color=col, lw=1.0, zorder=3); ax.plot(d, s, color=col, lw=1.0, ls=(0, (2.5, 1.5)), zorder=3)
    ax2 = ax.twinx(); ax2.fill_between(d, 0, hgt, color='#D9D4CB', lw=0, zorder=0); ax2.set_ylim(0, 6 * 1.0); ax2.set_yticks([]); ax2.spines[['top', 'right', 'left']].set_visible(False); ax2.set_zorder(0); ax.set_zorder(1); ax.patch.set_visible(False)
    ax.axvline(0, color=RED, lw=0.6, zorder=2)
    ax.set_xlim(-150, 150); ymax = max(float(np.nanmax(field(z, r_, 'rain_mmh') + field(z, r_, 'snow_mmh'))) for r_ in ('ctl', 'pgw585')); ax.set_ylim(0, ymax * (1.5 if i == 0 else 1.1))
    ax.set_xticks([-100, 0, 100]); ax.spines[['top', 'right']].set_visible(False)
    if i == n - 1: ax.set_xlabel('Distance (km)', labelpad=2)
    else: ax.set_xticklabels([])
    ax.set_ylabel('Precipitation\n(mm per h)', labelpad=2, linespacing=1.0)
    fig.text((x0s[2] - 9.5) / W, (b + rh + 0.3) / H, next(letters), fontsize=9, fontweight='bold', va='bottom')
    if i == 0:
        ax.legend(handles=[Line2D([], [], color=BLUE, lw=1.0, label='Rain, control'), Line2D([], [], color=BLUE, lw=1.0, ls=(0, (2.5, 1.5)), label='Snow, control'), Line2D([], [], color=RED, lw=1.0, label='Rain, warmer'), Line2D([], [], color=RED, lw=1.0, ls=(0, (2.5, 1.5)), label='Snow, warmer')],
                  loc='upper right', frameon=False, fontsize=5.4, handlelength=1.6, borderaxespad=0.1, labelspacing=0.25, ncol=1)
# colour bar and key
cax = ax_mm(x0s[0] + 8.0, 7.0, 60.0, 1.5)
cb = fig.colorbar(cf, cax=cax, orientation='horizontal', ticks=levels); cb.ax.tick_params(length=1.5, pad=1); cb.outline.set_linewidth(0.4)
cb.ax.set_xticklabels([f'{v:g}' for v in levels]); cb.set_label('Rain water (g per kg)', labelpad=1.5)
fig.legend(handles=[Line2D([], [], color='#4292C6', lw=0.7, label='Snow, graupel and ice (0.1, 0.3, 0.6 g per kg)'), Line2D([], [], color='#111111', lw=1.1, label='0 °C'), Line2D([], [], color=RED, lw=1.0, label='Landslide centroid')],
           loc='lower left', bbox_to_anchor=((x0s[1] + 20.0) / W, 3.2 / H), frameon=False, fontsize=5.6, handlelength=1.8, ncol=3, columnspacing=1.2)
bump_fonts(fig)
print(overlap_check(fig))
fig.savefig(f'{FIG}/FigS11_xsec.png', dpi=300); fig.savefig(f'{FIG}/FigS11_xsec.pdf'); print('saved', [e[0] for e in EVS])
