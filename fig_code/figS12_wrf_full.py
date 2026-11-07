import os as _os
REPO = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
DATA = _os.path.join(REPO, 'data'); FIG = _os.environ.get('FIGOUT', _os.path.join(REPO, 'fig')); CODE = _os.path.join(REPO, 'fig_code')
import sys; sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
"""Fig. 5 (v06) The same AR storms 4 degC warmer: AR plume -> heavy rain -> landslides, and how the heavy-rain area spreads.
a-e Maps: AR vapour transport of the control run (9 km domain, event maximum IVT), the land area with >=100 mm of rain in 48 h in the control
run (amber) and the area added in the warmer run (red), the 3 km domain and the recorded landslides.
f Change in snowfall, total precipitation and rain. g Land area with >=100 mm of rain in 48 h, control -> warmer (absolute, log scale).
h Rain share of precipitation by elevation. i Ratio of the land area above a 48-h rain threshold, warmer over control."""
import os, re, glob
from arcommon import *
import cartopy.crs as ccrs, cartopy.feature as cfeature
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.ndimage import gaussian_filter
import xarray as xr

WP = os.environ.get('WRFPOST', f'{DATA}/inputs/wrf/post')
CS = f'{DATA}/inputs/cases'
EVS = [('NOR', 'Western Norway', 'Dec 2011', '#0072B2'), ('BC', 'British Columbia', 'Nov 2021', '#8C6BB1'),
       ('ORE', 'Oregon', 'Nov 1996', '#E69F00'), ('LIG', 'Liguria and Piedmont', 'Nov 2014', '#009E73'),
       ('RSB', 'Rio Grande do Sul', 'Apr–May 2024', '#A6761D')]
EDGE = 10
AMB, RED, IVTC = '#F2A63B', '#B2182B', ['#DCE8F3', '#B3CDE8', '#7FA8D4', '#4E7FBF']
def _find(ev, suffix):
    for pat in (f'{WP}/{ev}*/{ev}*_{suffix}', f'{WP}/{ev}*_{suffix}'):
        g = sorted(glob.glob(pat))
        if g: return g[0]
def load(ev, arm):
    z = np.load(_find(ev, f'{arm}.npz')); return {k: z[k][EDGE:-EDGE, EDGE:-EDGE].astype(float) for k in z.files if z[k].ndim == 2}
_LV = pd.read_csv(f'{DATA}/derived/n6/levels_summary.csv')
def summary_dt(ev):  # simulated 2-m warming over land during the event (NC definition)
    q = _LV[(_LV.event.str.startswith(ev + '_')) & (_LV.run == 'pgw585')]; return float(q.d02_dT2m_land.iloc[0])
D = {ev: (load(ev, 'ctl'), load(ev, 'pgw585'), summary_dt(ev), xr.open_dataset(_find(ev, 'ctl_d01_ivt.nc'))) for ev, *_ in EVS}

THR = (50, 100, 150, 200)
def stats(c, w):
    land = (c['landmask'] > 0.5); A = c['cell_area_km2']
    def amean(x): return (x * A)[land].sum() / A[land].sum()
    r = {}
    for k_, v_ in (('tot', 'precip_total'), ('liq', 'liq_total'), ('snow', 'snowfall'), ('run', 'runoff_total')):
        r[k_] = (amean(w[v_]) / amean(c[v_]) - 1) * 100
    for t in THR:
        b0, b1 = A[land & (c['liq_max48h'] >= t)].sum(), A[land & (w['liq_max48h'] >= t)].sum()
        r[f'a{t}_c'], r[f'a{t}_w'] = b0, b1; r[f'ratio{t}'] = b1 / b0 if b0 > 0 else (np.inf if b1 > 0 else np.nan)
    return r
S = {ev: stats(c, w) for ev, (c, w, _, _) in D.items()}
for ev in S: print(ev, {k: round(float(v), 2) for k, v in S[ev].items()}, 'dT', D[ev][2])

W, H = 183, 130
fig = plt.figure(figsize=(W * MM, H * MM))
def ax_mm(l, b, w, h, **k): return fig.add_axes([l / W, b / H, w / W, h / H], **k)
PC = ccrs.PlateCarree()
gap = 1.6; mw_ = (W - 1.0 - 4 * gap) / 5; mh_ = mw_; mb_ = H - 9.5 - mh_
for k, (ev, name, when, col) in enumerate(EVS):
    x0 = 0.5 + k * (mw_ + gap)
    fig.text((x0 + 3.4) / W, (mb_ + mh_ + 0.5) / H, f'{name}\n{when}', fontsize=6.2, va='bottom', linespacing=1.0)
    fig.text((x0 - 0.3) / W, (H - 0.3) / H, 'abcde'[k], fontsize=9, fontweight='bold', va='top')
    c, w, dT, d1 = D[ev]
    la, lo = c['lat'], c['lon']; la0, lo0 = float(np.median(la)), float(np.median(lo))
    prj = ccrs.AzimuthalEquidistant(central_longitude=lo0, central_latitude=la0)
    ax = ax_mm(x0, mb_, mw_, mh_, projection=prj)
    edge = np.r_[np.c_[lo[0], la[0]], np.c_[lo[-1], la[-1]], np.c_[lo[:, 0], la[:, 0]], np.c_[lo[:, -1], la[:, -1]]]
    xy = prj.transform_points(PC, edge[:, 0], edge[:, 1])
    cx, cy = (xy[:, 0].min() + xy[:, 0].max()) / 2, (xy[:, 1].min() + xy[:, 1].max()) / 2
    hw = max(xy[:, 0].max() - xy[:, 0].min(), xy[:, 1].max() - xy[:, 1].min()) / 2 * 1.25
    ax.set_xlim(cx - hw, cx + hw); ax.set_ylim(cy - hw, cy + hw)
    ax.add_feature(cfeature.LAND.with_scale('50m'), facecolor=LAND_C, edgecolor='none', zorder=0)
    ax.add_feature(cfeature.OCEAN.with_scale('50m'), facecolor='#FFFFFF', edgecolor='none', zorder=0)
    ext = (float(lo.min()) - 8, float(lo.max()) + 8, float(la.min()) - 5, float(la.max()) + 5)
    hillshade(ax, extent=ext, dark=0.5, light=0.4, zorder=0.4)
    # AR plume: event-maximum IVT of the control run on the 9 km domain
    iv = gaussian_filter(d1.ivt_max_event.values, 1.0)
    ax.contourf(d1.lon.values, d1.lat.values, iv, levels=[500, 750, 1000, 1300, 5000], colors=IVTC, alpha=0.55, transform=PC, zorder=0.6)
    ax.contour(d1.lon.values, d1.lat.values, iv, levels=[500, 750, 1000, 1300], colors=['#4E7FBF'], linewidths=0.3, alpha=0.8, transform=PC, zorder=0.7)
    land = c['landmask'] > 0.5
    hc = land & (c['liq_max48h'] >= 100); hw_ = land & (w['liq_max48h'] >= 100)
    ax.pcolormesh(lo, la, np.where(hw_ & ~hc, 1.0, np.nan), cmap=ListedColormap([RED]), vmin=0, vmax=1, transform=PC, shading='auto', rasterized=True, zorder=1.1)
    ax.pcolormesh(lo, la, np.where(hc, 1.0, np.nan), cmap=ListedColormap([AMB]), vmin=0, vmax=1, transform=PC, shading='auto', rasterized=True, zorder=1.2)
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), lw=0.35, edgecolor='#333333', zorder=2)
    ax.add_feature(cfeature.BORDERS.with_scale('50m'), lw=0.2, edgecolor='#777777', zorder=2)
    ax.plot(np.r_[lo[0], lo[:, -1], lo[-1][::-1], lo[::-1, 0]], np.r_[la[0], la[:, -1], la[-1][::-1], la[::-1, 0]], color='#333333', lw=0.5, ls=(0, (2.5, 1.5)), transform=PC, zorder=3)
    p = pd.read_csv(f'{CS}/{ev}_points.csv'); plon = p.longitude if 'longitude' in p else p.lon; plat = p.latitude if 'latitude' in p else p.lat
    ax.scatter(plon, plat, s=0.9 if len(p) > 500 else 1.4, c='#111111', edgecolor='none', transform=PC, zorder=4, alpha=0.75 if len(p) > 500 else 0.95)
    ax.text(0.03, 0.03, f'{len(p):,} landslides', transform=ax.transAxes, fontsize=5.8, va='bottom', ha='left', color='#111111', path_effects=HALO, zorder=6)
    ax.text(0.97, 0.97, f'+{dT:.1f} °C', transform=ax.transAxes, fontsize=6.2, va='top', ha='right', color=RED, fontweight='bold', path_effects=HALO, zorder=6)
    ax.spines['geo'].set_linewidth(0.5)

# key row under the maps
ky = mb_ - 3.6
def key(x, h, lab, dx=5.0):
    fig.add_artist(h); fig.text((x + dx) / W, ky / H, lab, fontsize=5.9, va='center')
x_ = 2.0
for j, (lab, c_) in enumerate((('500', IVTC[0]), ('750', IVTC[1]), ('1,000', IVTC[2]), ('1,300', IVTC[3]))):
    fig.add_artist(plt.Rectangle(((x_ + j * 6.0) / W, (ky - 1.1) / H), 6.0 / W, 2.2 / H, facecolor=c_, alpha=0.75, lw=0, transform=fig.transFigure))
    fig.text((x_ + j * 6.0) / W, (ky - 1.5) / H, lab, fontsize=5.2, ha='center', va='top')
fig.text((x_ + 25.5) / W, ky / H, 'AR vapour transport, control run\n(event maximum, kg m$^{-1}$ s$^{-1}$)', fontsize=5.8, va='center', linespacing=1.0)
key(70.0, plt.Rectangle((70.0 / W, (ky - 1.1) / H), 3.5 / W, 2.2 / H, facecolor=AMB, lw=0, transform=fig.transFigure), '≥100 mm of rain in 48 h, control run', dx=4.5)
key(115.0, plt.Rectangle((115.0 / W, (ky - 1.1) / H), 3.5 / W, 2.2 / H, facecolor=RED, lw=0, transform=fig.transFigure), 'Added in the warmer run', dx=4.5)
key(146.5, Line2D([146.5 / W, 150.5 / W], [ky / H] * 2, color='#333333', lw=0.5, ls=(0, (2.5, 1.5))), '3 km domain', dx=5.0)
key(166.5, Line2D([167.5 / W], [ky / H], marker='o', ms=1.8, mfc='#111111', mec='none', ls='none'), 'Landslides', dx=3.0)

# ------------------------------------------------ row 2
rb = 10.0; yl = ky - 5.2; rh = yl - 6.5 - rb
pw = 33.0; xs0 = [21.0, 65.5, 107.5, 149.5]
# f: dot chart of the changes per event
axf = ax_mm(xs0[0], rb, pw, rh)
MK = [('snow', 'Snowfall', dict(marker='v', mfc='#9ECAE1', mec='#3182BD', mew=0.5)),
      ('tot', 'Total precipitation', dict(marker='o', mfc='white', mec='k', mew=0.6)),
      ('liq', 'Rain', dict(marker='o', mfc='k', mec='k', mew=0.6))]
for i, (ev, name, when, col) in enumerate(EVS):
    y = len(EVS) - 1 - i; v = [S[ev][k_] for k_, *_ in MK]
    axf.plot([np.nanmin(v), np.nanmax(v)], [y, y], color='#BDBDBD', lw=1.4, solid_capstyle='round', zorder=1)
    for (k_, lab, st_), vv in zip(MK, v): axf.plot(vv, y, ls='none', ms=3.4, zorder=3, **st_)
    cc = (1.07 ** D[ev][2] - 1) * 100
    axf.plot([cc, cc], [y - 0.28, y + 0.28], color='#7F7F7F', lw=1.0, zorder=2, solid_capstyle='butt')
    axf.text(-0.03, y, name.replace(' and ', ' & '), transform=axf.get_yaxis_transform(), ha='right', va='center', fontsize=5.9, color=col)
axf.axvline(0, color='#7F7F7F', lw=0.5, zorder=0)
axf.set_yticks([]); axf.set_ylim(-0.6, len(EVS) - 0.4); axf.spines[['top', 'right', 'left']].set_visible(False)
axf.set_xlim(-100, 100); axf.set_xticks([-100, -50, 0, 50, 100]); axf.set_xlabel('Change, warmer vs control (%)', fontsize=6.6)
axf.legend(handles=[Line2D([], [], ls='none', ms=3.4, label=lab, **st_) for _, lab, st_ in MK] + [Line2D([], [], color='#7F7F7F', lw=1.0, marker='|', ls='none', ms=5, label='7% per °C')],
           loc='lower left', bbox_to_anchor=(-0.62, 1.0), ncol=2, frameon=False, fontsize=5.5, handletextpad=0.2, columnspacing=0.7, borderaxespad=0.0, labelspacing=0.25)

# g: absolute land area with >=100 mm of rain in 48 h, control -> warmer
axg = ax_mm(xs0[1], rb, pw, rh)
for i, (ev, name, when, col) in enumerate(EVS):
    y = len(EVS) - 1 - i; a0, a1 = max(S[ev]['a100_c'], 1), S[ev]['a100_w']
    axg.plot([a0, a1], [y, y], color='#BDBDBD', lw=1.6, solid_capstyle='round', zorder=1)
    axg.plot(a0, y, 'o', ms=3.6, mfc=AMB, mec='white', mew=0.4, zorder=3); axg.plot(a1, y, 'o', ms=3.6, mfc=RED, mec='white', mew=0.4, zorder=3)
    axg.text(a1 * 1.6, y, f'×{a1 / a0:.1f}' if a1 / a0 < 10 else f'×{a1 / a0:.0f}', fontsize=5.8, va='center', ha='left', color=RED)
axg.set_xscale('log'); axg.set_xlim(80, 3e6); axg.set_xticks([100, 1000, 1e4, 1e5, 1e6]); axg.set_xticklabels(['10²', '10³', '10⁴', '10⁵', '10⁶'])
axg.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
axg.set_yticks([]); axg.set_ylim(-0.6, len(EVS) - 0.4); axg.spines[['top', 'right', 'left']].set_visible(False)
axg.set_xlabel('Area with ≥100 mm in 48 h (km²)', fontsize=6.6)
axg.legend(handles=[Line2D([], [], ls='none', marker='o', ms=3.4, mfc=AMB, mec='white', label='Control'), Line2D([], [], ls='none', marker='o', ms=3.4, mfc=RED, mec='white', label='Warmer')],
           loc='lower left', bbox_to_anchor=(0.0, 1.0), ncol=2, frameon=False, fontsize=5.5, handletextpad=0.2, columnspacing=0.8, borderaxespad=0.0)

# h: rain share by elevation
axh = ax_mm(xs0[2], rb, pw, rh)
eb = np.array([0, 250, 500, 750, 1000, 1250, 1500, 2000, 3000])
for ev, name, when, col in EVS:
    c, w, _, _ = D[ev]; land = c['landmask'] > 0.5; A = c['cell_area_km2']
    for arm, ls_ in ((c, (0, (2.2, 1.2))), (w, '-')):
        sh, xm = [], []
        for e0, e1 in zip(eb[:-1], eb[1:]):
            m = land & (c['hgt'] >= e0) & (c['hgt'] < e1)
            if m.sum() < 30: continue
            sh.append((arm['liq_total'] * A)[m].sum() / (arm['precip_total'] * A)[m].sum() * 100); xm.append((e0 + e1) / 2)
        axh.plot(xm, sh, color=col, lw=1.0 if ls_ == '-' else 0.8, ls=ls_)
axh.set_xlim(0, 2600); axh.set_ylim(0, 103); axh.set_xticks([0, 1000, 2000]); axh.set_yticks([0, 25, 50, 75, 100])
axh.set_xlabel('Terrain elevation (m)'); axh.set_ylabel('Rain share (%)'); axh.spines[['top', 'right']].set_visible(False)
axh.legend(handles=[Line2D([], [], color='#4D4D4D', lw=0.8, ls=(0, (2.2, 1.2)), label='Control'), Line2D([], [], color='#4D4D4D', lw=1.0, label='Warmer')],
           loc='lower left', frameon=False, fontsize=5.8, handlelength=1.8, borderaxespad=0.3)

# i: ratio of the area above a 48-h rain threshold
axi = ax_mm(xs0[3], rb, pw - 1, rh)
for ev, name, when, col in EVS:
    r = np.array([S[ev][f'ratio{t}'] for t in THR], float); fin = np.isfinite(r)
    axi.plot(np.array(THR)[fin], r[fin], color=col, lw=1.0, marker='o', ms=3.0, mfc=col, mec='white', mew=0.4, zorder=3)
    for t, rr in zip(THR, r):
        if np.isinf(rr): axi.annotate('', xy=(t, 46), xytext=(t, 30), arrowprops=dict(arrowstyle='-|>', color=col, lw=0.8, mutation_scale=5), zorder=3)
axi.set_yscale('log'); axi.set_ylim(1, 50); axi.set_yticks([1, 2, 5, 10, 20, 50]); axi.set_yticklabels(['1', '2', '5', '10', '20', '50'])
axi.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
axi.set_xlim(40, 210); axi.set_xticks(THR); axi.axhline(1, color='#BDBDBD', lw=0.5, zorder=0)
axi.set_xlabel('48-h rain threshold (mm)'); axi.set_ylabel('Area ratio, warmer/control'); axi.spines[['top', 'right']].set_visible(False)

for s_, x_ in zip('fghi', [0.3, xs0[1] - 3.5, xs0[2] - 9.5, xs0[3] - 9.0]): fig.text(x_ / W, yl / H, s_, fontsize=9, fontweight='bold', va='top')
bump_fonts(fig)
print(overlap_check(fig))
out = os.environ.get('FIG5OUT', f'{FIG}/parts/FigS12_wrf_full')
fig.savefig(out + '.png', dpi=300); fig.savefig(out + '.pdf'); print('saved', out)
