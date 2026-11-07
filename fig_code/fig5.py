"""NC Fig. 5 The same AR storms in a warmer climate: heavy rain climbs onto higher, steeper ground, fastest in the coldest storms.
a-d Maps (3 km WRF): >=100 mm of rain in 48 h in the control run (amber) and the area added in the SSP5-8.5 warmer run (red), the terrain
contour at the control-run rain line (where half of the heavy precipitation fell as rain), landslides.
e Area with >=100 mm of rain in 48 h relative to the control run against the warming of the 3 km domain (control, x0.25, x0.5, x0.75, x1).
f Rain share of precipitation above 1,000 m against the warming. g Growth of the heavy-rain area per degree against the control rain share.
h Exposure inside >=100 mm of rain in 48 h, warmer/control: steep land (slope >=25 deg), roads and population."""
import os as _os
REPO = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
DATA = _os.path.join(REPO, 'data'); FIG = _os.environ.get('FIGOUT', _os.path.join(REPO, 'fig')); CODE = _os.path.join(REPO, 'fig_code')
import os, glob, sys
sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from arcommon import *
import cartopy.crs as ccrs, cartopy.feature as cfeature
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
WP = f'{DATA}/inputs/wrf/post'; CS = f'{DATA}/inputs/cases'; EDGE = 10
EVS = [('NOR', 'NOR_2011', 'Western Norway', 'Dec 2011', '#0072B2'), ('BC', 'BC_2021', 'British Columbia', 'Nov 2021', '#8C6BB1'),
       ('LIG', 'LIG_2014', 'Liguria and Piedmont', 'Nov 2014', '#009E73'), ('ORE', 'ORE_1996', 'Oregon', 'Nov 1996', '#E69F00'),
       ('RSB', 'RSB_2024', 'Rio Grande do Sul', 'Apr–May 2024', '#A6761D')]
AMB, RED = '#F2A63B', '#B2182B'
L = pd.read_csv(f'{DATA}/derived/n7/levels_summary.csv')
MOD = pd.read_csv(f'{DATA}/proc/cluster/final/WRF_MODELS_NOR_2011.csv')   # western Norway with the change of single CMIP6 models
def load(ev, arm):
    z = np.load(glob.glob(f'{WP}/{ev}_*/{ev}_*_{arm}.npz')[0]); return {k: z[k][EDGE:-EDGE, EDGE:-EDGE].astype(float) for k in z.files if z[k].ndim == 2}
D = {ev: (load(ev, 'ctl'), load(ev, 'pgw585'), np.load(f'{DATA}/inputs/dem/{ev}_steep.npz')) for ev, *_ in EVS}
import xarray as xr
from scipy.ndimage import gaussian_filter
IVTC = ['#DCE8F3', '#B3CDE8', '#7FA8D4', '#4E7FBF']
def ivt(ev): return xr.open_dataset(glob.glob(f'{DATA}/inputs/wrf/post/{ev}_*/*ctl_d01_ivt.nc')[0])
W, H = 183, 125
fig = plt.figure(figsize=(W * MM, H * MM))
def ax_mm(l, b, w, h, **k): return fig.add_axes([l / W, b / H, w / W, h / H], **k)
PC = ccrs.PlateCarree()
ORDER = ['NOR', 'BC', 'ORE', 'LIG', 'RSB']
gap = 1.6; mw_ = (W - 1.0 - 4 * gap) / 5; mh_ = mw_; mb_ = H - 9.0 - mh_
for k, ev in enumerate(ORDER):
    _, key, name, when, col = next(e for e in EVS if e[0] == ev)
    x0 = 0.5 + k * (mw_ + gap)
    fig.text((x0 + 3.4) / W, (mb_ + mh_ + 0.5) / H, f'{name}\n{when}', fontsize=6.2, va='bottom', color=col, linespacing=1.0)
    fig.text((x0 - 0.3) / W, (H - 0.3) / H, 'abcde'[k], fontsize=9, fontweight='bold', va='top')
    c, w, s = D[ev]; la, lo = c['lat'], c['lon']; la0, lo0 = float(np.median(la)), float(np.median(lo))
    prj = ccrs.AzimuthalEquidistant(central_longitude=lo0, central_latitude=la0)
    ax = ax_mm(x0, mb_, mw_, mh_, projection=prj)
    edge = np.r_[np.c_[lo[0], la[0]], np.c_[lo[-1], la[-1]], np.c_[lo[:, 0], la[:, 0]], np.c_[lo[:, -1], la[:, -1]]]
    xy = prj.transform_points(PC, edge[:, 0], edge[:, 1])
    cx, cy = (xy[:, 0].min() + xy[:, 0].max()) / 2, (xy[:, 1].min() + xy[:, 1].max()) / 2
    hw = max(xy[:, 0].max() - xy[:, 0].min(), xy[:, 1].max() - xy[:, 1].min()) / 2 * 1.25
    ax.set_xlim(cx - hw, cx + hw); ax.set_ylim(cy - hw, cy + hw)
    ax.add_feature(cfeature.LAND.with_scale('50m'), facecolor=LAND_C, edgecolor='none', zorder=0)
    ax.add_feature(cfeature.OCEAN.with_scale('50m'), facecolor='#FFFFFF', edgecolor='none', zorder=0)
    hillshade(ax, extent=(float(lo.min()) - 8, float(lo.max()) + 8, float(la.min()) - 5, float(la.max()) + 5), dark=0.5, light=0.4, zorder=0.4)
    d1 = ivt(ev); iv = gaussian_filter(d1.ivt_max_event.values, 1.0)
    ax.contourf(d1.lon.values, d1.lat.values, iv, levels=[500, 750, 1000, 1300, 5000], colors=IVTC, alpha=0.55, transform=PC, zorder=0.6)
    ax.contour(d1.lon.values, d1.lat.values, iv, levels=[500, 750, 1000, 1300], colors=['#4E7FBF'], linewidths=0.3, alpha=0.8, transform=PC, zorder=0.7)
    land = c['landmask'] > 0.5; hc = land & (c['liq_max48h'] >= 100); hw_ = land & (w['liq_max48h'] >= 100)
    ax.pcolormesh(lo, la, np.where(hw_ & ~hc, 1.0, np.nan), cmap=ListedColormap([RED]), vmin=0, vmax=1, transform=PC, shading='auto', rasterized=True, zorder=1.1)
    ax.pcolormesh(lo, la, np.where(hc, 1.0, np.nan), cmap=ListedColormap([AMB]), vmin=0, vmax=1, transform=PC, shading='auto', rasterized=True, zorder=1.2)
    A = c['cell_area_km2']; z0 = np.average(c['hgt'][hc], weights=A[hc]); z1 = np.average(c['hgt'][hw_ & ~hc], weights=A[hw_ & ~hc])
    print(ev, 'mean elevation control/added', round(z0), round(z1))
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), lw=0.35, edgecolor='#333333', zorder=2)
    ax.add_feature(cfeature.BORDERS.with_scale('50m'), lw=0.2, edgecolor='#777777', zorder=2)
    ax.plot(np.r_[lo[0], lo[:, -1], lo[-1][::-1], lo[::-1, 0]], np.r_[la[0], la[:, -1], la[-1][::-1], la[::-1, 0]], color='#333333', lw=0.5, ls=(0, (2.5, 1.5)), transform=PC, zorder=3)
    p = pd.read_csv(f'{CS}/{ev}_points.csv'); plon = p.longitude if 'longitude' in p else p.lon; plat = p.latitude if 'latitude' in p else p.lat
    ax.scatter(plon, plat, s=0.9 if len(p) > 500 else 1.4, c='#111111', edgecolor='none', transform=PC, zorder=4, alpha=0.75 if len(p) > 500 else 0.95)
    dT = L[(L.event == key) & (L.run == 'pgw585')].d02_dT2m_land.iloc[0]
    ax.text(0.03, 0.03, f'{len(p):,} landslides', transform=ax.transAxes, fontsize=5.8, va='bottom', ha='left', color='#111111', path_effects=HALO, zorder=6)
    ax.text(0.97, 0.97, f'+{dT:.1f} °C', transform=ax.transAxes, fontsize=6.2, va='top', ha='right', color=RED, fontweight='bold', path_effects=HALO, zorder=6)
    ax.spines['geo'].set_linewidth(0.5)
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
ky2 = ky - 2.4
# ------------------------------------------------------------- row 2
rb = 12.5; yl = ky2 - 4.0; rh = yl - 5.5 - rb
xs = [12.0, 66.0, 132.0]; pw = 40.0
axe = ax_mm(xs[0], rb, pw, rh); axg = ax_mm(xs[1], rb, pw, rh); axh = ax_mm(xs[2], rb, 48.0, rh)
gr = {}
CC = pd.read_csv(f'{DATA}/proc/rev/cc_rates.csv').set_index('event')   # control rain scaled by 7% per degree
for ev, key, name, when, col in EVS:
    g = L[L.event == key].sort_values('d02_dT2m_land'); a0 = g.area_liq48_ge100.iloc[0]
    axe.plot(g.d02_dT2m_land, g.area_liq48_ge100 / a0, '-o', color=col, lw=1.0, ms=2.8, mfc=col, mec='white', mew=0.4, zorder=3)
    b = np.polyfit(g.d02_dT2m_land, np.log(g.area_liq48_ge100), 1)[0]; gr[ev] = (np.exp(b) - 1) * 100
    axg.plot(g.rain_share_ge1000m.iloc[0] * 100, gr[ev], 'o', ms=4.2, color=col, mec='white', mew=0.5, zorder=3)
    xc_ = g.rain_share_ge1000m.iloc[0] * 100; cc_ = float(CC.loc[key, 'cc'])
    axg.plot([xc_, xc_], [cc_, gr[ev]], color=col, lw=0.7, alpha=0.6, zorder=2)
    axg.plot(xc_, cc_, 'o', ms=3.4, mfc='white', mec=col, mew=0.8, zorder=3)
axe.plot(MOD.dT2m_land, MOD.A100_ratio, 'o', ms=3.0, mfc='white', mec=EVS[0][4], mew=0.8, ls='none', zorder=2.5)
axe.plot([2.55], [5.6], 'o', ms=3.0, mfc='white', mec=EVS[0][4], mew=0.8, zorder=2.5)
axe.text(2.72, 5.6, 'Norway, single\nCMIP6 models', fontsize=5.9, color=EVS[0][4], ha='left', va='center', linespacing=0.95)
axe.set_yscale('log'); axe.set_ylim(0.9, 80); axe.set_yticks([1, 2, 5, 10, 20, 50]); axe.set_yticklabels(['1', '2', '5', '10', '20', '50'])
axe.yaxis.set_minor_locator(matplotlib.ticker.NullLocator()); axe.set_xlim(-0.15, 4.9); axe.set_xticks([0, 1, 2, 3, 4])
axe.set_xlabel('Warming (°C)'); axe.set_ylabel('Area with ≥100 mm in 48 h,\nrelative to control')
nm = {'NOR': 'Norway', 'BC': 'British Columbia', 'LIG': 'Liguria and Piedmont', 'ORE': 'Oregon', 'RSB': 'Rio Grande do Sul'}
off = {'NOR': (4, 0, 'left'), 'BC': (0, 9, 'center'), 'LIG': (5, 3, 'left'), 'ORE': (5, -4, 'left'), 'RSB': (-4, -6, 'right')}
for ev, key, name, when, col in EVS:
    x = L[(L.event == key) & (L.run == 'ctl')].rain_share_ge1000m.iloc[0] * 100; dx, dy, ha = off[ev]
    axg.annotate({'BC': 'British\nColumbia', 'LIG': 'Liguria &\nPiedmont', 'RSB': 'Rio Grande\ndo Sul'}.get(ev, nm[ev]), (x, gr[ev]), xytext=(dx, dy), textcoords='offset points', fontsize=5.9, color=col, ha=ha, va='center', linespacing=0.95)
axg.set_yscale('log'); axg.set_ylim(3, 400); axg.set_yticks([5, 10, 20, 50, 100, 200]); axg.set_yticklabels(['5', '10', '20', '50', '100', '200'])
axg.yaxis.set_minor_locator(matplotlib.ticker.NullLocator()); axg.set_xlim(0, 108); axg.set_xticks([0, 25, 50, 75, 100])
from matplotlib.lines import Line2D as _L2
axg.legend(handles=[_L2([], [], marker='o', ls='none', ms=4.0, color='#555555', mec='white', label='WRF runs'), _L2([], [], marker='o', ls='none', ms=3.4, mfc='white', mec='#555555', mew=0.8, label='Control rain\n+7% per °C')],
           loc='lower left', frameon=False, fontsize=5.8, handletextpad=0.3, borderaxespad=0.2, labelspacing=0.5)
axg.set_xlabel('Control rain share above 1,000 m (%)', labelpad=2); axg.set_ylabel('Growth of the area with\n≥100 mm in 48 h (% per °C)')
# h: exposure ratios
E = pd.read_csv(f'{DATA}/inputs/dem/steep_exposure.csv'); MK = [('steep', 'Steep land (≥25°)', 'o', 'k'), ('road', 'Roads', 's', '#4D4D4D'), ('pop', 'Population', 'D', '#9A9A9A')]
for i, (ev, key, name, when, col) in enumerate(EVS):
    y = len(EVS) - 1 - i
    st = E[(E.event == ev) & (E.threshold == '100 mm') & (E.steep == 'slope>=25')].iloc[0]
    c = L[(L.event == key) & (L.run == 'ctl')].iloc[0]; w = L[(L.event == key) & (L.run == 'pgw585')].iloc[0]
    v = dict(steep=st.steep_w / max(st.steep_c, 1), road=w.road_km_liq48_ge100 / c.road_km_liq48_ge100, pop=w.pop_liq48_ge100 / c.pop_liq48_ge100)
    axh.plot([min(v.values()), max(v.values())], [y, y], color='#D0D0D0', lw=1.4, solid_capstyle='round', zorder=1)
    for (k_, lab, m, cc), dy in zip(MK, (0.0, 0.0, 0.0)): axh.plot(v[k_], y + dy, marker=m, ms=3.2, ls='none', mfc=cc if k_ != 'pop' else 'white', mec=cc if k_ != 'pop' else '#4D4D4D', mew=0.6, zorder=3)
    axh.text(-0.04, y, {'LIG': 'Liguria &\nPiedmont', 'RSB': 'Rio Grande\ndo Sul', 'BC': 'British\nColumbia', 'NOR': 'Western\nNorway'}.get(ev, nm[ev]), fontsize=5.9, color=col, ha='right', va='center', transform=axh.get_yaxis_transform(), linespacing=0.95)
axh.set_xscale('log'); axh.set_xlim(0.7, 50); axh.set_xticks([1, 2, 5, 10, 30]); axh.set_xticklabels(['1', '2', '5', '10', '30'])
axh.xaxis.set_minor_locator(matplotlib.ticker.NullLocator()); axh.axvline(1, color='#BDBDBD', lw=0.5, zorder=0)
axh.set_yticks([]); axh.set_ylim(-0.6, len(EVS) - 0.4); axh.spines[['top', 'right', 'left']].set_visible(False)
axh.set_xlabel('Ratio, warmer/control')
axh.legend(handles=[Line2D([], [], marker=m, ls='none', ms=3.2, mfc=cc if k_ != 'pop' else 'white', mec=cc if k_ != 'pop' else '#4D4D4D', mew=0.6, label=lab) for k_, lab, m, cc in MK],
           loc='lower right', bbox_to_anchor=(1.0, 1.0), ncol=3, frameon=False, fontsize=5.6, handletextpad=0.1, columnspacing=0.4, borderaxespad=0.0)
for a in (axe, axg): a.spines[['top', 'right']].set_visible(False)
for s_, x_ in zip('fgh', [0.3, xs[1] - 12.5, xs[2] - 15.5]): fig.text(x_ / W, yl / H, s_, fontsize=9, fontweight='bold', va='top')
bump_fonts(fig)
print(overlap_check(fig)); print({k: round(v) for k, v in gr.items()})
fig.savefig(f'{FIG}/Fig5.png', dpi=300); fig.savefig(f'{FIG}/Fig5.pdf'); print('saved')
