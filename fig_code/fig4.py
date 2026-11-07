"""NC Fig. 4 (mechanism) Warming turns the storms' snow to rain.
a-d Maps (3 km WRF, four near-freezing storms): snowfall lost between the control and the warmer run (shading), the elevation at which
half of the storm's precipitation fell as snow in the control run (blue line) and in the warmer run (red line), landslides.
e-h Rain share of precipitation by 100 m elevation band in the storm core (control precipitation >= 100 mm) for the control run and the warmer runs.
i Rain line (elevation where half of the heavy precipitation fell as rain) against the warming for every run.
j-k Hourly series in the storm core (British Columbia, Western Norway): precipitation rate split into rain and snow in the control run with the rain rate of the warmer run, and 2-m temperature in both runs.
"""
import os as _os
REPO = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
DATA = _os.path.join(REPO, 'data'); FIG = _os.environ.get('FIGOUT', _os.path.join(REPO, 'fig')); CODE = _os.path.join(REPO, 'fig_code')
import os, glob, sys
sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from arcommon import *
import cartopy.crs as ccrs, cartopy.feature as cfeature
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter
WP = f'{DATA}/inputs/wrf/post'; CS = f'{DATA}/inputs/cases'; EDGE = 10
EVS = [('NOR', 'NOR_2011', 'Western Norway', 'Dec 2011', '#0072B2'), ('BC', 'BC_2021', 'British Columbia', 'Nov 2021', '#8C6BB1'),
       ('ORE', 'ORE_1996', 'Oregon', 'Nov 1996', '#E69F00'), ('LIG', 'LIG_2014', 'Liguria and Piedmont', 'Nov 2014', '#009E73')]
BLUE, RED = '#2166AC', '#B2182B'
L = pd.read_csv(f'{DATA}/derived/n7/levels_summary.csv')
PR = pd.read_csv(f'{DATA}/proc/cluster/WRF_ELEVBANDS.csv'); PR = PR[(PR['mask'] == 'core100') & (PR.band_lo_m != 'all')].copy()
PR['band_lo'] = PR.band_lo_m.astype(float); PR['precip'] = PR.precip_mm; PR['rain_share'] = PR.rain_mm / PR.precip_mm; PR['ncell'] = PR.land_area_km2 / 9.0
RSL = pd.read_csv(f'{DATA}/proc/rsl_all_runs.csv'); RSL = RSL[RSL['mask'] == 'core100']
CSX = pd.read_csv(f'{DATA}/proc/core_stats_local.csv'); CSX = CSX[(CSX.core_thr == 100) & CSX.rsl_m.notna() | (CSX.core_thr == 100) & CSX.core_km2.notna()]
def load(ev, arm):
    z = np.load(f'{WP}/{ev}/{ev}_{arm}.npz'); return {k: z[k][EDGE:-EDGE, EDGE:-EDGE].astype(float) for k in z.files if z[k].ndim == 2}
W, H = 183, 130
fig = plt.figure(figsize=(W * MM, H * MM))
def ax_mm(l, b, w, h, **k): return fig.add_axes([l / W, b / H, w / W, h / H], **k)
PC = ccrs.PlateCarree()
SNOWB = '#8FBBD9'
gap = 2.2; mw_ = (W - 1.0 - 3 * gap) / 4; mh_ = mw_ * 0.86; mb_ = H - 8.0 - mh_
rsl = {}
for k, (ev, key, name, when, col) in enumerate(EVS):
    x0 = 0.5 + k * (mw_ + gap)
    fig.text((x0 + 3.4) / W, (mb_ + mh_ + 0.5) / H, f'{name}, {when}', fontsize=6.4, va='bottom', color=col)
    fig.text((x0 - 0.3) / W, (H - 0.3) / H, 'abcd'[k], fontsize=9, fontweight='bold', va='top')
    c, w = load(key, 'ctl'), load(key, 'pgw585'); la, lo = c['lat'], c['lon']; la0, lo0 = float(np.median(la)), float(np.median(lo))
    prj = ccrs.AzimuthalEquidistant(central_longitude=lo0, central_latitude=la0)
    ax = ax_mm(x0, mb_, mw_, mh_, projection=prj)
    edge = np.r_[np.c_[lo[0], la[0]], np.c_[lo[-1], la[-1]], np.c_[lo[:, 0], la[:, 0]], np.c_[lo[:, -1], la[:, -1]]]
    xy = prj.transform_points(PC, edge[:, 0], edge[:, 1])
    cx, cy = (xy[:, 0].min() + xy[:, 0].max()) / 2, (xy[:, 1].min() + xy[:, 1].max()) / 2
    hwx = (xy[:, 0].max() - xy[:, 0].min()) / 2 * 1.02; hwy = hwx * mh_ / mw_
    ax.set_xlim(cx - hwx, cx + hwx); ax.set_ylim(cy - hwy, cy + hwy)
    ax.add_feature(cfeature.LAND.with_scale('50m'), facecolor=LAND_C, edgecolor='none', zorder=0)
    ax.add_feature(cfeature.OCEAN.with_scale('50m'), facecolor='#FFFFFF', edgecolor='none', zorder=0)
    hillshade(ax, extent=(float(lo.min()) - 3, float(lo.max()) + 3, float(la.min()) - 2, float(la.max()) + 2), dark=0.5, light=0.4, zorder=0.4)
    land = c['landmask'] > 0.5
    sc = c['snowfall'] + c['graupel_hail']; sw = w['snowfall'] + w['graupel_hail']
    mask = land & (c['precip_total'] >= 25)
    ssc = sc / np.maximum(c['precip_total'], 1e-6); ssw = sw / np.maximum(w['precip_total'], 1e-6)
    snow_c = mask & (ssc >= 0.5); snow_w = mask & (ssw >= 0.5); turned = snow_c & ~snow_w
    ax.pcolormesh(lo, la, np.where(snow_c & snow_w, 1.0, np.nan), cmap=ListedColormap([SNOWB]), vmin=0, vmax=1, transform=PC, shading='auto', rasterized=True, zorder=1.1)
    ax.pcolormesh(lo, la, np.where(turned, 1.0, np.nan), cmap=ListedColormap([RED]), vmin=0, vmax=1, transform=PC, shading='auto', rasterized=True, zorder=1.2)
    A = c['cell_area_km2']; ft = A[turned].sum() / A[snow_c].sum() * 100; fl = ((sc - sw) * A)[land].sum() / (sc * A)[land].sum() * 100
    print(ev, 'snow-dominated land km2 %.0f, turned %.0f%%, snowfall lost %.0f%%' % (A[snow_c].sum(), ft, fl))
    for run, cc in (('ctl', BLUE), ('pgw585', RED)):
        r = CSX[(CSX.event == key) & (CSX.run == run)].rsl_m.dropna()
        if len(r) and np.isfinite(r.iloc[0]): rsl[(ev, run)] = float(r.iloc[0])
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), lw=0.35, edgecolor='#333333', zorder=2)
    ax.add_feature(cfeature.BORDERS.with_scale('50m'), lw=0.2, edgecolor='#777777', zorder=2)
    p = pd.read_csv(f'{CS}/{ev}_points.csv'); plon = p.longitude if 'longitude' in p else p.lon; plat = p.latitude if 'latitude' in p else p.lat
    ax.scatter(plon, plat, s=0.9 if len(p) > 500 else 1.4, c='#111111', edgecolor='none', transform=PC, zorder=4, alpha=0.75 if len(p) > 500 else 0.95)
    dT = L[(L.event == key) & (L.run == 'pgw585')].d02_dT2m_land.iloc[0]
    ax.text(0.97, 0.97, f'+{dT:.1f} °C', transform=ax.transAxes, fontsize=6.2, va='top', ha='right', color=RED, fontweight='bold', path_effects=HALO, zorder=6)
    ax.text(0.03, 0.03, f'{ft:.0f}% of the snow-dominated\nland turned to rain', transform=ax.transAxes, fontsize=5.8, va='bottom', ha='left', color=RED, path_effects=HALO, zorder=6, linespacing=1.0)
    ax.spines['geo'].set_linewidth(0.5)
# key under the maps
ky = mb_ - 3.4
def key_(x, h, lab, dx=5.0):
    fig.add_artist(h); fig.text((x + dx) / W, ky / H, lab, fontsize=5.8, va='center')
key_(2.0, plt.Rectangle((2.0 / W, (ky - 1.1) / H), 3.5 / W, 2.2 / H, facecolor=SNOWB, lw=0, transform=fig.transFigure), 'Mainly snow in the control run and in the warmer run', dx=4.5)
key_(74.0, plt.Rectangle((74.0 / W, (ky - 1.1) / H), 3.5 / W, 2.2 / H, facecolor=RED, lw=0, transform=fig.transFigure), 'Mainly snow in the control run, mainly rain in the warmer run', dx=4.5)
key_(160.0, Line2D([161.0 / W], [ky / H], marker='o', ms=1.8, mfc='#111111', mec='none', ls='none'), 'Landslides', dx=3.0)
# ------------------------------------------------------------- row 2: profiles under each map
pb = 49.5; ph = ky - 9.0 - pb
for k, (ev, key, name, when, col) in enumerate(EVS):
    x0 = 0.5 + k * (mw_ + gap) + 12.0
    ax = ax_mm(x0, pb, mw_ - 13.5 - (1.6 if k == 3 else 0), ph)
    fig.text((x0 - 12.3) / W, (pb + ph + 4.0) / H, 'efgh'[k], fontsize=9, fontweight='bold', va='top')
    for run, ls, lw, al in (('ctl', '-', 1.1, 1.0), ('pgw585_s050', '-', 0.8, 0.7), ('pgw585', '-', 1.1, 0.4)):
        d = PR[(PR.event == key) & (PR.run == run) & (PR.precip > 5) & (PR.ncell >= 20)].sort_values('band_lo')
        ax.plot(d.rain_share * 100, d.band_lo + 50, ls=ls, lw=lw, color=col, alpha=al, solid_capstyle='round')
    for run, cc in (('ctl', BLUE), ('pgw585', RED)):
        if (ev, run) in rsl: ax.plot([50], [rsl[(ev, run)]], marker='_', ms=5, mew=1.0, color=cc, ls='none', zorder=4)
    ax.plot([50, 50], [0, 2150], color='#BDBDBD', lw=0.5, ls=(0, (2, 1.5)), zorder=0)
    ax.set_xlim(0, 100); ax.set_xticks([0, 50, 100]); ax.set_ylim(0, 3200); ax.set_yticks([0, 1000, 2000, 3000]); ax.set_yticklabels(['0', '1,000', '2,000', '3,000'])
    ax.set_xlabel('Rain share (%)', labelpad=1.5)
    if k == 0: ax.set_ylabel('Elevation (m)')
    ax.spines[['top', 'right']].set_visible(False)
    if k == 0:
        ax.legend(handles=[Line2D([], [], color='#555555', lw=1.1, label='Control run'), Line2D([], [], color='#555555', lw=0.8, alpha=0.7, label='Half the change'), Line2D([], [], color='#555555', lw=1.1, alpha=0.4, label='Full change')],
                  loc='upper right', bbox_to_anchor=(1.08, 1.04), frameon=False, fontsize=5.6, handlelength=1.4, borderaxespad=0.1, labelspacing=0.3)
# ------------------------------------------------------------- row 3: rain line against warming
rb = 7.5; rh = pb - 13.0 - rb
axi = ax_mm(17.0, rb, 38.0, rh)
SL = {}
for ev, key, name, when, col in EVS:
    g = L[L.event == key][['run', 'd02_dT2m_land']].merge(RSL[RSL.event == key][['run', 'rsl_m']], on='run').sort_values('d02_dT2m_land'); g = g[g.rsl_m.notna()]
    axi.plot(g.d02_dT2m_land, g.rsl_m, '-o', color=col, lw=1.0, ms=2.8, mfc=col, mec='white', mew=0.4, zorder=3)
    if len(g) > 1:
        b = np.polyfit(g.d02_dT2m_land, g.rsl_m, 1)[0]; SL[ev] = b
        dx, dy, ha = {'ORE': (-1, 9, 'right'), 'LIG': (2, 7, 'right'), 'BC': (2, 7, 'right'), 'NOR': (2, 7, 'right')}[ev]
        axi.annotate(f'{b:.0f} m per °C', (g.d02_dT2m_land.iloc[-1], g.rsl_m.iloc[-1]), xytext=(dx, dy), textcoords='offset points', fontsize=5.6, color=col, va='center', ha=ha)
print('slopes', {k: round(v) for k, v in SL.items()})
axi.set_xlim(-0.15, 4.3); axi.set_xticks([0, 1, 2, 3, 4]); axi.set_ylim(500, 3000); axi.set_yticks([1000, 2000, 3000]); axi.set_yticklabels(['1,000', '2,000', '3,000'])
axi.set_xlabel('Warming (°C)'); axi.set_ylabel('Rain–snow\nline (m)', labelpad=2); axi.spines[['top', 'right']].set_visible(False)
fig.text(0.3 / W, (rb + rh + 2.5) / H, 'i', fontsize=9, fontweight='bold', va='top')

# ------------------------------------------------------------- row 3: hourly series in the storm core (j, k)
HR = pd.read_csv(f'{DATA}/proc/cluster/WRF_HOURLY.csv'); HR['t'] = pd.to_datetime(HR.time, format='%Y-%m-%dT%H')
AMB = '#F2A63B'
def hourly(x0, wdt, key, name, col, letter, ylab):
    c = HR[(HR.event == key) & (HR.run == 'ctl')].sort_values('t'); w = HR[(HR.event == key) & (HR.run == 'pgw585')].sort_values('t')
    m = c.merge(w, on='t', suffixes=('_c', '_w')).sort_values('t'); t0 = m.t.iloc[0].normalize(); h = (m.t - t0).dt.total_seconds() / 3600
    hh = 12.0; g = 2.0
    at = ax_mm(x0, rb + hh + g, wdt, rh - hh - g - 1.0); ab = ax_mm(x0, rb, wdt, hh)
    at.fill_between(h, 0, m.rain_mmh_c, color=AMB, lw=0, zorder=1); at.fill_between(h, m.rain_mmh_c, m.rain_mmh_c + m.snow_mmh_c, color=SNOWB, lw=0, zorder=1)
    at.plot(h, m.rain_mmh_w, color=RED, lw=0.9, zorder=3)
    ab.axhline(0, color='#999999', lw=0.5, ls=(0, (2, 1.5)), zorder=0)
    ab.plot(h, m.t2_c_c, color=BLUE, lw=0.9, zorder=3); ab.plot(h, m.t2_c_w, color=RED, lw=0.9, zorder=3)
    days = pd.date_range(t0, m.t.iloc[-1].normalize(), freq='1D'); hd = [(d - t0).total_seconds() / 3600 for d in days]
    for a in (at, ab):
        a.set_xlim(h.iloc[0], h.iloc[-1]); a.spines[['top', 'right']].set_visible(False); a.set_xticks(hd)
    at.set_xticklabels([]); at.tick_params(axis='x', length=1.5)
    ab.set_xticklabels([d.strftime('%-d %b') if i == 0 else d.strftime('%-d') for i, d in enumerate(days)]); ab.tick_params(axis='x', labelsize=5.6)
    ymax = float(np.nanmax(np.r_[m.rain_mmh_c + m.snow_mmh_c, m.rain_mmh_w])) * 1.3; at.set_ylim(0, ymax)
    tmin = float(np.nanmin(np.r_[m.t2_c_c, m.t2_c_w])); tmax = float(np.nanmax(np.r_[m.t2_c_c, m.t2_c_w])); ab.set_ylim(np.floor(tmin) - 0.5, np.ceil(tmax) + 0.5)
    at.set_ylabel('mm per h', labelpad=2); ab.set_ylabel('°C', labelpad=2)
    fig.text((x0 - 13.0) / W, (rb + rh + 2.5) / H, letter, fontsize=9, fontweight='bold', va='top')
    fig.text((x0 + 1.0) / W, (rb + rh + 2.2) / H, name, fontsize=6.6, color=col, va='top')
    return at, ab
atj, abj = hourly(78.0, 45.0, 'BC_2021', 'British Columbia', EVS[1][4], 'j', True)
atk, abk = hourly(137.0, 44.0, 'NOR_2011', 'Western Norway', EVS[0][4], 'k', False)
atj.legend(handles=[plt.Rectangle((0, 0), 1, 1, facecolor=AMB, lw=0, label='Rain, control'), plt.Rectangle((0, 0), 1, 1, facecolor=SNOWB, lw=0, label='Snow, control'), Line2D([], [], color=RED, lw=0.9, label='Rain, warmer')],
           loc='upper left', bbox_to_anchor=(0.0, 1.02), frameon=False, fontsize=5.4, handlelength=1.0, handleheight=0.8, borderaxespad=0.1, labelspacing=0.2, handletextpad=0.4)
abj.legend(handles=[Line2D([], [], color=BLUE, lw=0.9, label='Control'), Line2D([], [], color=RED, lw=0.9, label='Warmer')], loc='lower left', ncol=2, frameon=False, fontsize=5.4, handlelength=1.0, borderaxespad=0.1, columnspacing=0.8, handletextpad=0.4)
bump_fonts(fig)
print(overlap_check(fig)); print(rsl)
fig.savefig(f'{FIG}/Fig4.png', dpi=300); fig.savefig(f'{FIG}/Fig4.pdf'); print('saved')
