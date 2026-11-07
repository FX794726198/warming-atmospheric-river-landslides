"""NC Fig. 3 Freezing restrains AR landslide events, and the brake has begun to loosen.
a Change in the snow-dominated share of cold-season AR extreme precipitation over mountains (>=300 m), 1995-2024 minus 1941-1970 (ERA5 0.25 deg, 1 deg cells).
b Landslide chance of a cold-season AR extreme-rain episode by 2-day mean T2m, relative to 6 C (pooled, cluster-bootstrap band; inventories thin).
c Odds ratios of a landslide by temperature class (cell fixed effects) and per inventory (below vs above 0 C).
d Snow-dominated share of AR extreme cell-days, 11-year running mean, per region.
e Share of mountain land where most AR extreme precipitation falls as snow, 1941-1970 and 1995-2024."""
import os as _os
REPO = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
DATA = _os.path.join(REPO, 'data'); FIG = _os.environ.get('FIGOUT', _os.path.join(REPO, 'fig')); CODE = _os.path.join(REPO, 'fig_code')
import sys; sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from arcommon import *
import cartopy.crs as ccrs, cartopy.feature as cfeature
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.lines import Line2D
PC, ROB = ccrs.PlateCarree(), ccrs.Robinson()
N7 = f'{DATA}/derived/n7'; N3 = f'{DATA}/derived/n3'
M = pd.read_csv(f'{N7}/map_change_1deg.csv'); M['diff'] = M.s2 - M.s1
import os
S = pd.read_csv(f'{N7}/regional_series.csv' if os.path.exists(f'{N7}/regional_series.csv') else f'{N7}/regional_series6.csv')
PA = pd.read_csv(f'{N7}/protected_area_base7922.csv'); PA['frac'] = PA.area_snow_dominated_km2 / PA.area_assessed_km2 * 100
BC_ = pd.read_csv(f'{N3}/brake_curve_boot.csv'); BI = pd.read_csv(f'{N3}/brake_curve.csv'); FE = pd.read_csv(f'{DATA}/proc/rev/brake_robust.csv'); FE = FE[FE.model == '+ month FE + duration + year'].set_index('term').loc[['p0_4', 'm4_0', 'lt_m4']].reset_index()

REG = {  # code: (name, short label, box (lat0, lat1, lon0, lon1), colour)
    'SCA': ('Scandinavia', 'Scandinavia', (57, 71, 4, 20), '#0072B2'),
    'ALP': ('Alps', 'Alps', (43, 48, 5, 16), '#009E73'),
    'APE': ('Apennines', 'Apennines', (38, 45, 7.5, 18.5), '#66C2A5'),
    'CAU': ('Caucasus and Turkiye', 'Caucasus–Türkiye', (36, 44, 26, 48), '#CC79A7'),
    'BCA': ('British Columbia and Alaska coast', 'BC–Alaska coast', (48, 62, -140, -115), '#8C6BB1'),
    'PNW': ('US Pacific Northwest and Rockies', 'US Northwest', (40, 49, -125, -105), '#E69F00'),
    'NEA': ('Northeastern North America', 'NE North America', (40, 50, -80, -60), '#D55E00'),
    'HKH': ('Hindu Kush and Himalaya', 'Hindu Kush–Himalaya', (27, 38, 65, 95), '#7F7F7F'),
    'SAN': ('Southern Andes', 'Southern Andes', (-55, -33, -75, -66), '#56B4E9'),
    'NH': ('All land 25-72 N', 'All land 25–72° N', None, '#000000'),
    'SH': ('All land 25-56 S', 'All land 25–56° S', None, '#555555')}

W, H = 183, 110.5
fig = plt.figure(figsize=(W * MM, H * MM))
def ax_mm(l, b, w, h, **k): return fig.add_axes([l / W, b / H, w / W, h / H], **k)

# ---------------------------------------------------------------- a: map of change
LATS, LATN = -58, 75
xr_, yr_ = 2 * 1.70058333e7, 7.70743810e6 + 6.13884072e6
mw = 117; mh = mw * yr_ / xr_; mb = H - 1.5 - mh
axm = ax_mm(1, mb, mw, mh, projection=ROB)
axm.set_extent([-180, 180, LATS, LATN], PC)
axm.add_feature(cfeature.LAND.with_scale('110m'), facecolor=LAND_C, edgecolor='none', zorder=0)
bounds = [-25, -15, -10, -5, -2, 2, 5, 10, 15, 25]
cols = ['#B2182B', '#D6604D', '#F4A582', '#FDDBC7', '#F2F2F2', '#D1E5F0', '#92C5DE', '#4393C3', '#2166AC']
cmap = ListedColormap(cols); cmap.set_under('#67001F'); cmap.set_over('#053061'); norm = BoundaryNorm(bounds, cmap.N)
M['lon'] = (M.lon + 180) % 360 - 180; M['la2'] = np.floor(M.lat / 2) * 2 + 1; M['lo2'] = np.floor(M.lon / 2) * 2 + 1
M['w1'] = M.s1 * M.n1; M['w2'] = M.s2 * M.n2
M2 = M.groupby(['la2', 'lo2'])[['w1', 'w2', 'n1', 'n2']].sum().reset_index(); M2 = M2[(M2.n1 >= 40) & (M2.n2 >= 40)]
M2['d'] = M2.w2 / M2.n2 - M2.w1 / M2.n1
la2 = np.arange(-71, 77, 2); lo2 = np.arange(-179, 180, 2); G = np.full((len(la2), len(lo2)), np.nan)
G[((M2.la2 + 71) // 2).astype(int).values, ((M2.lo2 + 179) // 2).astype(int).values] = M2.d.values
axm.pcolormesh(lo2, la2, G, cmap=cmap, norm=norm, transform=PC, zorder=0.2, rasterized=True, shading='nearest')
print('map cells', len(M2), 'NH mean', np.average(M2[M2.la2 > 0].d, weights=np.cos(np.radians(M2[M2.la2 > 0].la2))).round(2), 'share<0', (M2[M2.la2 > 0].d < 0).mean().round(2))
hillshade(axm, dark=0.6, light=0.45)
axm.add_feature(cfeature.COASTLINE.with_scale('110m'), lw=0.3, edgecolor='#5A5A5A', zorder=2)
axm.spines['geo'].set_linewidth(0.5)
LBL = {'SCA': (2, 66, 'right', 'center'), 'ALP': (-6, 50, 'right', 'center'), 'CAU': (38, 33.5, 'center', 'top'),
       'BCA': (-128, 63.5, 'center', 'bottom'), 'PNW': (-127, 44.5, 'right', 'center'), 'NEA': (-58, 45, 'left', 'center'),
       'HKH': (80, 25.5, 'center', 'top'), 'SAN': (-64, -44, 'left', 'center')}
for k, (nm, sh, box, c) in REG.items():
    if box is None or k == 'APE': continue
    a0, a1, b0, b1 = box
    xs = np.r_[np.linspace(b0, b1, 20), np.full(20, b1), np.linspace(b1, b0, 20), np.full(20, b0)]
    ys = np.r_[np.full(20, a0), np.linspace(a0, a1, 20), np.full(20, a1), np.linspace(a1, a0, 20)]
    axm.plot(xs, ys, color=c, lw=0.9, transform=PC, zorder=3)
    tx, ty, ha, va = LBL[k]
    axm.text(tx, ty, sh.replace('–', '–\n') if k in ('HKH',) else sh, color=c, fontsize=5.6, ha=ha, va=va, transform=PC, zorder=6, path_effects=HALO, linespacing=0.95)
_p = ROB.transform_points(PC, np.array([-168.0, -86.0]), np.array([-42.0, -42.0]))
_x0, _x1 = axm.get_xlim(); _y0, _y1 = axm.get_ylim()
fx = (_p[:, 0] - _x0) / (_x1 - _x0); fy = (_p[0, 1] - _y0) / (_y1 - _y0)
cax = ax_mm(1 + mw * fx[0], mb + mh * fy, mw * (fx[1] - fx[0]), 1.6)
cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation='horizontal', extend='both', ticks=[-25, -10, 0, 10, 25], extendfrac=0.06)
cb.outline.set_linewidth(0.3); cax.tick_params(length=1.5, labelsize=6, pad=1)
cax.set_title('Change in snow-dominated\nAR extremes (%)', fontsize=6.2, pad=2, linespacing=1.0)

# ---------------------------------------------------------------- b: brake curve
axb = ax_mm(131, mb + 6.5, 49, mh - 7.5)
axb.axvspan(-12, 0, color='#DCE6F2', lw=0, zorder=0)
for s_, (lab, col) in INV.items():
    q = BI[BI.src == s_]
    axb.plot(q.t2, q.rel, color=col, lw=0.7, alpha=0.9, zorder=2)
axb.fill_between(BC_.t2, BC_.lo, BC_.hi, color=NAVY, alpha=0.18, lw=0, zorder=3)
axb.plot(BC_.t2, BC_.rel, color=NAVY, lw=1.8, zorder=4)
axb.axhline(1, color='#999999', lw=0.4, ls=(0, (2, 1.5)), zorder=1)
axb.set_xlim(-12, 12); axb.set_ylim(0, 1.25); axb.set_xticks([-10, -5, 0, 5, 10])
axb.set_xlabel('2-day mean T$_{2m}$ (°C)'); axb.set_ylabel('Landslide chance\n(relative to 6 °C)', linespacing=1.0)
axb.spines[['top', 'right']].set_visible(False)
axb.text(-11.4, 1.2, 'Below freezing', fontsize=5.8, color='#3B5B85', va='top')
v0 = BC_.loc[(BC_.t2 - 0).abs().idxmin()]; v4 = BC_.loc[(BC_.t2 + 4).abs().idxmin()]
for v_ in (v0, v4): axb.plot(v_.t2, v_.rel, 'o', color=NAVY, ms=3.2, mec='white', mew=0.5, zorder=5)
axb.text(-11.4, 0.96, f'{v0.rel:.2f} at 0 °C\n{v4.rel:.2f} at −4 °C', fontsize=5.6, color=NAVY, va='top', linespacing=1.1)
axb.legend(handles=[Line2D([], [], color=NAVY, lw=1.6, label='All, 95% band')] + [Line2D([], [], color=c, lw=0.8, label=l) for l, c in INV.values()],
           loc='lower right', frameon=False, fontsize=5.2, handlelength=1.0, labelspacing=0.12, borderaxespad=0, handletextpad=0.3)

# ---------------------------------------------------------------- c: odds ratios
rb, rh = 11.5, 35.5
axc = ax_mm(20, rb, 24, rh)
rows = [('0 to 4 °C', *FE.loc[0, ['OR', 'lo', 'hi']].values, NAVY), ('−4 to 0 °C', *FE.loc[1, ['OR', 'lo', 'hi']].values, NAVY),
        ('Below −4 °C', *FE.loc[2, ['OR', 'lo', 'hi']].values, NAVY)]
INVOR = [('NVE', 0.45, 0.28, 0.73), ('USGS', 0.55, 0.27, 1.13), ('ITALICA', 0.63, 0.24, 1.67), ('GLC', 0.52, 0.24, 1.15), ('GFLD', 0.27, 0.05, 1.54)]
rows += [(INV[s][0], o_, l_, h_, INV[s][1]) for s, o_, l_, h_ in INVOR]
ys = [9, 8, 7, 4.6, 3.6, 2.6, 1.6, 0.6]
for (lab, o_, l_, h_, c), y in zip(rows, ys):
    axc.plot([l_, h_], [y, y], color=c, lw=0.9); axc.plot(o_, y, 'o', color=c, ms=3.4, mec='white', mew=0.4)
    axc.text(0.032, y, lab, ha='right', va='center', fontsize=6, clip_on=False)
axc.axvline(1, color='#777777', lw=0.5, ls=(0, (2, 1.5)))
axc.set_xscale('log'); axc.set_xlim(0.04, 2.2); axc.set_xticks([0.05, 0.2, 0.5, 1, 2]); axc.set_xticklabels(['0.05', '0.2', '0.5', '1', '2']); axc.minorticks_off()
axc.set_ylim(0, 10.6); axc.set_yticks([])
axc.spines[['top', 'right', 'left']].set_visible(False)
axc.text(0.0035, 10.35, 'Same cell, vs ≥4 °C', fontsize=6, fontweight='bold', va='center', clip_on=False)
axc.text(0.0035, 5.8, 'Per inventory, <0 vs ≥0 °C', fontsize=6, fontweight='bold', va='center', clip_on=False)
axc.set_xlabel('Odds ratio of a landslide')

# ---------------------------------------------------------------- d: regional series
axd = ax_mm(56.0, rb, 38.5, rh)
SER = [k for k in ['SCA', 'BCA', 'PNW', 'ALP', 'NEA', 'SAN'] if k in set(S.region)]
end = {}
for k in SER:
    q = S[S.region == k].sort_values('year')
    y = q.share_s50.values * 100; r = pd.Series(y).rolling(11, center=True, min_periods=7).mean().values
    axd.plot(q.year, y, color=REG[k][3], lw=0.3, alpha=0.35)
    axd.plot(q.year, r, color=REG[k][3], lw=1.4)
    end[k] = r[np.isfinite(r)][-1]
lab_y = dict(end)
o_ = sorted(lab_y, key=lab_y.get)
for i in range(1, len(o_)):
    if lab_y[o_[i]] - lab_y[o_[i - 1]] < 5.5: lab_y[o_[i]] = lab_y[o_[i - 1]] + 5.5
axd.set_xlim(1940, 2025); axd.set_ylim(0, 100); axd.set_xticks([1950, 1970, 1990, 2010])
axd.set_ylabel('Snow-dominated AR\nextremes (%)', linespacing=1.0)
axd.spines[['top', 'right']].set_visible(False)

# ---------------------------------------------------------------- e: protected land
axe = ax_mm(118.5, rb, 27, rh)
ORD = ['NEA', 'ALP', 'CAU', 'SAN', 'PNW', 'SCA', 'HKH', 'BCA', 'NH', 'SH']
for i, k in enumerate(ORD[::-1]):
    q = PA[PA.region == REG[k][0]].set_index('period')
    a_, b_ = q.loc['1941-1970', 'frac'], q.loc['1995-2024', 'frac']
    c = REG[k][3]
    axe.plot([a_, b_], [i, i], color=c, lw=1.0, alpha=0.6)
    axe.plot(a_, i, 'o', mfc='white', mec=c, mew=0.8, ms=3.2)
    axe.plot(b_, i, 'o', color=c, mec='white', mew=0.3, ms=3.6)
    axe.text(-3, i, REG[k][1], ha='right', va='center', fontsize=5.6, color=c if k not in ('NH', 'SH') else '#262626', clip_on=False)
axe.set_xlim(0, 110); axe.set_ylim(-0.7, len(ORD) - 0.3); axe.set_yticks([]); axe.set_xticks([0, 40, 80])
axe.spines[['top', 'right', 'left']].set_visible(False)
axe.set_xlabel('Snow-dominated land (%)')
axe.legend(handles=[Line2D([], [], marker='o', ls='none', mfc='white', mec='k', mew=0.8, ms=3.2, label='1941–1970'),
                    Line2D([], [], marker='o', ls='none', color='k', ms=3.4, label='1995–2024')],
           loc='lower left', bbox_to_anchor=(-0.9, 0.98), ncol=2, frameon=False, fontsize=5.6, handletextpad=0.1, columnspacing=0.6, borderaxespad=0)

# ---------------------------------------------------------------- f: change by initial share
OB = pd.read_csv(f'{DATA}/proc/obs_bins.csv'); OB = OB[OB.hemi == 'NH'].reset_index(drop=True)
axf = ax_mm(155.0, rb, 24.0, rh)
EDGES = [0, 5, 20, 40, 60, 80, 95, 100]; mids = [(a + b) / 2 for a, b in zip(EDGES[:-1], EDGES[1:])]
axf.axhline(0, color='#777777', lw=0.5, ls=(0, (2, 1.5)), zorder=0)
for i, r in OB.iterrows():
    axf.plot([mids[i]] * 2, [r.lo, r.hi], color=NAVY, lw=0.9, zorder=2); axf.plot(mids[i], r.dmean, 'o', color=NAVY, ms=3.2, mec='white', mew=0.4, zorder=3)
axf.set_xticks([0, 20, 40, 60, 80, 100]); axf.set_xlim(0, 100); axf.set_ylim(-13, 4)
axf.set_xlabel('Snow share in\n1971–1994 (%)', linespacing=1.0)
axf.set_ylabel('Change in snow-\ndominated share (%)', linespacing=1.0)
axf.spines[['top', 'right']].set_visible(False)
axf.text(0.04, 0.04, 'Mountains\n25–72° N', transform=axf.transAxes, fontsize=5.4, va='bottom', color='#4D4D4D', linespacing=1.0)

def L(s, x, y): fig.text(x / W, y / H, s, fontsize=9, fontweight='bold', va='top')
L('a', 0.3, H - 0.3); L('b', 121.5, H - 0.3)
yl = rb + rh + 7.0
L('c', 0.3, yl); L('d', 46.5, yl); L('e', 96.5, yl); L('f', 146.0, yl)
if __name__ == '__main__':
    bump_fonts(fig)
    print(overlap_check(fig))
    out = f'{FIG}/Fig3'
    fig.savefig(out + '.png', dpi=300); fig.savefig(out + '.pdf'); print('saved')
