"""NC Fig. 1 (condensed from CEE Figs 1 and 2). Atmospheric rivers (ARs) and the size of landslide events.
a Regional landslide events (>=20 landslides) by driver, over the AR share of extreme-rain days (ERA5, 1979-2022).
b Zonal profile aligned with the map. c AR share of events by size, per inventory. d Size distribution of AR and other events.
e Driver composition by event size, extratropics and tropics."""
import os as _os
REPO = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
DATA = _os.path.join(REPO, 'data'); FIG = _os.environ.get('FIGOUT', _os.path.join(REPO, 'fig')); CODE = _os.path.join(REPO, 'fig_code')
import sys; sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from arcommon import *
import cartopy.crs as ccrs, cartopy.feature as cfeature
from matplotlib.colors import ListedColormap, BoundaryNorm
PC, ROB = ccrs.PlateCarree(), ccrs.Robinson()

B = events_all(); RG = regional(B)
# events ending by 2022 (ERA5 window), as in the chance table
EV = pd.read_parquet(f'{O}/events_fp.parquet'); EV['arm'] = EV.ar_frac >= 0.5; EV['lat'] = EV.ci - 70
EVp = EV[pool_mask(EV)]  # pooled statistics: GLC/GFLD events inside the national inventories' countries left out
z = np.load(f'{O}/cell_ex_ar.npz'); nex, nar = z['nex'], z['nar']
lsm = np.load(f'{DATA}/derived/a2/lsm.npy') > 0.5
share = np.where(lsm & (nex > 0), nar / np.maximum(nex, 1) * 100, np.nan)
lat1 = np.arange(-70, 76); lon1 = np.arange(360)
freq = np.load(f'{DATA}/inputs/arcat/clim_daily.npz')['p1'].mean(0) * 100

W, H = 183, 116
fig = plt.figure(figsize=(W * MM, H * MM))
def ax_mm(l, b, w, h, **k): return fig.add_axes([l / W, b / H, w / W, h / H], **k)
LATS, LATN = -58, 75
xr_, yr_ = 2 * 1.70058333e7, 7.70743810e6 + 6.13884072e6
mw = 128; mh = mw * yr_ / xr_; mb = H - 1.5 - mh
axm = ax_mm(1, mb, mw, mh, projection=ROB)
axm.set_extent([-180, 180, LATS, LATN], PC)
axm.add_feature(cfeature.LAND.with_scale('110m'), facecolor=LAND_C, edgecolor='none', zorder=0)
bounds = [0, 20, 40, 60, 80, 100]
cmap = ListedColormap(RAMP5); norm = BoundaryNorm(bounds, cmap.N)
axm.pcolormesh(lon1 - 0.5, lat1 - 0.5, share, cmap=cmap, norm=norm, transform=PC, zorder=0.2, rasterized=True, shading='auto')
hillshade(axm, dark=0.7, light=0.5)
axm.add_feature(cfeature.COASTLINE.with_scale('110m'), lw=0.3, edgecolor='#5A5A5A', zorder=2)
for yy in (25, -25): axm.plot([-180, 180], [yy, yy], color='#4D4D4D', lw=0.45, ls=(0, (3, 2)), transform=PC, zorder=2.5)
axm.text(178, 27, '25° N', fontsize=5.4, ha='right', va='bottom', color='#4D4D4D', transform=PC, zorder=6, path_effects=HALO)
axm.text(178, -27, '25° S', fontsize=5.4, ha='right', va='top', color='#4D4D4D', transform=PC, zorder=6, path_effects=HALO)
axm.spines['geo'].set_linewidth(0.5)
def msize(n): return 3 + 8 * np.log10(np.asarray(n) / 10) ** 1.7
st = dict(AR=dict(c=NAVY, ec='white', lw=0.5), TC=dict(c=TC_C, ec='white', lw=0.5), other=dict(c=OT_C, ec='white', lw=0.5))
for d in ('other', 'TC', 'AR'):
    q = RG[RG.drv == d]
    axm.scatter(q.lon, q.lat, s=msize(q.n), c=st[d]['c'], edgecolor=st[d]['ec'], linewidth=st[d]['lw'], transform=PC, zorder=4 if d == 'AR' else 3)
ANN = [('BC_2021-11-15', 'British Columbia\n2021', -150, 30, 'center'), ('Emilia-Romagna_May16', 'Emilia-Romagna 2023', -28, 33, 'center'),
       ('Brazil_Teresopolis', 'Teresópolis\n2011', -14, -12, 'left'), ('Typhoon RS', 'Rio Grande\ndo Sul 2024', -26, -34, 'left'),
       ('Taiwan_Morakot', 'Morakot\n2009', 162, 18, 'center'), ('Japan_Hiroshima', 'Hiroshima\n2018', 162, 45, 'center'), ('Dominica_Maria', 'Maria 2017', -42, 24, 'left')]
for key, txt, tx, ty, ha in ANN:
    if key == 'Typhoon RS': r = RG[(RG.src == 'COOLR_ev') & (RG.lat.between(-30, -28)) & (RG.date.dt.year == 2024)].iloc[0]
    else: r = RG[RG.name == key].iloc[0]
    axm.plot([r.lon, tx], [r.lat, ty], color='#4D4D4D', lw=0.35, transform=PC, zorder=3.5)
    axm.text(tx + (1.5 if ha == 'left' else 0), ty, txt, fontsize=5.6, ha=ha, va='center', linespacing=0.95, color='#262626', transform=PC, zorder=6, path_effects=HALO)
lx = -165
for k, (d, lab) in enumerate((('AR', 'Atmospheric river'), ('TC', 'Tropical cyclone'), ('other', 'Other storm'))):
    axm.scatter(lx, -8 - 7 * k, s=16, c=st[d]['c'], edgecolor='white', linewidth=0.4, transform=PC, zorder=5)
    axm.text(lx + 5, -8 - 7 * k, lab, va='center', fontsize=6.5, transform=PC, zorder=5)
for k, (n, t) in enumerate(((20, '20'), (200, '200'), (2000, '2k'), (20000, '20k'))):
    x = -163 + 17 * k
    axm.scatter(x, -38, s=msize(n), facecolor='none', edgecolor='k', linewidth=0.4, transform=PC, zorder=5)
    axm.text(x, -46, t, ha='center', va='center', fontsize=6, transform=PC, zorder=5)
axm.text(-166, -30.5, 'Landslides per event', fontsize=6.5, va='center', transform=PC)
_p = ROB.transform_points(PC, np.array([36.0, 104.0]), np.array([-48.0, -48.0]))
_x0, _x1 = axm.get_xlim(); _y0, _y1 = axm.get_ylim()
fx = (_p[:, 0] - _x0) / (_x1 - _x0); fy = (_p[0, 1] - _y0) / (_y1 - _y0)
cax = ax_mm(1 + mw * fx[0], mb + mh * fy, mw * (fx[1] - fx[0]), 1.6)
cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation='horizontal', ticks=[0, 20, 40, 60, 80, 100])
cb.outline.set_linewidth(0.3); cax.tick_params(length=1.5, labelsize=6, pad=1)
cax.set_title('Extreme-precipitation\ndays under an AR (%)', fontsize=6.3, pad=2, linespacing=1.0, x=0.57)

# b: zonal profile aligned with the Robinson y axis
def ry(lat): return ROB.transform_points(PC, np.zeros_like(np.asarray(lat, float)), np.asarray(lat, float))[:, 1]
axz = ax_mm(1 + mw + 8.5, mb, 39, mh)
y0, y1 = axm.get_ylim(); axz.set_ylim(y0, y1)
zs = np.array([nar[i][lsm[i]].sum() / max(nex[i][lsm[i]].sum(), 1) * 100 if lsm[i].sum() >= 3 else np.nan for i in range(146)])
zs_s = pd.Series(zs).rolling(5, center=True, min_periods=3).mean().values
zf = np.array([np.nanmean(freq[i][lsm[i]]) if lsm[i].sum() >= 3 else np.nan for i in range(146)])
zf_s = pd.Series(zf).rolling(5, center=True, min_periods=3).mean().values
m = (lat1 >= LATS) & (lat1 <= LATN)
FILL = '#BFD6DB'
axz.fill_betweenx(ry(lat1[m]), 0, zs_s[m], color=FILL, lw=0, zorder=1)
axz.plot(zf_s[m], ry(lat1[m]), color='k', lw=0.7, ls=(0, (2.5, 1.5)), zorder=2)
for lo in np.arange(-50, 70, 10):
    hi = lo + 10; yc = ry([lo + 5])[0]
    s1 = EVp[(EVp.n == 1) & (EVp.lat >= lo) & (EVp.lat < hi)]
    if len(s1) >= 20: axz.plot(s1.arm.mean() * 100, yc, 'o', ms=3.4, mfc='white', mec='#555555', mew=0.7, zorder=3)
    q = RG[(RG.lat >= lo) & (RG.lat < hi)]
    if len(q) >= 3:
        k = (q.drv == 'AR').sum(); l, u = wilson(k, len(q))
        axz.plot([l, u], [yc, yc], color=NAVY, lw=0.8, zorder=3)
        axz.plot(k / len(q) * 100, yc, 'o', ms=4.2, color=NAVY, mec='white', mew=0.4, zorder=4)

        axz.text(105.5, yc, f'{len(q)}', fontsize=5.5, va='center', color='#555555', clip_on=False)
yt = [-40, -20, 0, 20, 40, 60]
axz.set_yticks(ry(yt)); axz.set_yticklabels(['40° S', '20° S', '0°', '20° N', '40° N', '60° N'])
axz.set_xlim(0, 100); axz.set_xticks([0, 50, 100]); axz.set_xlabel('Under an AR (%)')
axz.spines[['top', 'right']].set_visible(False); axz.tick_params(axis='y', labelsize=6)
axz.text(105.5, ry([71])[0], 'n', fontsize=5.5, va='center', color='#555555', clip_on=False)
kr = ry([12, 3, -6, -15])
axz.plot(63, kr[0], 'o', ms=4.2, color=NAVY, mec='white', mew=0.4); axz.text(67, kr[0], 'Events ≥20', fontsize=5.8, va='center')
axz.plot(63, kr[1], 'o', ms=3.4, mfc='white', mec='#555555', mew=0.7); axz.text(67, kr[1], 'Single slides', fontsize=5.8, va='center')
axz.fill_between([60, 66], kr[2] - 1.9e5, kr[2] + 1.9e5, color=FILL, lw=0); axz.text(67, kr[2], 'Extreme days', fontsize=5.8, va='center')
axz.plot([60, 66], [kr[3], kr[3]], color='k', lw=0.7, ls=(0, (2.5, 1.5))); axz.text(67, kr[3], 'All days', fontsize=5.8, va='center')

cats = ['1', '2–4', '5–19', '≥20']
EV['szc'] = pd.cut(EV.n, [0, 1, 4, 19, 1e9], labels=cats); EVp = EV[pool_mask(EV)]
EV['date'] = D0 + pd.to_timedelta(EV.d0, 'D'); EV['month'] = EV.date.dt.month; EV['cold'] = cold_season(EV.lat, EV.month)

# ------------------------------------------------ row 2
rb, rh = 13.0, 33.0
# f: pooled share by size, against the chance AR share under the same rules (events moved to other years; v6_perm.py)
axc = ax_mm(11, rb, 43, rh)
g = EVp.groupby('szc', observed=True).arm.agg(['sum', 'size'])
ci = np.array([wilson(a, b) for a, b in zip(g['sum'], g['size'])]); pv = (g['sum'] / g['size'] * 100).values
CH = pd.read_csv(f'{DATA}/derived/v6/chance_inventory.csv'); CH = CH[CH.group == 'pooled'].set_index('size').reindex(['1', '2-4', '5-19', '>=20'])
axc.fill_between(range(4), CH.chance.values, pv, color=NAVY, alpha=0.10, lw=0)
axc.fill_between(range(4), ci[:, 0], ci[:, 1], color=NAVY, alpha=0.22, lw=0)
axc.fill_between(range(4), CH.chance_lo.values, CH.chance_hi.values, color='#8C8C8C', alpha=0.30, lw=0)
axc.plot(range(4), CH.chance.values, color='#4D4D4D', lw=0.9, ls=(0, (2.5, 1.5)))
axc.text(3.25, 15, 'Chance: same cells, dates\nand duration, other years', ha='right', va='top', fontsize=5.6, color='#333333', linespacing=1.0)
for k_, (s_, (lab, col)) in enumerate(INV.items()):
    q = EV[EV.src == s_]; t = q.groupby('szc', observed=True).arm.agg(['mean', 'size']); t = t[t['size'] >= 3]
    xx = np.array([cats.index(c) for c in t.index]) + (k_ - 2) * 0.07
    axc.scatter(xx, t['mean'] * 100, s=7, color=col, edgecolor='white', linewidth=0.3, zorder=3)
axc.plot(range(4), pv, '-o', color=NAVY, lw=1.6, ms=3.6, mec='white', mew=0.5, zorder=4)
axc.text(-0.1, pv[0], f'{pv[0]:.0f}%', ha='right', va='center', fontsize=6, color=NAVY)
axc.text(2.88, pv[3] + 2, f'{pv[3]:.0f}%', ha='right', va='bottom', fontsize=6, color=NAVY)
axc.text(3.27, 47, 'Excess over\nchance', ha='right', va='center', fontsize=5.6, color=NAVY, linespacing=1.0)
axc.set_xticks(range(4)); axc.set_xticklabels(cats); axc.set_xlim(-0.45, 3.3); axc.set_ylim(0, 105)
axc.set_xlabel('Landslides per event'); axc.set_ylabel('Events under an AR (%)')
axc.spines[['top', 'right']].set_visible(False)
from matplotlib.lines import Line2D
axc.legend(handles=[Line2D([], [], marker='o', ls='none', ms=2.6, color=c_, mec='white', mew=0.3, label=l_) for l_, c_ in INV.values()] +
           [Line2D([], [], marker='o', color=NAVY, lw=1.4, ms=3, mec='white', mew=0.4, label='All')],
           loc='upper left', ncol=2, frameon=False, fontsize=5.4, handlelength=1.0, columnspacing=0.6, borderaxespad=0.1, labelspacing=0.15, handletextpad=0.3)

# h: AR share of inventory events by size (inventory rule, with the chance share) poleward and equatorward of 25 degrees,
# and the storms behind the regional events of all sources (stacked bar at the right)
lab5 = cats
CHh = pd.read_csv(f'{DATA}/derived/v6/chance_inventory.csv')
RGall = B[B.n >= 20]
COLS = {'AR': NAVY, 'TC': TC_C, 'other': '#CFCFCF'}
for gi, (nm, grp, q, r) in enumerate((('Poleward of 25°', 'poleward of 25', EVp[EVp.lat.abs() >= 25], RGall[RGall.lat.abs() >= 25]),
                                       ('Within 25°', 'within 25', EVp[EVp.lat.abs() < 25], RGall[RGall.lat.abs() < 25]))):
    axh = ax_mm(66 + gi * 29.0, rb, 25.5, rh)
    g = q.groupby('szc', observed=True).arm.agg(['sum', 'size']).reindex(cats)
    pv = (g['sum'] / g['size'] * 100).values; ci = np.array([wilson(a, b) for a, b in zip(g['sum'], g['size'])])
    ch = CHh[CHh.group == grp].set_index('size').reindex(['1', '2-4', '5-19', '>=20'])
    xs = np.arange(4)
    axh.fill_between(xs, ch.chance_lo.values, ch.chance_hi.values, color='#8C8C8C', alpha=0.30, lw=0)
    axh.plot(xs, ch.chance.values, color='#4D4D4D', lw=0.8, ls=(0, (2.5, 1.5)))
    axh.fill_between(xs, ci[:, 0] * 100 if ci.max() <= 1 else ci[:, 0], ci[:, 1] * 100 if ci.max() <= 1 else ci[:, 1], color=NAVY, alpha=0.2, lw=0)
    axh.plot(xs, pv, '-o', color=NAVY, lw=1.4, ms=3.2, mec='white', mew=0.5, zorder=4)
    if gi == 0: axh.text(0.15, pv[0] - 2.5, f'{pv[0]:.0f}%', ha='left', va='top', fontsize=5.6, color=NAVY)
    else: axh.text(-0.42, pv[0] + 3, f'{pv[0]:.0f}%', ha='left', va='bottom', fontsize=5.6, color=NAVY)
    axh.text(2.8, pv[3] + 3, f'{pv[3]:.0f}%', ha='right', va='bottom', fontsize=5.6, color=NAVY)
    t = r.drv.value_counts(normalize=True).reindex(['AR', 'TC', 'other']).fillna(0) * 100; bot = 0
    for d in ('AR', 'TC', 'other'):
        axh.bar(4.3, t[d], bottom=bot, width=0.82, color=COLS[d], lw=0); bot += t[d]
    axh.text(4.3, t['AR'] / 2, f"AR\n{t['AR']:.0f}%", ha='center', va='center', fontsize=5.2, color='white', linespacing=0.95)
    if t['TC'] >= 12: axh.text(4.3, t['AR'] + t['TC'] / 2, f"TC\n{t['TC']:.0f}%", ha='center', va='center', fontsize=5.2, color='white', linespacing=0.95)
    axh.text(4.3, t['AR'] + t['TC'] + t['other'] / 2, 'All\nsources', ha='center', va='center', fontsize=4.9, color='#333333', linespacing=0.95)
    nn = list(g['size'].values) + [len(r)]
    fmt = lambda v: f'{v:,}' if v < 1000 else f'{v / 1000:.1f}k'
    axh.set_xlim(-0.45, 4.75); axh.set_ylim(0, 100); axh.set_xticks(list(xs) + [4.3])
    axh.set_xticklabels([f'{l_}\n{fmt(nn[j])}' for j, l_ in enumerate(lab5)] + [f'≥20\n{fmt(nn[4])}'], fontsize=5.2, linespacing=1.1)
    axh.set_title(nm, fontsize=6, pad=2)
    if gi == 0: axh.set_ylabel('Events under an AR (%)')
    else: axh.set_yticklabels([])
    axh.spines[['top', 'right']].set_visible(False)
    if gi == 0:
        axh.text(-0.05, 74, 'Inventory\nevents', ha='left', va='center', fontsize=5.4, color=NAVY, linespacing=0.95)
        axh.text(1.9, 12, 'Chance', ha='center', va='center', fontsize=5.2, color='#4D4D4D')
fig.text((66 + 27.0) / W, 1.4 / H, 'Landslides per event', fontsize=7.3, ha='center', va='bottom')


# e: cold vs warm season poleward of 25 degrees, per inventory (events with >=5 landslides) and all regional events
axs = ax_mm(134.5, rb, 44.0, rh - 1.5)
rows = []
for s_, (lab, col) in [(k_, v_) for k_, v_ in INV.items() if k_ != 'GFLD']:
    qq = EV[(EV.src == s_) & (EV.lat.abs() >= 25) & (EV.n >= 5)]
    rows.append((lab, [qq[qq.cold == c].arm.mean() * 100 for c in (True, False)], [qq[qq.cold == c].pc.mean() * 100 for c in (True, False)]))
rr = RG[RG.lat.abs() >= 25]
rows.append(('All regional events (≥20)', [(rr[rr.cold == c].drv == 'AR').mean() * 100 for c in (True, False)], [np.nan, np.nan]))
SC_ = {True: '#1F5A8C', False: '#E08A1E'}
yy = len(rows) * 3.0
BARNUM = []
for lab, v, pc in rows:
    axs.text(0, yy + 0.12, lab, ha='left', va='center', fontsize=6.0, color='#262626', fontweight='bold' if lab.startswith('All') else None)
    for k_, cold in enumerate((True, False)):
        y_ = yy - 1.0 - k_ * 0.95
        axs.barh(y_, 100, height=0.78, color='#F0F0F0', lw=0)
        axs.barh(y_, v[k_], height=0.78, color=SC_[cold], lw=0)
        BARNUM.append(axs.text(v[k_] - 1.5, y_ + 0.02, f'{v[k_]:.0f}', ha='right', va='center', fontsize=5.0, color='white'))
        if not np.isnan(pc[k_]): axs.plot([pc[k_]] * 2, [y_ - 0.36, y_ + 0.36], color='k', lw=0.9, solid_capstyle='butt')
    yy -= 3.0
axs.set_xlim(0, 100); axs.set_ylim(0.2, len(rows) * 3.0 + 0.6); axs.set_yticks([]); axs.set_xticks([0, 50, 100])
axs.spines[['top', 'right', 'left']].set_visible(False); axs.set_xlabel('Events under an AR (%)')
from matplotlib.patches import Patch
axs.legend(handles=[Patch(color=SC_[True], label='Cold season'), Patch(color=SC_[False], label='Warm season'), Line2D([], [], color='k', lw=0.9, marker='|', ls='none', ms=5, label='AR frequency, all days')],
           loc='lower left', bbox_to_anchor=(-0.02, 0.98), ncol=2, frameon=False, fontsize=5.8, handlelength=0.9, columnspacing=0.6, borderaxespad=0.2, handletextpad=0.3)
for r_ in rows: print('season', r_[0], np.round(r_[1]), np.round(r_[2]))

def L(s, x, y): fig.text(x / W, y / H, s, fontsize=9, fontweight='bold', va='top')
L('a', 0.3, H - 0.3); L('b', 1 + mw + 1.0, H - 0.3)
yl = rb + rh + 7.0
L('c', 0.3, yl); L('d', 57.5, yl); L('e', 125.5, yl)
if __name__ == '__main__':
    bump_fonts(fig)
    for t_ in BARNUM: t_.set_fontsize(5.2)   # numbers inside the bars stay smaller than the bar height
    print(overlap_check(fig))
    out = f'{FIG}/Fig1'
    fig.savefig(out + '.png', dpi=300); fig.savefig(out + '.pdf'); print('saved')
