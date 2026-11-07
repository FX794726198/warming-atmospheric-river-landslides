"""Fig. 1 Atmospheric rivers (ARs) and the size of landslide events.
a Regional landslide events (>=20 landslides) by driver, over the AR share of extreme-rain days (ERA5, 1979-2022).
b Zonal profile aligned with the map. c AR share of events by size, per inventory. d Size distribution of AR and other events.
e Driver composition by event size, extratropics and tropics."""
import os as _os
REPO = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
DATA = _os.path.join(REPO, 'data'); FIG = _os.environ.get('FIGOUT', _os.path.join(REPO, 'fig')); CODE = _os.path.join(REPO, 'fig_code')
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

W, H = 183, 173
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
cax.set_title('Extreme-rain days\nunder an AR (%)', fontsize=6.3, pad=2, linespacing=1.0)

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

        axz.text(103, yc, f'{len(q)}', fontsize=5.5, va='center', color='#555555', clip_on=False)
yt = [-40, -20, 0, 20, 40, 60]
axz.set_yticks(ry(yt)); axz.set_yticklabels(['40° S', '20° S', '0°', '20° N', '40° N', '60° N'])
axz.set_xlim(0, 100); axz.set_xticks([0, 50, 100]); axz.set_xlabel('Under an AR (%)')
axz.spines[['top', 'right']].set_visible(False); axz.tick_params(axis='y', labelsize=6)
axz.text(103, ry([71])[0], 'n', fontsize=5.5, va='center', color='#555555', clip_on=False)
kr = ry([12, 3, -6, -15])
axz.plot(63, kr[0], 'o', ms=4.2, color=NAVY, mec='white', mew=0.4); axz.text(67, kr[0], 'Events ≥20', fontsize=5.8, va='center')
axz.plot(63, kr[1], 'o', ms=3.4, mfc='white', mec='#555555', mew=0.7); axz.text(67, kr[1], 'Single slides', fontsize=5.8, va='center')
axz.fill_between([60, 66], kr[2] - 1.9e5, kr[2] + 1.9e5, color=FILL, lw=0); axz.text(67, kr[2], 'Extreme rain', fontsize=5.8, va='center')
axz.plot([60, 66], [kr[3], kr[3]], color='k', lw=0.7, ls=(0, (2.5, 1.5))); axz.text(67, kr[3], 'All days', fontsize=5.8, va='center')

# ------------------------------------------------ row 2: national inventories, events at their landslide centroids
EPT = pd.read_parquet(f'{DATA}/derived/s1/ev_points.parquet')
EPT['drv'] = np.where(EPT.drv == 'TC', 'TC', np.where(EPT.arm, 'AR', 'other'))  # same AR rule as the insets and panel f
cats = ['1', '2–4', '5–19', '≥20']
EV['szc'] = pd.cut(EV.n, [0, 1, 4, 19, 1e9], labels=cats); EVp = EV[pool_mask(EV)]
def msz(n): return 0.6 + 5.5 * np.log10(np.asarray(n, float)) ** 1.45
rb2, h2 = 62.0, 45.5
REG = [('USGS', 'Contiguous United States', ccrs.LambertConformal(central_longitude=-96, standard_parallels=(33, 45)), (-123.5, -68.5, 23.0, 50.0), 'll'),
       ('NVE', 'Norway', ccrs.TransverseMercator(central_longitude=15, central_latitude=64), (-1.5, 31.2, 57.7, 71.3), 'ul'),
       ('ITALICA', 'Italy', ccrs.TransverseMercator(central_longitude=12.5, central_latitude=41.5), (1.5, 20.0, 34.0, 47.3), 'll')]
x0 = 1.0; regax = []; ASP = []
for s_, nm, prj, ext, corner in REG:
    tmp = fig.add_axes([0, 0, 0.1, 0.1], projection=prj); tmp.set_extent(ext, PC); xl, yl_ = tmp.get_xlim(), tmp.get_ylim(); fig.delaxes(tmp)
    ASP.append((xl[1] - xl[0]) / (yl_[1] - yl_[0]))
h2 = (W - 1.2 - 2 * 3.2) / sum(ASP); print('h2', round(h2, 1))
for (s_, nm, prj, ext, corner), asp in zip(REG, ASP):
    w_ = h2 * asp
    ax = ax_mm(x0, rb2, w_, h2, projection=prj); ax.set_extent(ext, PC)
    ax.add_feature(cfeature.LAND.with_scale('50m'), facecolor=LAND_C, edgecolor='none', zorder=0)
    hillshade(ax, extent=(ext[0] - 3, ext[1] + 3, ext[2] - 2, ext[3] + 2))
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), lw=0.3, edgecolor='#5A5A5A', zorder=1)
    ax.add_feature(cfeature.BORDERS.with_scale('50m'), lw=0.3, edgecolor='#8C8C8C', zorder=1)
    if s_ == 'USGS': ax.add_feature(cfeature.NaturalEarthFeature('cultural', 'admin_1_states_provinces_lines', '50m'), facecolor='none', edgecolor='#B3B3B3', lw=0.2, zorder=1)
    ax.spines['geo'].set_linewidth(0.5)
    q = EPT[EPT.src == s_].sort_values('n')
    for d in ('other', 'TC', 'AR'):
        qq = q[q.drv == d]
        ax.scatter(qq.lon, qq.lat, s=msz(qq.n), c=st[d]['c'], edgecolor='white', linewidth=0.15 if d != 'AR' else 0.25, alpha=0.85 if d == 'other' else 0.95,
                   transform=PC, zorder=3 if d != 'AR' else 4)
    big = q[q.n >= 20]
    ax.scatter(big.lon, big.lat, s=msz(big.n), facecolor='none', edgecolor='k', linewidth=0.3, transform=PC, zorder=5)
    fig.text((x0 + 3.6) / W, (rb2 + h2 + 0.9) / H, f'{nm} · {len(q):,} events', fontsize=7, va='bottom', color='#262626')
    # inset: AR share by event size against the AR frequency of the same cells and months
    iw, ih = 12.5, 9.6
    if s_ == 'USGS': ix, iy = x0 + w_ * 0.40, rb2 + 1.2
    elif corner == 'ul': ix, iy = x0 + 7.4, rb2 + h2 - ih - 5.0
    else: ix, iy = x0 + 7.4, rb2 + 1.2
    card = ax_mm(ix - 6.2, iy - 1.0, iw + 11.6, ih + 5.0); card.set_facecolor('white'); card.patch.set_alpha(0.94)
    card.set_xticks([]); card.set_yticks([]); [sp.set_visible(False) for sp in card.spines.values()]
    ai = ax_mm(ix, iy, iw, ih)
    qe = EV[EV.src == s_]; t = qe.groupby('szc', observed=True).arm.agg(['mean', 'size'])
    pc_ = qe.pc.mean() * 100
    for j, c in enumerate(cats):
        y_ = 3 - j
        ai.barh(y_, 100, height=0.58, color='#E6E6E6', lw=0)
        if c in t.index and t.loc[c, 'size'] >= 3:
            v = t.loc[c, 'mean'] * 100
            ai.barh(y_, v, height=0.58, color=NAVY, lw=0)
            ai.text(104, y_, f'{v:.0f}', ha='left', va='center', fontsize=5.8, color=NAVY)
        ai.text(-5, y_, c, ha='right', va='center', fontsize=5.8)
    ai.plot([pc_, pc_], [-0.42, 3.42], color=TC_C, lw=0.9)
    ai.set_xlim(0, 100); ai.set_ylim(-0.5, 3.5); ai.axis('off')
    ai.text(-48, 4.0, 'Under an AR (%)', ha='left', va='bottom', fontsize=5.8)
    regax.append((x0, w_)); x0 += w_ + 3.2
# legend strip for c-e, under the maps
ly2 = rb2 - 3.6; lx = 30
fig.text(lx / W, ly2 / H, 'c–e: landslides per event', fontsize=6, va='center', ha='right')
for k, (n, t_) in enumerate(((1, '1'), (10, '10'), (100, '100'))):
    xx_ = lx + 3 + k * 11
    fig.add_artist(matplotlib.lines.Line2D([xx_ / W], [ly2 / H], marker='o', ls='none', ms=np.sqrt(msz(n)), mfc='none', mec='#333333', mew=0.5))
    fig.text((xx_ + 2.6) / W, ly2 / H, t_, fontsize=6, va='center')
xx_ = lx + 36
fig.add_artist(matplotlib.lines.Line2D([xx_ / W], [ly2 / H], marker='o', ls='none', ms=np.sqrt(msz(40)), mfc='#BDBDBD', mec='k', mew=0.7))
fig.text((xx_ + 2.8) / W, ly2 / H, '≥20 landslides (ringed)', fontsize=6, va='center')
fig.text((lx + 76) / W, ly2 / H, 'Colours as in a', fontsize=6, va='center', color='#333333')
fig.add_artist(matplotlib.lines.Line2D([(lx + 98) / W] * 2, [(ly2 - 1.2) / H, (ly2 + 1.2) / H], color=TC_C, lw=0.9))
fig.text((lx + 100) / W, ly2 / H, 'AR frequency, all days', fontsize=6, va='center', color='#333333')

# ------------------------------------------------ row 3
rb, rh = 13.0, 33.0
# f: pooled share by size, against the chance AR share under the same rules (events moved to other years; v6_perm.py)
axc = ax_mm(11, rb, 44, rh)
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
axc.text(2.62, 44, 'Excess over\nchance', ha='center', va='center', fontsize=5.6, color=NAVY, linespacing=1.0)
axc.set_xticks(range(4)); axc.set_xticklabels(cats); axc.set_xlim(-0.45, 3.3); axc.set_ylim(0, 105)
axc.set_xlabel('Landslides per event'); axc.set_ylabel('Events under an AR (%)')
axc.spines[['top', 'right']].set_visible(False)
from matplotlib.lines import Line2D
axc.legend(handles=[Line2D([], [], marker='o', ls='none', ms=2.6, color=c_, mec='white', mew=0.3, label=l_) for l_, c_ in INV.values()] +
           [Line2D([], [], marker='o', color=NAVY, lw=1.4, ms=3, mec='white', mew=0.4, label='All')],
           loc='upper left', ncol=2, frameon=False, fontsize=5.4, handlelength=1.0, columnspacing=0.6, borderaxespad=0.1, labelspacing=0.15, handletextpad=0.3)

# g: exceedance curves of event size
axd = ax_mm(66, rb, 41, rh)
cur = {}
for arm, col, lab in ((True, NAVY, 'AR events'), (False, '#6E6E6E', 'Other events')):
    for s_ in INV:
        v = np.sort(EV[(EV.src == s_) & (EV.arm == arm)].n.values)
        x = np.unique(v); p = np.array([(v >= u).mean() for u in x])
        axd.plot(x, p, color=col, lw=0.4, alpha=0.3, drawstyle='steps-post')
    v = np.sort(EVp[EVp.arm == arm].n.values); x = np.unique(v); p = np.array([(v >= u).mean() for u in x])
    cur[arm] = (x, p); axd.plot(x, p, color=col, lw=1.6, drawstyle='steps-post', label=lab, zorder=3)
xx = np.unique(np.concatenate([cur[True][0], cur[False][0]]))
pa = np.array([(EVp[EVp.arm].n >= u).mean() for u in xx]); po = np.array([(EVp[~EVp.arm].n >= u).mean() for u in xx])
axd.fill_between(xx, np.maximum(po, 1e-5), np.maximum(pa, 1e-5), step='post', color=NAVY, alpha=0.10, lw=0, zorder=1)
r5a = (EVp[EVp.arm].n >= 20).mean() * 100; r5o = (EVp[~EVp.arm].n >= 20).mean() * 100
axd.axvline(20, color='#999999', lw=0.5, ls=(0, (2, 1.5)), zorder=0)
axd.plot([20, 20], [r5a / 100, r5a / 100], 'o', color=NAVY, ms=3.4, mec='white', mew=0.5, zorder=5)
axd.plot([20], [r5o / 100], 'o', color='#6E6E6E', ms=3.4, mec='white', mew=0.5, zorder=5)
axd.text(24, r5a / 100 * 1.35, f'{r5a:.1f}%', fontsize=6, color=NAVY, va='bottom', path_effects=HALO)
axd.text(24, r5o / 100 * 0.62, f'{r5o:.1f}%', fontsize=6, color='#555555', va='top', path_effects=HALO)
axd.text(17, 1.5e-4, '20', fontsize=5.6, color='#777777', ha='right', va='bottom')
axd.set_xscale('log'); axd.set_yscale('log'); axd.set_xlim(1, 300); axd.set_ylim(1e-4, 1.3)
axd.set_xticks([1, 10, 100]); axd.set_xticklabels(['1', '10', '100'])
axd.set_yticks([1e-4, 1e-3, 1e-2, 1e-1, 1]); axd.set_yticklabels(['0.01', '0.1', '1', '10', '100'])
axd.set_xlabel('Landslides per event, n'); axd.set_ylabel('Events with ≥ n (%)')
axd.legend(loc='upper right', frameon=False, fontsize=6, handlelength=1.4, borderaxespad=0.1)
axd.spines[['top', 'right']].set_visible(False)

# h: AR share of inventory events by size (inventory rule, with the chance share) poleward and equatorward of 25 degrees,
# and the storms behind the regional events of all sources (stacked bar at the right)
lab5 = cats
CHh = pd.read_csv(f'{DATA}/derived/v6/chance_inventory.csv')
RGall = B[B.n >= 20]
COLS = {'AR': NAVY, 'TC': TC_C, 'other': '#CFCFCF'}
for gi, (nm, grp, q, r) in enumerate((('Poleward of 25°', 'poleward of 25', EVp[EVp.lat.abs() >= 25], RGall[RGall.lat.abs() >= 25]),
                                       ('Within 25°', 'within 25', EVp[EVp.lat.abs() < 25], RGall[RGall.lat.abs() < 25]))):
    axh = ax_mm(119 + gi * 30.5, rb, 26.5, rh)
    g = q.groupby('szc', observed=True).arm.agg(['sum', 'size']).reindex(cats)
    pv = (g['sum'] / g['size'] * 100).values; ci = np.array([wilson(a, b) for a, b in zip(g['sum'], g['size'])])
    ch = CHh[CHh.group == grp].set_index('size').reindex(['1', '2-4', '5-19', '>=20'])
    xs = np.arange(4)
    axh.fill_between(xs, ch.chance_lo.values, ch.chance_hi.values, color='#8C8C8C', alpha=0.30, lw=0)
    axh.plot(xs, ch.chance.values, color='#4D4D4D', lw=0.8, ls=(0, (2.5, 1.5)))
    axh.fill_between(xs, ci[:, 0] * 100 if ci.max() <= 1 else ci[:, 0], ci[:, 1] * 100 if ci.max() <= 1 else ci[:, 1], color=NAVY, alpha=0.2, lw=0)
    axh.plot(xs, pv, '-o', color=NAVY, lw=1.4, ms=3.2, mec='white', mew=0.5, zorder=4)
    axh.text(-0.42, pv[0] + 3, f'{pv[0]:.0f}%', ha='left', va='bottom', fontsize=5.6, color=NAVY)
    axh.text(3.0, pv[3] + 4, f'{pv[3]:.0f}%', ha='center', va='bottom', fontsize=5.6, color=NAVY)
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
fig.text((119 + 28.5) / W, 1.4 / H, 'Landslides per event', fontsize=7.3, ha='center', va='bottom')

def L(s, x, y): fig.text(x / W, y / H, s, fontsize=9, fontweight='bold', va='top')
L('a', 0.3, H - 0.3); L('b', 1 + mw + 1.0, H - 0.3)
yl2 = rb2 + h2 + 5.2
for k, (xx0, ww) in enumerate(regax): L('cde'[k], xx0 - 0.7 if k else 0.3, yl2)
yl = rb + rh + 6.0
L('f', 0.3, yl); L('g', 57.5, yl); L('h', 111.0, yl)
if __name__ == '__main__':
    bump_fonts(fig)
    print(overlap_check(fig))
    out = f'{FIG}/parts/Fig1'
    fig.savefig(out + '.png', dpi=300); fig.savefig(out + '.pdf'); print('saved')
