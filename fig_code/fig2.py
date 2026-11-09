"""Fig. 3 ARs make large landslide events by spreading extreme rain over large areas.
a-c Three cold-season AR events: integrated water vapour transport (IVT, ARCO ERA5, 0.25 deg) at the 6-h step nearest the landslide peak,
the AR object of the Chen et al. catalogue, event-total rain >= 100 mm and the landslides.
d Extreme-rain footprint of landslide events (land cells with 2-day rain >= local P99 within +-4 deg), AR vs other events.
e Share of events with >=5 landslides by footprint. f Relative probability of an event of >=5 landslides under an AR, before and after adjusting for the
footprint and the peak rain (logistic regression with month fixed effects)."""
import os as _os
REPO = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
DATA = _os.path.join(REPO, 'data'); FIG = _os.environ.get('FIGOUT', _os.path.join(REPO, 'fig')); CODE = _os.path.join(REPO, 'fig_code')
from arcommon import *
import cartopy.crs as ccrs, cartopy.feature as cfeature
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy import ndimage
PC = ccrs.PlateCarree()
lat1 = np.arange(-70, 76)
W, H = 183, 153
fig = plt.figure(figsize=(W * MM, H * MM))
def ax_mm(l, b, w, h, **k): return fig.add_axes([l / W, b / H, w / W, h / H], **k)

cases = [('ORE', '1996-11-18T18', (-150, -108, 28.5, 57), 'Oregon, USA · 18 Nov 1996'),
         ('NOR', '2011-12-26T12', (-19, 22, 44, 72), 'Western Norway · 26 Dec 2011'),
         ('LIG', '2014-11-15T12', (-8, 24, 30, 52), 'Liguria and Piedmont, Italy · 15 Nov 2014'),
         ('KZN', '2022-04-11T18', (10, 52, -44, -15.5), 'KwaZulu-Natal, South Africa · 11 Apr 2022'),
         ('KOR', '2023-07-15T00', (106, 148, 20, 48.5), 'Central South Korea · 15 Jul 2023'),
         ('RSB', '2024-04-30T06', (-76, -34, -40.5, -12), 'Rio Grande do Sul, Brazil · 30 Apr 2024')]
ivb = [250, 400, 550, 700, 850, 1000, 1200]
icm = ListedColormap(['#E4ECF5', '#C3D7EA', '#9BBEDF', '#6F9FD0', '#4E7FBF', '#2F5FA3']); icm.set_over('#1B3F7A'); inorm = BoundaryNorm(ivb, icm.N)
RAIN_C = '#E08A1E'; LSK = '#111111'
cw, ch, cg = 58.5, 40.0, 2.5; rows_b = [H - 5.0 - ch, H - 5.0 - ch - 5.6 - ch]; cb2 = rows_b[1]
for k_, (key, tt, ext, title) in enumerate(cases):
    k = k_ % 3; yb_ = rows_b[k_ // 3]
    ax = ax_mm(1 + k * (cw + cg), yb_, cw, ch, projection=PC)
    ax.set_extent(ext, PC)
    zc = np.load(f'{CS}/{key}.npz'); ti = list(zc['time']).index(tt)
    u, v = zc['u'][ti], zc['v'][ti]; iv = np.hypot(u, v); la, lo = zc['lat'], zc['lon']
    ax.add_feature(cfeature.LAND.with_scale('50m'), facecolor=LAND_C, edgecolor='none', zorder=0)
    hillshade(ax, extent=(ext[0] - 1, ext[1] + 1, ext[2] - 1, ext[3] + 1), dark=0.5, light=0.4, step=2, zorder=0.4)
    ax.pcolormesh(lo, la, np.where(iv >= 250, iv, np.nan), cmap=icm, norm=inorm, transform=PC, shading='auto', zorder=1, rasterized=True, alpha=0.82)
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), lw=0.35, edgecolor='#333333', zorder=2)
    ax.add_feature(cfeature.BORDERS.with_scale('50m'), lw=0.2, edgecolor='#777777', zorder=2)
    ax.contour(lo, la, ndimage.gaussian_filter(zc['pr'], 1.0), levels=[100], colors=RAIN_C, linewidths=1.0, transform=PC, zorder=4)
    arr = np.load(f'{DATA}/inputs/arcat/ar_case_masks.npz')[key][None]; s = 0
    p = pd.read_csv(f'{CS}/{key}_points.csv'); plon = p.longitude if 'longitude' in p else p.lon; plat = p.latitude if 'latitude' in p else p.lat
    lon360 = np.arange(360); lonp = np.where(lon360 > 180, lon360 - 360, lon360); o = np.argsort(lonp)
    lab_, _ = ndimage.label((np.asarray(arr[s]) > 0)[:, o], structure=np.ones((3, 3)))
    pi_ = np.clip(np.round(plat.values + 70).astype(int), 0, 145); pj_ = np.clip(np.round(plon.values).astype(int) + 179, 0, 359)
    keep = set()
    for di in range(-2, 3):
        for dj in range(-2, 3): keep |= set(lab_[np.clip(pi_ + di, 0, 145), np.clip(pj_ + dj, 0, 359)].tolist())
    keep.discard(0); msk = np.isin(lab_, list(keep)).astype(float)
    ax.contour(lonp[o], lat1, msk, levels=[0.5], colors='k', linewidths=0.8, transform=PC, zorder=3)
    sk = 10; uu = np.where(iv >= 250, u / np.maximum(iv, 1), np.nan); vv = np.where(iv >= 250, v / np.maximum(iv, 1), np.nan)
    ax.quiver(lo[::sk], la[::sk], uu[::sk, ::sk], vv[::sk, ::sk], transform=PC, color='white', scale=34, width=0.004, headwidth=3.5, headlength=3.5, zorder=3, alpha=0.85)
    if plon.std() < 0.5: ax.scatter(plon.median(), plat.median(), s=40, facecolor='none', edgecolor=LSK, linewidth=0.7, transform=PC, zorder=5)
    ax.scatter(plon, plat, s=3, c=LSK, edgecolor='none', transform=PC, zorder=5, alpha=0.9)
    ax.spines['geo'].set_linewidth(0.5)
    fig.text((1 + k * (cw + cg) + 4.2) / W, (yb_ + ch + 0.9) / H, title, fontsize=7, va='bottom', ha='left')
    fig.text((0.5 + k * (cw + cg)) / W, (yb_ + ch + 0.9) / H, 'abcdef'[k_], fontsize=9, fontweight='bold', va='bottom')
    ax.text(0.98, 0.03, f'{len(p):,} landslides', transform=ax.transAxes, fontsize=6.5, va='bottom', ha='right', color=LSK, path_effects=HALO, zorder=6)
cax2 = ax_mm(W / 2 - 8, cb2 - 4.6, 56, 1.6)
cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=inorm, cmap=icm), cax=cax2, orientation='horizontal', extend='max', ticks=ivb)
cb.outline.set_linewidth(0.3); cax2.tick_params(length=1.5, labelsize=6, pad=1)
cax2.text(1.06, 0.5, 'IVT (kg m$^{-1}$ s$^{-1}$)', transform=cax2.transAxes, fontsize=6.5, va='center')
kx = 1 + 2
fig.add_artist(matplotlib.lines.Line2D([(kx + 2) / W], [(cb2 - 3.8) / H], color=LSK, marker='o', ls='none', ms=2.4, mec='none'))
fig.text((kx + 5.5) / W, (cb2 - 3.8) / H, 'Landslides', fontsize=6.5, va='center')
kx = 23
fig.add_artist(matplotlib.lines.Line2D([kx / W, (kx + 4) / W], [(cb2 - 3.8) / H] * 2, color='k', lw=0.8))
fig.text((kx + 5.5) / W, (cb2 - 3.8) / H, 'AR outline', fontsize=6.5, va='center')
kx = 44
fig.add_artist(matplotlib.lines.Line2D([kx / W, (kx + 4) / W], [(cb2 - 3.8) / H] * 2, color=RAIN_C, lw=1.0))
fig.text((kx + 5.5) / W, (cb2 - 3.8) / H, 'Event precipitation ≥100 mm', fontsize=6.5, va='center')

# ------------------------------------------------ row 2
EF = pd.read_parquet(f'{DATA}/derived/v6/events_fp48.parquet'); EF['arm'] = EF.ar_frac >= 0.5; EF = EF[pool_mask(EF)]
rb, rh = 9.5, cb2 - 16 - 9.5
edges = [-1, 0.5, 10000, 30000, 60000, 120000, 1e12]; labs = ['0', '<10', '10–30', '30–60', '60–120', '>120']
# g: simultaneous footprint distribution, mirrored (AR events up, other events down), with each inventory as small markers
axd = ax_mm(11, rb, 50, rh)
for sign, (arm, col, lab) in zip((1, -1), ((True, NAVY, 'AR events'), (False, '#9A9A9A', 'Other events'))):
    qq = EF[EF.arm == arm]; idx = np.digitize(qq.fp_sim_km2, edges) - 1
    sh = np.array([(idx == j).mean() * 100 for j in range(6)])
    axd.bar(np.arange(6), sign * sh, width=0.72, color=col, lw=0, alpha=0.9)
    for j, v in enumerate(sh):
        if v > 30: axd.text(j - 0.08, sign * (v - 2.5), f'{v:.0f}', ha='center', va='top' if sign > 0 else 'bottom', fontsize=5.3, color='white')
        else: axd.text(j - 0.08, sign * (v + 1.5), f'{v:.0f}', ha='center', va='bottom' if sign > 0 else 'top', fontsize=5.3, color='#333333')
    for s_, (nm, c_) in INV.items():
        q2 = qq[qq.src == s_]; i2 = np.digitize(q2.fp_sim_km2, edges) - 1
        v2 = np.array([(i2 == j).mean() * 100 for j in range(6)])
        axd.scatter(np.arange(6) + 0.24, sign * v2, s=4, color=c_, edgecolor='white', linewidth=0.2, zorder=3)
    axd.text(5.45, sign * 48, f'{lab}\n(n = {len(qq):,})', ha='right', va='center', fontsize=6, color=col if arm else '#555555', linespacing=0.95)
axd.axhline(0, color='k', lw=0.5)
axd.set_ylim(-80, 75); axd.set_yticks([-60, -30, 0, 30, 60]); axd.set_yticklabels(['60', '30', '0', '30', '60'])
axd.set_xticks(range(6)); axd.set_xticklabels(labs, fontsize=6.5); axd.set_xlim(-0.5, 5.5); axd.set_ylabel('Share of events (%)')
axd.set_xlabel('Extreme-precipitation area (10³ km²)')
axd.spines[['top', 'right']].set_visible(False)
fa = (EF[EF.arm].fp_sim_km2 >= 30000).mean() * 100; fn = (EF[~EF.arm].fp_sim_km2 >= 30000).mean() * 100
axd.text(3.3, 70, f'≥30,000 km²: AR {fa:.0f}%, other {fn:.0f}%', ha='center', va='top', fontsize=6)
print('fp_sim >= 30k:', fa, fn, 'mean', EF[EF.arm].fp_sim_km2.mean(), EF[~EF.arm].fp_sim_km2.mean(), 'median', EF[EF.arm].fp_sim_km2.median(), EF[~EF.arm].fp_sim_km2.median())
# h: P(>=5 landslides | footprint) and P(>=20)
axe = ax_mm(73, rb, 48, rh)
for arm, col, lab in ((True, NAVY, 'AR events'), (False, '#7F7F7F', 'Other events')):
    qq = EF[EF.arm == arm]; idx = np.digitize(qq.fp_sim_km2, edges) - 1
    kk = np.array([(qq.n.values[idx == j] >= 5).sum() for j in range(6)]); nn = np.array([(idx == j).sum() for j in range(6)])
    ci = np.array([wilson(a, b) for a, b in zip(kk, nn)])
    axe.fill_between(range(6), ci[:, 0], ci[:, 1], color=col, alpha=0.18, lw=0)
    axe.plot(range(6), kk / nn * 100, '-o', color=col, ms=2.8, lw=1.2, label=lab + ', ≥5')
    k2 = np.array([(qq.n.values[idx == j] >= 20).sum() for j in range(6)])
    axe.plot(range(6), k2 / nn * 100, ':o', color=col, ms=2.2, lw=1.0, mfc='white', mew=0.7, label=lab + ', ≥20')
    print(lab, 'P>=5', np.round(kk / nn * 100, 1), 'P>=20', np.round(k2 / nn * 100, 1), nn)
axe.set_xticks(range(6)); axe.set_xticklabels(labs, fontsize=6.5); axe.set_xlim(-0.3, 5.3); axe.set_ylim(0, 36)
axe.set_xlabel('Extreme-precipitation area (10³ km²)'); axe.set_ylabel('Share of events (%)')
axe.legend(loc='upper left', frameon=False, fontsize=5.6, handlelength=1.6, borderaxespad=0.2, labelspacing=0.25, ncol=1)
axe.spines[['top', 'right']].set_visible(False)
# i: relative probabilities (v06 models, all events as AR vs all as other: cluster-robust by inventory and year; area and duration as logarithms)
R = pd.read_csv(f'{DATA}/proc/prob/ar_size_prob.csv')
axf = ax_mm(147, rb, 34, rh)
names = {'NVE': 'Norway', 'USGS': 'United States', 'ITALICA': 'Italy', 'GLC': 'Global (GLC)', 'GFLD': 'Global fatal', 'ALL': 'All events, ≥5', 'ALL20': 'All events, ≥20'}
models = [('raw', 'Unadjusted', NAVY), ('+duration', '+ duration', '#6BAED6'), ('+simultaneous footprint+duration', '+ area, duration', '#E08214'),
          ('+simultaneous footprint+peak+duration', '+ area, peak, duration', '#4D4D4D')]
rows_ = [('ALL20', 'big20', 'ALL'), ('ALL', 'big5', 'ALL'), ('GFLD', 'big5', 'GFLD'), ('GLC', 'big5', 'GLC'), ('ITALICA', 'big5', 'ITALICA'), ('USGS', 'big5', 'USGS'), ('NVE', 'big5', 'NVE')]
for i, (key_, out_, src_) in enumerate(rows_):
    for j, (mk, ml, col) in enumerate(models):
        r = R[(R.src == src_) & (R.outcome == out_) & (R.model == mk)].iloc[0]; y = i + 0.27 - j * 0.18
        axf.plot([max(r.lo, 0.3), min(r.hi, 9.5)], [y, y], color=col, lw=0.8)
        axf.plot(r.RR, y, 'o', color=col, ms=2.6 if not key_.startswith('ALL') else 3.2, zorder=3, label=ml if i == 0 else None)
    axf.text(0.27, i, names[key_], ha='right', va='center', fontsize=6, fontweight='bold' if key_.startswith('ALL') else None)
axf.axvline(1, color='#999999', lw=0.6, ls=(0, (2, 1.5)), zorder=0)
axf.set_xscale('log'); axf.set_xlim(0.3, 10); axf.set_xticks([0.5, 1, 2, 4, 8]); axf.set_xticklabels(['0.5', '1', '2', '4', '8'])
axf.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
axf.set_ylim(-0.55, len(rows_) - 0.45); axf.set_yticks([]); axf.spines[['top', 'right', 'left']].set_visible(False)
axf.set_xlabel('AR relative probability')
axf.legend(loc='lower left', bbox_to_anchor=(-0.62, 0.99), ncol=2, frameon=False, fontsize=5.6, handlelength=0.8, columnspacing=0.6, borderaxespad=0.1, handletextpad=0.3)

def L(s, x, y): fig.text(x / W, y / H, s, fontsize=9, fontweight='bold', va='top')
yl = rb + rh + 5.2
L('g', 0.3, yl); L('h', 62.5, yl); L('i', 121.5, yl)
bump_fonts(fig)
print(overlap_check(fig))
out = f'{FIG}/Fig2'
fig.savefig(out + '.png', dpi=300); fig.savefig(out + '.pdf'); print('saved')
