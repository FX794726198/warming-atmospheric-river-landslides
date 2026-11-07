"""Fig. 4 Where AR extreme precipitation sits near the snow-rain threshold.
a Share of AR extreme-precipitation cell-days (2-day precipitation >= local P99, AR on day d or d-1) with a 2-day mean
2-m air temperature of -4 to 0 degC (ERA5, 1979-2022, land cells with >= 20 AR extreme-precipitation cell-days); triangles mark the WRF cases.
b Temperature classes of AR extreme precipitation by latitude (land). c Share of cold-season landslide events (|lat| >= 25 deg)
with >= 5 landslides by the 2-day mean temperature at the event, AR and other events."""
import os as _os
REPO = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
DATA = _os.path.join(REPO, 'data'); FIG = _os.environ.get('FIGOUT', _os.path.join(REPO, 'fig')); CODE = _os.path.join(REPO, 'fig_code')
from arcommon import *
import cartopy.crs as ccrs, cartopy.feature as cfeature
from matplotlib.colors import ListedColormap, BoundaryNorm
PC, ROB = ccrs.PlateCarree(), ccrs.Robinson()
z = np.load(f'{O}/ar_ex_tclass.npz'); Hc = z['H'].astype(float); Hn = z['Hn'].astype(float)
lsm = np.load(f'{DATA}/derived/a2/lsm.npy') > 0.5
lat1 = np.arange(-70, 76); lon1 = np.arange(360)
ok = lsm & (Hn >= 20)
nf = np.where(ok, Hc[1] / np.maximum(Hn, 1) * 100, np.nan)
E = pd.read_parquet(f'{O}/events_t2.parquet'); E = E[pool_mask(E)]; E = E[E.cold & (E.lat.abs() >= 25)]

W, H = 183, 114
fig = plt.figure(figsize=(W * MM, H * MM))
def ax_mm(l, b, w, h, **k): return fig.add_axes([l / W, b / H, w / W, h / H], **k)
LATS, LATN = -58, 75
xr_, yr_ = 2 * 1.70058333e7, 7.70743810e6 + 6.13884072e6
mw = 134; mh = mw * yr_ / xr_; mb = H - 1.5 - mh
axm = ax_mm(1, mb, mw, mh, projection=ROB)
axm.set_extent([-180, 180, LATS, LATN], PC)
axm.add_feature(cfeature.LAND.with_scale('110m'), facecolor=LAND_C, edgecolor='none', zorder=0)
bounds = [0, 2, 5, 10, 15, 20, 30, 40]
cmap = ListedColormap(['#EEEBE1', '#DCD6E8', '#C2B3D8', '#A58CC6', '#8664B0', '#6A3D9A', '#4A1C73']); cmap.set_over('#2E0A4F')
norm = BoundaryNorm(bounds, cmap.N)
axm.pcolormesh(lon1 - 0.5, lat1 - 0.5, nf, cmap=cmap, norm=norm, transform=PC, zorder=0.2, rasterized=True, shading='auto')
hillshade(axm, dark=0.6, light=0.45)
axm.add_feature(cfeature.COASTLINE.with_scale('110m'), lw=0.3, edgecolor='#4D4D4D', zorder=2)
axm.spines['geo'].set_linewidth(0.5)
cases = {'BC': (-121.4, 49.6), 'ORE': (-123.9, 43.7), 'NOR': (6.5, 61.0), 'LIG': (8.7, 44.7), 'RSB': (-51.9, -29.1)}
CASEL = {'BC': ('British Columbia', -150, 58), 'ORE': ('Oregon', -150, 38), 'NOR': ('Norway', -18, 66), 'LIG': ('Liguria', -22, 40), 'RSB': ('Rio Grande do Sul', -28, -38)}
for k_, (x_, y_) in cases.items():
    axm.plot(x_, y_, '^', ms=4.6, mfc='#F2A63B', mec='k', mew=0.5, transform=PC, zorder=5)
    nm_, tx_, ty_ = CASEL[k_]
    axm.plot([x_, tx_], [y_, ty_], color='#4D4D4D', lw=0.35, transform=PC, zorder=4)
    axm.text(tx_, ty_, nm_, fontsize=5.6, ha='right' if tx_ < x_ else 'left', va='center', transform=PC, zorder=6, path_effects=HALO)
axm.plot(-163, -14, '^', ms=4.2, mfc='#F2A63B', mec='k', mew=0.5, transform=PC, zorder=5)
axm.text(-157, -14, 'WRF cases', fontsize=6, va='center', transform=PC, zorder=5)
_p = ROB.transform_points(PC, np.array([-160.0, -95.0]), np.array([-40.0, -40.0]))
_x0, _x1 = axm.get_xlim(); _y0, _y1 = axm.get_ylim()
fx = (_p[:, 0] - _x0) / (_x1 - _x0); fy = (_p[0, 1] - _y0) / (_y1 - _y0)
cax = ax_mm(1 + mw * fx[0], mb + mh * fy, mw * (fx[1] - fx[0]), 1.5)
cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation='horizontal', extend='max', ticks=[0, 5, 10, 20, 40])
cb.outline.set_linewidth(0.3); cax.tick_params(length=1.5, labelsize=5.5, pad=1)
cax.set_title('AR extreme precipitation\nat −4 to 0 °C (%)', fontsize=6, pad=2, linespacing=1.0)

# b: zonal stacked temperature classes, aligned with the map
def ry(lat): return ROB.transform_points(PC, np.zeros_like(np.asarray(lat, float)), np.asarray(lat, float))[:, 1]
axz = ax_mm(1 + mw + 10, mb, 34.5, mh)
y0, y1 = axm.get_ylim(); axz.set_ylim(y0, y1)
TCOL = ['#1F5A8C', '#8664B0', '#F2A63B', '#DDDDDD']; TLAB = ['< −4 °C', '−4 to 0 °C', '0 to 4 °C', '≥ 4 °C']
for b0 in range(-55, 75, 5):
    m = (lat1 >= b0) & (lat1 < b0 + 5)
    tot = Hn[m][lsm[m]].sum()
    if tot < 500: continue
    sh = Hc[:, m][:, lsm[m]].sum(1) / tot * 100
    yb, yt = ry([b0])[0], ry([b0 + 5])[0]; left = 0
    for c in range(4):
        axz.barh((yb + yt) / 2, sh[c], height=(yt - yb) * 0.86, left=left, color=TCOL[c], lw=0)
        left += sh[c]
yt_ = [-40, -20, 0, 20, 40, 60]
axz.set_yticks(ry(yt_)); axz.set_yticklabels(['40° S', '20° S', '0°', '20° N', '40° N', '60° N'])
axz.set_xlim(0, 100); axz.set_xticks([0, 50, 100]); axz.set_xlabel('Share (%)')
axz.spines[['top', 'right']].set_visible(False); axz.tick_params(axis='y', labelsize=6)

# ------------------------------------------------ row 2
from scipy.stats import gaussian_kde
EP = pd.read_parquet(f'{DATA}/derived/s1/episodes_extreme_cold.parquet', columns=['src', 't2', 'rain_rel', 'cold', 'hit'])
EPc = EP[EP.cold & (EP.rain_rel >= 1)].dropna(subset=['t2'])
rb, rh = 9.5, mb - 9.5 - 13.5
# c: ridgelines of temperature for AR episodes with extreme rain (grey) and those followed by landslides (colour)
axr = ax_mm(13, rb, 48, rh)
xs = np.linspace(-15, 25, 300); srcs = list(INV)
for i, s_ in enumerate(srcs):
    y0_ = len(srcs) - 1 - i; q = EPc[EPc.src == s_]; h = q[q.hit.astype(bool)]
    for d_, fc, ec, lw_, a_ in ((q.t2.values, '#D9D9D9', '#A6A6A6', 0.4, 1.0), (h.t2.values, INV[s_][1], INV[s_][1], 0.9, 0.35)):
        kd = gaussian_kde(np.clip(d_, -15, 25), bw_method=0.25)(xs); kd = kd / kd.max() * 0.9
        axr.fill_between(xs, y0_, y0_ + kd, color=fc, alpha=a_, lw=0, zorder=2 + (fc != '#D9D9D9'))
        axr.plot(xs, y0_ + kd, color=ec, lw=lw_, zorder=4)
    fa, fh = (q.t2 < 0).mean() * 100, (h.t2 < 0).mean() * 100
    axr.text(-14.5, y0_ + 0.66, INV[s_][0], fontsize=5.8, va='center', color=INV[s_][1])
    axr.text(-14.5, y0_ + 0.34, f'{fa:.0f}% → {fh:.0f}%', fontsize=5.4, va='center', ha='left', color='#333333', path_effects=HALO, zorder=6)
axr.plot([0, 0], [-0.05, len(srcs) - 0.05], color='#555555', lw=0.6, ls=(0, (2, 1.5)), zorder=5)
axr.set_xlim(-15, 25); axr.set_ylim(-0.05, len(srcs) + 0.32); axr.set_yticks([]); axr.spines[['top', 'right', 'left']].set_visible(False)
axr.set_xlabel('2-day mean T$_\\mathrm{2m}$ (°C)')
axr.text(-14.5, len(srcs) + 0.30, 'Below 0 °C: all → with landslides', fontsize=5.4, ha='left', va='top', color='#333333')
from matplotlib.patches import Patch
axr.legend(handles=[Patch(fc='#D9D9D9', ec='#A6A6A6', lw=0.4, label='All episodes'),
                    Patch(fc=INV['NVE'][1], ec=INV['NVE'][1], alpha=0.45, lw=0.6, label='With landslides')],
           loc='lower left', bbox_to_anchor=(-0.01, 1.0), ncol=2, frameon=False, fontsize=5.6, handlelength=1.1, handleheight=0.7,
           borderaxespad=0.1, columnspacing=1.0, handletextpad=0.4)
# d: share of AR episodes with extreme rain followed by a landslide, by temperature
axy = ax_mm(77, rb, 44, rh)
tbe = [-30, -4, -2, 0, 2, 4, 6, 8, 12, 30]; tmid = np.array([-6, -3, -1, 1, 3, 5, 7, 10, 15])
for s_, (nm, col) in INV.items():
    q = EPc[EPc.src == s_]; idx = np.digitize(q.t2, tbe) - 1
    kk = np.array([q.hit.values[idx == j].sum() for j in range(9)]); nn = np.array([(idx == j).sum() for j in range(9)])
    ok_ = (nn >= 30) & (kk > 0)
    axy.plot(tmid[ok_], kk[ok_] / nn[ok_] * 100, '-o', color=col, ms=2.4, lw=1.0, label=nm)
axy.axvline(0, color='#555555', lw=0.6, ls=(0, (2, 1.5)), zorder=0)
axy.set_yscale('log'); axy.set_ylim(0.15, 40); axy.set_yticks([0.2, 0.5, 1, 2, 5, 10, 20]); axy.set_yticklabels(['0.2', '0.5', '1', '2', '5', '10', '20'])
axy.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
axy.set_xlim(-7.5, 16.5); axy.set_xticks([-5, 0, 5, 10, 15])
axy.set_xlabel('2-day mean T$_\\mathrm{2m}$ (°C)'); axy.set_ylabel('Followed by a landslide (%)')
axy.spines[['top', 'right']].set_visible(False)
# e: share of events with >=5 landslides by event temperature
axc = ax_mm(137, rb, 44, rh)
tb = [-40, -2, 0, 2, 4, 6, 8, 12, 40]; xm = np.array([-3, -1, 1, 3, 5, 7, 10, 14])
for arm, col, lab in ((True, NAVY, 'AR events'), (False, '#7F7F7F', 'Other events')):
    q = E[E.arm == arm]; idx = np.digitize(q.t2, tb) - 1
    kk = np.array([(q.n.values[idx == j] >= 5).sum() for j in range(8)]); nn = np.array([(idx == j).sum() for j in range(8)])
    ci = np.array([wilson(a, b) for a, b in zip(kk, nn)])
    axc.fill_between(xm, ci[:, 0], ci[:, 1], color=col, alpha=0.18, lw=0)
    axc.plot(xm, kk / nn * 100, '-o', color=col, ms=2.6, lw=1.2, label=lab)
    print(lab, list(zip(xm, nn, np.round(kk / nn * 100, 1))))
axc.axvline(0, color='#555555', lw=0.6, ls=(0, (2, 1.5)), zorder=0)
axc.set_xlim(-4, 15); axc.set_xticks([0, 5, 10, 15]); axc.set_ylim(0, 25)
axc.set_xlabel('Event T$_\\mathrm{2m}$ (°C)'); axc.set_ylabel('Events with ≥5 landslides (%)')
axc.legend(loc='upper right', frameon=False, fontsize=6, handlelength=1.3, borderaxespad=0.1)
axc.spines[['top', 'right']].set_visible(False)

# key for b, below the map
kx = 40; ky = mb - 3.8
fig.text((kx - 2) / W, ky / H, 'AR extreme precipitation in b:', fontsize=6, va='center', ha='right')
for c in range(4):
    fig.add_artist(plt.Rectangle(((kx + c * 18.5) / W, (ky - 1.1) / H), 3.2 / W, 2.2 / H, color=TCOL[c], lw=0, transform=fig.transFigure))
    fig.text((kx + c * 18.5 + 4.2) / W, ky / H, TLAB[c], fontsize=6, va='center')

def L(s, x, y): fig.text(x / W, y / H, s, fontsize=9, fontweight='bold', va='top')
L('a', 0.3, H - 0.3); L('b', 1 + mw + 0.8, H - 0.3)
yl = rb + rh + 5.0
L('c', 0.3, yl); L('d', 64.5, yl); L('e', 124.5, yl)
out = f'{FIG}/parts/Fig4'
if __name__ == '__main__':
    bump_fonts(fig)
    print(overlap_check(fig))
    fig.savefig(out + '.png', dpi=300); fig.savefig(out + '.pdf'); print('saved')
