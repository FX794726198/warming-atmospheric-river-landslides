"""Fig. 2 A season switch in the storms behind landslide events.
a,b Events with >=5 landslides in the cold (a) and warm (b) season, coloured by storm and sized by landslides; insets: storms behind regional
events (>=20) poleward and equatorward of 25 degrees. c Regional events by month and latitude. d Storms behind events with >=5 landslides
north of 25 N by month, with the AR frequency. e AR share of events in the cold and warm seasons poleward of 25 degrees, per inventory."""
import os as _os
REPO = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
DATA = _os.path.join(REPO, 'data'); FIG = _os.environ.get('FIGOUT', _os.path.join(REPO, 'fig')); CODE = _os.path.join(REPO, 'fig_code')
from arcommon import *
import cartopy.crs as ccrs, cartopy.feature as cfeature
from matplotlib.colors import ListedColormap, BoundaryNorm
zm = np.load(f'{O}/cell_ex_ar_month.npz'); NEX, NAR = zm['nex'].astype(float), zm['nar'].astype(float)
lsm = np.load(f'{DATA}/derived/a2/lsm.npy') > 0.5
lat1 = np.arange(-70, 76); lon1 = np.arange(360)
NH = (lat1 >= 0)[:, None]
cmon = np.array([[m in (10, 11, 12, 1, 2, 3) for m in range(1, 13)]]).T  # NH cold months
def season_share(cold):
    selN = cmon[:, 0] if cold else ~cmon[:, 0]; selS = ~selN
    ne = np.where(NH, NEX[selN].sum(0), NEX[selS].sum(0)); na = np.where(NH, NAR[selN].sum(0), NAR[selS].sum(0))
    return np.where(lsm & (ne >= 8), na / np.maximum(ne, 1) * 100, np.nan)
bounds = [0, 20, 40, 60, 80, 100]
cmapR = ListedColormap(RAMP5); normR = BoundaryNorm(bounds, cmapR.N)
PC, ROB = ccrs.PlateCarree(), ccrs.Robinson()
B = events_all(); RG = regional(B)
EV = pd.read_parquet(f'{O}/events.parquet'); EV['arm'] = EV.ar_frac >= 0.5; EV['lat'] = EV.ci - 70
EV['date'] = D0 + pd.to_timedelta(EV.d0, 'D'); EV['month'] = EV.date.dt.month; EV['cold'] = cold_season(EV.lat, EV.month)
B5 = B[B.n >= 5]
COLS = {'AR': NAVY, 'TC': TC_C, 'other': OT_C}

W, H = 183, 124
fig = plt.figure(figsize=(W * MM, H * MM))
def ax_mm(l, b, w, h, **k): return fig.add_axes([l / W, b / H, w / W, h / H], **k)
LATS, LATN = -58, 75
xr_, yr_ = 2 * 1.70058333e7, 7.70743810e6 + 6.13884072e6
mw = 90.5; mh = mw * yr_ / xr_; mb = H - 4.0 - mh
def msize(n): return 1.2 + 6.5 * np.log10(np.asarray(n) / 2.5) ** 1.6
st = dict(AR=dict(c=NAVY, ec='white', lw=0.3), TC=dict(c=TC_C, ec='white', lw=0.3), other=dict(c=OT_C, ec='white', lw=0.3))
for k, (cold, title) in enumerate(((True, 'Cold season'), (False, 'Warm season'))):
    ax = ax_mm(1 + k * (mw + 1), mb, mw, mh, projection=ROB); ax.set_extent([-180, 180, LATS, LATN], PC)
    ax.add_feature(cfeature.LAND.with_scale('110m'), facecolor=LAND_C, edgecolor='none', zorder=0)
    ax.pcolormesh(lon1 - 0.5, lat1 - 0.5, season_share(cold), cmap=cmapR, norm=normR, transform=PC, zorder=0.2, rasterized=True, shading='auto')
    hillshade(ax, dark=0.6, light=0.45)
    ax.add_feature(cfeature.COASTLINE.with_scale('110m'), lw=0.25, edgecolor='#5A5A5A', zorder=1)
    ax.spines['geo'].set_linewidth(0.5)
    q = B5[B5.cold == cold].sort_values('n', ascending=False)
    for d in ('other', 'TC', 'AR'):
        m = q.drv == d
        ax.scatter(q.lon[m], q.lat[m], s=msize(q.n[m]), c=st[d]['c'], edgecolor=st[d]['ec'], linewidth=st[d]['lw'], transform=PC, zorder=3 if d == 'AR' else 2)
    fig.text((1 + k * (mw + 1) + 6) / W, (mb + mh + 0.6) / H, title + (' (Oct–Mar north, Apr–Sep south)' if cold else ' (Apr–Sep north, Oct–Mar south)'),
             fontsize=6.5, va='bottom')
    # inset: storms behind regional events in this season (South Pacific, clear of land)
    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
    def ll2mm(lo_, la_):
        p_ = ROB.transform_point(lo_, la_, PC); return (1 + k * (mw + 1) + mw * (p_[0] - x0) / (x1 - x0), mb + mh * (p_[1] - y0) / (y1 - y0))
    xa, yb_ = ll2mm(-147, -48); xb, _ = ll2mm(-101, -48)
    axi = ax_mm(xa, yb_, xb - xa, 6.4)
    r = RG[RG.cold == cold]
    for j, (nm, g) in enumerate((('≥25°', r[r.lat.abs() >= 25]), ('<25°', r[r.lat.abs() < 25]))):
        sh = g.drv.value_counts(normalize=True).reindex(['AR', 'TC', 'other']).fillna(0) * 100; left = 0
        for d in ('AR', 'TC', 'other'):
            axi.barh(1 - j, sh[d], left=left, height=0.66, color=COLS[d] if d != 'other' else '#CFCFCF', lw=0); left += sh[d]
        if sh['AR'] >= 30: axi.text(sh['AR'] / 2, 1 - j, f"{sh['AR']:.0f}%", ha='center', va='center', fontsize=5.6, color='white')
        axi.text(-4, 1 - j, nm, ha='right', va='center', fontsize=5.6)
        axi.text(104, 1 - j, f'{len(g)}', ha='left', va='center', fontsize=5.6, color='#555555')
    axi.set_xlim(0, 100); axi.set_ylim(-0.5, 1.5); axi.axis('off')
    axi.text(-30, 1.8, 'Events ≥20 (n)', fontsize=5.6, va='bottom')
# shared legend strip under the maps
ly = mb - 3.2
def lmark(x, d, lab):
    fig.add_artist(matplotlib.lines.Line2D([x / W], [ly / H], marker='o', ms=4, mfc=COLS[d], mec='white', mew=0.3, ls='none'))
    fig.text((x + 1.8) / W, ly / H, lab, fontsize=6, va='center')
lmark(20, 'AR', 'Atmospheric river'); lmark(48, 'TC', 'Tropical cyclone'); lmark(76, 'other', 'Other storm')
fig.text(104 / W, ly / H, 'Landslides per event', fontsize=6, va='center')
for j, n in enumerate((5, 50, 500, 5000)):
    x = 131 + j * 12
    fig.add_artist(matplotlib.lines.Line2D([x / W], [ly / H], marker='o', ms=np.sqrt(msize(n)), mfc='none', mec='k', mew=0.4, ls='none'))
    fig.text((x + 2.2) / W, ly / H, f'{n:,}', fontsize=5.8, va='center')

# ------------------------------------------------ row 2
rb, rh = 9.5, mb - 8.5 - 9.5 - 5
# c: month x latitude, zonal land share of extreme-rain days under an AR, with the regional events
axa = ax_mm(10, rb, 56, rh)
NE_l = np.array([[NEX[m, i_][lsm[i_]].sum() for m in range(12)] for i_ in range(146)])
NA_l = np.array([[NAR[m, i_][lsm[i_]].sum() for m in range(12)] for i_ in range(146)])
k5 = np.ones(5)
NE_s = np.apply_along_axis(lambda v: np.convolve(v, k5, 'same'), 0, NE_l); NA_s = np.apply_along_axis(lambda v: np.convolve(v, k5, 'same'), 0, NA_l)
NE_s = (np.roll(NE_s, 1, 1) + 2 * NE_s + np.roll(NE_s, -1, 1)); NA_s = (np.roll(NA_s, 1, 1) + 2 * NA_s + np.roll(NA_s, -1, 1))
hov = np.where(NE_s >= 10, NA_s / np.maximum(NE_s, 1) * 100, np.nan)
hxm = np.arange(0, 14) + 0.5; hv = np.c_[hov[:, -1:], hov, hov[:, :1]]
axa.contourf(hxm, lat1, hv, levels=bounds, cmap=cmapR, norm=normR, zorder=0, extend='neither')
for yy in (-25, 25): axa.axhline(yy, color='#7F7F7F', lw=0.5, ls=(0, (2, 1.5)), zorder=1)
x = RG.month + (RG.date.dt.day - 1) / 31
for d in ('other', 'TC', 'AR'):
    m = RG.drv == d
    axa.scatter(x[m], RG.lat[m], s=msize(RG.n[m]) * 0.8, c=st[d]['c'], edgecolor='white', linewidth=0.35, zorder=3 if d == 'AR' else 2)
axa.set_xlim(1, 13); axa.set_ylim(-45, 75)
axa.set_xticks(np.arange(1.5, 13)); axa.set_xticklabels(list('JFMAMJJASOND'))
axa.set_yticks([-25, 0, 25, 50, 75]); axa.set_yticklabels(['25° S', '0°', '25° N', '50° N', '75° N'])
axa.set_xlabel('Month'); axa.spines[['top', 'right']].set_visible(False)
# shared colour scale of a-c, above c
cax = ax_mm(36.5, rb + rh + 2.4, 29.5, 1.5)
cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=normR, cmap=cmapR), cax=cax, orientation='horizontal', ticks=bounds)
cb.outline.set_linewidth(0.3); cax.tick_params(length=1.2, labelsize=5.6, pad=0.6); cax.xaxis.set_ticks_position('top')
fig.text(35.0 / W, (rb + rh + 3.15) / H, 'Extreme-rain days\nunder an AR (%), a–c', fontsize=5.8, ha='right', va='center', linespacing=1.0)

# d: clock of the storms behind events with >=5 landslides north of 25 N, with the AR frequency of the same cells and months
DS = 55.0
axd = ax_mm(103.5 - DS / 2, rb + rh / 2 - DS / 2 + 1.0, DS, DS, projection='polar')
q = B5[B5.lat >= 25]; mo = np.arange(1, 13)
t = q.groupby('month').drv.value_counts(normalize=True).unstack().reindex(index=mo, columns=['AR', 'TC', 'other']).fillna(0) * 100
nn = q.groupby('month').size().reindex(mo, fill_value=0)
th = (mo - 1) / 12 * 2 * np.pi; wd = 2 * np.pi / 12 * 0.9; R0, SC = 52, 0.52
axd.set_theta_zero_location('N'); axd.set_theta_direction(-1)
bot = np.full(12, R0, float)
for d in ('AR', 'TC', 'other'):
    axd.bar(th, t[d].values * SC, width=wd, bottom=bot, color=COLS[d] if d != 'other' else '#CFCFCF', edgecolor='white', lw=0.5); bot += t[d].values * SC
EVp = EV[pool_mask(EV)]; qe = EVp[(EVp.lat >= 25) & (EVp.n >= 5)]
pcm = qe.groupby('month').pc.mean().reindex(mo).values * 100
tt = np.linspace(0, 2 * np.pi, 361); pp = np.interp(tt, np.r_[th, 2 * np.pi], np.r_[pcm, pcm[0]]) * SC + R0
axd.plot(tt, pp, color='white', lw=1.8, zorder=3); axd.plot(tt, pp, color='k', lw=0.6, ls=(0, (2.2, 1.2)), zorder=4)
print('pcm', np.round(pcm))
for th_, v, f_, tc_ in zip(th, t['AR'].values, pcm, t['TC'].values):
    rg = R0 + f_ * SC; hi_ = R0 + v * SC
    if hi_ - rg >= 18: axd.text(th_, (rg + hi_) / 2, f'{v:.0f}', ha='center', va='center', fontsize=7.0, color='white')
    else: axd.text(th_, hi_ + tc_ * SC + 5.6, f'{v:.0f}', ha='center', va='center', fontsize=7.0, color=NAVY)
for th_, ml in zip(th, 'JFMAMJJASOND'): axd.text(th_, R0 + 100 * SC + 8, ml, ha='center', va='center', fontsize=6)
axd.set_ylim(0, R0 + 100 * SC + 3); axd.set_yticks([]); axd.set_xticks([]); axd.spines['polar'].set_visible(False); axd.grid(False)
axd.patch.set_alpha(0)
axd.text(0, 0, 'Events ≥5\nnorth of 25° N\n(% under an AR)', ha='center', va='center', fontsize=6.6, linespacing=1.05)
# key for d
kx_, ky_ = 80.0, rb - 6.3
fig.add_artist(matplotlib.lines.Line2D([kx_ / W, (kx_ + 4) / W], [ky_ / H] * 2, color='k', lw=0.6, ls=(0, (2.2, 1.2))))
fig.text((kx_ + 5) / W, ky_ / H, 'AR frequency, all days', fontsize=5.6, va='center')
fig.add_artist(plt.Rectangle(((kx_ + 31) / W, (ky_ - 1) / H), 2.6 / W, 2.0 / H, color='#CFCFCF', lw=0, transform=fig.transFigure))
fig.text((kx_ + 34.3) / W, ky_ / H, 'Other storms', fontsize=5.6, va='center')

# e: cold vs warm season per inventory (paired bars), with the AR frequency of the same cells and seasons
axc = ax_mm(141.5, rb, 39.0, rh + 1.0)
rows = []
for s_, (lab, col) in [(k_, v_) for k_, v_ in INV.items() if k_ != 'GFLD']:
    qq = EV[(EV.src == s_) & (EV.lat.abs() >= 25) & (EV.n >= 5)]
    rows.append((lab + ', ≥5 landslides', [qq[qq.cold == c].arm.mean() * 100 for c in (True, False)], [qq[qq.cold == c].pc.mean() * 100 for c in (True, False)]))
rr = RG[RG.lat.abs() >= 25]
rows.append(('All regional events, ≥20', [(rr[rr.cold == c].drv == 'AR').mean() * 100 for c in (True, False)], [np.nan, np.nan]))
SC_ = {True: '#1F5A8C', False: '#E08A1E'}
yy = len(rows) * 3.0
for lab, v, pc in rows:
    axc.text(0, yy, lab, ha='left', va='center', fontsize=6.2, color='#262626', fontweight='bold' if lab.startswith('All') else None)
    for k_, cold in enumerate((True, False)):
        y_ = yy - 1.0 - k_ * 0.95
        axc.barh(y_, 100, height=0.78, color='#F0F0F0', lw=0)
        axc.barh(y_, v[k_], height=0.78, color=SC_[cold], lw=0)
        axc.text(v[k_] - 2, y_, f'{v[k_]:.0f}', ha='right', va='center', fontsize=6.0, color='white')
        if not np.isnan(pc[k_]): axc.plot([pc[k_]] * 2, [y_ - 0.42, y_ + 0.42], color='k', lw=0.9, solid_capstyle='butt')
    yy -= 3.0
axc.set_xlim(0, 100); axc.set_ylim(0.2, len(rows) * 3.0 + 0.6); axc.set_yticks([]); axc.set_xticks([0, 50, 100])
axc.spines[['top', 'right', 'left']].set_visible(False); axc.set_xlabel('Events under an AR (%)')
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
axc.legend(handles=[Patch(color=SC_[True], label='Cold'), Patch(color=SC_[False], label='Warm'), Line2D([], [], color='k', lw=0.9, marker='|', ls='none', ms=5, label='AR frequency')],
           loc='lower left', bbox_to_anchor=(-0.02, 1.0), ncol=3, frameon=False, fontsize=6, handlelength=0.9, columnspacing=0.7, borderaxespad=0.2, handletextpad=0.35)

def L(s, x, y): fig.text(x / W, y / H, s, fontsize=9, fontweight='bold', va='top')
L('a', 0.3, H - 0.3); L('b', 1 + mw + 1.2, H - 0.3)
yl = rb + rh + 7.5
L('c', 0.3, yl); L('d', 70.5, yl); L('e', 133.5, yl)
bump_fonts(fig)
print(overlap_check(fig))
for r in rows: print(r[0], np.round(r[1]), np.round(r[2]))
print(t.round(0).assign(n=nn).to_string())
out = f'{FIG}/parts/Fig2'
fig.savefig(out + '.png', dpi=300); fig.savefig(out + '.pdf'); print('saved')
