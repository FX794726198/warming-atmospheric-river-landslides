"""Supplementary figures S1-S4 of the AR-landslide CEE draft."""
import os as _os
REPO = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
DATA = _os.path.join(REPO, 'data'); FIG = _os.environ.get('FIGOUT', _os.path.join(REPO, 'fig')); CODE = _os.path.join(REPO, 'fig_code')
from arcommon import *
from scipy.stats import theilslopes
S = f'{DATA}/derived/s1'; OUT = f'{FIG}/parts'
W = 183
def newfig(h):
    f = plt.figure(figsize=(W * MM, h * MM)); return f, (lambda l, b, w, hh, **k: f.add_axes([l / W, b / h, w / W, hh / h], **k))
def L(f, h, s, x, y): f.text(x / W, y / h, s, fontsize=9, fontweight='bold', va='top')
SZ = ['1', '2–4', '5–19', '≥20']

# ---------------- Fig. S1 clustering rules
CL = pd.read_csv(f'{S}/cluster_rules.csv')
rules = ['±1 day, 3 × 3 cells (main)', 'same day, same cell', '±1 day, 5 × 5 cells', '±2 days, 3 × 3 cells']
H = 52; f, ax_mm = newfig(H)
a = ax_mm(10, 10, 36, H - 17)
RC = ['k', '#1B9E77', '#D95F02', '#7570B3']
for r, c in zip(rules, RC):
    q = CL[CL.rule == r]; g = q.groupby('size').apply(lambda d: (d.ar * d.n).sum() / d.n.sum()).reindex(SZ)
    a.plot(range(4), g.values, '-o', color=c, ms=2.6, lw=1.6 if 'main' in r else 1.0, label=r.replace(' (main)', ''))
a.set_xticks(range(4)); a.set_xticklabels(SZ); a.set_ylim(0, 100); a.set_xlabel('Landslides per event'); a.set_ylabel('Events under an AR (%)')
a.legend(loc='upper left', frameon=False, fontsize=5.5, handlelength=1.2, borderaxespad=0.1, labelspacing=0.2)
a.spines[['top', 'right']].set_visible(False)
for k, r in enumerate(rules[1:]):
    b = ax_mm(60 + k * 41.5, 10, 36, H - 17)
    for s_, (lab, col) in INV.items():
        q = CL[(CL.rule == r) & (CL.src == s_)].set_index('size').reindex(SZ); ok = q.n >= 3
        b.plot(np.arange(4)[ok.values], q.ar[ok], '-o', color=col, ms=2.2, lw=0.9, label=lab)
    b.set_xticks(range(4)); b.set_xticklabels(SZ); b.set_ylim(0, 100); b.set_xlabel('Landslides per event')
    b.set_title(r, fontsize=6.5, pad=2); b.spines[['top', 'right']].set_visible(False)
    if k == 1: b.legend(loc='upper left', frameon=False, fontsize=5.5, handlelength=1.2, borderaxespad=0.1, labelspacing=0.2)
for k, x in enumerate((0.3, 51, 92.5, 134)): L(f, H, 'abcd'[k], x, H - 0.3)
print('S1', overlap_check(f)); f.savefig(f'{OUT}/FigS1.png', dpi=300); plt.close(f)

# ---------------- Fig. S2 rain-matched odds
RM = pd.read_csv(f'{S}/rain_matched.csv'); MH = pd.read_csv(f'{S}/rain_matched_mh.csv').set_index('src').OR_MH
RL = ['<5', '5–10', '10–20', '20–40', '40–80', '≥80']
H = 56; f, ax_mm = newfig(H)
a = ax_mm(12, 11, 70, H - 15); b = ax_mm(103, 11, 70, H - 15)
for s_, (lab, col) in INV.items():
    q = RM[RM.src == s_].set_index('rain').reindex(RL)
    a.plot(range(6), q.p_ar * 100, '-o', color=col, ms=2.4, lw=1.0, label=lab)
    a.plot(range(6), q.p_no * 100, '--', color=col, lw=0.8)
    b.plot(range(6), q.OR, '-o', color=col, ms=2.4, lw=1.0, label=f'{lab} (MH {MH[s_]:.2f})')
a.set_yscale('log'); a.set_xticks(range(6)); a.set_xticklabels(RL); a.set_xlabel('2-day rain at the cell (mm)'); a.set_ylabel('Cell-days with a landslide (%)')
a.plot([], [], 'k-', lw=1.0, label='AR days'); a.plot([], [], 'k--', lw=0.8, label='Other days')
a.legend(loc='upper left', frameon=False, fontsize=5.5, ncol=2, handlelength=1.4, borderaxespad=0.1, labelspacing=0.2)
b.axhline(1, color='#999999', lw=0.6, ls=(0, (2, 1.5)))
b.set_xticks(range(6)); b.set_xticklabels(RL); b.set_xlabel('2-day rain at the cell (mm)'); b.set_ylabel('Odds ratio, AR vs other days'); b.set_ylim(0.5, 3.5)
b.legend(loc='upper left', frameon=False, fontsize=5.5, handlelength=1.2, borderaxespad=0.1, labelspacing=0.2)
for x in (a, b): x.spines[['top', 'right']].set_visible(False)
L(f, H, 'a', 0.3, H - 0.3); L(f, H, 'b', 91, H - 0.3)
print('S2', overlap_check(f)); f.savefig(f'{OUT}/FigS2.png', dpi=300); plt.close(f)

# ---------------- Fig. S3 which ARs set off landslides
YD = pd.read_csv(f'{S}/yield.csv'); ST = pd.read_csv(f'{S}/ar_storms_region.csv'); LG = pd.read_csv(f'{S}/episode_logit.csv')
EP = pd.read_parquet(f'{S}/episodes_min.parquet')
H = 54; f, ax_mm = newfig(H)
a = ax_mm(10, 10, 34, H - 15); b = ax_mm(54, 10, 32, H - 15); c = ax_mm(106, 10, 30, H - 15); d = ax_mm(149, 10, 32, H - 15)
for s_, (lab, col) in INV.items():
    q = YD[(YD.src == s_) & (YD.episodes >= 50)]
    a.plot(q.cat, q.p_any, '-o', color=col, ms=2.2, lw=0.9, label=lab)
a.set_ylim(0, 8.8); a.set_xticks(range(6)); a.set_xticklabels(['<1', '1', '2', '3', '4', '5']); a.set_xlabel('AR category'); a.set_ylabel('AR episodes at a cell\nwith ≥1 landslide (%)', linespacing=1.0)
a.legend(loc='upper left', frameon=False, fontsize=5.2, handlelength=1.0, borderaxespad=0.1, labelspacing=0.15)
RGN = ['NVE', 'ITALICA', 'US West', 'US East']; RLB = ['Norway', 'Italy', 'US\nwest', 'US\neast']
for k, (thr, colr, lab) in enumerate(((1, '#C6DBEF', '≥1'), (5, '#6BAED6', '≥5'), (20, NAVY, '≥20'))):
    v = [(ST[ST.region == r].nls >= thr).mean() * 100 for r in RGN]
    b.bar(np.arange(4) + (k - 1) * 0.27, v, 0.26, color=colr, lw=0, label=f'{lab} landslides')
b.set_xticks(range(4)); b.set_xticklabels(RLB, fontsize=5.5, linespacing=0.9); b.set_ylim(0, 100); b.set_ylabel('AR storms over the region (%)')
b.legend(loc='upper right', frameon=False, fontsize=5.2, handlelength=1.0, borderaxespad=0.1, labelspacing=0.15)
VN = [('z_rain_rel', '2-day rain'), ('z_dur_h', 'Duration'), ('z_ivt_max', 'Peak IVT'), ('z_a30', '30-day rain'), ('z_a7', '7-day rain'), ('frz', 'Below 0 °C')]
for i, (v, lab) in enumerate(VN[::-1]):
    for j, (s_, (nm, col)) in enumerate(INV.items()):
        r = LG[(LG.src == s_) & (LG['var'] == v)].iloc[0]; y = i + 0.28 - j * 0.14
        c.plot([r.lo, r.hi], [y, y], color=col, lw=0.7); c.plot(r.OR, y, 'o', color=col, ms=2.0)
    c.text(0.085, i, lab, ha='right', va='center', fontsize=5.6)
c.axvline(1, color='#999999', lw=0.6, ls=(0, (2, 1.5))); c.set_xscale('log'); c.set_xlim(0.1, 20); c.set_xticks([0.2, 1, 5]); c.set_xticklabels(['0.2', '1', '5'])
c.xaxis.set_minor_locator(matplotlib.ticker.NullLocator()); c.set_yticks([]); c.spines['left'].set_visible(False); c.set_xlabel('Odds ratio (per SD)')
q = EP[EP.cold & EP.t2ge0 & EP.ext].copy(); q['ac'] = pd.cut(q.a30, [-1, 33.3, 66.7, 101], labels=['Dry', 'Normal', 'Wet'])
ACOL = ['#F4A582', '#BDBDBD', '#4393C3']
for k, ac in enumerate(['Dry', 'Normal', 'Wet']):
    v = [q[(q.src == s_) & (q.ac == ac)].hit.mean() * 100 for s_ in INV]
    d.bar(np.arange(5) + (k - 1) * 0.27, v, 0.26, color=ACOL[k], lw=0, label=ac)
d.set_xticks(range(5)); d.set_xticklabels(['NO', 'US', 'IT', 'GLC', 'GFLD'], fontsize=5.5); d.set_ylabel('Episodes with ≥1 landslide (%)')
d.legend(loc='upper right', frameon=False, fontsize=5.2, handlelength=1.0, borderaxespad=0.1, labelspacing=0.15, title='30 days before', title_fontsize=5.2)
for x in (a, b, c, d): x.spines[['top', 'right']].set_visible(False)
L(f, H, 'a', 0.3, H - 0.3); L(f, H, 'b', 45, H - 0.3); L(f, H, 'c', 88.5, H - 0.3); L(f, H, 'd', 139.5, H - 0.3)
print('S3', overlap_check(f)); f.savefig(f'{OUT}/FigS3.png', dpi=300); plt.close(f)

# ---------------- Fig. S4 land surface
stt = np.load(f'{O}/era5_static.npz'); SD = stt['standard_deviation_of_orography']
EP['sdor'] = SD[EP.ci.values, EP.cj.values]
LS = pd.read_csv(f'{S}/episode_logit_surface.csv')
FS = pd.read_parquet(f'{S}/events_fp_steep.parquet'); FS['arm'] = FS.ar_frac >= 0.5
H = 52; f, ax_mm = newfig(H)
a = ax_mm(11, 10, 44, H - 14); b = ax_mm(92, 10, 36, H - 14); c = ax_mm(141, 10, 40, H - 14)
for s_, (lab, col) in INV.items():
    q = EP[EP.src == s_]; cell = q.groupby(['ci', 'cj']).sdor.first().reset_index()
    cell['rq'] = pd.qcut(cell.sdor, 4, labels=False); q = q.merge(cell[['ci', 'cj', 'rq']], on=['ci', 'cj'])
    v = q[q.ext].groupby('rq').hit.mean() * 100; m = cell.groupby('rq').sdor.median()
    a.plot(v.index, v.values, '-o', color=col, ms=2.4, lw=0.9, label=f'{lab} ({m.iloc[0]:.0f}–{m.iloc[-1]:.0f} m)')
a.set_yscale('log'); a.set_ylim(0.6, 40); a.set_yticks([1, 2, 5, 10, 20]); a.set_yticklabels(['1', '2', '5', '10', '20'])
a.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
a.set_xticks(range(4)); a.set_xticklabels(['Q1\nflattest', 'Q2', 'Q3', 'Q4\nsteepest'], linespacing=0.9); a.set_xlim(-0.2, 3.2)
a.set_xlabel('Relief quartile of the cell', labelpad=1); a.set_ylabel('Episodes with ≥1 landslide (%)')
a.legend(loc='upper left', bbox_to_anchor=(1.0, 1.02), frameon=False, fontsize=5.2, handlelength=1.0, borderaxespad=0.1, labelspacing=0.3, title='Median relief Q1–Q4', title_fontsize=5.2, alignment='left')
VS = [('z_rain_rel', '2-day rain'), ('z_sdor', 'Relief'), ('z_elev', 'Elevation'), ('z_forest', 'Forest cover')]
for i, (v, lab) in enumerate(VS[::-1]):
    for j, (s_, (nm, col)) in enumerate(INV.items()):
        r = LS[(LS.src == s_) & (LS['var'] == v)].iloc[0]; y = i + 0.28 - j * 0.14
        b.plot([r.lo, r.hi], [y, y], color=col, lw=0.7); b.plot(r.OR, y, 'o', color=col, ms=2.0)
    b.text(0.24, i, lab, ha='right', va='center', fontsize=5.6)
b.axvline(1, color='#999999', lw=0.6, ls=(0, (2, 1.5))); b.set_xscale('log'); b.set_xlim(0.3, 18); b.set_xticks([0.5, 1, 2, 5, 10]); b.set_xticklabels(['0.5', '1', '2', '5', '10'])
b.xaxis.set_minor_locator(matplotlib.ticker.NullLocator()); b.set_yticks([]); b.spines['left'].set_visible(False); b.set_xlabel('Odds ratio (per SD)')
for k, (am, colr, lab) in enumerate(((True, NAVY, 'AR events'), (False, '#BDBDBD', 'Other events'))):
    v = []
    for s_ in INV:
        q = FS[(FS.src == s_) & (FS.arm == am)]; v.append(q.fp_steep.sum() / (q.fp_steep.sum() + q.fp_gentle.sum()) * 100)
    c.bar(np.arange(5) + (k - 0.5) * 0.38, v, 0.36, color=colr, lw=0, label=lab)
c.set_xticks(range(5)); c.set_xticklabels(['NO', 'US', 'IT', 'GLC', 'GFLD'], fontsize=5.5); c.set_ylim(0, 70)
c.set_ylabel('Footprint on steep cells (%)'); c.legend(loc='upper left', frameon=False, fontsize=5.2, handlelength=1.0, borderaxespad=0.1, labelspacing=0.15, ncol=2)
for x in (a, b, c): x.spines[['top', 'right']].set_visible(False)
L(f, H, 'a', 0.3, H - 0.3); L(f, H, 'b', 79, H - 0.3); L(f, H, 'c', 131, H - 0.3)
print('S4', overlap_check(f)); f.savefig(f'{OUT}/FigS4.png', dpi=300); plt.close(f)

# ---------------- Fig. S5 AR share of extreme-rain area
T = pd.read_csv(f'{O}/trend_extreme_ar.csv'); T = T[T['mask'] == 'land'].copy(); T['sh'] = T.ex_ar / T.ex * 100
RN = {'WNA': 'W North America', 'ENA': 'E North America', 'EUR': 'Europe', 'EAS': 'East Asia', 'SAS': 'South Asia', 'SAM': 'S South America', 'SAF': 'Southern Africa', 'ANZ': 'Australia–NZ'}
RCOL = dict(zip(RN, ['#1F77B4', '#AEC7E8', '#2CA02C', '#D62728', '#FF7F0E', '#9467BD', '#8C564B', '#17BECF']))
D = pd.read_parquet(f'{O}/extreme_objects.parquet')
D['band'] = pd.cut(D.lat.abs(), [0, 15, 25, 35, 45, 55, 90], labels=['0–15°', '15–25°', '25–35°', '35–45°', '45–55°', '≥55°'])
D['sb'] = pd.cut(D['size'], [0, 1, 2, 5, 10, 19, 1e9], labels=['1', '2', '3–5', '6–10', '11–19', '≥20'])
H = 56; f, ax_mm = newfig(H)
a = ax_mm(11, 10, 66, H - 14); b = ax_mm(92, 10, 40, H - 14); c = ax_mm(145, 10, 36, H - 14)
for r, nm in RN.items():
    q = T[T.reg == r].sort_values('y'); a.plot(q.y, q.sh, color=RCOL[r], lw=0.8, label=nm)
    sl = theilslopes(q.sh, q.y); a.plot(q.y, sl[1] + sl[0] * q.y, color=RCOL[r], lw=0.5, ls=(0, (2, 1.5)))
a.set_xlim(1979, 2022); a.set_ylim(0, 100); a.set_ylabel('Extreme-rain area under an AR (%)'); a.set_xlabel('Year')
a.legend(loc='lower left', ncol=4, frameon=False, fontsize=5.2, handlelength=1.0, columnspacing=0.6, borderaxespad=0.1, labelspacing=0.15)
a.spines[['top', 'right']].set_visible(False)
BC = plt.get_cmap('viridis')(np.linspace(0.05, 0.9, 6))
for k, (bd, q) in enumerate(D.groupby('band', observed=True)):
    g = q.groupby('sb', observed=True).apply(lambda d: (d.arfrac * d['size']).sum() / d['size'].sum() * 100)
    b.plot(range(len(g)), g.values, '-o', color=BC[k], ms=2.2, lw=1.0, label=bd)
b.set_xticks(range(6)); b.set_xticklabels(['1', '2', '3–5', '6–10', '11–19', '≥20'], fontsize=5.5); b.set_ylim(0, 100)
b.set_xlabel('Extreme-rain area (1° cells)'); b.set_ylabel('Under an AR (%)')
b.legend(loc='upper left', ncol=2, frameon=False, fontsize=5.3, handlelength=1.0, columnspacing=0.5, borderaxespad=0.1, labelspacing=0.15)
b.spines[['top', 'right']].set_visible(False)
# c: regional events vs extreme rain by latitude band (poleward)
B = events_all(); RG = regional(B)
xs = []
for k, (lo, hi, lab) in enumerate(((25, 35, '25–35°'), (35, 45, '35–45°'), (45, 90, '≥45°'))):
    q = D[(D.lat.abs() >= lo) & (D.lat.abs() < hi)]; ex = (q.arfrac * q['size']).sum() / q['size'].sum() * 100
    big = q[q['size'] >= 20]; exb = (big.arfrac * big['size']).sum() / big['size'].sum() * 100
    rg = RG[(RG.lat.abs() >= lo) & (RG.lat.abs() < hi)]; ev = (rg.drv == 'AR').mean() * 100
    c.bar(k - 0.27, ex, 0.26, color='#BDD7E7', lw=0, label='All extreme rain' if k == 0 else None)
    c.bar(k, exb, 0.26, color='#6BAED6', lw=0, label='Extreme rain ≥20 cells' if k == 0 else None)
    c.bar(k + 0.27, ev, 0.26, color=NAVY, lw=0, label='Regional landslide events' if k == 0 else None)
    xs.append(lab)
c.set_xticks(range(3)); c.set_xticklabels(xs); c.set_ylim(0, 115); c.set_yticks([0, 50, 100]); c.set_ylabel('Under an AR (%)'); c.set_xlabel('Latitude')
c.legend(loc='upper left', frameon=False, fontsize=5.3, handlelength=1.0, borderaxespad=0.1, labelspacing=0.15)
c.spines[['top', 'right']].set_visible(False)
L(f, H, 'a', 0.3, H - 0.3); L(f, H, 'b', 81, H - 0.3); L(f, H, 'c', 135.5, H - 0.3)
print('S5', overlap_check(f)); f.savefig(f'{OUT}/FigS5.png', dpi=300); plt.close(f)
