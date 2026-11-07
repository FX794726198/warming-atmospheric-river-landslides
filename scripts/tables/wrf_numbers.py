"""Per-event numbers of the WRF pseudo-global-warming runs, from the post-processed npz and summary files.
Land cells of the 3 km domain, ten cells along each edge left out, area-weighted. Writes
out/wrf/wrf_events.csv (all numbers used in the text) and parts/tab_wrfres.md (Supplementary table)."""
import os, re, glob, json
import numpy as np, pandas as pd

WP = os.environ.get('WRFPOST', 'work/data/wrf/post_fig')
OUT = 'work/out/wrf'; PARTS = 'work/paper/parts'
EVS = [('NOR', 'Western Norway, Dec 2011'), ('BC', 'British Columbia, Nov 2021'), ('ORE', 'Oregon, Nov 1996'),
       ('LIG', 'Liguria and Piedmont, Nov 2014'), ('RSB', 'Rio Grande do Sul, Apr–May 2024')]
EDGE = 10


def find(ev, suf):
    g = sorted(glob.glob(f'{WP}/{ev}_*/{ev}_*_{suf}'))
    return g[0] if g else None


def load(ev, arm):
    p = find(ev, f'{arm}.npz')
    if not p: return None
    z = np.load(p)
    return {k: z[k][EDGE:-EDGE, EDGE:-EDGE].astype(float) for k in z.files if z[k].ndim == 2}


rows = []
for ev, name in EVS:
    c, w = load(ev, 'ctl'), load(ev, 'pgw585')
    if c is None or w is None: continue
    s = pd.read_csv(find(ev, 'summary.csv')).set_index('arm')
    land = c['landmask'] > 0.5; A = c['cell_area_km2']
    am = lambda x: float((x * A)[land].sum() / A[land].sum())
    r = dict(event=ev, name=name, dT2m=float(s.loc['pgw585', 'd02_dT2m']), dT850=float(s.loc['pgw585', 'd02_dT850']),
             dT2m_d01=float(s.loc['pgw585', 'd01_dT2m']))
    for k in ('peak_ivt_d01', 'peak_ivt_d02footprint'):
        r[k + '_c'], r[k + '_w'] = float(s.loc['ctl', k]), float(s.loc['pgw585', k])
    for k in ('precip_total', 'liq_total', 'snowfall', 'graupel_hail', 'snowmelt', 'runoff_total'):
        r[k + '_c'], r[k + '_w'] = am(c[k]), am(w[k])
    for t in (50, 100, 150, 200):
        r[f'a{t}_c'] = float(A[land & (c['liq_max48h'] >= t)].sum()); r[f'a{t}_w'] = float(A[land & (w['liq_max48h'] >= t)].sum())
        r[f'a24_{t}_c'] = float(A[land & (c['liq_max24h'] >= t)].sum()); r[f'a24_{t}_w'] = float(A[land & (w['liq_max24h'] >= t)].sum())
        r[f'p{t}_c'] = float(A[land & (c['precip_max48h'] >= t)].sum()); r[f'p{t}_w'] = float(A[land & (w['precip_max48h'] >= t)].sum())
    v = c['liq_max48h'][land]; wts = A[land]; o = np.argsort(v); cw = np.cumsum(wts[o]) / wts.sum()
    p99 = float(v[o][np.searchsorted(cw, 0.99)]); r['p99_c_mm'] = p99
    r['ap99_c'] = float(A[land & (c['liq_max48h'] >= p99)].sum()); r['ap99_w'] = float(A[land & (w['liq_max48h'] >= p99)].sum())
    h = c['hgt']; r['land_km2'] = float(A[land].sum())
    for lo_, hi_ in ((0, 500), (500, 1000), (1000, 1500), (1500, 9000)):
        m = land & (h >= lo_) & (h < hi_)
        if A[m].sum() < 500: continue
        f = lambda d: float((d['liq_total'] * A)[m].sum() / (d['precip_total'] * A)[m].sum())
        r[f'liqshare_{lo_}_c'], r[f'liqshare_{lo_}_w'] = f(c), f(w)
        r[f'liq_{lo_}_c'], r[f'liq_{lo_}_w'] = float((c['liq_total'] * A)[m].sum() / A[m].sum()), float((w['liq_total'] * A)[m].sum() / A[m].sum())
    rows.append(r)

R = pd.DataFrame(rows).set_index('event'); os.makedirs(OUT, exist_ok=True); R.to_csv(f'{OUT}/wrf_events.csv')
pc = lambda a, b: (b / a - 1) * 100 if a > 0 else float('nan')
for ev, r in R.iterrows():
    print(f"{ev}: dT {r.dT2m:.1f}  IVT {pc(r.peak_ivt_d01_c, r.peak_ivt_d01_w):+.0f}%  tot {r.precip_total_c:.1f}->{r.precip_total_w:.1f} ({pc(r.precip_total_c, r.precip_total_w):+.0f}%, CC {7 * r.dT2m:.0f}%)  "
          f"liq {pc(r.liq_total_c, r.liq_total_w):+.0f}%  snow {pc(r.snowfall_c, r.snowfall_w):+.0f}%  run {pc(r.runoff_total_c, r.runoff_total_w):+.0f}%  "
          f"a100 {r.a100_c:.0f}->{r.a100_w:.0f} (x{r.a100_w / max(r.a100_c, 1):.1f})  a200 {r.a200_c:.0f}->{r.a200_w:.0f}  "
          f"p99 {r.p99_c_mm:.0f} mm a {r.ap99_c:.0f}->{r.ap99_w:.0f} (x{r.ap99_w / r.ap99_c:.1f})  a100 total-precip {pc(r.p100_c, r.p100_w):+.0f}%")


def chg(a, b, fmt='{:.1f}'):
    if a < 0.05 and b < 0.05: return '0'
    return f'{fmt.format(a)} ({(b / a - 1) * 100:+.0f}%)'.replace('-', '−')


def area(a, b):
    return f'{a:,.0f} (×{b / a:.1f})' if a > 0 else f'0 ({b:,.0f} km² in warmer run)'


SHORT = {'NOR': 'Western Norway 2011', 'BC': 'British Columbia 2021', 'ORE': 'Oregon 1996', 'LIG': 'Liguria and Piedmont 2014', 'RSB': 'Rio Grande do Sul 2024'}
lines = ['| Event | Warming (°C) | Peak IVT (kg m⁻¹ s⁻¹) | Total precipitation (mm) | Rain (mm) | Snowfall (mm) | Runoff (mm) | Area ≥100 mm in 48 h (km²) | Area ≥200 mm in 48 h (km²) |',
         '|---|---|---|---|---|---|---|---|---|']
for ev, r in R.iterrows():
    lines.append(f"| {SHORT[ev]} | {r.dT2m:.1f} | {chg(r.peak_ivt_d01_c, r.peak_ivt_d01_w, '{:.0f}')} | {chg(r.precip_total_c, r.precip_total_w)} | "
                 f"{chg(r.liq_total_c, r.liq_total_w)} | {chg(r.snowfall_c, r.snowfall_w)} | {chg(r.runoff_total_c, r.runoff_total_w)} | "
                 f"{area(r.a100_c, r.a100_w)} | {area(r.a200_c, r.a200_w)} |")
open(f'{PARTS}/tab_wrfres.md', 'w').write('\n'.join(lines) + '\n')
print('\n'.join(lines))


# ---------------------------------------------------------------- control runs against rain gauges (GHCN-Daily)
EV = 'work/data/wrf/eval'
if os.path.exists(f'{EV}/summary.csv'):
    E = pd.read_csv(f'{EV}/summary.csv').set_index('event')
    FIPS = {'NO': 'Norway', 'SW': 'Sweden', 'DA': 'Denmark', 'UK': 'United Kingdom', 'US': 'United States', 'CA': 'Canada', 'GM': 'Germany',
            'FR': 'France', 'IT': 'Italy', 'SZ': 'Switzerland', 'SI': 'Slovenia', 'AU': 'Austria', 'BR': 'Brazil', 'UY': 'Uruguay', 'AR': 'Argentina', 'PA': 'Paraguay'}
    DAYS = {'NOR': '24–27 Dec 2011', 'BC': '13–16 Nov 2021', 'ORE': '16–19 Nov 1996', 'LIG': '9–16 Nov 2014', 'RSB': '27 Apr–2 May 2024'}
    L2 = ['| Event | Days | Gauges | Observed median (mm) | Simulated median (mm) | Mean bias (%) | Spearman correlation | Within a factor of two (%) |',
          '|---|---|---|---|---|---|---|---|']
    for ev, name in EVS:
        k = [i for i in E.index if i.startswith(ev + '_') and float(E.loc[i, 'n_stations'] or 0) > 0]
        if not k: continue
        k = k[0]; e = E.loc[k]
        if 'chirps' in k: n_ = f'{int(e.n_stations):,} CHIRPS cells'
        else:
            st = pd.read_csv(f'{EV}/{k}_stations.csv'); cc = st['id'].str[:2].value_counts()
            print(ev, ', '.join(f'{FIPS.get(c, c)} {n}' for c, n in cc.items())); n_ = f'{int(e.n_stations):,}'
        L2.append(f"| {SHORT[ev]} | {DAYS[ev]} | {n_} | {float(e.median_obs_mm):.1f} | {float(e.median_ctl_mm):.1f} | {float(e.station_mean_bias_pct):+.0f} | {float(e.spearman) + 1e-9:.2f} | {100 * float(e.share_within_factor2):.0f} |".replace('-', '−'))
    open(f'{PARTS}/tab_wrfeval.md', 'w').write('\n'.join(L2) + '\n'); print('\n'.join(L2))
