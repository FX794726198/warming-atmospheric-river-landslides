# Which ARs set off landslides and which do not?
# (1) AR storms over each inventory region (consecutive days with an AR over >=1 landslide cell of the region): share with 0, >=1, >=5, >=20 landslides.
# (2) AR episodes at landslide cells: compare episodes followed by landslides with those that were not, by AR strength (IVT max, duration,
#     category), rain at the cell (max 2-day rain / local P99), antecedent rain (30-day and 7-day totals before the episode, as percentiles of
#     the same cell and calendar month), 2-day mean T2m, and season. Logistic model per inventory with standardized predictors.
import numpy as np, pandas as pd, statsmodels.formula.api as smf
O = 'work/out/g1'; E = 'work/data/era5'; S = 'work/out/s1'
T0 = pd.Timestamp('1979-01-01'); days = pd.date_range('1979-01-01', '2022-12-31'); ND = len(days)
z = np.load(f'{O}/panel.npz'); ci, cj, W = z['ci'], z['cj'], z['W']
col = {(a, b): k for k, (a, b) in enumerate(zip(ci, cj))}
PR = []; TT = []
for y in range(1979, 2023):
    PR.append(np.nan_to_num(np.load(f'{E}/pr_{y}.npy', mmap_mode='r')[:, ci, cj].astype(np.float32)))
    TT.append(np.load(f'{E}/t2m_{y}.npy', mmap_mode='r')[:, 0, ci, cj].astype(np.float32))
PR = np.concatenate(PR); TT = np.concatenate(TT); assert len(PR) == ND
p99 = np.load(f'{O}/r2_p99.npy')[ci, cj]
R2 = PR.copy(); R2[1:] += PR[:-1]
cs = np.vstack([np.zeros((1, PR.shape[1])), np.cumsum(PR, 0)])
def ante(n):  # total of days d-n-1 .. d-2 (excludes the 2-day window of the event)
    A = np.full(PR.shape, np.nan, np.float32); A[n + 1:] = cs[n:ND - 1] - cs[0:ND - n - 1]; return A
A30 = ante(30); A7 = ante(7)
mon = days.month.values
def pct_by_month(A):  # percentile of each value within the same cell and calendar month
    P = np.full(A.shape, np.nan, np.float32)
    for m in range(1, 13):
        r = np.flatnonzero(mon == m); a = A[r]
        rk = np.argsort(np.argsort(np.where(np.isnan(a), -1, a), 0), 0).astype(np.float32)
        P[r] = rk / (len(r) - 1) * 100
    return P
P30 = pct_by_month(A30); P7 = pct_by_month(A7)
L = pd.read_parquet(f'{O}/ls_ar.parquet')
L = L[~((L.src == 'USGS') & L.inv.isin(['USGS Earthquake-Triggered Ground Failure', 'USGS Seismogenic Mass Movements']))]
win = {'NVE': ('2000', '2022'), 'USGS': ('1979', '2022'), 'ITALICA': ('1996', '2021'), 'GLC': ('2007', '2017'), 'GFLD': ('2004', '2017')}

# ---------------- (2) episodes
rows = []
for s, (a, b) in win.items():
    ep = pd.read_parquet(f'{O}/ep_{s}.parquet')
    ep = ep[(ep.t0 >= a) & (ep.t1 <= b + '-12-31T18')].copy()
    d0 = ((ep.t0.dt.floor('D') - T0).dt.days).values; d1 = ((ep.t1.dt.floor('D') - T0).dt.days).values + 1
    d1 = np.minimum(d1, ND - 1); k = np.array([col[(x, y)] for x, y in zip(ep.ci, ep.cj)])
    rmax = np.array([R2[u:v + 1, kk].max() for u, v, kk in zip(d0, d1, k)])
    ep['rain_rel'] = rmax / np.maximum(p99[k], 1); ep['rain'] = rmax
    ep['a30'] = P30[d0, k]; ep['a7'] = P7[d0, k]; ep['a30mm'] = A30[d0, k]
    ep['t2'] = np.array([TT[max(u - 1, 0):v + 1, kk].mean() for u, v, kk in zip(d0, d1, k)])
    ep['month'] = ep.t0.dt.month; ep['lat'] = ep.ci - 70
    ep['cold'] = np.where(ep.lat >= 0, ep.month.isin([10, 11, 12, 1, 2, 3]), ep.month.isin([4, 5, 6, 7, 8, 9]))
    ep['hit'] = (ep.nls > 0).astype(int); ep['src'] = s
    rows.append(ep)
EP = pd.concat(rows, ignore_index=True); EP.to_parquet(f'{S}/episodes_drivers.parquet')
pd.set_option('display.width', 220)
print('\n== episodes with vs without landslides (medians)')
g = EP.groupby(['src', 'hit']).agg(n=('hit', 'size'), ivt=('ivt_max', 'median'), dur=('dur_h', 'median'), cat=('cat', 'mean'),
                                     rain=('rain', 'median'), rain_rel=('rain_rel', 'median'), a30=('a30', 'median'), a7=('a7', 'median'),
                                     t2=('t2', 'median'), cold=('cold', 'mean'))
print(g.round(2).to_string())
print('\n== logistic, standardized predictors (OR per SD), month FE')
out = []
for s in win:
    q = EP[EP.src == s].dropna(subset=['a30', 'a7', 't2']).copy()
    for v in ('ivt_max', 'dur_h', 'rain_rel', 'a30', 'a7', 't2'):
        x = np.log(q[v].clip(lower=1e-3)) if v in ('ivt_max', 'dur_h', 'rain_rel') else q[v]
        q['z_' + v] = (x - x.mean()) / x.std()
    q['frz'] = (q.t2 < 0).astype(int)
    m = smf.logit('hit ~ z_ivt_max + z_dur_h + z_rain_rel + z_a30 + z_a7 + frz + C(month)', q).fit(disp=0)
    ci_ = m.conf_int()
    for v in ('z_ivt_max', 'z_dur_h', 'z_rain_rel', 'z_a30', 'z_a7', 'frz'):
        out.append((s, v, np.exp(m.params[v]), np.exp(ci_.loc[v, 0]), np.exp(ci_.loc[v, 1])))
LG = pd.DataFrame(out, columns=['src', 'var', 'OR', 'lo', 'hi']); LG.to_csv(f'{S}/episode_logit.csv', index=False)
print(LG.pivot(index='var', columns='src', values='OR').round(2).to_string())
print('\n== P(landslide) by rain class x antecedent 30-day tercile (cold season, T2m >= 0)')
q = EP[EP.cold & (EP.t2 >= 0)].copy()
q['rc'] = pd.cut(q.rain_rel, [0, 0.5, 1, 1e9], labels=['<0.5 P99', '0.5–1 P99', '≥P99'])
q['ac'] = pd.cut(q.a30, [-1, 33.3, 66.7, 101], labels=['dry', 'normal', 'wet'])
print((q.groupby(['src', 'rc', 'ac'], observed=True).hit.mean() * 100).unstack().round(2).to_string())
print('\n== share of episodes by outcome')
print(EP.groupby('src').agg(n=('hit', 'size'), any=('hit', 'mean'), p5=('nls', lambda v: (v >= 5).mean()), p20=('nls', lambda v: (v >= 20).mean())).round(4))

# ---------------- (1) AR storms over inventory regions
reg = {'NVE': ('NVE', lambda la, lo: np.ones(len(la), bool)), 'ITALICA': ('ITALICA', lambda la, lo: np.ones(len(la), bool)),
       'US West': ('USGS', lambda la, lo: lo < -115), 'US East': ('USGS', lambda la, lo: lo > -100)}
rows = []
for nm, (s, f) in reg.items():
    a, b = win[s]
    g = L[(L.src == s) & (L.date >= a) & (L.date <= b + '-12-31')]
    lon_ = np.where(g.cj > 180, g.cj - 360, g.cj); g = g[f(g.ci - 70, lon_)]
    cells = sorted({col[(x, y)] for x, y in zip(g.ci, g.cj) if (x, y) in col})
    t0 = (pd.Timestamp(a + '-01-01') - T0).days; t1 = (pd.Timestamp(b + '-12-31') - T0).days + 1
    arday = W[t0:t1][:, cells].any(1); frac = W[t0:t1][:, cells].mean(1)
    nday = np.zeros(t1 - t0, int)
    for d in g.date: nday[(d - T0).days - t0] += 1
    lab = np.cumsum(np.r_[1, np.diff(arday.astype(int)) == 1]) * arday   # storm id per AR day
    for sid in np.unique(lab[lab > 0]):
        ii = np.flatnonzero(lab == sid); lo_, hi_ = ii[0], min(ii[-1] + 1, len(nday) - 1)
        n = nday[lo_:hi_ + 1].sum()
        rows.append((nm, sid, len(ii), frac[ii].max(), n))
ST = pd.DataFrame(rows, columns=['region', 'storm', 'days', 'maxcover', 'nls']); ST.to_csv(f'{S}/ar_storms_region.csv', index=False)
print('\n== AR storms over inventory regions')
print(ST.groupby('region').agg(storms=('nls', 'size'), any=('nls', lambda v: (v > 0).mean() * 100), p5=('nls', lambda v: (v >= 5).mean() * 100),
                                p20=('nls', lambda v: (v >= 20).mean() * 100)).round(1).to_string())
ST['cov'] = pd.cut(ST.maxcover, [0, 0.1, 0.25, 0.5, 1.01], labels=['<10%', '10–25%', '25–50%', '≥50%'])
print((ST.groupby(['region', 'cov'], observed=True).agg(n=('nls', 'size'), p20=('nls', lambda v: (v >= 20).mean() * 100), any=('nls', lambda v: (v > 0).mean() * 100))).round(1).to_string())
