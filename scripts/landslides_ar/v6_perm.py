"""v06: chance AR share under the actual attribution rules, by moving each event to other years with its calendar dates, cells and duration.
(a) inventory events (landslide counted under an AR when an AR covered its cell on the day or the day before; event under an AR when at least
half of its landslide cell-days were), (b) regional landslide events (AR at the centre cell from the day before the event to its last day)."""
import numpy as np, pandas as pd, sys, calendar
sys.path.insert(0, 'fig_code')
from arcommon import pool_mask, national_mask, dedupe, cold_season
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
O = 'work/out/g1'; I = 'work/data/arcat'; V = 'work/out/v6'
T0 = pd.Timestamp('1979-01-01'); days = pd.date_range('1979-01-01', '2022-12-31'); ND = len(days)
rng = np.random.default_rng(1)

# ---------------------------------------------------------------- (a) cell-day membership of the inventory events (as in g3_events.py)
L = pd.read_parquet(f'{O}/ls_ar.parquet')
L = L[~((L.src == 'USGS') & L.inv.isin(['USGS Earthquake-Triggered Ground Failure', 'USGS Seismogenic Mass Movements']))]
win = {'NVE': (2000, 2022), 'USGS': (1979, 2024), 'ITALICA': (1996, 2021), 'GLC': (2007, 2017), 'GFLD': (2004, 2017)}
CD = []
for s, (a, b) in win.items():
    g = L[(L.src == s) & (L.date >= str(a)) & (L.date <= f'{b}-12-31')].copy().reset_index(drop=True)
    g['dn'] = (g.date - T0).dt.days
    cd = g.groupby(['dn', 'ci', 'cj']).agg(n=('lat', 'size'), ar=('ar', 'max')).reset_index()
    idx = {k: i for i, k in enumerate(zip(cd.dn, cd.ci, cd.cj))}
    r = []; c = []
    for i, (d, x, y) in enumerate(zip(cd.dn, cd.ci, cd.cj)):
        for dd in (0, 1):
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    j = idx.get((d + dd, x + di, (y + dj) % 360))
                    if j is not None and j != i: r.append(i); c.append(j)
    A = coo_matrix((np.ones(len(r)), (r, c)), shape=(len(cd), len(cd)))
    nc, lab = connected_components(A, directed=False); cd['ev'] = lab; cd['src'] = s; CD.append(cd)
CD = pd.concat(CD, ignore_index=True); CD.to_parquet(f'{V}/celldays.parquet')
EV = pd.read_parquet(f'{O}/events_fp.parquet')   # events with d1 <= 2022
key = EV.set_index(['src', 'ev']).index
CD = CD[pd.MultiIndex.from_arrays([CD.src, CD.ev]).isin(key)].reset_index(drop=True)
# check: reproduces ar_frac
chk = CD.groupby(['src', 'ev']).ar.mean(); m = EV.set_index(['src', 'ev']).ar_frac
print('ar_frac reproduced:', np.allclose(chk.reindex(m.index).values, m.values))

# AR window flag W[t, ci, cj] = AR on day t or t-1 (daily masks)
W = np.zeros((ND, 146, 360), bool); t = 0; prev = None
for y in range(1979, 2023):
    nd = 366 if calendar.isleap(y) else 365
    a = np.unpackbits(np.load(f'{I}/ard_{y}.npy'), axis=0)[:nd].astype(bool)
    w = a.copy(); w[1:] |= a[:-1]
    if prev is not None: w[0] |= prev
    W[t:t + nd] = w; t += nd; prev = a[-1]
print('W built', W.shape, W.mean().round(3))
dd = days[CD.dn.values]; mo = dd.month.values; da = dd.day.values
ev_id = pd.factorize(pd.MultiIndex.from_arrays([CD.src, CD.ev]))[0]; nev = ev_id.max() + 1
evkey = pd.MultiIndex.from_arrays([CD.src, CD.ev]).unique()
ci = CD.ci.values.astype(int); cj = CD.cj.values.astype(int)
own_year = days[CD.groupby(ev_id).dn.transform('min').values].year.values
src_ev = CD.groupby(ev_id).src.first().values; yr_ev = pd.Series(own_year).groupby(ev_id).first().values
cnt = np.bincount(ev_id, minlength=nev)
P = np.zeros(nev); K = np.zeros(nev)     # P: number of alternative years with the event under an AR, K: alternative years tried
ARM = {}   # year -> arm per event (for permutation samples)
for y in range(1979, 2023):
    d = np.minimum(da, np.array([calendar.monthrange(y, int(mm))[1] for mm in mo]))
    dn2 = (pd.to_datetime(pd.DataFrame({'year': y, 'month': mo, 'day': d})) - T0).dt.days.values
    ok = (dn2 < ND)
    fr = np.bincount(ev_id, weights=W[np.clip(dn2, 0, ND - 1), ci, cj] & ok, minlength=nev) / cnt
    arm = fr >= 0.5
    valid = np.array([win[s][0] <= y <= win[s][1] for s in src_ev]) & (yr_ev != y)
    P += arm & valid; K += valid; ARM[y] = (arm, valid)
EV2 = EV.set_index(['src', 'ev']).loc[evkey].reset_index()
EV2['p_chance'] = P / K; EV2['arm'] = (EV2.ar_frac >= 0.5).astype(int); EV2['lat'] = EV2.ci - 70
EV2['sz'] = pd.cut(EV2.n, [0, 1, 4, 19, 1e9], labels=['1', '2-4', '5-19', '>=20'])
EV2.to_parquet(f'{V}/events_chance.parquet')
# permutation samples: one random valid year per event
years = np.arange(1979, 2023); NP = 1000
def perm_shares(mask):
    out = []
    for _ in range(NP):
        yy = rng.integers(0, len(years), nev); s = np.zeros(nev, bool)
        for k, y in enumerate(years):
            sel = (yy == k)
            arm, valid = ARM[y]; s[sel] = arm[sel]
            # invalid draws: redraw by taking the event's own chance (small bias); keep simple: mark invalid as chance-based Bernoulli
            bad = sel & ~valid; s[bad] = rng.random(bad.sum()) < (P / np.maximum(K, 1))[bad]
        out.append(s[mask].mean())
    return np.percentile(out, [2.5, 97.5])
pm = np.asarray(pool_mask(EV2))
lines = []
for name, m in [('pooled', pm), ('poleward of 25', pm & (np.abs(EV2.lat) >= 25)), ('within 25', pm & (np.abs(EV2.lat) < 25))] + [(s, (EV2.src == s).values) for s in win]:
    for sz in ['1', '2-4', '5-19', '>=20']:
        mm = m & (EV2.sz == sz).values
        if mm.sum() == 0: continue
        lo, hi = perm_shares(mm)
        lines.append(dict(group=name, size=sz, events=int(mm.sum()), observed=EV2.arm[mm].mean() * 100, chance=EV2.p_chance[mm].mean() * 100, chance_lo=lo * 100, chance_hi=hi * 100))
T = pd.DataFrame(lines); pd.set_option('display.width', 250); print(T.round(1).to_string()); T.to_csv(f'{V}/chance_inventory.csv', index=False)

# ---------------------------------------------------------------- (b) regional landslide events: AR at centre cell, day before to last day, 6-hourly masks
A = pd.read_parquet(f'{O}/outbreaks_all.parquet'); A['date'] = pd.to_datetime(A.date0)
A['drv'] = np.where(A.tcd2 <= 500, 'TC', np.where(A.ar0, 'AR', 'other'))
A = A[~(A.src.isin(['GLC', 'GFLD']).values & national_mask(A.lat.values, A.lon.values))]
A['d0'] = pd.to_datetime(A.date0); A['d1'] = pd.to_datetime(A.date1); A['name'] = A.title
X = pd.read_parquet(f'{O}/extra_outbreaks.parquet')
X = pd.DataFrame(dict(src=X.source, n=X.n, lat=X.lat, lon=X.lon, drv=X.drv, date=pd.to_datetime(X.date_peak), name=X.name, d0=pd.to_datetime(X.date_start), d1=pd.to_datetime(X.date_end)))
B = pd.concat([A[['src', 'n', 'lat', 'lon', 'drv', 'date', 'name', 'd0', 'd1']], X], ignore_index=True)
B = dedupe(B); B = B[B.n >= 20].sort_values('n', ascending=False).reset_index(drop=True)
B['month'] = B.date.dt.month; B['cold'] = cold_season(B.lat, B.month)
print('regional events:', len(B), B.drv.value_counts().to_dict())
import os
YRS = sorted(int(f[3:7]) for f in os.listdir(I) if f.startswith('ar_') and f.endswith('.npy'))
cache = {}
def arr(y):
    if y not in cache: cache[y] = np.load(f'{I}/ar_{y}.npy', mmap_mode='r')
    return cache[y]
def arflag(d0, d1, lat, lon):
    ci_ = int(round(lat + 70)); cj_ = int(round(lon)) % 360
    d = d0 - pd.Timedelta(days=1)
    while d <= d1:
        if d.year not in cache and d.year not in YRS: return np.nan
        a = arr(d.year); s = (d.dayofyear - 1) * 4
        if (np.asarray(a[s:s + 4, ci_, cj_]) > 0).any(): return True
        d += pd.Timedelta(days=1)
    return False
def shift(d, y):
    return pd.Timestamp(year=y, month=d.month, day=min(d.day, calendar.monthrange(y, d.month)[1]))
pc = []; per = {}
for i, e in B.iterrows():
    f = []
    for y in YRS:
        if y == e.d0.year: continue
        a0 = shift(e.d0, y); a1 = a0 + (e.d1 - e.d0)
        if a1.year not in YRS or (a0 - pd.Timedelta(days=1)).year not in YRS: continue
        v = arflag(a0, a1, e.lat, e.lon)
        if v == v: f.append(bool(v))
    pc.append(np.mean(f)); per[i] = np.array(f)
B['p_chance'] = pc; B['ar'] = (B.drv == 'AR').astype(int)
B['dur'] = (B.d1 - B.d0).dt.days + 1
B.to_parquet(f'{V}/regional_chance.parquet')
def perm_reg(idx):
    out = []
    for _ in range(NP):
        out.append(np.mean([rng.choice(per[i]) for i in idx]))
    return np.percentile(out, [2.5, 97.5]) * 100
lines = []
pol = np.abs(B.lat) >= 25
for name, m in [('all 231', np.ones(len(B), bool)), ('poleward of 25', pol), ('poleward of 25, cold season', pol & B.cold), ('poleward of 25, warm season', pol & ~B.cold),
                ('poleward of 35', np.abs(B.lat) >= 35), ('25-35', pol & (np.abs(B.lat) < 35)), ('15-25', (np.abs(B.lat) >= 15) & ~pol), ('within 15', np.abs(B.lat) < 15)]:
    idx = np.where(m)[0]; lo, hi = perm_reg(idx)
    lines.append(dict(group=name, events=len(idx), AR=B.ar[m].mean() * 100, TC=(B.drv[m] == 'TC').mean() * 100, chance=B.p_chance[m].mean() * 100, chance_lo=lo, chance_hi=hi, dur_mean=B.dur[m].mean()))
T2 = pd.DataFrame(lines); print(T2.round(1).to_string()); T2.to_csv(f'{V}/chance_regional.csv', index=False)
