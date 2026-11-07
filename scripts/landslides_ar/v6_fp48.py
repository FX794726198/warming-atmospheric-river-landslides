"""v06: simultaneous extreme-rain footprint in km2 (largest land area within 4 deg of the event centre with 2-day rain >= local P99
on any single 2-day window of the event), union footprint in km2, event duration; logistic models for >=5 and >=20 landslides."""
import numpy as np, pandas as pd, sys
sys.path.insert(0, 'fig_code')
from arcommon import pool_mask
import statsmodels.formula.api as smf
E = 'work/data/era5'; O = 'work/out/g1'; V = 'work/out/v6'
lsm = np.load('work/out/a2/lsm.npy'); land = lsm > 0.5
p99 = np.load(f'{O}/r2_p99.npy')
lat = np.arange(146) - 70.0
AREA = np.broadcast_to((111.2 ** 2 * np.cos(np.radians(lat)))[:, None], (146, 360)).copy()
days = pd.date_range('1979-01-01', '2022-12-31')
PR = {y: np.load(f'{E}/pr_{y}.npy', mmap_mode='r') for y in range(1979, 2023)}
def day(dn):
    d = days[dn]; return np.nan_to_num(np.asarray(PR[d.year][d.dayofyear - 1], dtype=np.float32))
cache = {}
def r2(dn):
    if dn not in cache:
        if len(cache) > 400: cache.clear()
        cache[dn] = day(dn) + (day(dn - 1) if dn > 0 else 0)
    return cache[dn]
EV = pd.read_parquet(f'{O}/events_fp.parquet').sort_values('d0')
R = 4
sim = []; uni = []; simc = []
for e in EV.itertuples():
    i = int(round(e.ci)); j = int(round(e.cj))
    ii = slice(max(0, i - R), min(146, i + R + 1)); jj = [(j + o) % 360 for o in range(-R, R + 1)]
    L = land[ii][:, jj]; A = AREA[ii][:, jj]; q = p99[ii][:, jj]
    ex = np.zeros(L.shape, bool); best = 0.0; bestc = 0
    for dn in range(int(e.d0), int(e.d1) + 1):
        h = (r2(dn)[ii][:, jj] >= q) & L
        ex |= h; a = float(A[h].sum()); best = max(best, a); bestc = max(bestc, int(h.sum()))
    sim.append(best); uni.append(float(A[ex].sum())); simc.append(bestc)
EV['fp_sim_km2'] = sim; EV['fp_uni_km2'] = uni; EV['fp_sim_cells'] = simc; EV['dur'] = EV.d1 - EV.d0 + 1
EV.to_parquet(f'{V}/events_fp48.parquet')
print(EV[['fp', 'fp_sim_cells', 'fp_sim_km2', 'fp_uni_km2', 'dur']].describe().round(1))

# ------------------------------------------------------------------ logistic models
EF = EV.copy(); EF['arm'] = (EF.ar_frac >= 0.5).astype(int)
EF['lfs'] = np.log1p(EF.fp_sim_km2 / 1000); EF['lfu'] = np.log1p(EF.fp_uni_km2 / 1000); EF['lpk'] = np.log(EF.pk.clip(0.05)); EF['ldur'] = np.log(EF.dur)
d0 = pd.Timestamp('1979-01-01') + pd.to_timedelta(EF.d0, 'D'); EF['mo'] = d0.dt.month.astype(str); EF['yr'] = d0.dt.year
EF['lat'] = EF.ci - 70; EF['cold'] = np.where(EF.lat >= 0, EF.mo.astype(int).isin([10, 11, 12, 1, 2, 3]), EF.mo.astype(int).isin([4, 5, 6, 7, 8, 9])).astype(int)
EF['blk'] = EF.src + '_' + EF.yr.astype(str)   # clusters for robust standard errors: inventory x year
EF['big5'] = (EF.n >= 5).astype(int); EF['big20'] = (EF.n >= 20).astype(int)
print('AR share by size:', EF[pool_mask(EF)].groupby(pd.cut(EF[pool_mask(EF)].n, [0, 1, 4, 19, 1e9]), observed=True).arm.mean().round(3).to_dict())
rows = []
def fit(name, f, q, y):
    try:
        m = smf.logit(f, q).fit(disp=0, cov_type='cluster', cov_kwds={'groups': pd.factorize(q.blk)[0]}, maxiter=200)
        b = m.params['arm']; ci = m.conf_int().loc['arm']
        return dict(model=name, n=len(q), nbig=int(q[y].sum()), OR=np.exp(b), lo=np.exp(ci[0]), hi=np.exp(ci[1]), p=m.pvalues['arm'])
    except Exception as ex:
        return dict(model=name, n=len(q), nbig=int(q[y].sum()), OR=np.nan, lo=np.nan, hi=np.nan, p=np.nan, err=str(ex)[:60])
for y in ('big5', 'big20'):
    for s in ['NVE', 'USGS', 'ITALICA', 'GLC', 'GFLD', 'ALL']:
        q = EF[pool_mask(EF)] if s == 'ALL' else EF[EF.src == s]
        fe = '+C(src)' if s == 'ALL' else ''
        sea = 'C(mo)' if y == 'big5' else 'cold'   # for >=20 landslides the events are few, so a cold-season term replaces month effects
        for name, f in [('raw', f'{y}~arm+{sea}{fe}'), ('+duration', f'{y}~arm+ldur+{sea}{fe}'),
                        ('+simultaneous footprint', f'{y}~arm+lfs+{sea}{fe}'), ('+simultaneous footprint+duration', f'{y}~arm+lfs+ldur+{sea}{fe}'),
                        ('+union footprint+duration', f'{y}~arm+lfu+ldur+{sea}{fe}'), ('+peak', f'{y}~arm+lpk+{sea}{fe}'),
                        ('+simultaneous footprint+peak+duration', f'{y}~arm+lfs+lpk+ldur+{sea}{fe}')]:
            r = fit(name, f, q, y); r.update(outcome=y, src=s); rows.append(r)
R = pd.DataFrame(rows); pd.set_option('display.width', 250); print(R.round(3).to_string()); R.to_csv(f'{V}/logit_v6.csv', index=False)
# footprint by driver and size
P = EF[pool_mask(EF)]
print(P.groupby('arm')[['fp_sim_km2', 'fp_uni_km2', 'dur']].agg(['mean', 'median']).round(0))
print(P.groupby([pd.cut(P.n, [0, 1, 4, 19, 1e9]), 'arm'], observed=True)[['fp_sim_km2', 'dur']].mean().round(0))
# yield by simultaneous-footprint class
P['fc'] = pd.cut(P.fp_sim_km2, [-1, 0, 10000, 30000, 60000, 1e9], labels=['0', '<10k', '10-30k', '30-60k', '>60k'])
print(P.groupby(['fc', 'arm'], observed=True).agg(n=('n', 'size'), big5=('big5', 'mean'), big20=('big20', 'mean')).round(3))
