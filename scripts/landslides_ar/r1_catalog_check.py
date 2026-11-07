# Robustness to the AR catalogue: recompute the headline AR shares with another daily AR mask
# (same 1-degree grid: rows = lat -70..75, cols = lon 0..359; yearly files PRE{year}.npy = packbits(bool[days,146,360], axis=0)).
# Usage: python r1_catalog_check.py PREFIX LABEL [LAST_YEAR]
#   PREFIX=work/data/arcat/ard_  reproduces the Chen et al. numbers of the paper.
import sys; sys.path.insert(0, 'fig_code')
from arcommon import *
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
PRE, LAB = sys.argv[1], sys.argv[2]; Y1 = int(sys.argv[3]) if len(sys.argv) > 3 else 2024
S = 'work/out/s1'
days = pd.date_range('1979-01-01', f'{Y1}-12-31')
M = np.zeros((len(days), 146, 360), bool); k = 0
for y in range(1979, Y1 + 1):
    nd = 366 if y % 4 == 0 else 365
    M[k:k + nd] = np.unpackbits(np.load(f'{PRE}{y}.npy'), axis=0)[:nd].astype(bool); k += nd
W = M.copy(); W[1:] |= M[:-1]                      # AR on the day or the day before
mo = days.month.values - 1
P2 = np.stack([W[(mo == m) & (days.year.values <= 2024)].mean(0) for m in range(12)])   # climatology of the 2-day window
print(LAB, 'mean AR frequency (daily, all cells)', round(M.mean() * 100, 2), '%; DJF at 60N5E/44N9E/45N124W',
      [round(M[np.isin(mo, [11, 0, 1])][:, i, j].mean() * 100, 1) for i, j in ((130, 5), (114, 9), (115, 236))])

# (a) landslides and inventory events
L = pd.read_parquet(f'{O}/ls_ar.parquet')
L = L[~((L.src == 'USGS') & L.inv.isin(['USGS Earthquake-Triggered Ground Failure', 'USGS Seismogenic Mass Movements']))]
L = L[L.date <= f'{Y1}-12-31']
dn = (L.date - D0).dt.days.values
L['ar'] = W[dn, L.ci.values, L.cj.values]; L['pc'] = P2[L.date.dt.month.values - 1, L.ci.values, L.cj.values]
win = {'NVE': ('2000', '2022'), 'USGS': ('1979', '2024'), 'ITALICA': ('1996', '2021'), 'GLC': ('2007', '2017'), 'GFLD': ('2004', '2017')}
EV = []
for s, (a, b) in win.items():
    g = L[(L.src == s) & (L.date >= a) & (L.date <= b + '-12-31')].copy().reset_index(drop=True)
    g['dn'] = (g.date - D0).dt.days
    cd = g.groupby(['dn', 'ci', 'cj']).agg(n=('lat', 'size'), ar=('ar', 'max'), pc=('pc', 'mean')).reset_index()
    idx = {kk: i for i, kk in enumerate(zip(cd.dn, cd.ci, cd.cj))}; r = []; c = []
    for i, (d, x, y) in enumerate(zip(cd.dn, cd.ci, cd.cj)):
        for dd in (0, 1):
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    j = idx.get((d + dd, x + di, (y + dj) % 360))
                    if j is not None and j != i: r.append(i); c.append(j)
    _, lab = connected_components(coo_matrix((np.ones(len(r)), (r, c)), shape=(len(cd), len(cd))), directed=False)
    cd['ev'] = lab
    e = cd.groupby('ev').agg(n=('n', 'sum'), ar_frac=('ar', 'mean'), pc=('pc', 'mean'), ci=('ci', 'mean'), cj=('cj', 'mean')).reset_index()
    e['src'] = s; EV.append(e)
E = pd.concat(EV, ignore_index=True); E['arm'] = E.ar_frac >= 0.5
E['sz'] = pd.cut(E.n, [0, 1, 4, 19, 1e9], labels=['1', '2-4', '5-19', '>=20'])
Ep = E[pool_mask(E)]
rows = []
for s, q in list(E.groupby('src')) + [('Pooled', Ep)]:
    t = q.groupby('sz', observed=True).agg(n=('n', 'size'), ar=('arm', 'mean'))
    rows.append(dict(catalogue=LAB, group=s, clim=q.pc.mean() * 100, **{f'ar_{k}': v * 100 for k, v in t.ar.items()}, **{f'n_{k}': v for k, v in t.n.items()}))
T1 = pd.DataFrame(rows)

# (b) regional landslide events (>= 20 landslides): AR flag at the centre cell from the day before the event to its last day
def arflag(d0, d1, lat, lon):
    ci = int(round(lat + 70)); cj = int(round(lon)) % 360
    if ci < 0 or ci > 145: return np.nan
    a = (pd.Timestamp(d0) - D0).days - 1; b = (pd.Timestamp(d1) - D0).days
    if b >= len(days) or a < 0: return np.nan
    return bool(M[a:b + 1, ci, cj].any())
A = pd.read_parquet(f'{O}/outbreaks_all.parquet')
A = A[~(A.src.isin(['GLC', 'GFLD']).values & national_mask(A.lat.values, A.lon.values))]
A['ar0x'] = [arflag(a, b, la, lo) if n >= 20 else np.nan for a, b, la, lo, n in zip(A.date0, A.date1, A.lat, A.lon, A.n)]
A = pd.DataFrame(dict(src=A.src, n=A.n, lat=A.lat, lon=A.lon, date=pd.to_datetime(A.date0), ar0=A.ar0, ar0x=A.ar0x, tcd=A.tcd2))
X = pd.read_parquet(f'{O}/extra_outbreaks.parquet')
X['ar0x'] = [arflag(a, b, la, lo) for a, b, la, lo in zip(X.date_start, X.date_end, X.lat, X.lon)]
X = pd.DataFrame(dict(src=X.source, n=X.n, lat=X.lat, lon=X.lon, date=pd.to_datetime(X.date_peak), ar0=X.ar0, ar0x=X.ar0x, tcd=X.tcd))
B = dedupe(pd.concat([A, X], ignore_index=True)); B['cold'] = cold_season(B.lat, B.date.dt.month)
B = B[(B.n >= 20) & B.ar0x.notna()]
rows = []
for col, nm in (('ar0', 'Chen (paper)'), ('ar0x', LAB)):
    drv = np.where(B.tcd <= 500, 'TC', np.where(B[col].astype(bool), 'AR', 'other')); q = B.assign(drv=drv)
    po = q[q.lat.abs() >= 25]; pc = po[po.cold]; pw = po[~po.cold]; tr = q[q.lat.abs() < 25]
    rows.append(dict(catalogue=nm, n=len(q), poleward_ar=(po.drv == 'AR').mean() * 100, n_pole=len(po), cold_ar=(pc.drv == 'AR').mean() * 100, n_cold=len(pc),
                     warm_ar=(pw.drv == 'AR').mean() * 100, n_warm=len(pw), tropics_ar=(tr.drv == 'AR').mean() * 100, n_trop=len(tr),
                     agree=(q[col].astype(bool) == q.ar0x.astype(bool)).mean() * 100))
T2 = pd.DataFrame(rows)
pd.set_option('display.width', 250)
print(T1.round(1).to_string()); print(T2.round(1).to_string())
tag = LAB.replace(' ', '_')
T1.to_csv(f'{S}/catalog_check_sizes_{tag}.csv', index=False); T2.to_csv(f'{S}/catalog_check_regional_{tag}.csv', index=False)
