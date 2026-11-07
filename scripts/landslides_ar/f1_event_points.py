# Landslide events of the national inventories with landslide-centroid positions (for the regional maps of Fig. 1)
import sys; sys.path.insert(0, 'fig_code')
from arcommon import *
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
L = pd.read_parquet(f'{O}/ls_ar.parquet')
L = L[~((L.src == 'USGS') & L.inv.isin(['USGS Earthquake-Triggered Ground Failure', 'USGS Seismogenic Mass Movements']))]
win = {'NVE': ('2000', '2022'), 'USGS': ('1979', '2024'), 'ITALICA': ('1996', '2021')}
A = pd.read_parquet(f'{O}/outbreaks_all.parquet'); A['drv'] = np.where(A.tcd2 <= 500, 'TC', np.where(A.ar0, 'AR', 'other')); A['d0'] = pd.to_datetime(A.date0)
out = []
for s, (a, b) in win.items():
    g = L[(L.src == s) & (L.date >= a) & (L.date <= b + '-12-31')].copy().reset_index(drop=True)
    g['dn'] = (g.date - D0).dt.days
    cd = g.groupby(['dn', 'ci', 'cj']).agg(n=('lat', 'size'), ar=('ar', 'max')).reset_index()
    idx = {kk: i for i, kk in enumerate(zip(cd.dn, cd.ci, cd.cj))}; r = []; c = []
    for i, (d, x, y) in enumerate(zip(cd.dn, cd.ci, cd.cj)):
        for dd in (0, 1):
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    j = idx.get((d + dd, x + di, (y + dj) % 360))
                    if j is not None and j != i: r.append(i); c.append(j)
    _, lab = connected_components(coo_matrix((np.ones(len(r)), (r, c)), shape=(len(cd), len(cd))), directed=False)
    cd['ev'] = lab
    g = g.merge(cd[['dn', 'ci', 'cj', 'ev']], on=['dn', 'ci', 'cj'])
    e1 = cd.groupby('ev').agg(ar_frac=('ar', 'mean')).reset_index()
    e2 = g.groupby('ev').agg(n=('lat', 'size'), lat=('lat', 'mean'), lon=('lon', 'mean'), d0=('date', 'min')).reset_index()
    e = e1.merge(e2, on='ev'); e['src'] = s; e['arm'] = e.ar_frac >= 0.5
    a_ = A[A.src == s][['d0', 'n', 'drv', 'lat', 'lon']].rename(columns={'lat': 'la', 'lon': 'lo'})
    m = e.merge(a_, on=['d0', 'n'], how='left')
    m['dd'] = (m.lat - m.la) ** 2 + (m.lon - m.lo) ** 2
    m = m.sort_values('dd').drop_duplicates('ev').sort_values('ev')
    print(s, len(e), 'matched drv', m.drv.notna().mean().round(3), m.drv.value_counts().to_dict())
    m['drv'] = m.drv.fillna(pd.Series(np.where(m.arm, 'AR', 'other'), index=m.index))
    out.append(m[['src', 'ev', 'n', 'lat', 'lon', 'd0', 'ar_frac', 'arm', 'drv']])
pd.concat(out).to_parquet('work/out/s1/ev_points.parquet')
