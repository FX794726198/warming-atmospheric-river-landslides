# Supplementary analyses for the AR-landslide CEE draft.
# (1) AR share by event size under four clustering rules; (2) rain-matched landslide odds on AR days (Mantel-Haenszel, strata = rain bin x month);
# (3) AR episode yield by AR category; (4) sensitivity of the regional-event drivers to the TC distance, the AR window and the AR cell rule.
import numpy as np, pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
O = 'work/out/g1'; S = 'work/out/s1'
import os; os.makedirs(S, exist_ok=True)
T0 = pd.Timestamp('1979-01-01')
L = pd.read_parquet(f'{O}/ls_ar.parquet')
L = L[~((L.src == 'USGS') & L.inv.isin(['USGS Earthquake-Triggered Ground Failure', 'USGS Seismogenic Mass Movements']))]
win = {'NVE': ('2000', '2022'), 'USGS': ('1979', '2024'), 'ITALICA': ('1996', '2021'), 'GLC': ('2007', '2017'), 'GFLD': ('2004', '2017')}

# ---------------- (1) clustering rules
def events(g, dday, half):
    cd = g.groupby(['dn', 'ci', 'cj']).agg(n=('lat', 'size'), ar=('ar', 'max'), pc=('pc', 'mean')).reset_index()
    idx = {k: i for i, k in enumerate(zip(cd.dn, cd.ci, cd.cj))}; r = []; c = []
    for i, (d, x, y) in enumerate(zip(cd.dn, cd.ci, cd.cj)):
        for dd in range(0, dday + 1):
            for di in range(-half, half + 1):
                for dj in range(-half, half + 1):
                    j = idx.get((d + dd, x + di, (y + dj) % 360))
                    if j is not None and j != i: r.append(i); c.append(j)
    A = coo_matrix((np.ones(len(r)), (r, c)), shape=(len(cd), len(cd)))
    _, lab = connected_components(A, directed=False); cd['ev'] = lab
    return cd.groupby('ev').agg(n=('n', 'sum'), ar_frac=('ar', 'mean'), pc=('pc', 'mean')).reset_index()
rules = {'same day, same cell': (0, 0), '±1 day, 3 × 3 cells (main)': (1, 1), '±1 day, 5 × 5 cells': (1, 2), '±2 days, 3 × 3 cells': (2, 1)}
rows = []
for s, (a, b) in win.items():
    g = L[(L.src == s) & (L.date >= a) & (L.date <= b + '-12-31')].copy(); g['dn'] = (g.date - T0).dt.days
    for rn, (dd, hh) in rules.items():
        e = events(g, dd, hh); e['arm'] = e.ar_frac >= 0.5
        e['sz'] = pd.cut(e.n, [0, 1, 4, 19, 1e9], labels=['1', '2–4', '5–19', '≥20'])
        for sz, q in e.groupby('sz', observed=True):
            rows.append((s, rn, sz, len(q), q.arm.mean() * 100, q.pc.mean() * 100))
CL = pd.DataFrame(rows, columns=['src', 'rule', 'size', 'n', 'ar', 'pc']); CL.to_csv(f'{S}/cluster_rules.csv', index=False)
print(CL.pivot_table(index=['src', 'size'], columns='rule', values='ar').round(0).to_string())

# ---------------- (2) rain-matched odds (cell-days at cells with landslides)
z = np.load(f'{O}/panel.npz'); ci, cj, W, R2 = z['ci'], z['cj'], z['W'], z['R2'].astype(np.float32)
days = pd.date_range('1979-01-01', '2022-12-31'); mon = days.month.values
col = {(a, b): k for k, (a, b) in enumerate(zip(ci, cj))}
rb = [0, 5, 10, 20, 40, 80, 1e9]; rlab = ['<5', '5–10', '10–20', '20–40', '40–80', '≥80']
rows = []; mh = []
for s, (a, b) in win.items():
    g = L[(L.src == s) & (L.date >= a) & (L.date <= min(b, '2022') + '-12-31')]
    cols = sorted({col[(x, y)] for x, y in zip(g.ci, g.cj) if (x, y) in col})
    t0 = (pd.Timestamp(a + '-01-01') - T0).days; t1 = (pd.Timestamp(min(b, '2022') + '-12-31') - T0).days + 1
    Y = np.zeros((t1 - t0, len(cols)), bool); cpos = {c: k for k, c in enumerate(cols)}
    for x, y, d in zip(g.ci, g.cj, g.date):
        k = cpos.get(col.get((x, y)));
        if k is not None: Y[(d - T0).days - t0, k] = True
    w = W[t0:t1][:, cols]; r = R2[t0:t1][:, cols]; m = np.repeat(mon[t0:t1][:, None], len(cols), 1)
    bi = np.digitize(r, rb) - 1
    num = den = 0.0
    for j in range(6):
        sel = bi == j
        a1 = (Y & w & sel).sum(); b1 = (~Y & w & sel).sum(); c1 = (Y & ~w & sel).sum(); d1 = (~Y & ~w & sel).sum()
        p_ar = a1 / max(a1 + b1, 1); p_no = c1 / max(c1 + d1, 1)
        rows.append((s, rlab[j], a1 + c1, p_ar, p_no, (a1 * d1) / max(b1 * c1, 1)))
        for mm in range(1, 13):
            ss = sel & (m == mm)
            A_ = (Y & w & ss).sum(); B_ = (~Y & w & ss).sum(); C_ = (Y & ~w & ss).sum(); D_ = (~Y & ~w & ss).sum(); N_ = A_ + B_ + C_ + D_
            if N_ > 0: num += A_ * D_ / N_; den += B_ * C_ / N_
    mh.append((s, num / den))
RM = pd.DataFrame(rows, columns=['src', 'rain', 'nls_days', 'p_ar', 'p_no', 'OR']); RM.to_csv(f'{S}/rain_matched.csv', index=False)
MH = pd.DataFrame(mh, columns=['src', 'OR_MH']); MH.to_csv(f'{S}/rain_matched_mh.csv', index=False)
print(RM.round(5).to_string()); print(MH.round(2))

# ---------------- (3) yield by AR category
rows = []
for s in win:
    ep = pd.read_parquet(f'{O}/ep_{s}.parquet')
    for c, q in ep.groupby('cat'):
        rows.append((s, int(c), len(q), (q.nls > 0).mean() * 100, (q.nls >= 5).mean() * 100))
YD = pd.DataFrame(rows, columns=['src', 'cat', 'episodes', 'p_any', 'p5']); YD.to_csv(f'{S}/yield.csv', index=False)
print(YD.round(2).to_string())

# ---------------- (4) driver sensitivity for regional events
A = pd.read_parquet(f'{O}/outbreaks_all.parquet'); X = pd.read_parquet(f'{O}/extra_outbreaks.parquet')
A = A[['src', 'n', 'lat', 'lon', 'date0', 'ar0', 'ar1', 'tcd2']].rename(columns={'tcd2': 'tcd'})
A['arpk'] = A.ar0  # inventory events: peak day not separable at daily resolution; keep window flag
X = pd.DataFrame(dict(src=X.source, n=X.n, lat=X.lat, lon=X.lon, date0=pd.to_datetime(X.date_peak), ar0=X.ar0, ar1=X.ar1, tcd=X.tcd, arpk=X.ar_peak0 > 0))
B = pd.concat([A, X], ignore_index=True); B = B[B.n >= 20]
m = pd.to_datetime(B.date0).dt.month
B['cold'] = np.where(B.lat >= 0, m.isin([10, 11, 12, 1, 2, 3]), m.isin([4, 5, 6, 7, 8, 9]))
rows = []
for lab, tcd, arc in (('Main: TC 500 km, AR at centre cell', 500, 'ar0'), ('TC 300 km', 300, 'ar0'), ('TC 800 km', 800, 'ar0'),
                      ('AR in 3 × 3 cells', 500, 'ar1'), ('AR on peak day only (event inventories)', 500, 'arpk'), ('No TC rule (AR first)', -1, 'ar0')):
    drv = np.where((B.tcd <= tcd) & (tcd > 0), 'TC', np.where(B[arc], 'AR', 'other'))
    q = B.assign(drv=drv)
    pc = q[(q.lat.abs() >= 25) & q.cold]; pw = q[(q.lat.abs() >= 25) & ~q.cold]; tr = q[q.lat.abs() < 25]
    rows.append((lab, (pc.drv == 'AR').mean() * 100, len(pc), (pw.drv == 'AR').mean() * 100, len(pw), (tr.drv == 'AR').mean() * 100, (tr.drv == 'TC').mean() * 100, len(tr)))
SE = pd.DataFrame(rows, columns=['rule', 'cold_ar', 'n_cold', 'warm_ar', 'n_warm', 'trop_ar', 'trop_tc', 'n_trop']); SE.to_csv(f'{S}/driver_sensitivity.csv', index=False)
print(SE.round(0).to_string())
