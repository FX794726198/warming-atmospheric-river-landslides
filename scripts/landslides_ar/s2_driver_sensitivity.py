# Sensitivity of the regional-event drivers to the TC distance, the AR cell rule and the AR window (deduplicated regional events).
import sys; sys.path.insert(0, 'fig_code')
from arcommon import *
S = 'work/out/s1'
A = pd.read_parquet(f'{O}/outbreaks_all.parquet'); X = pd.read_parquet(f'{O}/extra_outbreaks.parquet')
A = A[~(A.src.isin(['GLC', 'GFLD']).values & national_mask(A.lat.values, A.lon.values))]
A = pd.DataFrame(dict(src=A.src, n=A.n, lat=A.lat, lon=A.lon, date=pd.to_datetime(A.date0), ar0=A.ar0, ar1=A.ar1, tcd=A.tcd2, arpk=A.ar0))
X = pd.DataFrame(dict(src=X.source, n=X.n, lat=X.lat, lon=X.lon, date=pd.to_datetime(X.date_peak), ar0=X.ar0, ar1=X.ar1, tcd=X.tcd, arpk=X.ar_peak0 > 0))
B = pd.concat([A, X], ignore_index=True); B = B[B.n >= 20]; B = dedupe(B)
B['cold'] = cold_season(B.lat, B.date.dt.month)
rows = []
for lab, tcd, arc in (('Main: TC within 500 km, AR at the centre cell', 500, 'ar0'), ('TC within 300 km', 300, 'ar0'), ('TC within 800 km', 800, 'ar0'),
                      ('AR anywhere in the 3 × 3 cells', 500, 'ar1'), ('AR on the peak day (event inventories)', 500, 'arpk'), ('No TC rule', -1, 'ar0')):
    drv = np.where((B.tcd <= tcd) & (tcd > 0), 'TC', np.where(B[arc].astype(bool), 'AR', 'other')); q = B.assign(drv=drv)
    pc = q[(q.lat.abs() >= 25) & q.cold]; pw = q[(q.lat.abs() >= 25) & ~q.cold]; tr = q[q.lat.abs() < 25]
    rows.append((lab, (pc.drv == 'AR').mean() * 100, len(pc), (pw.drv == 'AR').mean() * 100, len(pw), (tr.drv == 'AR').mean() * 100, (tr.drv == 'TC').mean() * 100, len(tr)))
SE = pd.DataFrame(rows, columns=['rule', 'cold_ar', 'n_cold', 'warm_ar', 'n_warm', 'trop_ar', 'trop_tc', 'n_trop']); SE.to_csv(f'{S}/driver_sensitivity.csv', index=False)
print(SE.round(0).to_string())
