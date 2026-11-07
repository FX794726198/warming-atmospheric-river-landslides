"""Phase of AR extreme precipitation over land 20-72 deg, ERA5 0.25 deg (ARCO), 1940-2024.
AR extreme cell-day: 2-day precipitation (day + previous day) >= the cell's 99th percentile of 2-day precipitation over all days
of the base period, and an AR (Chen et al. 2026, daily, 1 deg, nearest cell) on the day or the day before.
Rain fraction = 1 - 2-day snowfall / 2-day precipitation. Season year: NH cold season Oct(Y-1)-Mar(Y) -> Y; SH cold season Apr-Sep(Y) -> Y.
Outputs per season-year, season (0 cold, 1 warm) and cell: counts of AR extreme cell-days (nar), with rain fraction < 0.5 (ns50),
< 0.8 (ns80), sums of 2-day precipitation and snowfall on those days (ptot, psnow), and the same for all extreme cell-days (nex, nexs50)."""
import numpy as np, os, sys, datetime as dt
D = os.environ.get('NDIR', 'work/out/n4'); ARD = os.environ.get('ARDDIR', 'work/data/arcat'); ARD2 = os.environ.get('ARD_OLD', 'work/data/arcat_old')
Y0, Y1 = int(os.environ.get('Y0', 1940)), int(os.environ.get('Y1', 2024)); BASE = os.environ.get('BASE', 'all')
C = np.load(f'{D}/cells.npz'); idx = C['idx']; nlon = len(C['lon'])
la = C['lat'][idx // nlon]; lo = C['lon'][idx // nlon * 0 + idx % nlon]; nc = len(idx)
ai = np.clip(np.round(la).astype(int) + 70, 0, 145); aj = np.round(lo).astype(int) % 360
years = [y for y in range(Y0, Y1 + 1) if os.path.exists(f'{D}/daily_{y}.npz')]
print('years', years[0], '-', years[-1], len(years), flush=True)
def ar_year(y):
    for p in (f'{ARD}/ard_{y}.npy', f'{ARD2}/ard_{y}.npy'):
        if os.path.exists(p):
            a = np.load(p)
            nd = (dt.date(y + 1, 1, 1) - dt.date(y, 1, 1)).days
            if a.dtype == np.uint8 and a.shape[0] < nd: a = np.unpackbits(a, axis=0)[:nd]
            return a.astype(bool)[:, ai, aj]
    return None
# ---- pass 1: P99 of 2-day precipitation per cell (cell blocks to bound memory)
p99f = f'{D}/p99_{BASE}.npy'
if not os.path.exists(p99f):
    bys = years if BASE == 'all' else [y for y in years if 1979 <= y <= 2022]
    P99 = np.zeros(nc, np.float32); B = 30000
    for b0 in range(0, nc, B):
        b1 = min(nc, b0 + B); tp = np.concatenate([np.load(f'{D}/daily_{y}.npz')['tp'][:, b0:b1] for y in bys]).astype(np.float32) / 10
        r2 = tp.copy(); r2[1:] += tp[:-1]; P99[b0:b1] = np.percentile(r2[1:], 99, axis=0); print('p99 block', b0, flush=True)
    np.save(p99f, P99)
P99 = np.load(p99f)
# ---- pass 2: per year
ny = Y1 - Y0 + 2
out = {k: np.zeros((ny, 2, nc), np.int16) for k in ('nar', 'ns50', 'ns80', 'nex', 'nexs50')}
out.update({k: np.zeros((ny, 2, nc), np.float32) for k in ('ptot', 'psnow')})
prev_tp = prev_sf = prev_ar = None
for y in years:
    z = np.load(f'{D}/daily_{y}.npz'); tp = z['tp'].astype(np.float32) / 10; sf = z['sf'].astype(np.float32) / 10
    ar = ar_year(y)
    if ar is None: print('no AR for', y); prev_tp = None; continue
    days = [dt.date(y, 1, 1) + dt.timedelta(days=i) for i in range(len(tp))]
    T2 = tp.copy(); S2 = sf.copy(); A2 = ar.copy()
    T2[1:] += tp[:-1]; S2[1:] += sf[:-1]; A2[1:] |= ar[:-1]
    if prev_tp is not None: T2[0] += prev_tp; S2[0] += prev_sf; A2[0] |= prev_ar
    ex = T2 >= np.maximum(P99, 1.0); axe = ex & A2
    with np.errstate(invalid='ignore', divide='ignore'): rf = 1 - S2 / np.where(T2 > 0, T2, np.nan)
    mon = np.array([d.month for d in days])
    for i, d in enumerate(days):
        m = d.month
        # season and season-year per hemisphere
        nh = la > 0
        cold = np.where(nh, m in (10, 11, 12, 1, 2, 3), m in (4, 5, 6, 7, 8, 9))
        sy = np.where(nh & (m >= 10), y + 1, y) - Y0
        s = np.where(cold, 0, 1)
        e = ex[i]; a = axe[i]
        if not e.any(): continue
        r = rf[i]
        for key, mask in (('nex', e), ('nexs50', e & (r < 0.5)), ('nar', a), ('ns50', a & (r < 0.5)), ('ns80', a & (r < 0.8))):
            k = np.flatnonzero(mask); np.add.at(out[key], (sy[k], s[k], k), 1)
        k = np.flatnonzero(a); np.add.at(out['ptot'], (sy[k], s[k], k), T2[i, k]); np.add.at(out['psnow'], (sy[k], s[k], k), S2[i, k])
    prev_tp, prev_sf, prev_ar = tp[-1], sf[-1], ar[-1]
    print(y, 'AR extreme cell-days', int(axe.sum()), flush=True)
np.savez_compressed(f'{D}/snow_counts_{BASE}_{Y0}_{Y1}.npz', y0=Y0, la=la, lo=lo, zs=C['zs'], p99=P99, **out)
print('saved', flush=True)
