"""Change in the snow-dominated share of cold-season AR extreme days (1941-1970 -> 1995-2024) by the initial share of each 1-deg cell.
Uses OBS_CELLS.csv (cells grouped by an independent period, 1971-1994) when present, otherwise map_change_1deg.csv grouped by 1941-1970."""
import numpy as np, pandas as pd, os
rng = np.random.default_rng(1)
P = 'work/nc/proc/cluster/OBS_CELLS.csv'
if os.path.exists(P):
    M = pd.read_csv(P); M = M[(M.n_4170 >= 20) & (M.n_7194 >= 20) & (M.n_9524 >= 20)]
    M['s0'] = M.s_7194; M['d'] = M.s_9524 - M.s_4170; src = 'OBS_CELLS (grouped by 1971–1994)'
else:
    M = pd.read_csv('work/out/n7/map_change_1deg.csv'); M = M[(M.n1 >= 20) & (M.n2 >= 20)]
    M['s0'] = M.s1; M['d'] = M.s2 - M.s1; src = 'prototype (grouped by 1941–1970)'
M['w'] = M.mount_area_km2 if 'mount_area_km2' in M else np.cos(np.deg2rad(M.lat))
bins = [0, 5, 20, 40, 60, 80, 95, 100.01]; labs = ['0–5', '5–20', '20–40', '40–60', '60–80', '80–95', '95–100']
rows = []
for hemi, sel in [('NH', M.lat >= 25), ('SH', M.lat <= -25)]:
    d = M[sel].copy(); d['b'] = pd.cut(d.s0, bins, right=False, include_lowest=True, labels=labs)
    for b, x in d.groupby('b', observed=True):
        m = np.average(x.d, weights=x.w)
        bs = [np.average(x.d.values[i], weights=x.w.values[i]) for i in (rng.integers(0, len(x), len(x)) for _ in range(1000))]
        rows.append(dict(hemi=hemi, bin=b, n=len(x), dmean=m, lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5), frac_down=(x.d < 0).mean()))
R = pd.DataFrame(rows); R.to_csv('obs_bins.csv', index=False); print(src); print(R.round(2).to_string())
