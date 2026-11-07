"""Reference for the growth of the heavy-rain area: the control run's largest 48-h rain at every cell scaled by 7% per degree of the
simulated warming (phase and pattern unchanged), against the WRF warmer runs. Same land mask and edge as all WRF statistics."""
import numpy as np, pandas as pd
EDGE = 10; WP = 'work/data/wrf/post_fig'
L = pd.read_csv('work/out/n7/levels_summary.csv')
rows = []
for ev in ['NOR_2011', 'BC_2021', 'LIG_2014', 'ORE_1996', 'RSB_2024']:
    z = np.load(f'{WP}/{ev}/{ev}_ctl.npz'); g = lambda k: z[k][EDGE:-EDGE, EDGE:-EDGE].astype(float)
    land = g('landmask') > 0.5; A = g('cell_area_km2'); r48 = g('liq_max48h'); p48 = g('precip_max48h')
    a0 = A[land & (r48 >= 100)].sum()
    q = L[L.event == ev].sort_values('d02_dT2m_land')
    for r in q.itertuples():
        f = 1.07 ** r.d02_dT2m_land
        rows.append(dict(event=ev, run=r.run, dT=r.d02_dT2m_land, A_wrf=r.area_liq48_ge100, A_cc=A[land & (r48 * f >= 100)].sum(), A_ccall=A[land & (p48 * f >= 100)].sum(), A0_check=a0))
D = pd.DataFrame(rows); D.to_csv('work/nc/proc/rev/cc_area.csv', index=False)
out = []
for ev, q in D.groupby('event', sort=False):
    rate = lambda a: (np.exp(np.polyfit(q.dT, np.log(np.maximum(a, 1.0)), 1)[0]) - 1) * 100
    out.append(dict(event=ev, wrf=rate(q.A_wrf), cc=rate(q.A_cc), ccall=rate(q.A_ccall), A0=q.A_wrf.iloc[0], A0_npz=q.A0_check.iloc[0], full_wrf=q.A_wrf.iloc[-1], full_cc=q.A_cc.iloc[-1], add_wrf=q.A_wrf.iloc[-1] - q.A_wrf.iloc[0], add_cc=q.A_cc.iloc[-1] - q.A_cc.iloc[0]))
O = pd.DataFrame(out); print(O.round(1).to_string()); O.to_csv('work/nc/proc/rev/cc_rates.csv', index=False)
print(D.round(1).to_string())
