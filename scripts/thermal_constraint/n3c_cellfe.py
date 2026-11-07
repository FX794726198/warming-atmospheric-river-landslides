"""Sub-freezing brake with cell fixed effects (each cell compared with itself), SE clustered by storm week."""
import numpy as np, pandas as pd, statsmodels.api as sm, patsy
d = pd.read_parquet('work/out/s1/episodes_drivers.parquet')
x = d[(d.rain_rel >= 1) & d.cold].copy()
x['tc'] = pd.Categorical(pd.cut(x.t2, [-99, -4, 0, 4, 99], labels=['lt_m4', 'm4_0', 'p0_4', 'ge4']).astype(str), ['ge4', 'p0_4', 'm4_0', 'lt_m4'])
x['lrr'] = np.log(x.rain_rel); x['a30c'] = (x.a30 - 50) / 25
x['cell'] = x.ci.astype(str) + '_' + x.cj.astype(str); x['wk'] = x.t0.dt.strftime('%Y-%U') + x.src
g = x.groupby('cell').hit.agg(['min', 'max']); keep = g.index[(g['min'] == 0) & (g['max'] == 1)]
x = x[x.cell.isin(keep)].reset_index(drop=True)
print('episodes', len(x), 'cells', x.cell.nunique(), 'hits', x.hit.sum(), flush=True)
y, X = patsy.dmatrices('hit ~ C(tc) + lrr + a30c + C(cell)', x, return_type='dataframe')
m = sm.Logit(y, X).fit(disp=0, method='lbfgs', maxiter=3000, cov_type='cluster', cov_kwds={'groups': pd.factorize(x.wk)[0]})
rows = []
for k in [c for c in X.columns if c.startswith('C(tc)') or c in ('lrr', 'a30c')]:
    b = m.params[k]; s = m.bse[k]; rows.append(dict(term=k, OR=np.exp(b), lo=np.exp(b - 1.96 * s), hi=np.exp(b + 1.96 * s), p=m.pvalues[k]))
R = pd.DataFrame(rows); print(R.round(3)); R.to_csv('work/out/n3/brake_cellfe.csv', index=False)
# by inventory
for s_ in sorted(x.src.unique()):
    z = x[x.src == s_]
    try:
        y2, X2 = patsy.dmatrices('hit ~ I(t2 < 0) + lrr + a30c + C(cell)', z, return_type='dataframe')
        m2 = sm.Logit(y2, X2).fit(disp=0, method='lbfgs', maxiter=3000, cov_type='cluster', cov_kwds={'groups': pd.factorize(z.wk)[0]})
        k = [c for c in X2.columns if c.startswith('I(t2')][0]; b = m2.params[k]; s = m2.bse[k]
        print(s_, len(z), z.cell.nunique(), 'OR(<0 vs >=0) %.2f (%.2f-%.2f)' % (np.exp(b), np.exp(b - 1.96 * s), np.exp(b + 1.96 * s)), flush=True)
    except Exception as e: print(s_, 'fail', e)
