"""Reviewer check: the sub-freezing constraint with cell fixed effects plus calendar-month fixed effects and episode duration,
and the continuous temperature curve estimated in the same sample with the same controls."""
import numpy as np, pandas as pd, statsmodels.api as sm, patsy
d = pd.read_parquet('work/out/s1/episodes_drivers.parquet')
x = d[(d.rain_rel >= 1) & d.cold].copy()
x['tc'] = pd.Categorical(pd.cut(x.t2, [-99, -4, 0, 4, 99], labels=['lt_m4', 'm4_0', 'p0_4', 'ge4']).astype(str), ['ge4', 'p0_4', 'm4_0', 'lt_m4'])
x['lrr'] = np.log(x.rain_rel); x['a30c'] = (x.a30 - 50) / 25; x['ldur'] = np.log(x.dur_h + 6.0)
x['cell'] = x.ci.astype(str) + '_' + x.cj.astype(str); x['wk'] = x.t0.dt.strftime('%Y-%U') + x.src
g = x.groupby('cell').hit.agg(['min', 'max']); keep = g.index[(g['min'] == 0) & (g['max'] == 1)]
x = x[x.cell.isin(keep)].reset_index(drop=True)
print('episodes', len(x), 'cells', x.cell.nunique(), 'hits', x.hit.sum(), flush=True)
cl = pd.factorize(x.wk)[0]
def fit(f, z=x, grp=None):
    y, X = patsy.dmatrices(f, z, return_type='dataframe')
    return sm.Logit(y, X).fit(disp=0, method='lbfgs', maxiter=5000, cov_type='cluster', cov_kwds={'groups': cl if grp is None else grp}), X
rows = []
for name, f in [('cell FE (paper)', 'hit ~ C(tc) + lrr + a30c + C(cell)'),
                ('+ month FE', 'hit ~ C(tc) + lrr + a30c + C(month) + C(cell)'),
                ('+ month FE + duration', 'hit ~ C(tc) + lrr + a30c + ldur + C(month) + C(cell)'),
                ('+ month FE + duration + year', 'hit ~ C(tc) + lrr + a30c + ldur + I((t0.dt.year - 2000) / 10) + C(month) + C(cell)')]:
    m, X = fit(f)
    for k in [c for c in X.columns if c.startswith('C(tc)')]:
        b, s = m.params[k], m.bse[k]; rows.append(dict(model=name, term=k.split('T.')[1].rstrip(']'), OR=np.exp(b), lo=np.exp(b - 1.96 * s), hi=np.exp(b + 1.96 * s), p=m.pvalues[k]))
    if 'ldur' in X.columns: print(name, 'ldur OR per log unit %.2f' % np.exp(m.params['ldur']), flush=True)
R = pd.DataFrame(rows); print(R.round(3).to_string()); R.to_csv('work/nc/proc/rev/brake_robust.csv', index=False)
# continuous curve in the same sample, with cell and month fixed effects and duration
f = "hit ~ cr(t2, knots=(-8, -4, -2, 0, 2, 4, 8), lower_bound=-15, upper_bound=15) + lrr + a30c + ldur + C(month) + C(cell)"
z = x[(x.t2 > -15) & (x.t2 < 15)].reset_index(drop=True)
m, X = fit(f, z, pd.factorize(z.wk)[0])
di = patsy.dmatrix(X.design_info, z.iloc[[0]].assign(t2=0.0), return_type='dataframe')
cols = [c for c in X.columns if c.startswith('cr(')]
def row(t):
    r = patsy.build_design_matrices([X.design_info], z.iloc[[0]].assign(t2=float(t)), return_type='dataframe')[0]; return r[cols].values[0]
out = []
for t in (-8, -6, -4, -2, 0, 2, 4):
    dv = row(t) - row(6.0); b = dv @ m.params[cols].values; v = dv @ m.cov_params().loc[cols, cols].values @ dv; s = np.sqrt(v)
    out.append(dict(t2=t, OR_vs_6=np.exp(b), lo=np.exp(b - 1.96 * s), hi=np.exp(b + 1.96 * s)))
C = pd.DataFrame(out); print('curve, cell+month FE, same sample (n=%d)' % len(z)); print(C.round(3).to_string()); C.to_csv('work/nc/proc/rev/brake_curve_cellfe.csv', index=False)
# support of each temperature class
S = x.groupby('tc', observed=True).agg(episodes=('hit', 'size'), landslides=('hit', 'sum'), cells=('cell', 'nunique')).reset_index()
comp = x.groupby(['tc', 'src'], observed=True).size().unstack(fill_value=0)
S = S.merge(comp, left_on='tc', right_index=True); print(S.to_string()); S.to_csv('work/nc/proc/rev/brake_support.csv', index=False)
