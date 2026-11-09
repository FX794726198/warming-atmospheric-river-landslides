"""Thermal constraint as relative probabilities: classes vs >=4 C (four models), halving the precipitation,
within-cell spline vs 6 C, per inventory below vs above 0 C. Same samples and models as brake_robust.py / n3c_cellfe.py."""
import sys; sys.path.insert(0, 'scripts/probability')
import numpy as np, pandas as pd, statsmodels.api as sm, patsy
from probratio import design, ratio, prob
O = 'data/proc/prob'
d = pd.read_parquet('work/out/s1/episodes_drivers.parquet')
x = d[(d.rain_rel >= 1) & d.cold].copy()
CATS = ['ge4', 'p0_4', 'm4_0', 'lt_m4']
x['tc'] = pd.Categorical(pd.cut(x.t2, [-99, -4, 0, 4, 99], labels=['lt_m4', 'm4_0', 'p0_4', 'ge4']).astype(str), CATS)
x['lrr'] = np.log(x.rain_rel); x['a30c'] = (x.a30 - 50) / 25; x['ldur'] = np.log(x.dur_h + 6.0)
x['cell'] = x.ci.astype(str) + '_' + x.cj.astype(str); x['wk'] = x.t0.dt.strftime('%Y-%U') + x.src
g = x.groupby('cell').hit.agg(['min', 'max']); keep = g.index[(g['min'] == 0) & (g['max'] == 1)]
x = x[x.cell.isin(keep)].reset_index(drop=True)
print('episodes', len(x), 'cells', x.cell.nunique(), 'hits', x.hit.sum(), 'rate %.4f' % x.hit.mean(), flush=True)
cl = pd.factorize(x.wk)[0]
def fit(f, z=x, grp=None):
    y, X = patsy.dmatrices(f, z, return_type='dataframe')
    m = sm.Logit(y, X).fit(disp=0, method='lbfgs', maxiter=5000, cov_type='cluster', cov_kwds={'groups': cl if grp is None else grp})
    return m, X
rows = []; halves = []
MODELS = [('cell FE (paper)', 'hit ~ C(tc) + lrr + a30c + C(cell)'),
                ('+ month FE', 'hit ~ C(tc) + lrr + a30c + C(month) + C(cell)'),
                ('+ month FE + duration', 'hit ~ C(tc) + lrr + a30c + ldur + C(month) + C(cell)'),
                ('+ month FE + duration + year', 'hit ~ C(tc) + lrr + a30c + ldur + I((t0.dt.year - 2000) / 10) + C(month) + C(cell)')]
for name, f in (MODELS if 'skipclass' not in sys.argv else []):
    m, X = fit(f); b = m.params.values; V = m.cov_params().values; di = X.design_info
    Xc = {c: design(di, x.assign(tc=pd.Categorical([c] * len(x), CATS))) for c in CATS}
    for c in CATS:
        r = ratio(b, V, Xc[c], Xc['ge4']); P, plo, phi = prob(b, V, Xc[c])
        k = [kk for kk in X.columns if kk == f'C(tc)[T.{c}]']
        orr = np.exp(m.params[k[0]]) if k else 1.0
        rows.append(dict(model=name, term=c, P=P, Plo=plo, Phi=phi, RR=r['RR'], lo=r['lo'], hi=r['hi'], OR=orr))
    # halving the precipitation relative to P99 (observed episodes, all else as observed)
    X1 = design(di, x.assign(lrr=x.lrr - np.log(2))); X0 = np.asarray(X, float)
    r = ratio(b, V, X1, X0); r.update(model=name, coef=m.params['lrr'], OR=0.5 ** m.params['lrr']); halves.append(r)
    print(name, 'halving: RR %.2f (%.2f-%.2f), OR %.2f' % (r['RR'], r['lo'], r['hi'], r['OR']), flush=True)
if rows:
    R = pd.DataFrame(rows); print(R.round(4).to_string()); R.to_csv(f'{O}/brake_prob.csv', index=False)
    pd.DataFrame(halves).to_csv(f'{O}/brake_halving_prob.csv', index=False)
# within-cell spline, relative probability vs 6 C
f = "hit ~ cr(t2, knots=(-8, -4, -2, 0, 2, 4, 8), lower_bound=-15, upper_bound=15, constraints='center') + lrr + a30c + ldur + C(month) + C(cell)"
z = x[(x.t2 > -15) & (x.t2 < 15)].reset_index(drop=True)
m, X = fit(f, z, pd.factorize(z.wk)[0]); b = m.params.values; V = m.cov_params().values; di = X.design_info
X6 = design(di, z.assign(t2=6.0)); out = []
for t in (-8, -6, -4, -2, 0, 2, 4):
    r = ratio(b, V, design(di, z.assign(t2=float(t))), X6); r.update(t2=t); out.append(r)
CV = pd.DataFrame(out); print("within-cell spline vs 6 C (n=%d)" % len(z)); print(CV.round(3).to_string()); CV.to_csv(f'{O}/brake_curve_cellfe_prob.csv', index=False)
# per inventory, below vs above 0 C, cell FE, same covariates as the paper's per-inventory models
inv = []
for s_ in ['NVE', 'USGS', 'ITALICA', 'GLC', 'GFLD']:
    zz = x[x.src == s_].reset_index(drop=True).copy(); zz['frz'] = (zz.t2 < 0).astype(float)
    m2, X2 = fit('hit ~ frz + lrr + a30c + C(cell)', zz, pd.factorize(zz.wk)[0]); b2 = m2.params.values; V2 = m2.cov_params().values
    di2 = X2.design_info
    r = ratio(b2, V2, design(di2, zz.assign(frz=1.0)), design(di2, zz.assign(frz=0.0)))
    r.update(src=s_, n=len(zz), cells=zz.cell.nunique(), OR=np.exp(m2.params['frz']), p=m2.pvalues['frz']); inv.append(r)
I = pd.DataFrame(inv); print(I.round(3).to_string()); I.to_csv(f'{O}/brake_inv_prob.csv', index=False)
