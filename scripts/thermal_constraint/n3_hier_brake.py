"""NC: hierarchical model of the sub-freezing brake. Cold-season AR episodes with extreme precipitation (2-day rain >= local P99)
at landslide cells; outcome = landslide recorded (hit). Fixed: temperature class, log rain/P99, antecedent 30-day percentile, inventory.
Random: cell intercept, storm-week intercept (crossed), inventory-specific sub-freezing slope (Bayesian mixed GLM, variational Bayes)."""
import numpy as np, pandas as pd, statsmodels.formula.api as smf, scipy.sparse as sp, patsy
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
d = pd.read_parquet('work/out/s1/episodes_drivers.parquet')
x = d[(d.rain_rel >= 1) & d.cold].copy().reset_index(drop=True)
x['tc'] = pd.Categorical(pd.cut(x.t2, [-99, -4, 0, 4, 99], labels=['lt_m4', 'm4_0', 'p0_4', 'ge4']).astype(str), ['ge4', 'p0_4', 'm4_0', 'lt_m4'])
x['lrr'] = np.log(x.rain_rel); x['a30c'] = (x.a30 - 50) / 25
x['cell'] = x.ci.astype(str) + '_' + x.cj.astype(str); x['wk'] = x.t0.dt.strftime('%Y-%U') + x.src
x['sub0'] = (x.t2 < 0).astype(float)
print(len(x), x.cell.nunique(), x.wk.nunique(), flush=True)
rows = []
m0 = smf.logit('hit ~ C(tc) + lrr + a30c + C(src)', x).fit(disp=0, cov_type='cluster', cov_kwds={'groups': pd.factorize(x.wk)[0]})
print(m0.summary2().tables[1].round(3), flush=True)
ci0 = m0.conf_int()
for k in m0.params.index:
    if 'tc' in k: rows.append(dict(model='logit, inventory FE, SE clustered by storm week', term=k, OR=np.exp(m0.params[k]), lo=np.exp(ci0.loc[k, 0]), hi=np.exp(ci0.loc[k, 1])))
# Bayesian mixed GLM with sparse random-effect design
y, X = patsy.dmatrices('hit ~ C(tc) + lrr + a30c + C(src)', x, return_type='dataframe')
blocks = []; ident = []
for gi, col in enumerate(['cell', 'wk']):
    codes, uniq = pd.factorize(x[col]); blocks.append(sp.csr_matrix((np.ones(len(x)), (np.arange(len(x)), codes)), shape=(len(x), len(uniq)))); ident += [gi] * len(uniq)
codes, uniq = pd.factorize(x.src); blocks.append(sp.csr_matrix((x.sub0.values, (np.arange(len(x)), codes)), shape=(len(x), len(uniq)))); ident += [2] * len(uniq)
Z = sp.hstack(blocks).tocsr(); ident = np.array(ident)
m1 = BinomialBayesMixedGLM(y.values.ravel(), X.values, Z, ident, fe_p=2, vcp_p=1, fep_names=list(X.columns), vcp_names=['cell', 'storm_week', 'inventory_subfreezing_slope'], vc_names=None)
r1 = m1.fit_vb(verbose=False)
fe = pd.DataFrame({'mean': r1.fe_mean, 'sd': r1.fe_sd}, index=X.columns); print(fe.round(3), flush=True)
print('random-effect SD (posterior mean of log SD -> SD):', dict(zip(['cell', 'storm_week', 'inv_slope'], np.exp(r1.vcp_mean).round(3))), flush=True)
for k in fe.index:
    if 'tc' in k: rows.append(dict(model='Bayesian mixed GLM: cell + storm week + inventory sub-freezing slope', term=k, OR=np.exp(fe.loc[k, 'mean']), lo=np.exp(fe.loc[k, 'mean'] - 1.96 * fe.loc[k, 'sd']), hi=np.exp(fe.loc[k, 'mean'] + 1.96 * fe.loc[k, 'sd'])))
nslope = len(uniq); vm = r1.vc_mean[-nslope:]; vs = r1.vc_sd[-nslope:]
S = pd.DataFrame({'inventory': uniq, 'slope_dev': vm, 'sd': vs}); print(S.round(3), flush=True)
R = pd.DataFrame(rows); print(R.round(3), flush=True); R.to_csv('work/out/n3/hier_brake.csv', index=False)
S.to_csv('work/out/n3/hier_brake_inventory_slopes.csv', index=False)
