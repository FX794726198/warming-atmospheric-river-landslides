"""0.25 deg area check as relative probabilities (reconstructs the fp025 logit: big20 ~ arm [+ lfs + ldur] + cold + C(src), clusters inventory x year)."""
import sys; sys.path.insert(0, 'scripts/probability')
import numpy as np, pandas as pd, statsmodels.api as sm, patsy
from probratio import design, ratio
E = pd.read_csv('work/arco/fp025_events.csv')
E = E[E.pooled.astype(bool)].reset_index(drop=True); print(len(E), (E.n >= 20).sum())
d0 = pd.Timestamp('1979-01-01') + pd.to_timedelta(E.d0, 'D'); mo = d0.dt.month; E['yr'] = d0.dt.year
E['cold'] = np.where(E.lat >= 0, mo.isin([10, 11, 12, 1, 2, 3]), mo.isin([4, 5, 6, 7, 8, 9])).astype(int)
E['blk'] = E.src + '_' + E.yr.astype(str); E['big20'] = (E.n >= 20).astype(int)
E['l1'] = np.log1p(E.fp_sim_km2_1deg / 1000); E['l025'] = np.log1p(E.fp_sim_km2_025 / 1000); E['ldur'] = np.log(E.dur)
rows = []
for arm_def, a in [('>=0.5', (E.ar_frac >= 0.5)), ('>0', (E.ar_frac > 0))]:
    E['arm'] = a.astype(int)
    for name, f in [('raw', 'big20 ~ arm + cold + C(src)'), ('1deg', 'big20 ~ arm + l1 + ldur + cold + C(src)'), ('025', 'big20 ~ arm + l025 + ldur + cold + C(src)')]:
        y, X = patsy.dmatrices(f, E, return_type='dataframe'); q = E.loc[X.index]
        m = sm.Logit(y, X).fit(disp=0, cov_type='cluster', cov_kwds={'groups': pd.factorize(q.blk)[0]})
        r = ratio(m.params.values, m.cov_params().values, design(X.design_info, q.assign(arm=1)), design(X.design_info, q.assign(arm=0)))
        r.update(arm_def=arm_def, model=name, n=len(q), OR=np.exp(m.params['arm'])); rows.append(r)
R = pd.DataFrame(rows); print(R.round(3).to_string()); R.to_csv('data/proc/prob/fp025_prob.csv', index=False)
