"""AR effect on event size as relative probability (all events set to AR vs all set to non-AR), same models as v6_fp48.py."""
import sys; sys.path.insert(0, 'scripts/probability'); sys.path.insert(0, 'fig_code')
import numpy as np, pandas as pd, statsmodels.api as sm, patsy
from arcommon import pool_mask
from probratio import design, ratio
V_ = 'work/out/v6'; O = 'data/proc/prob'
EF = pd.read_parquet(f'{V_}/events_fp48.parquet'); EF['arm'] = (EF.ar_frac >= 0.5).astype(int)
EF['lfs'] = np.log1p(EF.fp_sim_km2 / 1000); EF['lfu'] = np.log1p(EF.fp_uni_km2 / 1000); EF['lpk'] = np.log(EF.pk.clip(0.05)); EF['ldur'] = np.log(EF.dur)
d0 = pd.Timestamp('1979-01-01') + pd.to_timedelta(EF.d0, 'D'); EF['mo'] = d0.dt.month.astype(str); EF['yr'] = d0.dt.year
EF['lat'] = EF.ci - 70; EF['cold'] = np.where(EF.lat >= 0, EF.mo.astype(int).isin([10, 11, 12, 1, 2, 3]), EF.mo.astype(int).isin([4, 5, 6, 7, 8, 9])).astype(int)
EF['blk'] = EF.src + '_' + EF.yr.astype(str); EF['big5'] = (EF.n >= 5).astype(int); EF['big20'] = (EF.n >= 20).astype(int)
rows = []
for y in ('big5', 'big20'):
    for s in ['NVE', 'USGS', 'ITALICA', 'GLC', 'GFLD', 'ALL']:
        q = (EF[pool_mask(EF)] if s == 'ALL' else EF[EF.src == s]).reset_index(drop=True)
        fe = '+C(src)' if s == 'ALL' else ''; sea = 'C(mo)' if y == 'big5' else 'cold'
        for name, f in [('raw', f'{y}~arm+{sea}{fe}'), ('+duration', f'{y}~arm+ldur+{sea}{fe}'),
                        ('+simultaneous footprint', f'{y}~arm+lfs+{sea}{fe}'), ('+simultaneous footprint+duration', f'{y}~arm+lfs+ldur+{sea}{fe}'),
                        ('+union footprint+duration', f'{y}~arm+lfu+ldur+{sea}{fe}'), ('+peak', f'{y}~arm+lpk+{sea}{fe}'),
                        ('+simultaneous footprint+peak+duration', f'{y}~arm+lfs+lpk+ldur+{sea}{fe}')]:
            r = dict(model=name, outcome=y, src=s, n=len(q), nbig=int(q[y].sum()), share=q[y].mean())
            try:
                yy, X = patsy.dmatrices(f, q, return_type='dataframe'); qq = q.loc[X.index]; di = X.design_info
                m = sm.Logit(yy, X).fit(disp=0, cov_type='cluster', cov_kwds={'groups': pd.factorize(qq.blk)[0]}, maxiter=200)
                rr = ratio(m.params.values, m.cov_params().values, design(di, qq.assign(arm=1)), design(di, qq.assign(arm=0)))
                r.update(rr); r.update(OR=np.exp(m.params['arm']), p=m.pvalues['arm'])
            except Exception as ex: r.update(err=str(ex)[:80])
            rows.append(r)
R = pd.DataFrame(rows); pd.set_option('display.width', 250)
print(R[['outcome', 'src', 'model', 'n', 'nbig', 'P1', 'P0', 'RR', 'lo', 'hi', 'OR']].round(3).to_string()); R.to_csv(f'{O}/ar_size_prob.csv', index=False)
