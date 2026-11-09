"""Per-SD drivers of the AR episode yield (Supplementary Figs. yield c and surface b) as relative probabilities:
mean predicted probability with the variable raised by one SD (or freezing on vs off) over the observed mean prediction."""
import sys; sys.path.insert(0, 'scripts/probability')
import numpy as np, pandas as pd, statsmodels.api as sm, patsy
from probratio import design, ratio
S = 'work/out/s1'; O = 'work/out/g1'
EP = pd.read_parquet(f'{S}/episodes_drivers.parquet')
st = np.load(f'{O}/era5_static.npz'); i, j = EP.ci.values, EP.cj.values
EP['sdor'] = st['standard_deviation_of_orography'][i, j]; EP['elev'] = st['geopotential_at_surface'][i, j] / 9.81; EP['forest'] = st['high_vegetation_cover'][i, j]
def rel(m, X, q, v):
    di = X.design_info; b = m.params.values; V = m.cov_params().values
    if v == 'frz': return ratio(b, V, design(di, q.assign(frz=1)), design(di, q.assign(frz=0)))
    return ratio(b, V, design(di, q.assign(**{v: q[v] + 1.0})), np.asarray(X, float))
out = []
for tag, f, vs in [('yield', 'hit ~ z_ivt_max + z_dur_h + z_rain_rel + z_a30 + z_a7 + frz + C(month)', ('z_ivt_max', 'z_dur_h', 'z_rain_rel', 'z_a30', 'z_a7', 'frz')),
                   ('surface', 'hit ~ z_rain_rel + z_dur_h + z_ivt_max + z_a30 + z_a7 + frz + z_sdor + z_forest + z_elev + C(month)', ('z_rain_rel', 'z_a30', 'z_sdor', 'z_forest', 'z_elev'))]:
    for s in ['NVE', 'USGS', 'ITALICA', 'GLC', 'GFLD']:
        q = EP[EP.src == s].dropna(subset=['a30', 'a7', 't2']).copy()
        if tag == 'yield':
            for v in ('ivt_max', 'dur_h', 'rain_rel', 'a30', 'a7', 't2'):
                x = np.log(q[v].clip(lower=1e-3)) if v in ('ivt_max', 'dur_h', 'rain_rel') else q[v]; q['z_' + v] = (x - x.mean()) / x.std()
        else:
            for v in ('rain_rel', 'dur_h', 'ivt_max'): x = np.log(q[v].clip(lower=1e-3)); q['z_' + v] = (x - x.mean()) / x.std()
            for v in ('a30', 'a7', 'forest', 'elev'): q['z_' + v] = (q[v] - q[v].mean()) / q[v].std()
            x = np.log(q.sdor.clip(lower=1e-4)); q['z_sdor'] = (x - x.mean()) / x.std()
        q['frz'] = (q.t2 < 0).astype(int); q = q.reset_index(drop=True)
        y, X = patsy.dmatrices(f, q, return_type='dataframe'); qq = q.loc[X.index]
        m = sm.Logit(y, X).fit(disp=0)
        for v in vs:
            r = rel(m, X, qq, v); r.update(fig=tag, src=s, var=v, OR=np.exp(m.params[v]), n=len(qq), base=qq.hit.mean()); out.append(r)
R = pd.DataFrame(out); pd.set_option('display.width', 200); print(R[['fig', 'src', 'var', 'n', 'base', 'RR', 'lo', 'hi', 'OR']].round(3).to_string())
R.to_csv('data/proc/prob/episode_prob.csv', index=False)
