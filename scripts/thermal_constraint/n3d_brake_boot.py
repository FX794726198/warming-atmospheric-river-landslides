"""Cluster bootstrap (storm weeks) of the pooled brake curve in n3b: 95% band of the landslide probability relative to 6 C."""
import numpy as np, pandas as pd, statsmodels.formula.api as smf, warnings
warnings.filterwarnings('ignore')
d = pd.read_parquet('work/out/s1/episodes_drivers.parquet')
x = d[(d.rain_rel >= 1) & d.cold & d.t2.between(-15, 15)].copy()
x['lrr'] = np.log(x.rain_rel); x['a30c'] = (x.a30 - 50) / 25; x['wk'] = x.t0.dt.strftime('%Y-%U') + x.src
g = np.arange(-12, 12.01, 0.5); srcs = sorted(x.src.unique()); w = np.array([(x.src == s).mean() for s in srcs])
def curve(df):
    m = smf.logit('hit ~ cr(t2, df=6) + lrr + a30c + C(src)', df).fit(disp=0, maxiter=200)
    P = np.sum([m.predict(pd.DataFrame(dict(t2=g, src=s, lrr=0.0, a30c=0.0))).values * wi for s, wi in zip(srcs, w)], 0)
    return P / P[np.argmin(np.abs(g - 6))]
rng = np.random.default_rng(1); wk = x.wk.values; uw = np.unique(wk); groups = {k: np.flatnonzero(wk == k) for k in uw}
B = []
for b in range(200):
    pick = rng.choice(uw, len(uw), replace=True); idx = np.concatenate([groups[k] for k in pick])
    try: B.append(curve(x.iloc[idx]))
    except Exception as e: pass
B = np.array(B); print('reps', len(B))
R = pd.DataFrame(dict(t2=g, rel=curve(x), lo=np.percentile(B, 2.5, 0), hi=np.percentile(B, 97.5, 0)))
R.to_csv('work/out/n3/brake_curve_boot.csv', index=False)
print(R[R.t2.isin([-8, -4, -2, 0, 2, 4, 6, 8, 12])].round(2).to_string())
