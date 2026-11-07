"""Landslide probability per cold-season AR extreme-precipitation episode as a smooth function of 2-day mean T2m
(natural cubic spline, df=6), adjusted for rain/P99, antecedent rain and inventory; per-inventory curves too."""
import numpy as np, pandas as pd, statsmodels.formula.api as smf
d = pd.read_parquet('work/out/s1/episodes_drivers.parquet')
x = d[(d.rain_rel >= 1) & d.cold & d.t2.between(-15, 15)].copy()
x['lrr'] = np.log(x.rain_rel); x['a30c'] = (x.a30 - 50) / 25; x['wk'] = x.t0.dt.strftime('%Y-%U') + x.src
m = smf.logit('hit ~ cr(t2, df=6) + lrr + a30c + C(src)', x).fit(disp=0, cov_type='cluster', cov_kwds={'groups': pd.factorize(x.wk)[0]})
g = np.arange(-12, 12.01, 0.5); rows = []
for s in ['pooled'] + sorted(x.src.unique()):
    sub = x if s == 'pooled' else x[x.src == s]
    mm = m if s == 'pooled' else smf.logit('hit ~ cr(t2, df=4) + lrr + a30c', sub).fit(disp=0)
    base = dict(lrr=0.0, a30c=0.0)
    if s == 'pooled':
        P = []
        for src in sorted(x.src.unique()):
            nd = pd.DataFrame(dict(t2=g, src=src, **base)); P.append(mm.predict(nd).values * (x.src == src).mean())
        p = np.sum(P, 0)
    else:
        p = mm.predict(pd.DataFrame(dict(t2=g, **base))).values
    ref = p[np.argmin(np.abs(g - 6))]
    for t, pp in zip(g, p): rows.append(dict(src=s, t2=t, p=pp, rel=pp / ref, n=len(sub)))
R = pd.DataFrame(rows); R.to_csv('work/out/n3/brake_curve.csv', index=False)
print(R[R.t2.isin([-8, -4, -2, 0, 1, 2, 4, 6, 8])].pivot(index='t2', columns='src', values='rel').round(2))
