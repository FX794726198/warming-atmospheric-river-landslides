"""Rain-halving equivalent of the temperature classes: lrr coefficient in the cell-FE and full-control models."""
import numpy as np, pandas as pd, statsmodels.api as sm, patsy
exec(open('brake_robust.py').read().split('rows = []')[0])
out = []
for name, f in [('cell FE (paper)', 'hit ~ C(tc) + lrr + a30c + C(cell)'),
                ('+ month FE + duration + year', 'hit ~ C(tc) + lrr + a30c + ldur + I((t0.dt.year - 2000) / 10) + C(month) + C(cell)')]:
    m, X = fit(f)
    b, s = m.params['lrr'], m.bse['lrr']
    out.append(dict(model=name, lrr_b=b, lrr_se=s, OR_halving=0.5 ** b, OR_halving_lo=0.5 ** (b + 1.96 * s), OR_halving_hi=0.5 ** (b - 1.96 * s)))
R = pd.DataFrame(out); print(R.round(3).to_string()); R.to_csv('lrr_equiv.csv', index=False)
