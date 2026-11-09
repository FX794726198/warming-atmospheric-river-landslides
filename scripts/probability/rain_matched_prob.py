"""Rain-matched comparison as probabilities: per rain class p_AR / p_other, and Mantel-Haenszel risk ratios
(Greenland-Robins variance) with strata of cell x rain class x calendar month (and rain class x month, as before)."""
import numpy as np, pandas as pd
O = 'work/out/g1'; T0 = pd.Timestamp('1979-01-01')
L = pd.read_parquet(f'{O}/ls_ar.parquet')
L = L[~((L.src == 'USGS') & L.inv.isin(['USGS Earthquake-Triggered Ground Failure', 'USGS Seismogenic Mass Movements']))]
win = {'NVE': ('2000', '2022'), 'USGS': ('1979', '2024'), 'ITALICA': ('1996', '2021'), 'GLC': ('2007', '2017'), 'GFLD': ('2004', '2017')}
z = np.load(f'{O}/panel.npz'); ci, cj, W, R2 = z['ci'], z['cj'], z['W'], z['R2'].astype(np.float32)
days = pd.date_range('1979-01-01', '2022-12-31'); mon = days.month.values
col = {(a, b): k for k, (a, b) in enumerate(zip(ci, cj))}
rb = [0, 5, 10, 20, 40, 80, 1e9]; rlab = ['<5', '5–10', '10–20', '20–40', '40–80', '≥80']
def mh(A, B, C, D):
    N = A + B + C + D; ok = N > 0; A, B, C, D, N = A[ok], B[ok], C[ok], D[ok], N[ok]
    n1 = A + B; n0 = C + D; R = (A * n0 / N).sum(); S = (C * n1 / N).sum()
    v = ((n1 * n0 * (A + C) - A * C * N) / N ** 2).sum() / (R * S); rr = R / S; se = np.sqrt(v)
    orr = (A * D / N).sum() / (B * C / N).sum()
    return rr, rr * np.exp(-1.96 * se), rr * np.exp(1.96 * se), orr
rows = []; out = []
for s, (a, b) in win.items():
    g = L[(L.src == s) & (L.date >= a) & (L.date <= min(b, '2022') + '-12-31')]
    cols = sorted({col[(x, y)] for x, y in zip(g.ci, g.cj) if (x, y) in col})
    t0 = (pd.Timestamp(a + '-01-01') - T0).days; t1 = (pd.Timestamp(min(b, '2022') + '-12-31') - T0).days + 1
    Y = np.zeros((t1 - t0, len(cols)), bool); cpos = {c: k for k, c in enumerate(cols)}
    for x, y, d in zip(g.ci, g.cj, g.date):
        k = cpos.get(col.get((x, y)))
        if k is not None: Y[(d - T0).days - t0, k] = True
    w = W[t0:t1][:, cols].astype(bool); r = R2[t0:t1][:, cols]; m = np.repeat(mon[t0:t1][:, None], len(cols), 1)
    bi = np.digitize(r, rb) - 1; K = len(cols)
    cellidx = np.repeat(np.arange(K)[None, :], t1 - t0, 0)
    for j in range(6):
        sel = bi == j
        a1 = (Y & w & sel).sum(); b1 = (~Y & w & sel).sum(); c1 = (Y & ~w & sel).sum(); d1 = (~Y & ~w & sel).sum()
        p_ar = a1 / max(a1 + b1, 1); p_no = c1 / max(c1 + d1, 1)
        # Katz interval for the class risk ratio
        se = np.sqrt(1 / max(a1, 1) - 1 / max(a1 + b1, 1) + 1 / max(c1, 1) - 1 / max(c1 + d1, 1)) if a1 > 0 and c1 > 0 else np.nan
        rr = p_ar / p_no if p_no > 0 else np.nan
        rows.append((s, rlab[j], a1 + c1, p_ar, p_no, rr, rr * np.exp(-1.96 * se), rr * np.exp(1.96 * se)))
    for name, key in [('cell x rain x month', (cellidx * 6 + bi) * 12 + (m - 1)), ('rain x month', bi * 12 + (m - 1))]:
        ok = ((bi >= 0) & (bi < 6)).ravel(); key = key.ravel()[ok]; nb = key.max() + 1; yy = Y.ravel()[ok]; ww = w.ravel()[ok]
        A = np.bincount(key, (yy & ww), nb); B = np.bincount(key, (~yy & ww), nb); C = np.bincount(key, (yy & ~ww), nb); D = np.bincount(key, (~yy & ~ww), nb)
        rr, lo, hi, orr = mh(A.astype(float), B.astype(float), C.astype(float), D.astype(float))
        out.append((s, name, rr, lo, hi, orr)); print(s, name, 'RR_MH %.2f (%.2f-%.2f)  OR_MH %.2f' % (rr, lo, hi, orr), flush=True)
RM = pd.DataFrame(rows, columns=['src', 'rain', 'nls_days', 'p_ar', 'p_no', 'RR', 'lo', 'hi']); RM.to_csv('data/proc/prob/rain_matched_prob.csv', index=False)
MH = pd.DataFrame(out, columns=['src', 'strata', 'RR_MH', 'lo', 'hi', 'OR_MH']); MH.to_csv('data/proc/prob/rain_matched_mh_prob.csv', index=False)
print(RM.round(4).to_string())
