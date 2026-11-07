"""Regional analysis of the phase of AR extreme precipitation, 1940-2024, from snow_counts_<BASE>_1940_2024.npz.
Usage: python snow_regions.py <snow_counts.npz> <outdir>
Cold season only (NH Oct(Y-1)-Mar(Y) -> Y, SH Apr-Sep). Land cells with orography >= 300 m. Weights = cos(lat).
Outputs: regional_series.csv, regional_trends.csv, regional_periods.csv, snowline_series.csv, map_change_1deg.csv, protected_area.csv"""
import numpy as np, pandas as pd, sys, os
from scipy import stats
import statsmodels.api as sm
f, out = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True)
z = np.load(f); y0 = int(z['y0']); la = z['la']; lo = np.where(z['lo'] > 180, z['lo'] - 360, z['lo']); zs = z['zs']
C = {k: z[k][:, 0, :].astype(np.float64) for k in ('nar', 'ns50', 'ns80', 'nex', 'nexs50', 'ptot', 'psnow')}   # [year, cell], cold season
ny = C['nar'].shape[0]; years = y0 + np.arange(ny)
w = np.cos(np.radians(la)); cellarea = 27.83 ** 2 * w   # km2 of a 0.25 deg cell
R = {'Scandinavia': (57, 71, 4, 20), 'Iceland and Britain': (50, 66, -25, 0), 'Alps': (43, 48, 5, 16), 'Apennines': (38, 45, 7.5, 18.5),
     'Caucasus and Turkiye': (36, 44, 26, 48), 'British Columbia and Alaska coast': (48, 62, -140, -115), 'US Pacific Northwest and Rockies': (40, 49, -125, -105),
     'Northeastern North America': (40, 50, -80, -60), 'Hindu Kush and Himalaya': (27, 38, 65, 95), 'Japan': (33, 45, 130, 145),
     'Southern Andes': (-55, -33, -75, -66), 'New Zealand': (-47, -34, 166, 179), 'All land 25-72 N': (25, 72, -180, 180), 'All land 25-56 S': (-56, -25, -180, 180)}
mount = zs >= 300
# complete cold seasons: NH needs Oct-Mar -> first complete season-year y0+1, last 2024 (Oct 2023-Mar 2024); SH Apr-Sep complete y0..2024
rows = []; sl = []
for name, (a0, a1, b0, b1) in R.items():
    m = mount & (la >= a0) & (la <= a1) & (lo >= b0) & (lo <= b1)
    if m.sum() == 0: continue
    nh = a0 > 0
    for i, yy in enumerate(years):
        if yy > 2024 or (nh and yy == y0): continue
        S = {k: (C[k][i, m] * w[m]).sum() for k in C}
        rows.append(dict(region=name, year=yy, ncell=int(m.sum()), n_ar=float((C['nar'][i, m]).sum()), share_s50=S['ns50'] / S['nar'] if S['nar'] > 0 else np.nan,
                         share_s80=S['ns80'] / S['nar'] if S['nar'] > 0 else np.nan, snowfrac_amount=S['psnow'] / S['ptot'] if S['ptot'] > 0 else np.nan,
                         share_ex_s50=S['nexs50'] / S['nex'] if S['nex'] > 0 else np.nan, n_ex=float(C['nex'][i, m].sum())))
        # AR snow line: binomial GLM of snow-dominated AR extreme cell-days on elevation (km), cells with >=1 AR extreme cell-day
        k = m & (C['nar'][i] > 0)
        if k.sum() >= 30 and C['ns50'][i, k].sum() >= 5 and (C['nar'][i, k] - C['ns50'][i, k]).sum() >= 5:
            Y = np.c_[C['ns50'][i, k], C['nar'][i, k] - C['ns50'][i, k]]; X = sm.add_constant(zs[k] / 1000.0)
            try:
                g = sm.GLM(Y, X, family=sm.families.Binomial()).fit(); b0_, b1_ = g.params
                z50 = -b0_ / b1_ * 1000 if b1_ > 0 else np.nan
            except Exception: z50 = np.nan
        else: z50 = np.nan
        sl.append(dict(region=name, year=yy, snowline_m=z50))
S = pd.DataFrame(rows); SL = pd.DataFrame(sl); S = S.merge(SL, on=['region', 'year'])
S.to_csv(f'{out}/regional_series.csv', index=False)
def trend(x, y):
    ok = np.isfinite(y) & np.isfinite(x)
    if ok.sum() < 10: return np.nan, np.nan, np.nan
    r = stats.linregress(x[ok], y[ok]); ts = stats.theilslopes(y[ok], x[ok])[0]
    return r.slope * 10, r.pvalue, ts * 10
tr = []; per = []
for name, g in S.groupby('region', sort=False):
    for p0, p1 in ((1941, 2024), (1979, 2024)):
        h = g[(g.year >= p0) & (g.year <= p1)]
        d = dict(region=name, period=f'{p0}-{p1}')
        for v in ('share_s50', 'share_s80', 'snowfrac_amount', 'share_ex_s50', 'snowline_m'):
            s_, p_, t_ = trend(h.year.values.astype(float), h[v].values); d[v + '_per_decade'] = s_; d[v + '_p'] = p_; d[v + '_theilsen'] = t_
        tr.append(d)
    for p0, p1 in ((1941, 1970), (1971, 2000), (2001, 2024), (1995, 2024)):
        h = g[(g.year >= p0) & (g.year <= p1)]
        per.append(dict(region=name, period=f'{p0}-{p1}', share_s50=h.share_s50.mean(), snowfrac_amount=h.snowfrac_amount.mean(), share_ex_s50=h.share_ex_s50.mean(), snowline_m=h.snowline_m.mean(), n_ar_per_season=h.n_ar.mean()))
pd.DataFrame(tr).to_csv(f'{out}/regional_trends.csv', index=False); pd.DataFrame(per).to_csv(f'{out}/regional_periods.csv', index=False)
# change map on a 1 deg grid (mountain cells only), 1941-1970 vs 1995-2024
def psum(k, p0, p1):
    i = (years >= p0) & (years <= p1); return C[k][i].sum(0)
a1, s1 = psum('nar', 1941, 1970), psum('ns50', 1941, 1970); a2, s2 = psum('nar', 1995, 2024), psum('ns50', 1995, 2024)
gi = np.floor(la).astype(int); gj = np.floor(lo).astype(int)
df = pd.DataFrame(dict(lat=gi + 0.5, lon=gj + 0.5, a1=a1 * w, s1=s1 * w, a2=a2 * w, s2=s2 * w, n1=a1, n2=a2, area=cellarea))[mount]
M = df.groupby(['lat', 'lon']).sum().reset_index()
M = M[(M.n1 >= 20) & (M.n2 >= 20)]; M['share1'] = M.s1 / M.a1; M['share2'] = M.s2 / M.a2; M['diff'] = M.share2 - M.share1
M[['lat', 'lon', 'share1', 'share2', 'diff', 'n1', 'n2', 'area']].round(4).to_csv(f'{out}/map_change_1deg.csv', index=False)
# mountain area where most AR extreme cell-days are snow-dominated, per 30-year period and region
pa = []
for name, (a0, a1_, b0, b1) in R.items():
    m = mount & (la >= a0) & (la <= a1_) & (lo >= b0) & (lo <= b1)
    for p0, p1 in ((1941, 1970), (1971, 2000), (1995, 2024)):
        A, Sx = psum('nar', p0, p1), psum('ns50', p0, p1); ok = m & (A >= 10)
        pa.append(dict(region=name, period=f'{p0}-{p1}', area_snow_dominated_km2=float(cellarea[ok & (Sx / np.maximum(A, 1) >= 0.5)].sum()), area_assessed_km2=float(cellarea[ok].sum())))
pd.DataFrame(pa).to_csv(f'{out}/protected_area.csv', index=False)
T = pd.DataFrame(tr); print(T[['region', 'period', 'share_s50_per_decade', 'share_s50_p', 'snowfrac_amount_per_decade', 'snowfrac_amount_p', 'share_ex_s50_per_decade', 'share_ex_s50_p', 'snowline_m_per_decade', 'snowline_m_p']].round(4).to_string())
print(pd.DataFrame(per).round(3).to_string()); print(pd.DataFrame(pa).round(0).to_string())
