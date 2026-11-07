# Land-surface differences: (1) AR episodes at landslide cells with vs without landslides by the cell's relief (std of sub-grid orography),
# sub-grid slope, elevation and forest (high-vegetation) cover; logistic model adding these to rain, antecedent rain and freezing.
# (2) Event footprints split into steep (relief >= 150 m) and gentle cells, AR vs other events, and their link with event size.
import numpy as np, pandas as pd, statsmodels.formula.api as smf
O = 'work/out/g1'; S = 'work/out/s1'; E = 'work/data/era5'
st = np.load(f'{O}/era5_static.npz'); sd = st['standard_deviation_of_orography']; sl = st['slope_of_sub_gridscale_orography']
el = st['geopotential_at_surface'] / 9.81; hv = st['high_vegetation_cover']; lv = st['low_vegetation_cover']
EP = pd.read_parquet(f'{S}/episodes_drivers.parquet')
i, j = EP.ci.values, EP.cj.values
EP['sdor'] = sd[i, j]; EP['slor'] = sl[i, j]; EP['elev'] = el[i, j]; EP['forest'] = hv[i, j]
pd.set_option('display.width', 220)
print('== medians, episodes with vs without landslides')
print(EP.groupby(['src', 'hit'])[['sdor', 'slor', 'elev', 'forest']].median().round(3).to_string())
print('== cells: AR episodes with landslides (%) by relief quartile within inventory, for episodes with 2-day rain >= P99 (and all)')
for s, q in EP.groupby('src'):
    cell = q.groupby(['ci', 'cj']).agg(sdor=('sdor', 'first')).reset_index()
    qq = pd.qcut(cell.sdor, 4, labels=['Q1 flat', 'Q2', 'Q3', 'Q4 steep']); cell['rq'] = qq
    q = q.merge(cell[['ci', 'cj', 'rq']], on=['ci', 'cj'])
    a = q.groupby('rq', observed=True).hit.mean() * 100; b = q[q.rain_rel >= 1].groupby('rq', observed=True).hit.mean() * 100
    r = q.groupby('rq', observed=True).sdor.median()
    print(s, 'relief m', r.round(0).to_dict(), '| all', a.round(2).to_dict(), '| rain>=P99', b.round(1).to_dict())
out = []
for s in EP.src.unique():
    q = EP[EP.src == s].dropna(subset=['a30', 'a7', 't2']).copy()
    for v in ('rain_rel', 'dur_h', 'ivt_max'): x = np.log(q[v].clip(lower=1e-3)); q['z_' + v] = (x - x.mean()) / x.std()
    for v in ('a30', 'a7', 'forest', 'elev'): q['z_' + v] = (q[v] - q[v].mean()) / q[v].std()
    for v in ('sdor', 'slor'): x = np.log(q[v].clip(lower=1e-4)); q['z_' + v] = (x - x.mean()) / x.std()
    q['frz'] = (q.t2 < 0).astype(int)
    m = smf.logit('hit ~ z_rain_rel + z_dur_h + z_ivt_max + z_a30 + z_a7 + frz + z_sdor + z_forest + z_elev + C(month)', q).fit(disp=0)
    c = m.conf_int()
    for v in ('z_rain_rel', 'z_a30', 'z_sdor', 'z_forest', 'z_elev'):
        out.append((s, v, np.exp(m.params[v]), np.exp(c.loc[v, 0]), np.exp(c.loc[v, 1])))
LG = pd.DataFrame(out, columns=['src', 'var', 'OR', 'lo', 'hi']); LG.to_csv(f'{S}/episode_logit_surface.csv', index=False)
print('== logistic with surface (OR per SD)'); print(LG.pivot(index='var', columns='src', values='OR').round(2).to_string())
print(LG.round(2).to_string())

# (2) steep vs gentle footprint
lsm = np.load('work/out/a2/lsm.npy') > 0.5; p99 = np.load(f'{O}/r2_p99.npy'); steep = sd >= 150
EV = pd.read_parquet(f'{O}/events_fp.parquet'); EV = EV[EV.d1 <= (pd.Timestamp('2022-12-31') - pd.Timestamp('1979-01-01')).days]
days = pd.date_range('1979-01-01', '2022-12-31'); PR = {y: np.load(f'{E}/pr_{y}.npy', mmap_mode='r') for y in range(1979, 2023)}
cache = {}
def r2(dn):
    if dn not in cache:
        if len(cache) > 400: cache.clear()
        d = days[dn]; a = np.nan_to_num(np.asarray(PR[d.year][d.dayofyear - 1], dtype=np.float32))
        if dn > 0: d1 = days[dn - 1]; a = a + np.nan_to_num(np.asarray(PR[d1.year][d1.dayofyear - 1], dtype=np.float32))
        cache[dn] = a
    return cache[dn]
R = 4; fs = []; fg = []
EV = EV.sort_values('d0')
for e in EV.itertuples():
    ii = slice(max(0, int(round(e.ci)) - R), min(146, int(round(e.ci)) + R + 1)); jj = [(int(round(e.cj)) + o) % 360 for o in range(-R, R + 1)]
    ex = np.zeros((ii.stop - ii.start, len(jj)), bool)
    for dn in range(int(e.d0), int(e.d1) + 1): ex |= (r2(dn)[ii][:, jj] >= p99[ii][:, jj]) & lsm[ii][:, jj]
    stp = steep[ii][:, jj]; fs.append(int((ex & stp).sum())); fg.append(int((ex & ~stp).sum()))
EV['fp_steep'] = fs; EV['fp_gentle'] = fg; EV.to_parquet(f'{S}/events_fp_steep.parquet')
EV['arm'] = EV.ar_frac >= 0.5; EV['big'] = (EV.n >= 5).astype(int)
print('== footprint steep/gentle, AR vs other (means), and share of footprint on steep cells')
g = EV.groupby(['src', 'arm']).agg(steep=('fp_steep', 'mean'), gentle=('fp_gentle', 'mean')); g['steep_share'] = g.steep / (g.steep + g.gentle) * 100
print(g.round(2).to_string())
out = []
EV['mo'] = ((pd.Timestamp('1979-01-01') + pd.to_timedelta(EV.d0, 'D')).dt.month).astype(str)
EV['ls'] = np.log1p(EV.fp_steep); EV['lg'] = np.log1p(EV.fp_gentle); EV['lpk'] = np.log(EV.pk.clip(0.05))
for s in ['NVE', 'USGS', 'ITALICA', 'GLC', 'GFLD']:
    q = EV[EV.src == s]
    m = smf.logit('big ~ arm + ls + lg + lpk + C(mo)', q).fit(disp=0); c = m.conf_int()
    out.append((s, *[np.exp(m.params[v]) for v in ('arm[T.True]', 'ls', 'lg')], *[f'{np.exp(c.loc[v,0]):.2f}–{np.exp(c.loc[v,1]):.2f}' for v in ('ls', 'lg')]))
print(pd.DataFrame(out, columns=['src', 'OR_AR', 'OR_steep_log', 'OR_gentle_log', 'ci_steep', 'ci_gentle']).round(2).to_string())
