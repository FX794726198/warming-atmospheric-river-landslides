"""v06: the 231 regional landslide events as a supplementary data table (with duplicates removed across sources listed), region labels,
per-region storm shares, leave-one-region-out and leave-one-source-out checks of the cold-season AR share and of the size gradient."""
import numpy as np, pandas as pd, sys
sys.path.insert(0, 'fig_code')
from arcommon import pool_mask, national_mask, cold_season
from collections import defaultdict
O = 'work/out/g1'; V = 'work/out/v6'
A = pd.read_parquet(f'{O}/outbreaks_all.parquet'); A['date'] = pd.to_datetime(A.date0)
A['drv'] = np.where(A.tcd2 <= 500, 'TC', np.where(A.ar0, 'AR', 'other'))
A = A[~(A.src.isin(['GLC', 'GFLD']).values & national_mask(A.lat.values, A.lon.values))]
A['d0'] = pd.to_datetime(A.date0); A['d1'] = pd.to_datetime(A.date1); A['name'] = A.title; A['tc_km'] = A.tcd2; A['ar_centre'] = A.ar0; A['ar_3x3'] = A.ar1
X = pd.read_parquet(f'{O}/extra_outbreaks.parquet')
X = pd.DataFrame(dict(src=X.source, n=X.n, lat=X.lat, lon=X.lon, drv=X.drv, date=pd.to_datetime(X.date_peak), name=X.name, d0=pd.to_datetime(X.date_start), d1=pd.to_datetime(X.date_end),
                      tc_km=X.tcd, ar_centre=X.ar0, ar_3x3=X.ar1, storm=X.storm))
B = pd.concat([A[['src', 'n', 'lat', 'lon', 'drv', 'date', 'name', 'd0', 'd1', 'tc_km', 'ar_centre', 'ar_3x3', 'storm']], X], ignore_index=True)
B = B[B.n >= 20].sort_values('n', ascending=False).reset_index(drop=True)
# duplicates across sources (dates within 3 days, centres within 2 deg), keeping the row with more landslides; record what was dropped
d = B.date.values.astype('datetime64[D]').astype(int); la = B.lat.values; lo = B.lon.values; sr = B.src.values
keep = np.ones(len(B), bool); dup_of = np.full(len(B), -1); idx = defaultdict(list)
for i in range(len(B)):
    hit = -1
    for dd in range(d[i] - 3, d[i] + 4):
        for j in idx.get(dd, ()):
            if sr[j] != sr[i] and abs(la[j] - la[i]) <= 2 and abs(((lo[j] - lo[i] + 180) % 360) - 180) <= 2: hit = j; break
        if hit >= 0: break
    if hit >= 0: keep[i] = False; dup_of[i] = hit
    else: idx[d[i]].append(i)
B['kept'] = keep; B['duplicate_of'] = dup_of
K = B[keep].reset_index(drop=True); K['id'] = [f'R{i + 1:03d}' for i in range(len(K))]
pos = {old: new for new, old in enumerate(np.where(keep)[0])}
D = B[~keep].copy(); D['duplicate_of_id'] = [K.id[pos[j]] for j in D.duplicate_of]
print('regional events kept', len(K), 'duplicates dropped', len(D))
print(D[['src', 'n', 'date', 'lat', 'lon', 'duplicate_of_id']].assign(kept_src=[K.src[pos[j]] for j in D.duplicate_of], kept_n=[K.n[pos[j]] for j in D.duplicate_of]).to_string())
K['month'] = K.date.dt.month; K['cold'] = cold_season(K.lat, K.month)
def region(lat, lon):
    if lat >= 25 and -170 <= lon < -100: return 'North America, Pacific side'
    if lat >= 25 and -100 <= lon < -50: return 'North America, east and interior'
    if 5 < lat < 25 and -120 <= lon < -55: return 'Central America and Caribbean'
    if lat <= 12 and -85 <= lon < -30: return 'South America'
    if lat >= 35 and -12 <= lon < 48: return 'Europe and Türkiye'
    if -36 <= lat < 37 and -20 <= lon < 55: return 'Africa'
    if lat >= 5 and lon >= 45: return 'Asia'
    if lat < 5 and lon >= 95: return 'Oceania and Maritime Southeast Asia'
    return 'other'
K['region'] = [region(a, b) for a, b in zip(K.lat, K.lon)]
print(K.region.value_counts())
print(K[K.region == 'other'][['src', 'name', 'lat', 'lon']])
SRC = {'NVE': 'Norway (NVE)', 'USGS': 'United States (LIUS)', 'ITALICA': 'Italy (ITALICA)', 'GLC': 'Global Landslide Catalog', 'GFLD': 'Global Fatal Landslide Database', 'COOLR_ev': 'COOLR event inventory',
       'emberson2022': 'Emberson et al. 2022', 'canada_pcld': 'Canadian Landslide Database', 'er2023': 'Emilia-Romagna 2023', 'turkey_blacksea': 'Eastern Black Sea (Çömert et al. 2025)'}
T = pd.DataFrame(dict(id=K.id, source=K.src.map(SRC), event=K.name.fillna(''), date_first=K.d0.dt.date, date_last=K.d1.dt.date, date=K.date.dt.date, landslides=K.n,
                      latitude=K.lat.round(2), longitude=K.lon.round(2), region=K.region, season=np.where(K.cold, 'cold', 'warm'), ar_centre_cell=K.ar_centre.astype(bool),
                      ar_3x3_cells=K.ar_3x3.astype(bool), tc_distance_km=K.tc_km.round(0).replace(1e9, np.nan), tc_name=K.storm.fillna(''), storm=K.drv))
T.to_csv(f'{V}/SupplementaryData1_regional_events.csv', index=False)
D2 = pd.DataFrame(dict(source=D.src.map(SRC), event=D.name.fillna(''), date=D.date.dt.date, landslides=D.n, latitude=D.lat.round(2), longitude=D.lon.round(2), counted_as=D.duplicate_of_id))
D2.to_csv(f'{V}/SupplementaryData1_duplicates.csv', index=False)
# per-region shares
pol = np.abs(K.lat) >= 25
def shares(m):
    g = K[m]; return dict(events=len(g), AR=(g.drv == 'AR').mean() * 100, TC=(g.drv == 'TC').mean() * 100, other=(g.drv == 'other').mean() * 100)
rows = []
for r in K.region.unique():
    m = (K.region == r)
    rows.append(dict(region=r, **{k + '_all': v for k, v in shares(m).items()}, **{k + '_cold25': v for k, v in shares(m & pol & K.cold).items()}, **{k + '_warm25': v for k, v in shares(m & pol & ~K.cold).items()}))
RG = pd.DataFrame(rows).sort_values('events_all', ascending=False); pd.set_option('display.width', 250); print(RG.round(0).to_string()); RG.to_csv(f'{V}/regional_by_region.csv', index=False)
# leave-one-out of the cold-season AR share poleward of 25 and the overall shares
base = shares(pol & K.cold); print('cold season poleward 25:', base); print('all 231:', shares(np.ones(len(K), bool)))
loo = []
for r in K.region.unique():
    loo.append(dict(left_out=r, kind='region', **shares(pol & K.cold & (K.region != r))))
for s in K.src.unique():
    loo.append(dict(left_out=SRC[s], kind='source', **shares(pol & K.cold & (K.src != s))))
LOO = pd.DataFrame(loo); print(LOO.round(1).to_string()); LOO.to_csv(f'{V}/loo_cold_share.csv', index=False)
# size gradient poleward of 25 from the inventory events, per inventory and leaving one inventory out
EV = pd.read_parquet(f'{V}/events_chance.parquet') if __import__('os').path.exists(f'{V}/events_chance.parquet') else pd.read_parquet(f'{O}/events_fp.parquet')
EV['arm'] = (EV.ar_frac >= 0.5).astype(int); EV['lat'] = EV.ci - 70; pm = np.asarray(pool_mask(EV)); EV['sz'] = pd.cut(EV.n, [0, 1, 4, 19, 1e9], labels=['1', '2-4', '5-19', '>=20'])
P = EV[pm & (np.abs(EV.lat) >= 25)]
g = P.groupby('sz', observed=True).arm.agg(['size', 'mean']); print('poleward 25 gradient', (g['mean'] * 100).round(0).to_dict(), g['size'].to_dict())
rows = []
for s in ['NVE', 'USGS', 'ITALICA', 'GLC', 'GFLD']:
    for lab, q in [('only', P[P.src == s]), ('without', P[P.src != s])]:
        gg = q.groupby('sz', observed=True).arm.agg(['size', 'mean'])
        rows.append(dict(src=s, kind=lab, **{f'share_{k}': v * 100 for k, v in gg['mean'].items()}, **{f'n_{k}': v for k, v in gg['size'].items()}))
G = pd.DataFrame(rows); print(G.round(0).to_string()); G.to_csv(f'{V}/gradient_loo.csv', index=False)
