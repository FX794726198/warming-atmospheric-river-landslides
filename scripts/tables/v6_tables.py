"""v06 supplementary tables: logistic models with the simultaneous extreme-rain area (tab_logit6.md), chance AR shares (tab_chance.md),
regional landslide events by region with leave-one-out (tab_region.md)."""
import pandas as pd, numpy as np
V = 'work/out/v6'; P = 'work/paper/parts'
M = lambda s: s.replace('-', '−')
R = pd.read_csv(f'{V}/logit_v6.csv')
NM = {'NVE': 'Norway', 'USGS': 'United States', 'ITALICA': 'Italy', 'GLC': 'Global (GLC)', 'GFLD': 'Global fatal (GFLD)', 'ALL': 'Pooled'}
MD = [('raw', 'Unadjusted'), ('+duration', '+ duration'), ('+simultaneous footprint', '+ area'), ('+simultaneous footprint+duration', '+ area, duration'),
      ('+peak', '+ peak rain'), ('+simultaneous footprint+peak+duration', '+ area, peak rain, duration')]
rows = ['| Outcome | Inventory | Events | Events with outcome | ' + ' | '.join(m for _, m in MD) + ' |', '|' + '---|' * (4 + len(MD))]
for out, lab in (('big5', '≥5 landslides'), ('big20', '≥20 landslides')):
    for s in NM:
        q = R[(R.src == s) & (R.outcome == out)].set_index('model')
        c = lambda m: M(f"{q.loc[m, 'OR']:.2f} ({q.loc[m, 'lo']:.2f}–{q.loc[m, 'hi']:.2f})")
        rows.append(f"| {lab} | {NM[s]} | {int(q.loc['raw', 'n']):,} | {int(q.loc['raw', 'nbig'])} | " + ' | '.join(c(m) for m, _ in MD) + ' |')
open(f'{P}/tab_logit6.md', 'w').write('\n'.join(rows) + '\n')

C = pd.read_csv(f'{V}/chance_inventory.csv'); C2 = pd.read_csv(f'{V}/chance_regional.csv')
GN = {'pooled': 'Five inventories, pooled', 'poleward of 25': 'Inventories, poleward of 25°', 'within 25': 'Inventories, within 25° of the equator', 'NVE': 'Norway', 'USGS': 'United States', 'ITALICA': 'Italy', 'GLC': 'Global (GLC)', 'GFLD': 'Global fatal (GFLD)'}
rows = ['| Events | Size class | Events (n) | Under an AR (%) | Chance (%) | Chance, 95% range (%) |', '|---|---|---|---|---|---|']
for r in C.itertuples():
    if r.group in ('NVE', 'USGS', 'ITALICA', 'GLC', 'GFLD') and r.size not in ('1', '>=20'): continue
    rows.append(f"| {GN[r.group]} | {r.size.replace('>=', '≥')} | {r.events:,} | {r.observed:.0f} | {r.chance:.0f} | {r.chance_lo:.0f}–{r.chance_hi:.0f} |")
G2 = {'all 231': 'Regional events, all', 'poleward of 25': 'Regional events, poleward of 25°', 'poleward of 25, cold season': 'Regional events, poleward of 25°, cold season',
      'poleward of 25, warm season': 'Regional events, poleward of 25°, warm season', 'poleward of 35': 'Regional events, poleward of 35°', '25-35': 'Regional events, 25–35°', '15-25': 'Regional events, 15–25°', 'within 15': 'Regional events, within 15°'}
for r in C2.itertuples():
    rows.append(f"| {G2[r.group]} | ≥20 | {r.events} | {r.AR:.0f} | {r.chance:.0f} | {r.chance_lo:.0f}–{r.chance_hi:.0f} |")
open(f'{P}/tab_chance.md', 'w').write('\n'.join(rows) + '\n')

RG = pd.read_csv(f'{V}/regional_by_region.csv'); LOO = pd.read_csv(f'{V}/loo_cold_share.csv')
rows = ['| Region | Events | AR (%) | TC (%) | Other (%) | Cold season, poleward of 25°: events | AR (%) | Warm season, poleward of 25°: events | AR (%) | Cold-season AR share without this region (%) |', '|---|---|---|---|---|---|---|---|---|---|']
for r in RG.itertuples():
    lo = LOO[(LOO.kind == 'region') & (LOO.left_out == r.region)].iloc[0]
    f = lambda v: '–' if pd.isna(v) else f'{v:.0f}'
    rows.append(f"| {r.region} | {r.events_all} | {r.AR_all:.0f} | {r.TC_all:.0f} | {r.other_all:.0f} | {r.events_cold25} | {f(r.AR_cold25)} | {r.events_warm25} | {f(r.AR_warm25)} | {lo.AR:.0f} ({lo.events}) |")
rows.append(f"| All | {RG.events_all.sum()} | {(RG.AR_all * RG.events_all).sum() / RG.events_all.sum():.0f} | {(RG.TC_all * RG.events_all).sum() / RG.events_all.sum():.0f} | {(RG.other_all * RG.events_all).sum() / RG.events_all.sum():.0f} | {RG.events_cold25.sum()} | 95 | {RG.events_warm25.sum()} | 52 | – |")
open(f'{P}/tab_region.md', 'w').write('\n'.join(rows) + '\n')
print(open(f'{P}/tab_logit6.md').read()); print(open(f'{P}/tab_chance.md').read()); print(open(f'{P}/tab_region.md').read())
print(LOO[LOO.kind == 'source'].round(1).to_string())
