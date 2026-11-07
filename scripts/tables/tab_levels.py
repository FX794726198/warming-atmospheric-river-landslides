import pandas as pd, numpy as np
L=pd.read_csv('work/out/n7/levels_summary.csv')
RS=pd.read_csv('work/nc/proc/rsl_all_runs.csv'); RS=RS[RS['mask']=='core100']
CCA=pd.read_csv('work/nc/proc/rev/cc_area.csv')[['event','run','A_cc']]
L=L.drop(columns=['rain_line_m']).merge(RS[['event','run','rsl_m']].rename(columns={'rsl_m':'rain_line_m'}),on=['event','run'],how='left').merge(CCA,on=['event','run'],how='left')
NM={'NOR_2011':'Western Norway 2011','BC_2021':'British Columbia 2021','LIG_2014':'Liguria and Piedmont 2014','ORE_1996':'Oregon 1996','RSB_2024':'Rio Grande do Sul 2024'}
SC={'ctl':'Control','pgw585_s025':'× 0.25','pgw585_s050':'× 0.5','pgw585_s075':'× 0.75','pgw585':'× 1'}
hdr=['Event','Change','Warming (°C)','Area ≥50 mm (km²)','Area ≥100 mm (km²)','Area ≥100 mm, CC reference (km²)','Area ≥150 mm (km²)','Rain share above 1,000 m (%)','Rain–snow line (m)','Population (thousands)','Roads (km)']
rows=[]
for ev in NM:
    q=L[L.event==ev].copy(); q['o']=q.run.map(list(SC).index); q=q.sort_values('o')
    for i,r in enumerate(q.itertuples()):
        part=pd.isna(r.area_liq48_ge50)          # 0.25 and 0.75 steps of LIG and RSB: only warming, >=100 mm area and rain share processed
        na=lambda v,f: 'n.a.' if pd.isna(v) else f(v)
        rsl='n.a.' if (part and ev!='RSB_2024') else ('–' if pd.isna(r.rain_line_m) else f'{r.rain_line_m:,.0f}')
        rows.append([NM[ev] if i==0 else '', SC[r.run], f'{r.d02_dT2m_land:.1f}' if r.run!='ctl' else '–', na(r.area_liq48_ge50,lambda v:f'{v:,.0f}'), f'{r.area_liq48_ge100:,.0f}', f'{r.A_cc:,.0f}', na(r.area_liq48_ge150,lambda v:f'{v:,.0f}'),
                     f'{r.rain_share_ge1000m*100:.0f}', rsl, na(r.pop_liq48_ge100,lambda v:f'{v/1e3:,.0f}' if v>=1e4 else f'{v/1e3:.1f}'), na(r.road_km_liq48_ge100,lambda v:f'{v:,.0f}')])
out='| '+' | '.join(hdr)+' |\n|'+'---|'*len(hdr)+'\n'+'\n'.join('| '+' | '.join(r)+' |' for r in rows)+'\n'
open('work/tables/tab_levels.md','w').write(out); print(out)
