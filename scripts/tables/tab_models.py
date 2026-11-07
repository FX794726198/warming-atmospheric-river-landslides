import pandas as pd
M=pd.read_csv('work/nc/proc/cluster/final/WRF_MODELS_NOR_2011.csv').sort_values('dT2m_land')
hdr=['CMIP6 model','Warming over land (°C)','Area ≥100 mm (km²)','Area ratio','Precipitation ratio','Rain ratio','Snowfall ratio','Rain share above 1,000 m (%)']
L=pd.read_csv('work/out/n7/levels_summary.csv'); L=L[L.event=='NOR_2011'].set_index('run'); c,f=L.loc['ctl'],L.loc['pgw585']
E=pd.read_csv('work/out/wrf/wrf_events.csv').set_index('event').loc['NOR']   # unrounded totals, as in the WRF results table
rows=[['Multi-model mean',f'{f.d02_dT2m_land:.1f}',f'{E.a100_w:,.0f}',f'{E.a100_w/E.a100_c:.1f}',f'{E.precip_total_w/E.precip_total_c:.2f}',f'{E.liq_total_w/E.liq_total_c:.2f}',f'{E.snowfall_w/E.snowfall_c:.2f}',f'{f.rain_share_ge1000m*100:.0f}']]
for r in M.itertuples():
    rows.append([r.model,f'{r.dT2m_land:.1f}',f'{r.A100_km2:,.0f}',f'{r.A100_ratio:.1f}',f'{r.P_ratio:.2f}',f'{r.liq_ratio:.2f}',f'{r.snow_ratio:.2f}',f'{r.rain_share_ge1000m*100:.0f}'])
out='| '+' | '.join(hdr)+' |\n|'+'---|'*len(hdr)+'\n'+'\n'.join('| '+' | '.join(x)+' |' for x in rows)+'\n'
open('work/tables/tab_models.md','w').write(out); print(out)
