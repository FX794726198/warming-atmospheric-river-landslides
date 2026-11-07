import numpy as np, pandas as pd
from scipy.spatial import cKDTree
WP='work/data/wrf/post'; CS='work/data/cases'; EDGE=10
CSX=pd.read_csv('core_stats_local.csv')
NM={'NOR_2011':'Western Norway 2011','BC_2021':'British Columbia 2021','ORE_1996':'Oregon 1996','LIG_2014':'Liguria and Piedmont 2014','RSB_2024':'Rio Grande do Sul 2024'}
rows=[]
for ev,sh in [('NOR_2011','NOR'),('BC_2021','BC'),('ORE_1996','ORE'),('LIG_2014','LIG'),('RSB_2024','RSB')]:
    c=np.load(f'{WP}/{ev}/{ev}_ctl.npz'); w=np.load(f'{WP}/{ev}/{ev}_pgw585.npz')
    g=lambda z,k: z[k][EDGE:-EDGE,EDGE:-EDGE].astype(float)
    land=g(c,'landmask')>0.5; A=g(c,'cell_area_km2'); pc=g(c,'precip_total'); pw=g(w,'precip_total')
    sc=g(c,'snowfall')+g(c,'graupel_hail'); sw=g(w,'snowfall')+g(w,'graupel_hail')
    mask=land&(pc>=25); snow_c=mask&(sc/np.maximum(pc,1e-6)>=0.5); snow_w=mask&(sw/np.maximum(pw,1e-6)>=0.5); turned=snow_c&~snow_w
    sd=A[snow_c].sum(); ft=A[turned].sum()/sd*100 if sd>0 else np.nan
    r=lambda run: CSX[(CSX.event==ev)&(CSX.run==run)&(CSX.core_thr==100)]
    rsl=[r(run).rsl_m.dropna() for run in ('ctl','pgw585')]
    rsl=[f'{v.iloc[0]:,.0f}' if len(v) and np.isfinite(v.iloc[0]) else '–' for v in rsl]
    fi=[r(run).frac_points_in_liq48.dropna().iloc[0]*100 for run in ('ctl','pgw585')]
    p=pd.read_csv(f'{CS}/{sh}_points.csv'); plon=p.longitude if 'longitude' in p else p.lon; plat=p.latitude if 'latitude' in p else p.lat
    tree=cKDTree(np.c_[c['lat'].ravel(),c['lon'].ravel()]); _,idx=tree.query(np.c_[plat,plon])
    med=[np.median(z['liq_max48h'].ravel()[idx]) for z in (c,w)]
    rows.append([NM[ev], f'{sd:,.0f}' if sd>0 else '0', f'{ft:.0f}' if np.isfinite(ft) else '–', f'{rsl[0]} / {rsl[1]}', f'{fi[0]:.0f} / {fi[1]:.0f}', f'{med[0]:.0f} / {med[1]:.0f}'])
hdr=['Event','Snow-dominated land, control (km²)','Share turned to rain (%)','Rain–snow line, control / warmer (m)','Landslides inside ≥100 mm of rain in 48 h, control / warmer (%)','Largest 48-h rain at the landslide sites, control / warmer (mm)']
out='| '+' | '.join(hdr)+' |\n|'+'---|'*len(hdr)+'\n'+'\n'.join('| '+' | '.join(r)+' |' for r in rows)+'\n'
open('work/tables/tab_wrfphase.md','w').write(out); print(out)
