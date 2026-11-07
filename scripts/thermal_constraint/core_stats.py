import numpy as np, pandas as pd
from scipy.spatial import cKDTree
WP='work/data/wrf/post'; CS='work/data/cases'; EDGE=10
def load(ev,run):
    z=np.load(f'{WP}/{ev}/{ev}_{run}.npz'); return {k:z[k][EDGE:-EDGE,EDGE:-EDGE].astype(float) for k in z.files if z[k].ndim==2}
rows=[]; prof=[]
for ev,sh in [('NOR_2011','NOR'),('BC_2021','BC'),('ORE_1996','ORE'),('LIG_2014','LIG'),('RSB_2024','RSB')]:
    c=load(ev,'ctl'); w=load(ev,'pgw585'); lm=c['landmask']>0.5; A=c['cell_area_km2']; h=c['hgt']
    for thr in [50,100]:
        core=lm&(c['precip_total']>=thr)
        for run,g in [('ctl',c),('pgw585',w)]:
            tot=g['precip_total']; liq=g['liq_total']; sn=g['snowfall']+g['graupel_hail']
            hmax=int(np.ceil(h[core].max()/100)*100) if core.any() else 0
            d=[]
            for lo in range(0,hmax,100):
                m=core&(h>=lo)&(h<lo+100)
                if m.sum()<3: continue
                a=A[m].sum(); P=(tot*A)[m].sum()/a; R=(liq*A)[m].sum()/a; S=(sn*A)[m].sum()/a
                d.append(dict(event=ev,run=run,core_thr=thr,band_lo=lo,area_km2=a,ncell=int(m.sum()),precip=P,rain=R,snow=S,rain_share=R/P))
            d=pd.DataFrame(d); prof.append(d)
            ss=1-d.rain_share.values; mid=d.band_lo.values+50; rsl=np.nan
            for i in range(1,len(ss)):
                if ss[i-1]<0.5<=ss[i]: rsl=mid[i-1]+(0.5-ss[i-1])/(ss[i]-ss[i-1])*100; break
            # share of core precip that fell as rain, and above 1000 m
            a=A[core].sum(); rs=(liq*A)[core].sum()/(tot*A)[core].sum()
            m1=core&(h>=1000); rs1=(liq*A)[m1].sum()/(tot*A)[m1].sum() if m1.any() else np.nan
            rows.append(dict(event=ev,run=run,core_thr=thr,core_km2=a,rsl_m=rsl,rain_share_core=rs,rain_share_core_ge1000=rs1))
    # landslides inside >=100 mm/48 h rain area
    p=pd.read_csv(f'{CS}/{sh}_points.csv'); plon=p.longitude if 'longitude' in p else p.lon; plat=p.latitude if 'latitude' in p else p.lat
    tree=cKDTree(np.c_[c['lat'].ravel(),c['lon'].ravel()]); _,idx=tree.query(np.c_[plat,plon])
    for run,g in [('ctl',c),('pgw585',w)]:
        for thr in [50,100]:
            inside=(g['liq_max48h'].ravel()[idx]>=thr)
            rows.append(dict(event=ev,run=run,core_thr=thr,n_points=len(p),frac_points_in_liq48=inside.mean()))
    # elevation of landslides vs rain line
R=pd.DataFrame(rows); R.to_csv('core_stats_local.csv',index=False); pd.concat(prof).to_csv('core_profiles_local.csv',index=False)
pd.set_option('display.width',200); print(R.to_string())
