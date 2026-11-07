import numpy as np, pandas as pd, glob
from scipy.ndimage import gaussian_filter
WP='work/data/wrf/post'; EDGE=10
rows=[]; lines=[]
for ev in ['NOR_2011','BC_2021','ORE_1996','LIG_2014','RSB_2024']:
    for run in ['ctl','pgw585']:
        z=np.load(f'{WP}/{ev}/{ev}_{run}.npz')
        g={k:z[k][EDGE:-EDGE,EDGE:-EDGE].astype(float) for k in z.files if z[k].ndim==2}
        lm=g['landmask']>0.5; h=g['hgt']; A=g['cell_area_km2']
        tot=g['precip_total']; liq=g['liq_total']; sn=g['snowfall']+g['graupel_hail']
        hmax=int(np.ceil(h[lm].max()/100)*100)
        prev=None
        for lo in range(0,hmax,100):
            m=lm&(h>=lo)&(h<lo+100)
            if m.sum()==0: continue
            a=A[m].sum(); P=(tot*A)[m].sum()/a; R=(liq*A)[m].sum()/a; S=(sn*A)[m].sum()/a
            rows.append(dict(event=ev,run=run,band_lo=lo,band_hi=lo+100,area_km2=a,ncell=int(m.sum()),precip=P,rain=R,snow=S,rain_share=R/P if P>0 else np.nan))
        # rain-snow line: lowest band where snow share crosses 0.5 (interpolated), requiring precip>5 mm
        d=pd.DataFrame([r for r in rows if r['event']==ev and r['run']==run]); d=d[d.precip>5]
        ss=1-d.rain_share.values; mid=d.band_lo.values+50
        rsl=np.nan
        for i in range(1,len(ss)):
            if ss[i-1]<0.5<=ss[i]:
                rsl=mid[i-1]+(0.5-ss[i-1])/(ss[i]-ss[i-1])*100; break
        lines.append(dict(event=ev,run=run,rsl_m=rsl,hmax=h[lm].max()))
E=pd.DataFrame(rows); E.to_csv('elev_bands_local.csv',index=False)
print(pd.DataFrame(lines))
for ev in E.event.unique():
    d=E[(E.event==ev)]
    print(ev); print(d.pivot(index='band_lo',columns='run',values='rain_share').round(2).T.to_string())
