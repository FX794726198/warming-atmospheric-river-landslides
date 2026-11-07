"""Print 0 degC height (mean -60..+20 km), snow share within +-30 km, terrain max, for each event and run."""
import numpy as np, sys
R='work/nc/proc/cluster'
def field(z, run, var):
    a=z[f'{run}_{var}'].astype(float); a[a==float(z['fill'])]=np.nan; return a*float(z[f'scale_{var}'])
for ev in sys.argv[1:] or ['NOR_2011','BC_2021','ORE_1996','LIG_2014']:
    try: z=np.load(f'{R}/WRF_XSEC_{ev}.npz'); z['ctl_T']
    except Exception as e: print(ev,'not available'); continue
    d=z['dist_km']; zz=z['z_m'].astype(float); hgt=z['hgt_m'].astype(float)
    print(f'== {ev}  terrain max on section {np.nanmax(hgt):.0f} m  times', {k: str(z[k]) for k in z.files if 'times' in k})
    for run in ('ctl','pgw585_s050','pgw585'):
        if f'{run}_T' not in z.files: continue
        T=field(z,run,'T'); r=field(z,run,'rain_mmh'); s=field(z,run,'snow_mmh')
        fl=np.full(len(d),np.nan)
        for j in range(len(d)):
            col=T[:,j]; ok=~np.isnan(col)
            if ok.sum()<2: continue
            zc=zz[ok]; tc=col[ok]
            # lowest crossing from >0 below to <0 above
            idx=np.where((tc[:-1]>=0)&(tc[1:]<0))[0]
            if len(idx): k=idx[0]; fl[j]=zc[k]+(zc[k+1]-zc[k])*tc[k]/(tc[k]-tc[k+1])
            elif tc[0]<0: fl[j]=zc[0]
        m=(d>=-60)&(d<=20); core=np.abs(d)<=30
        share=np.nansum(s[core])/np.nansum((r+s)[core])*100
        print(f'  {run:12s} 0C height {np.nanmean(fl[m]):.0f} m (min {np.nanmin(fl[m]):.0f}, max {np.nanmax(fl[m]):.0f})  snow share +-30 km {share:.0f}%  rain max {np.nanmax(r):.1f} snow max {np.nanmax(s):.1f} mm/h')
