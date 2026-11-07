"""Regional yearly shares of AR extreme-precipitation cell-days by temperature class, with trends."""
import numpy as np, pandas as pd, sys
from scipy import stats
N='work/out/n1'; f=sys.argv[1] if len(sys.argv)>1 else f'{N}/arex_temp_1979_2022.npz'
z=np.load(f); cnt=z['cnt'].astype(np.int64); tsum=z['tsum'].astype(np.float64); years=z['years']
lsm=np.load('work/out/a2/lsm.npy')>0.5; zs=np.load('work/data/era5/zs_1deg.npy')
lat=np.arange(-70,76); lon=np.arange(0,360); LAT,LON=np.meshgrid(lat,lon,indexing='ij'); LONW=np.where(LON>180,LON-360,LON)
area=np.cos(np.deg2rad(LAT))
REG={'Scandinavia':(57,71,4,20),'British Columbia and Alaska coast':(48,62,-140,-115),'US Pacific Northwest and Rockies':(40,49,-125,-105),
     'Alps':(43,48,5,16),'Caucasus and Türkiye':(36,44,26,48),'Hindu Kush and Himalaya':(27,38,65,95),'Japan':(33,45,130,145),
     'Southern Andes':(-55,-33,-75,-66),'New Zealand':(-47,-34,166,179),'Iceland and Britain':(50,66,-25,0),'Northeastern North America':(40,50,-80,-60),
     'All land poleward of 25°':(-70,75,-180,180)}
rows=[]
def trend(y,v):
    m=np.isfinite(v)
    if m.sum()<10: return np.nan,np.nan,np.nan
    r=stats.linregress(y[m],v[m]); ts=stats.theilslopes(v[m],y[m]); return r.slope*10, r.pvalue, ts[0]*10
for nm,(la0,la1,lo0,lo1) in REG.items():
    box=(LAT>=la0)&(LAT<=la1)&(LONW>=lo0)&(LONW<=lo1)&lsm
    if nm.startswith('All'): box&=(np.abs(LAT)>=25)
    for elev,em in (('all',box),('>=300 m',box&(zs>=300))):
        for s,sn in ((0,'cold'),(1,'all year')):
            c=cnt[:,0] if s==0 else cnt.sum(1); t=tsum[:,0] if s==0 else tsum.sum(1)
            w=area*em
            n_ar=(c[:,1]*w).sum((1,2)); n_ex=(c[:,0]*w).sum((1,2)); sub0=((c[:,2]+c[:,3])*w).sum((1,2)); m40=(c[:,3]*w).sum((1,2)); tm=(t*w).sum((1,2))
            with np.errstate(invalid='ignore',divide='ignore'):
                sh0=100*sub0/n_ar; sh40=100*m40/n_ar; shar=100*n_ar/n_ex; tmean=tm/n_ar
            r=dict(region=nm,elev=elev,season=sn,cells=int(em.sum()),ar_celldays_per_yr=n_ar.mean()/area[em].sum() if em.sum() else np.nan,
                   sub0_share_first=np.nanmean(sh0[:12]),sub0_share_last=np.nanmean(sh0[-12:]),
                   sub0_trend_per_dec=trend(years,sh0)[0],sub0_p=trend(years,sh0)[1],sub0_theil=trend(years,sh0)[2],
                   m40_share_first=np.nanmean(sh40[:12]),m40_share_last=np.nanmean(sh40[-12:]),m40_trend=trend(years,sh40)[0],m40_p=trend(years,sh40)[1],
                   t_first=np.nanmean(tmean[:12]),t_last=np.nanmean(tmean[-12:]),t_trend=trend(years,tmean)[0],t_p=trend(years,tmean)[1],
                   arshare_first=np.nanmean(shar[:12]),arshare_last=np.nanmean(shar[-12:]),arshare_trend=trend(years,shar)[0],arshare_p=trend(years,shar)[1],
                   n_ar_trend_pct=trend(years,100*n_ar/np.nanmean(n_ar))[0],n_ar_p=trend(years,n_ar)[1])
            rows.append(r)
            if elev=='>=300 m' and s==0: pd.DataFrame(dict(year=years,sub0=sh0,m40=sh40,tmean=tmean,n_ar=n_ar,ar_share=shar)).to_csv(f'{N}/series_{nm[:12].replace(" ","_")}.csv',index=False)
R=pd.DataFrame(rows); R.to_csv(f'{N}/regional_trends.csv',index=False)
pd.set_option('display.width',250); pd.set_option('display.max_columns',30)
cols=['region','elev','season','cells','sub0_share_first','sub0_share_last','sub0_trend_per_dec','sub0_p','m40_share_first','m40_share_last','t_first','t_last','t_trend','t_p','arshare_first','arshare_last','n_ar_trend_pct']
print(R[(R.season=='cold')][cols].round(2).to_string())
print(R[(R.season=='all year')&(R.elev=='>=300 m')][cols].round(2).to_string())
