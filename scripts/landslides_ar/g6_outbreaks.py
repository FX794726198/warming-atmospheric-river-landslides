# global landslide outbreak catalogue and weather driver: AR (Chen 2026) vs TC (IBTrACS) vs other
import numpy as np, pandas as pd
O='work/out/g1'; I='work/data/arcat'
T0=pd.Timestamp('1979-01-01')
EV=pd.read_parquet(f'{O}/events.parquet')
EV['date0']=T0+pd.to_timedelta(EV.d0,'D'); EV['date1']=T0+pd.to_timedelta(EV.d1,'D')
EV['lat']=EV.ci-70; EV['lon']=np.where(EV.cj>180,EV.cj-360,EV.cj)
EV['kind']='cluster'
C=pd.read_parquet(f'{O}/coolr_events.parquet')
C=pd.DataFrame(dict(src='COOLR_ev',n=C.n,date0=C.d0,date1=C.d1,lat=C.lat,lon=C.lon,kind='inventory',title=C.event_title,storm=C.storm))
A=pd.concat([EV[['src','n','date0','date1','lat','lon','kind']],C],ignore_index=True)
A=A[(A.date0>='1980-01-03')&(A.date1<='2024-12-31')].reset_index(drop=True)
# AR flag: any AR step at centroid cell (nb=0) or 3x3 (nb=1) from date0-1 to date1
cache={}
def arr(y):
    if y not in cache: cache[y]=np.load(f'{I}/ar_{y}.npy',mmap_mode='r')
    return cache[y]
def arflag(d0,d1,lat,lon,nb):
    ci=int(round(lat+70)); cj=int(round(lon))%360
    if ci<0 or ci>145: return np.nan
    d=d0-pd.Timedelta(days=1); hit=False
    while d<=d1:
        a=arr(d.year); s=(d.dayofyear-1)*4
        blk=np.asarray(a[s:s+4,max(0,ci-nb):ci+nb+1][:,:,[(cj+o)%360 for o in range(-nb,nb+1)]])
        if (blk>0).any(): return True
        d+=pd.Timedelta(days=1)
    return False
A['ar0']=[arflag(a,b,la,lo,0) for a,b,la,lo in zip(A.date0,A.date1,A.lat,A.lon)]
A['ar1']=[arflag(a,b,la,lo,1) for a,b,la,lo in zip(A.date0,A.date1,A.lat,A.lon)]
# TC flag
tr=pd.read_parquet(f'{O}/ibtracs.parquet'); tr=tr.dropna(subset=['LAT','LON'])
tr['day']=tr.ISO_TIME.dt.floor('D')
byday={d:g[['LAT','LON']].values for d,g in tr.groupby('day')}
def tcdist(d0,d1,lat,lon):
    best=1e9; d=d0-pd.Timedelta(days=2)
    while d<=d1:
        p=byday.get(d)
        if p is not None:
            la1,lo1=np.radians(lat),np.radians(lon); la2,lo2=np.radians(p[:,0]),np.radians(p[:,1])
            h=np.sin((la2-la1)/2)**2+np.cos(la1)*np.cos(la2)*np.sin((lo2-lo1)/2)**2
            best=min(best,float((2*6371*np.arcsin(np.sqrt(h))).min()))
        d+=pd.Timedelta(days=1)
    return best
A['tcd']=[tcdist(a,b,la,lo) for a,b,la,lo in zip(A.date0,A.date1,A.lat,A.lon)]
A.to_parquet(f'{O}/outbreaks_all.parquet'); print(len(A))
