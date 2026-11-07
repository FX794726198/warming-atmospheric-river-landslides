# match landslides to Chen 2026 AR mask: AR on day d or d-1 at the cell (and 3x3 neighbourhood); climatological baseline by cell-month
import numpy as np, pandas as pd
I='work/data/arcat'; O='work/out/g1'
import os; os.makedirs(O,exist_ok=True)
L=pd.read_parquet('work/out/landslides_all.parquet')
L['date']=pd.to_datetime(L.date); L=L[(L.date>='1979-01-02')&(L.date<='2024-12-31')].copy()
L=L[(L.ci>=0)&(L.ci<146)].reset_index(drop=True)
D={}
for y in range(1979,2025):
    nd=366 if y%4==0 else 365
    D[y]=np.unpackbits(np.load(f'{I}/ard_{y}.npy'),axis=0)[:nd].astype(bool)
def ar_at(dates,ci,cj,nb=0):
    out=np.zeros(len(dates),bool)
    for k,(d,i,j) in enumerate(zip(dates,ci,cj)):
        for dd in (d,d-pd.Timedelta(days=1)):
            a=D[dd.year][dd.dayofyear-1]
            if nb==0: v=a[i,j]
            else: v=a[max(0,i-nb):i+nb+1,[(j+o)%360 for o in range(-nb,nb+1)]].any()
            if v: out[k]=True;break
    return out
L['ar']=ar_at(L.date,L.ci,L.cj,0)
L['ar3']=ar_at(L.date,L.ci,L.cj,1)
c=np.load(f'{I}/clim_daily.npz'); p2=c['p2']
L['pc']=p2[L.date.dt.month-1,L.ci,L.cj]
# baseline for 3x3: compute on the fly from daily arrays (cell-month climatology of neighbourhood window)
L.to_parquet(f'{O}/ls_ar.parquet')
print(L.groupby('src').agg(n=('ar','size'),ar=('ar','mean'),ar3=('ar3','mean'),pc=('pc','mean')).round(3))
