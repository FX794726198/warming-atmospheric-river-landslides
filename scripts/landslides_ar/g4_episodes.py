# per-cell AR episodes from Chen mask (6-h), with ERA5 IVT max/mean and Ralph-like category; cells = landslide cells
import numpy as np, pandas as pd
I='work/data/arcat'; E='work/data/era5'; O='work/out/g1'
P=np.load(f'{O}/panel.npz'); ci,cj=P['ci'],P['cj']
M=[];V=[]
for y in range(1979,2023):
    M.append((np.load(f'{I}/ar_{y}.npy',mmap_mode='r')[:,ci,cj]>0))
    V.append(np.load(f'{E}/ivt_{y}.npy',mmap_mode='r')[:,ci,cj])
M=np.concatenate(M); V=np.concatenate(V).astype(np.float32)
t6=pd.date_range('1979-01-01','2022-12-31T18',freq='6h'); assert len(t6)==len(M)
rows=[]
for k in range(len(ci)):
    m=M[:,k]
    if not m.any(): continue
    # fill single-step gaps
    mm=m.copy(); g=(~m[1:-1])&m[:-2]&m[2:]; mm[1:-1]|=g
    d=np.diff(np.r_[0,mm.astype(int),0]); st=np.flatnonzero(d==1); en=np.flatnonzero(d==-1)-1
    v=V[:,k]
    for s,e in zip(st,en):
        seg=v[s:e+1]; dur=(e-s+1)*6; mx=float(seg.max())
        cat=int(np.clip((mx-250)//250+1,0,5)) if mx>=250 else 0
        if cat>0:
            if dur>=48: cat=min(5,cat+1)
            elif dur<24: cat=cat-1
        rows.append((k,ci[k],cj[k],s,e,dur,mx,float(seg.mean()),cat))
A=pd.DataFrame(rows,columns=['k','ci','cj','s','e','dur_h','ivt_max','ivt_mean','cat'])
A['t0']=t6[A.s]; A['t1']=t6[A.e]
A.to_parquet(f'{O}/ar_episodes.parquet'); print(len(A)); print(A.groupby('cat').agg(n=('k','size'),dur=('dur_h','median'),ivt=('ivt_max','median')))
