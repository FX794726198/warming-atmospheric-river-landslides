# yearly extreme-rain (2-day >= local P99) land cell-days, and share under AR; per region; landslide-prone cells (any inventory) and all land
import numpy as np, pandas as pd
E='work/data/era5'; I='work/data/arcat'; O='work/out/g1'
lsm=np.load('work/out/a2/lsm.npy')>0.5
p99=np.load(f'{O}/r2_p99.npy')
L=pd.read_parquet(f'{O}/ls_ar.parquet'); prone=np.zeros((146,360),bool); prone[L.ci,L.cj]=True
lat=np.arange(-70,76)[:,None]*np.ones((1,360)); lon=np.ones((146,1))*np.where(np.arange(360)>180,np.arange(360)-360,np.arange(360))[None,:]
R={'WNA':(lat>=30)&(lat<=62)&(lon>=-140)&(lon<=-110),'ENA':(lat>=30)&(lat<=50)&(lon>=-95)&(lon<=-60),'EUR':(lat>=36)&(lat<=70)&(lon>=-12)&(lon<=30),
   'EAS':(lat>=25)&(lat<=45)&(lon>=100)&(lon<=145),'SAS':(lat>=25)&(lat<=40)&(lon>=60)&(lon<=100),'SAM':(lat<=-25)&(lat>=-56)&(lon>=-76)&(lon<=-40),
   'SAF':(lat<=-22)&(lat>=-35)&(lon>=15)&(lon<=35),'ANZ':(lat<=-28)&(lat>=-48)&(lon>=140)&(lon<=180)}
rows=[]; prev_pr=None; prev_ar=None
for y in range(1979,2023):
    pr=np.nan_to_num(np.load(f'{E}/pr_{y}.npy').astype(np.float32))
    nd=pr.shape[0]
    ar=np.unpackbits(np.load(f'{I}/ard_{y}.npy'),axis=0)[:nd].astype(bool)
    r2=pr.copy(); r2[1:]+=pr[:-1]
    if prev_pr is not None: r2[0]+=prev_pr
    w=ar.copy(); w[1:]|=ar[:-1]
    if prev_ar is not None: w[0]|=prev_ar
    prev_pr=pr[-1]; prev_ar=ar[-1]
    ex=(r2>=p99[None])&lsm[None]
    for k,m in R.items():
        for nm,mm in (('land',m&lsm),('prone',m&lsm&prone)):
            e=ex[:,mm]; a=w[:,mm]
            n_ex=e.sum(); n_exar=(e&a).sum()
            fp=e.sum(1); fpar=(e&a).sum(1)
            rows.append(dict(y=y,reg=k,mask=nm,ncell=int(mm.sum()),ex=int(n_ex),ex_ar=int(n_exar),big_days=int((fp>=max(3,int(0.1*mm.sum()))).sum()),big_days_ar=int(((fpar>=max(3,int(0.1*mm.sum()))) ).sum()),ar_days=int(a.any(1).sum()),ar_cells=int(a.sum())))
    print(y,flush=True)
pd.DataFrame(rows).to_csv(f'{O}/trend_extreme_ar.csv',index=False)
