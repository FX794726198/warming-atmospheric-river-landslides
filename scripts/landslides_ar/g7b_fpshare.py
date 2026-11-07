# AR share of extreme-rain cells as a function of the connected footprint size of extreme rain (8-connected land cells with 2-day rain >= P99)
import numpy as np, pandas as pd
from scipy import ndimage
E='work/data/era5'; I='work/data/arcat'; O='work/out/g1'
lsm=np.load('work/out/a2/lsm.npy')>0.5
p99=np.load(f'{O}/r2_p99.npy')
st=np.ones((3,3),bool)
rows=[]; prev_pr=None; prev_ar=None
for y in range(1979,2023):
    pr=np.nan_to_num(np.load(f'{E}/pr_{y}.npy').astype(np.float32)); nd=pr.shape[0]
    ar=np.unpackbits(np.load(f'{I}/ard_{y}.npy'),axis=0)[:nd].astype(bool)
    r2=pr.copy(); r2[1:]+=pr[:-1]
    if prev_pr is not None: r2[0]+=prev_pr
    w=ar.copy(); w[1:]|=ar[:-1]
    if prev_ar is not None: w[0]|=prev_ar
    prev_pr=pr[-1]; prev_ar=ar[-1]
    ex=(r2>=p99[None])&lsm[None]
    for t in range(nd):
        lab,n=ndimage.label(ex[t],structure=st)
        if n==0: continue
        # wrap-around not handled (minor)
        idx=np.arange(1,n+1)
        size=ndimage.sum(np.ones_like(lab),lab,idx)
        arfrac=ndimage.mean(w[t],lab,idx)
        cy=ndimage.mean(np.arange(146)[:,None]*np.ones((1,360)),lab,idx)
        for s,a,c in zip(size,arfrac,cy):
            rows.append((y,t,int(s),float(a),float(c)-70))
    print(y,len(rows),flush=True)
D=pd.DataFrame(rows,columns=['y','t','size','arfrac','lat'])
D.to_parquet(f'{O}/extreme_objects.parquet')
