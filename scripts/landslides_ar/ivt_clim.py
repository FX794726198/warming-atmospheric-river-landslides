import numpy as np, pandas as pd
E='work/data/era5/'
P85=np.zeros((12,146,360),np.float32); P95=np.zeros((12,146,360),np.float32)
for m in range(1,13):
    parts=[]
    for y in range(1980,2020):
        t=pd.date_range(f'{y}-01-01',f'{y}-12-31T18',freq='6h'); sel=np.flatnonzero(t.month==m)
        a=np.load(f'{E}ivt_{y}.npy',mmap_mode='r'); parts.append(np.asarray(a[sel[0]:sel[-1]+1]))
    X=np.concatenate(parts,0)
    P85[m-1]=np.percentile(X,85,axis=0); P95[m-1]=np.percentile(X,95,axis=0)
    print(m,flush=True)
np.savez('work/data/ivt_monthly_pct.npz',P85=P85,P95=P95)
