# daily AR presence (any 6-h step in day) per cell, and monthly climatology of 2-day window (d-1 or d) 1979-2024
import numpy as np, os
I='work/data/arcat'
Y0,Y1=1979,2024
prev=None
num=np.zeros((12,146,360)); den=np.zeros(12); num1=np.zeros((12,146,360))
for y in range(Y0,Y1+1):
    a=np.load(f'{I}/ar_{y}.npy',mmap_mode='r')
    nd=a.shape[0]//4
    d=(np.asarray(a).reshape(nd,4,146,360)>0).any(1)
    np.save(f'{I}/ard_{y}.npy',np.packbits(d,axis=0))
    w=d.copy(); w[1:]|=d[:-1]
    if prev is not None: w[0]|=prev
    prev=d[-1]
    days=np.arange(np.datetime64(f'{y}-01-01'),np.datetime64(f'{y+1}-01-01'))
    mo=days.astype('datetime64[M]').astype(int)%12
    for m in range(12):
        s=mo==m; num[m]+=w[s].sum(0); num1[m]+=d[s].sum(0); den[m]+=s.sum()
    print(y,nd,flush=True)
np.savez(f'{I}/clim_daily.npz',p2=(num/den[:,None,None]).astype(np.float32),p1=(num1/den[:,None,None]).astype(np.float32))
