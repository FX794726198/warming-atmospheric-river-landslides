# per-cell AR share of extreme-rain days (2-day rain >= local P99) 1979-2022, and AR day frequency
import numpy as np
E='work/data/era5'; I='work/data/arcat'; O='work/out/g1'
p99=np.load(f'{O}/r2_p99.npy')
nex=np.zeros((146,360)); nar=np.zeros((146,360)); nmon=np.zeros((12,146,360)); narm=np.zeros((12,146,360))
prev_pr=None; prev_ar=None
for y in range(1979,2023):
    pr=np.nan_to_num(np.load(f'{E}/pr_{y}.npy').astype(np.float32)); nd=pr.shape[0]
    ar=np.unpackbits(np.load(f'{I}/ard_{y}.npy'),axis=0)[:nd].astype(bool)
    r2=pr.copy(); r2[1:]+=pr[:-1]
    if prev_pr is not None: r2[0]+=prev_pr
    w=ar.copy(); w[1:]|=ar[:-1]
    if prev_ar is not None: w[0]|=prev_ar
    prev_pr=pr[-1]; prev_ar=ar[-1]
    ex=r2>=p99[None]
    nex+=ex.sum(0); nar+=(ex&w).sum(0)
    print(y,flush=True)
np.savez(f'{O}/cell_ex_ar.npz',nex=nex,nar=nar)
