import numpy as np
E='work/data/era5'
P=[np.load(f'{E}/pr_{y}.npy',mmap_mode='r') for y in range(1979,2023)]
out=np.zeros((146,360),np.float32)
for i0 in range(0,146,10):
    a=np.concatenate([np.asarray(p[:,i0:i0+10,:],dtype=np.float32) for p in P]); a=np.nan_to_num(a)
    r2=a.copy(); r2[1:]+=a[:-1]
    out[i0:i0+10]=np.percentile(r2,99,axis=0)
    print(i0,flush=True)
np.save('work/out/g1/r2_p99.npy',out)
