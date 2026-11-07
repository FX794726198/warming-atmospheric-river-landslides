# Chen et al. 2026 ESSD AR catalog -> per-year uint8 label arrays (T,146,360), lat -70..75 ascending, 1979-2024
import netCDF4 as nc, numpy as np, os, datetime as dt, time
F="work/data/ar_catalog/AR_1940-2024.nc"
O='work/data/arcat'
f=nc.Dataset(F); v=f['AR_map']; v.set_auto_maskandscale(False)
CH=15524
def idx(y): return (dt.date(y,1,1)-dt.date(1940,1,1)).days*4
Y0,Y1=1979,2024
s0=idx(Y0); s1=idx(Y1+1)
k0=s0//CH; k1=(s1-1)//CH
buf={}  # partial years
clim=np.zeros((12,146,360),np.float64); nmon=np.zeros(12)
for k in range(k0,k1+1):
    t=time.time()
    a,b=max(k*CH,s0),min((k+1)*CH,s1)
    blk=np.asarray(v[a:b,15:161,:])[:,::-1,:]   # lat 75..-70 -> -70..75
    for y in range(Y0,Y1+1):
        ya,yb=idx(y),idx(y+1)
        lo,hi=max(ya,a),min(yb,b)
        if lo>=hi: continue
        buf.setdefault(y,[]).append(blk[lo-a:hi-a])
        if hi==yb:
            arr=np.concatenate(buf.pop(y),0)
            assert arr.shape[0]==yb-ya
            np.save(f'{O}/ar_{y}.npy',np.ascontiguousarray(arr))
            # monthly AR frequency
            days=np.arange(np.datetime64(f'{y}-01-01'),np.datetime64(f'{y+1}-01-01'),np.timedelta64(6,'h'))
            mo=days.astype('datetime64[M]').astype(int)%12
            for m in range(12):
                clim[m]+= (arr[mo==m]>0).sum(0); nmon[m]+=(mo==m).sum()
            print(y,arr.shape,flush=True)
    print('chunk',k,round(time.time()-t,1),flush=True)
np.savez(f'{O}/ar_freq_monthly_1979_2024.npz',freq=(clim/nmon[:,None,None]).astype(np.float32),n=nmon)
print('done')
