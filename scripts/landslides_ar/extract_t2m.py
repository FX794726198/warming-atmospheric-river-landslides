import gcsfs, xarray as xr, numpy as np, sys, os, time
fs=gcsfs.GCSFileSystem(token='anon')
ds=xr.open_zarr(fs.get_mapper('weatherbench2/datasets/era5/1959-2023_01_10-6h-360x181_equiangular_with_poles_conservative.zarr'),consolidated=True)
out='work/data/era5'
for y in [int(a) for a in sys.argv[1:]]:
    f=f'{out}/t2m_{y}.npy'
    if os.path.exists(f): continue
    t=time.time()
    a=ds['2m_temperature'].sel(time=slice(f'{y}-01-01',f'{y}-12-31T18'), latitude=slice(-70,75)).transpose('time','latitude','longitude').values
    a=a.reshape(-1,4,146,360)
    np.save(f+'.tmp.npy',(np.stack([a.mean(1),a.max(1)],1)-273.15).astype(np.float16))  # day, (mean,max), lat, lon  in K
    os.rename(f+'.tmp.npy',f); print(y,round(time.time()-t,1),flush=True)
