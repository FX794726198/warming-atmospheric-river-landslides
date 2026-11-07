# Extract ERA5 (WB2 1 deg, 6-hourly) IVT (kg/m/s, uint16) and daily precip (mm, float16) for lat -70..75, per year
import gcsfs, xarray as xr, numpy as np, sys, os, time
fs=gcsfs.GCSFileSystem(token='anon')
ds=xr.open_zarr(fs.get_mapper('weatherbench2/datasets/era5/1959-2023_01_10-6h-360x181_equiangular_with_poles_conservative.zarr'),consolidated=True)
out='work/data/era5'; os.makedirs(out,exist_ok=True)
for y in [int(a) for a in sys.argv[1:]]:
    f1=f'{out}/ivt_{y}.npy'; f2=f'{out}/pr_{y}.npy'
    if os.path.exists(f2): continue
    t=time.time()
    s=ds.sel(time=slice(f'{y}-01-01',f'{y}-12-31T18'), latitude=slice(-70,75))
    ivt=s.integrated_vapor_transport.transpose('time','latitude','longitude').values*100.
    ivt=np.clip(np.nan_to_num(ivt),0,65000).round().astype(np.uint16)
    np.save(f1+'.tmp.npy',ivt); os.rename(f1+'.tmp.npy',f1)
    # daily precip: 24h accumulation ending at 00 UTC of day d+1 -> assign to day d
    p=ds.total_precipitation_24hr.sel(time=slice(f'{y}-01-02T00',f'{y+1}-01-01T00'), latitude=slice(-70,75))
    p=p.sel(time=p.time.dt.hour==0).transpose('time','latitude','longitude').values*1000.
    np.save(f2+'.tmp.npy',p.astype(np.float16)); os.rename(f2+'.tmp.npy',f2)
    print(y, ivt.shape, p.shape, round(time.time()-t,1), flush=True)
