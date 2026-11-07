import gcsfs, xarray as xr, numpy as np, pandas as pd
fs=gcsfs.GCSFileSystem(token='anon')
ds=xr.open_zarr(fs.get_mapper('gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3'),consolidated=True,chunks=None)
U='vertical_integral_of_eastward_water_vapour_flux'; V='vertical_integral_of_northward_water_vapour_flux'
C={'KZN':('2022-04-10','2022-04-12T18',(-46,-10),(5,58),(-29.9,30.9)),
   'KOR':('2023-07-13','2023-07-15T18',(18,54),(100,154),(36.9,128.5))}
for k,(t0,t1,la,lo,cen) in C.items():
    t=pd.date_range(t0,t1,freq='6h')
    lon=np.arange(lo[0],lo[1]+0.01,0.25)%360
    sub=ds[[U,V]].sel(time=t,latitude=slice(la[1],la[0])).sel(longitude=lon)
    u=sub[U].values; v=sub[V].values; ivt=np.hypot(u,v)
    th=pd.date_range(t0,pd.Timestamp(t1)+pd.Timedelta(hours=5),freq='1h')
    pr=ds['total_precipitation'].sel(time=th,latitude=slice(la[1],la[0])).sel(longitude=lon).sum('time').values*1000
    np.savez_compressed(f'work/data/cases/{k}.npz',u=u.astype(np.float32),v=v.astype(np.float32),pr=pr.astype(np.float32),
        lat=sub.latitude.values,lon=np.where(lon>180,lon-360,lon),time=t.values.astype('datetime64[h]').astype(str))
    i=np.argmin(abs(sub.latitude.values-cen[0])); j=np.argmin(abs(np.where(lon>180,lon-360,lon)-cen[1]))
    print(k, ivt.shape, 'IVT at centre by step', [(str(a)[5:13], int(b)) for a,b in zip(t, ivt[:,i,j])], 'max pr', round(float(pr.max())), flush=True)
