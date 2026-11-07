import gcsfs, xarray as xr, numpy as np, pandas as pd
fs=gcsfs.GCSFileSystem(token='anon')
ds=xr.open_zarr(fs.get_mapper('gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3'),consolidated=True,chunks=None)
U='vertical_integral_of_eastward_water_vapour_flux'; V='vertical_integral_of_northward_water_vapour_flux'
print(U in ds, V in ds)
C={'ORE':('1996-11-16','1996-11-19T18',(28,57),(-150,-108),(43.7,-123.9)),
   'NOR':('2011-12-24','2011-12-27T18',(44,72),(-35,28),(60.9,6.6)),
   'RSB':('2024-04-27','2024-05-02T18',(-42,-8),(-78,-32),(-29.1,-51.9)),
   'LIG':('2014-11-09','2014-11-16T18',(30,52),(-20,25),(44.5,8.8))}
for k,(t0,t1,la,lo,cen) in C.items():
    t=pd.date_range(t0,t1,freq='6h')
    lon=np.arange(lo[0],lo[1]+0.01,0.25)%360
    sub=ds[[U,V,'total_precipitation']].sel(time=t,latitude=slice(la[1],la[0]))
    sub=sub.sel(longitude=lon)
    u=sub[U].values; v=sub[V].values
    ivt=np.hypot(u,v)
    # precip: hourly accumulations; sum over full window hourly
    th=pd.date_range(t0,pd.Timestamp(t1)+pd.Timedelta(hours=5),freq='1h')
    pr=ds['total_precipitation'].sel(time=th,latitude=slice(la[1],la[0])).sel(longitude=lon).sum('time').values*1000
    np.savez_compressed(f'work/data/cases/{k}.npz',u=u.astype(np.float32),v=v.astype(np.float32),pr=pr.astype(np.float32),
        lat=sub.latitude.values,lon=np.where(lon>180,lon-360,lon),time=t.values.astype('datetime64[h]').astype(str))
    i=np.argmin(abs(sub.latitude.values-cen[0])); j=np.argmin(abs(np.where(lon>180,lon-360,lon)-cen[1]))
    print(k, ivt.shape, 'peak at centre', t[np.argmax(ivt[:,i,j])], round(float(ivt[:,i,j].max())), 'max pr', round(float(pr.max())),flush=True)
