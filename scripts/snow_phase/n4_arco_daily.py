"""Daily total precipitation and snowfall (ERA5, 0.25 deg) from the ARCO-ERA5 hourly store on Google Cloud, for land cells at
20-72 deg latitude in both hemispheres. Day D = accumulations ending at D 01:00 ... D+1 00:00 UTC. Stored per year as uint16 in 0.1 mm.
Resumable: a year whose file exists is skipped. Order: 1979-2024, then 1978 back to 1940."""
import urllib.request, ssl, numpy as np, numcodecs, datetime as dt, concurrent.futures as cf, os, time, shutil, sys
ctx = ssl.create_default_context(cafile='/root/.ccr/ca-bundle.crt')
BASE = 'https://storage.googleapis.com/gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3'
OUT = 'work/out/n4'; codec = numcodecs.Blosc(cname='lz4', clevel=5, shuffle=1)
S = np.load('work/data/era5_025_static.npz'); LA = np.broadcast_to(S['lat'][:, None], S['zs'].shape)
sel = (S['lsm'] > 0.5) & (np.abs(LA) >= 20) & (np.abs(LA) <= 72); IDX = np.flatnonzero(sel.ravel())
if not os.path.exists(f'{OUT}/cells.npz'): np.savez(f'{OUT}/cells.npz', idx=IDX, lat=S['lat'], lon=S['lon'], zs=S['zs'].ravel()[IDX], lsm=S['lsm'].ravel()[IDX])
H = lambda d: int((d - dt.datetime(1900, 1, 1)).total_seconds() // 3600)
def get(v, h):
    for k in range(6):
        try:
            b = urllib.request.urlopen(f'{BASE}/{v}/{h}.0.0', context=ctx, timeout=120).read()
            return np.frombuffer(codec.decode(b), '<f4').ravel()[IDX]
        except Exception as e:
            err = e; time.sleep(2 ** k)
    raise RuntimeError(f'{v} {h}: {err}')
EX = cf.ThreadPoolExecutor(64)
years = list(range(1979, 2025)) + list(range(1978, 1939, -1))
for y in years:
    f = f'{OUT}/daily_{y}.npz'
    if os.path.exists(f): continue
    if shutil.disk_usage('/').free < 1.0e9: print('disk nearly full, stopping', flush=True); sys.exit(1)
    days = [dt.datetime(y, 1, 1) + dt.timedelta(days=i) for i in range((dt.datetime(y + 1, 1, 1) - dt.datetime(y, 1, 1)).days)]
    TP = np.zeros((len(days), len(IDX)), np.uint16); SF = np.zeros_like(TP); nanh = np.zeros(len(days), np.int16); t0 = time.time()
    futs = {}
    def submit(i):
        hs = [H(days[i]) + k + 1 for k in range(24)]
        futs[i] = ([EX.submit(get, 'total_precipitation', h) for h in hs], [EX.submit(get, 'snowfall', h) for h in hs])
    for i in range(min(3, len(days))): submit(i)
    for i in range(len(days)):
        if i + 3 < len(days): submit(i + 3)
        ft, fs = futs.pop(i); tp = np.array([x.result() for x in ft]); sf = np.array([x.result() for x in fs])
        nanh[i] = int(np.isnan(tp).any(1).sum())
        TP[i] = np.round(np.nansum(tp, 0) * 1e4).clip(0, 65535).astype(np.uint16)   # m -> 0.1 mm
        SF[i] = np.round(np.nansum(sf, 0) * 1e4).clip(0, 65535).astype(np.uint16)
    np.savez_compressed(f + '.tmp.npz', tp=TP, sf=SF, nan_hours=nanh, year=y)
    os.replace(f + '.tmp.npz', f)
    print(y, 'done in %.0f s' % (time.time() - t0), 'nan hours', int(nanh.sum()), 'MB', round(os.path.getsize(f) / 1e6), 'free GB', round(shutil.disk_usage('/').free / 1e9, 1), flush=True)
print('all years done', flush=True)
