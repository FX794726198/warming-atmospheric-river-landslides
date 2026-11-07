"""Daily ERA5 total precipitation and snowfall (0.25 deg) from ARCO-ERA5 (public GCS), land cells at 20-72 deg latitude.
Day D = hourly accumulations ending D 01:00 ... D+1 00:00 UTC; stored per year as uint16 (0.1 mm). Usage: python arco_daily.py Y0 Y1 OUTDIR"""
import urllib.request, numpy as np, numcodecs, datetime as dt, concurrent.futures as cf, os, sys, time
BASE = 'https://storage.googleapis.com/gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3'
Y0, Y1, OUT = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]; os.makedirs(OUT, exist_ok=True)
codec = numcodecs.Blosc(cname='lz4', clevel=5, shuffle=1)
H = lambda d: int((d - dt.datetime(1900, 1, 1)).total_seconds() // 3600)
def raw(v, key):
    for k in range(6):
        try: return np.frombuffer(codec.decode(urllib.request.urlopen(f'{BASE}/{v}/{key}', timeout=120).read()), '<f4')
        except Exception as e: err = e; time.sleep(2 ** k)
    raise RuntimeError(f'{v} {key}: {err}')
if not os.path.exists(f'{OUT}/cells.npz'):
    lat = np.arange(90, -90.01, -0.25, dtype=np.float32); lon = np.arange(0, 360, 0.25, dtype=np.float32)
    h = H(dt.datetime(2000, 1, 1)); zs = raw('geopotential_at_surface', f'{h}.0.0') / 9.80665; lsm = raw('land_sea_mask', f'{h}.0.0')
    LA = np.repeat(lat, 1440); idx = np.flatnonzero((lsm > 0.5) & (np.abs(LA) >= 20) & (np.abs(LA) <= 72))
    np.savez(f'{OUT}/cells.npz.tmp.npz', idx=idx, lat=lat, lon=lon, zs=zs[idx], lsm=lsm[idx]); os.replace(f'{OUT}/cells.npz.tmp.npz', f'{OUT}/cells.npz')
IDX = np.load(f'{OUT}/cells.npz')['idx']
get = lambda v, h: raw(v, f'{h}.0.0')[IDX]
EX = cf.ThreadPoolExecutor(int(os.environ.get('NTHREADS', 48)))
for y in range(Y0, Y1 + 1):
    f = f'{OUT}/daily_{y}.npz'
    if os.path.exists(f): continue
    days = [dt.datetime(y, 1, 1) + dt.timedelta(days=i) for i in range((dt.datetime(y + 1, 1, 1) - dt.datetime(y, 1, 1)).days)]
    TP = np.zeros((len(days), len(IDX)), np.uint16); SF = np.zeros_like(TP); nanh = np.zeros(len(days), np.int16); t0 = time.time(); futs = {}
    def submit(i):
        hs = [H(days[i]) + k + 1 for k in range(24)]
        futs[i] = ([EX.submit(get, 'total_precipitation', h) for h in hs], [EX.submit(get, 'snowfall', h) for h in hs])
    for i in range(min(3, len(days))): submit(i)
    for i in range(len(days)):
        if i + 3 < len(days): submit(i + 3)
        ft, fs = futs.pop(i); tp = np.array([x.result() for x in ft]); sf = np.array([x.result() for x in fs])
        nanh[i] = int(np.isnan(tp).any(1).sum())
        TP[i] = np.round(np.nansum(tp, 0) * 1e4).clip(0, 65535).astype(np.uint16); SF[i] = np.round(np.nansum(sf, 0) * 1e4).clip(0, 65535).astype(np.uint16)
    np.savez_compressed(f + '.tmp.npz', tp=TP, sf=SF, nan_hours=nanh, year=y); os.replace(f + '.tmp.npz', f)
    print(y, 'done %.0f s' % (time.time() - t0), 'nan hours', int(nanh.sum()), flush=True)
