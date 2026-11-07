"""ERA5 pressure-level fields from ARCO-ERA5 (gs://gcp-public-data-arco-era5, anonymous) -> WPS intermediate files (version 5).
usage: arco_pl.py EVENT START(YYYY-mm-dd_HH) END OUTDIR [PREFIX=ARCO] [STEP_H=3]
Writes OUTDIR/PREFIX:YYYY-MM-DD_HH with TT, UU, VV, GHT, SPECHUMD and RH on the 37 ERA5 pressure levels over the event's
ERA5 area (wps/era5_areas.json, N/W/S/E), same as the CDS request. Conventions follow ungrib + Vtable.ECMWF:
GHT = geopotential / 9.81; RH from q, T and p with the IFS mixed-phase saturation (water above 0 C, ice below -23 C,
quadratic blend in between), which is how ERA5 relative humidity is defined. Surface fields still come from CDS via ungrib."""
import sys, os, json, struct, datetime as dt, numpy as np, xarray as xr
from concurrent.futures import ThreadPoolExecutor
ev, s, e, outd = sys.argv[1:5]; pre = sys.argv[5] if len(sys.argv) > 5 else "ARCO"; step = int(sys.argv[6]) if len(sys.argv) > 6 else 3
os.makedirs(outd, exist_ok=True)
N, W, S, E = json.load(open(os.path.expandvars("/scratch/$USER/ar_pgw/wps/era5_areas.json")))[ev]
ds = xr.open_zarr("gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3", chunks=None, storage_options=dict(token="anon"))
lat = ds.latitude.values; lon = ds.longitude.values; lev = ds.level.values.astype(float)
jj = np.where((lat >= S) & (lat <= N))[0]
lons = np.arange(W, E + 0.001, 0.25); ii = np.round((lons % 360) / 0.25).astype(int) % 1440
la = lat[jj][::-1]                                         # ascending (SW corner start)
t0 = dt.datetime.strptime(s, "%Y-%m-%d_%H"); t1 = dt.datetime.strptime(e, "%Y-%m-%d_%H")
TIMES = [t0 + dt.timedelta(hours=h) for h in range(0, int((t1 - t0).total_seconds() // 3600) + 1, step)]
V = {"TT": ("temperature", "K", "Temperature"), "UU": ("u_component_of_wind", "m s-1", "U"), "VV": ("v_component_of_wind", "m s-1", "V"),
     "GHT": ("geopotential", "m", "Height"), "SPECHUMD": ("specific_humidity", "kg kg-1", "Specific humidity")}
def rec(b): return struct.pack(">i", len(b)) + b + struct.pack(">i", len(b))
def field(hdate, name, units, desc, xlvl, a):
    ny, nx = a.shape
    h = hdate.ljust(24).encode() + struct.pack(">f", 0.0) + "ERA5 ARCO (ECMWF via Google)".ljust(32).encode() + name.ljust(9).encode() + \
        units.ljust(25).encode() + desc.ljust(46).encode() + struct.pack(">f", xlvl) + struct.pack(">3i", nx, ny, 0)
    p = "SWCORNER".encode() + struct.pack(">5f", float(la[0]), float(lons[0]), 0.25, 0.25, 6367.470)
    return rec(struct.pack(">i", 5)) + rec(h) + rec(p) + rec(struct.pack(">i", 1)) + rec(a.astype(">f4").tobytes())
def esat(T):                                               # IFS mixed-phase saturation vapour pressure (Pa)
    ew = 611.21 * np.exp(17.502 * (T - 273.16) / (T - 32.19)); ei = 611.21 * np.exp(22.587 * (T - 273.16) / (T + 0.7))
    a = np.clip((T - 250.16) / (273.16 - 250.16), 0, 1) ** 2
    return a * ew + (1 - a) * ei
def one(t):
    fn = os.path.join(outd, f"{pre}:{t:%Y-%m-%d_%H}")
    if os.path.exists(fn) and os.path.getsize(fn) > 0: return fn + " exists"
    D = {}
    for k, (v, u, d) in V.items():
        x = ds[v].sel(time=np.datetime64(t)).values[:, jj][:, ::-1][:, :, ii].astype("f8")   # (lev, lat asc, lon)
        D[k] = x / 9.81 if k == "GHT" else x
    p = lev[:, None, None] * 100.0; q = D["SPECHUMD"]; e_ = p * q / (0.621981 + 0.378019 * q)
    D["RH"] = np.clip(100.0 * e_ / esat(D["TT"]), 0, None)
    hd = f"{t:%Y-%m-%d_%H}:00:00"; out = bytearray()
    for k in ("TT", "RH", "SPECHUMD", "UU", "VV", "GHT"):
        u, d = (V[k][1], V[k][2]) if k in V else ("%", "Relative Humidity")
        for il, pl in enumerate(lev): out += field(hd, k, u, d, float(pl * 100.0), D[k][il])
    open(fn + ".part", "wb").write(out); os.replace(fn + ".part", fn)
    return f"{fn} {len(out)/1e6:.0f} MB"
with ThreadPoolExecutor(8) as ex:
    for r in ex.map(one, TIMES): print(r, flush=True)
print("DONE", len(TIMES), "times", flush=True)
