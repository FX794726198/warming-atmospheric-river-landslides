"""CMIP6 monthly-mean PGW deltas, (2071-2100) minus (1985-2014), multi-model mean of 10 models (r1i1p1*, Amon),
global 1 deg grid, months 4, 5, 11, 12; variables ta (plev), tas, ts.  Pangeo CMIP6 GCS zarr store,
global and per month.  Output: cmip6/delta_{ssp585,ssp245}_monthly.npz"""
import os
import gcsfs, xarray as xr, numpy as np, pandas as pd, warnings, sys
warnings.simplefilter("ignore")
fs = gcsfs.GCSFileSystem(token="anon")
with fs.open("cmip6/cmip6-zarr-consolidated-stores.csv") as f: C = pd.read_csv(f)
models = ["ACCESS-CM2","CNRM-CM6-1","EC-Earth3","GFDL-ESM4","IPSL-CM6A-LR","MIROC6","MPI-ESM1-2-HR","MRI-ESM2-0","NorESM2-MM","UKESM1-0-LL"]
MONTHS = [4, 5, 11, 12]
LAT = np.arange(-90, 90.1, 1.0); LON = np.arange(0, 360, 1.0)
OUT = os.path.expandvars("/scratch/$USER/ar_pgw/cmip6")
cache = {}
def clim(model, exp, var):
    key = (model, exp, var)
    if key in cache: return cache[key]
    x = C[(C.source_id == model) & (C.experiment_id == exp) & (C.variable_id == var) & (C.table_id == "Amon") & (C.member_id.str.startswith("r1i1p1"))]
    if len(x) == 0: raise ValueError(f"no store {model} {exp} {var}")
    x = x.sort_values("member_id")
    ds = xr.open_zarr(fs.get_mapper(x.zstore.iloc[0]), consolidated=True, decode_times=True, use_cftime=True)
    a = ds[var]
    y0, y1 = (1985, 2014) if exp == "historical" else (2071, 2100)
    t = a.time.dt; sel = np.isin(t.month.values, MONTHS) & (t.year.values >= y0) & (t.year.values <= y1)
    a = a.isel(time=np.where(sel)[0])
    nyr = len(np.unique(a.time.dt.year.values))
    m = a.groupby("time.month").mean("time").load()           # (month, [plev], lat, lon)
    m = m.sortby("lat"); m = m.assign_coords(lon=(m.lon % 360)).sortby("lon")
    lo = m.lon.values   # periodic pad, then interpolate to the 1-deg global grid
    m = xr.concat([m.isel(lon=[-1]).assign_coords(lon=[lo[-1] - 360]), m, m.isel(lon=[0]).assign_coords(lon=[lo[0] + 360])], "lon")
    # pad poles by nearest so the interpolation covers +-90
    la = m.lat.values
    m = xr.concat([m.isel(lat=[0]).assign_coords(lat=[-90.0]), m, m.isel(lat=[-1]).assign_coords(lat=[90.0])], "lat") if la[0] > -90 else m
    r = m.interp(lat=LAT, lon=LON)
    if "plev" in r.dims: r = r.sortby("plev", ascending=False)   # surface first (Pa)
    print(f"  {model} {exp} {var} years={nyr} shape={r.shape}", flush=True)
    cache[key] = r; return r
res = {}
for exp in ("ssp585", "ssp245"):
    acc = {"ta": [], "tas": [], "ts": []}; used = []; plev = None
    for mdl in models:
        try:
            d = {}
            for v in ("ta", "tas", "ts"):
                h = clim(mdl, "historical", v); f = clim(mdl, exp, v); d[v] = (f - h)
            if plev is None: plev = d["ta"].plev.values
            ta = d["ta"].interp(plev=plev) if not np.array_equal(d["ta"].plev.values, plev) else d["ta"]
            acc["ta"].append(ta.transpose("month", "plev", "lat", "lon").values)
            acc["tas"].append(d["tas"].transpose("month", "lat", "lon").values); acc["ts"].append(d["ts"].transpose("month", "lat", "lon").values)
            used.append(mdl); print(exp, mdl, "ok global-mean tas delta by month", np.round(np.nanmean(d["tas"].values, axis=(1, 2)), 2), flush=True)
        except Exception as e: print(exp, mdl, "skip", repr(e)[:200], flush=True)
    S = {v: np.stack(acc[v]) for v in acc}
    np.savez_compressed(f"{OUT}/delta_{exp}_monthly.npz", lat=LAT, lon=LON, plev=plev, months=np.array(MONTHS),
                        ta=np.nanmean(S["ta"], 0).astype("f4"), tas=np.nanmean(S["tas"], 0).astype("f4"), ts=np.nanmean(S["ts"], 0).astype("f4"),
                        spread_tas=np.nanstd(S["tas"], 0).astype("f4"), spread_ta=np.nanstd(S["ta"], 0).astype("f4"), models=np.array(used))
    # also keep each model's tas/ta delta (for later scaling / spread)
    np.savez_compressed(f"{OUT}/delta_{exp}_monthly_permodel.npz", lat=LAT, lon=LON, plev=plev, months=np.array(MONTHS),
                        ta=S["ta"].astype("f2"), tas=S["tas"].astype("f4"), ts=S["ts"].astype("f4"), models=np.array(used))
    print("saved", exp, "models", used, flush=True)
    # free future-period caches, keep historical for the second scenario
    for k in [k for k in cache if k[1] == exp]: del cache[k]
print("DONE", flush=True)
