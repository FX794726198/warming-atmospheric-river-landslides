"""Small per-event copies of the AR PGW post-processing outputs, for analysis off the cluster.
usage: make_light.py EVENT   -> /scratch/$USER/ar_pgw/post/light/EVENT/
  EVENT_d02_static.nc        HGT (m), LANDMASK, XLAT, XLONG (deg), CELL_AREA (km2), 301 x 301 d02 mass points (from the ctl run)
  EVENT_{ARM}.npz            as written by post_case.py (event totals, max 24/48 h, SWE, IVT-max time series)
  EVENT_{ARM}_d02_totals.nc  post_event.py d02 file without the hourly fields (liq_hourly, precip_hourly)
  EVENT_{ARM}_d01_ivt.nc     post_event.py d01 file without the 3-hourly IVT/IWV fields: event-max IVT and event precipitation on
                             the full d01 grid, d01 lat/lon/hgt/landmask, and the IVT/IWV time series (d02 footprint, target box)
  EVENT_summary.csv, EVENT_{ARM}_summary.json
Writes a marker .complete when done."""
import sys, os, shutil, numpy as np, xarray as xr
ev = sys.argv[1]; P = os.environ.get("POST_DIR", os.path.expandvars("/scratch/$USER/ar_pgw/post")); O = f"{P}/light/{ev}"; os.makedirs(O, exist_ok=True)
enc = lambda ds: {v: dict(zlib=True, complevel=4) for v in ds.data_vars}
z = np.load(f"{P}/{ev}_ctl.npz")
st = xr.Dataset(dict(HGT=(("south_north", "west_east"), z["hgt"].astype("f4"), dict(units="m")),
                     LANDMASK=(("south_north", "west_east"), z["landmask"].astype("f4"), dict(units="1 = land")),
                     XLAT=(("south_north", "west_east"), z["lat"].astype("f4"), dict(units="degrees_north")),
                     XLONG=(("south_north", "west_east"), z["lon"].astype("f4"), dict(units="degrees_east")),
                     CELL_AREA=(("south_north", "west_east"), z["cell_area_km2"].astype("f4"), dict(units="km2"))),
                attrs=dict(event=ev, grid="WRF d02 mass points, 3 km, Lambert conformal, 301 x 301"))
st.to_netcdf(f"{O}/{ev}_d02_static.nc", encoding=enc(st))
for f in [f"{ev}_summary.csv"] + [f"{ev}_{a}{s}" for a in ("ctl", "pgw585") for s in (".npz", "_summary.json")]:
    if os.path.exists(f"{P}/{f}"): shutil.copy2(f"{P}/{f}", O)
for a in ("ctl", "pgw585"):
    f2 = f"{P}/{ev}_{a}_d02.nc"
    if os.path.exists(f2):
        d = xr.open_dataset(f2).drop_vars(["liq_hourly", "precip_hourly", "time"], errors="ignore"); d.load()
        d.to_netcdf(f"{O}/{ev}_{a}_d02_totals.nc", encoding=enc(d)); d.close()
    f1 = f"{P}/{ev}_{a}_d01.nc"
    if os.path.exists(f1):
        d = xr.open_dataset(f1).drop_vars(["ivt_u", "ivt_v", "iwv", "lat3", "lon3"], errors="ignore"); d.load()
        d.to_netcdf(f"{O}/{ev}_{a}_d01_ivt.nc", encoding=enc(d)); d.close()
open(f"{O}/.complete", "w").write("ok\n")
for f in sorted(os.listdir(O)): print(f, os.path.getsize(f"{O}/{f}"))
