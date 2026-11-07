"""Post-process one AR PGW WRF run to small NetCDF files.
usage: post_event.py RUNDIR OUTDIR        (RUNDIR = /scratch/$USER/ar_pgw/runs/EVENT_ARM)
Writes OUTDIR/EVENT_ARM_d02.nc, OUTDIR/EVENT_ARM_d01.nc and OUTDIR/EVENT_ARM_summary.json.
Event window = event start (run start + 12 h spin-up) to run end.
d02: WRF RAINNC is total grid-scale precipitation (all phases); frozen = SNOWNC + GRAUPELNC + HAILNC; liquid = RAINNC + RAINC - frozen.
     Fields: event-total liquid and total precipitation, frozen precipitation (snowfall), max 24 h / 48 h liquid and total precipitation
     (running windows inside the event window), max hourly liquid rate, snowmelt (ACSNOM), surface/underground runoff (SFROFF/UDROFF),
     SWE at event start/end, terrain, land mask, hourly liquid and total precipitation (float32, compressed).
d01: IVT u/v and IWV every 3 h from QVAPOR, U, V, P+PB, PSFC (layer dp from mass-level mid-points, top at p_top); full-resolution
     event-max IVT and event-total precipitation, IVT fields subsampled every 3rd point (27 km), and time series of IVT averaged over
     the d02 footprint and over a 2-degree box around the landslide target."""
import sys, os, glob, json, numpy as np, xarray as xr, netCDF4, datetime as dt
sys.path.insert(0, os.path.expandvars("/scratch/$USER/ar_pgw/scripts")); from events import EVENTS, NEST
run, outd = sys.argv[1], sys.argv[2]; os.makedirs(outd, exist_ok=True)
name = os.path.basename(run.rstrip("/")); ev = "_".join(name.split("_")[:2]); E = EVENTS[ev]
t_run0 = dt.datetime.strptime(E["start"], "%Y-%m-%d_%H"); t_ev0 = t_run0 + dt.timedelta(hours=float(os.environ.get("POST_SPINUP_H", 12)))   # POST_SPINUP_H: test override only
t_end = dt.datetime.strptime(os.environ.get("POST_END", E["end"]), "%Y-%m-%d_%H")
g = 9.80665
def times(ds):
    return [dt.datetime.strptime(b"".join(t).decode(), "%Y-%m-%d_%H:%M:%S") for t in ds["Times"][:]]
def ordered_frames(pattern):
    """[(time, file, index)] for every output time, in time order. When two segments wrote the same time (an earlier segment
    ran past the restart its successor started from), the frame is taken from the file written last (mtime), i.e. from the
    segment that continued the run, so each run is read as one trajectory. (v1, to 2026-11-05 15Z, kept the first copy.)"""
    pick = {}
    for f in sorted(glob.glob(os.path.join(run, pattern)), key=lambda f: (os.path.getmtime(f), f)):
        with netCDF4.Dataset(f) as ds:
            for i, t in enumerate(times(ds)): pick[t] = (f, i)
    return [(t,) + pick[t] for t in sorted(pick)]
def stack(pattern, vars_):
    fs = sorted(glob.glob(os.path.join(run, pattern))); T = []; D = {v: [] for v in vars_}; cur = ds = None
    for t, f, i in ordered_frames(pattern):
        if f != cur:
            if ds is not None: ds.close()
            ds, cur = netCDF4.Dataset(f), f
        T.append(t)
        for v in vars_: D[v].append(np.asarray(ds[v][i]) if v in ds.variables else None)
    if ds is not None: ds.close()
    return T, D, fs
enc = dict(zlib=True, complevel=4)
summary = dict(run=name, event=ev, event_start=str(t_ev0), event_end=str(t_end))
# ---------------- d02 ----------------
V2 = ["RAINNC", "RAINC", "SNOWNC", "GRAUPELNC", "HAILNC", "ACSNOM", "SFROFF", "UDROFF", "SNOW", "SNOWH", "T2"]
T, D, fs = stack("wrfaux5_d02_*", V2)
if T:
    with netCDF4.Dataset(fs[0]) as ds:
        hgt = np.asarray(ds["HGT"][0]); lm = np.asarray(ds["LANDMASK"][0]); lat = np.asarray(ds["XLAT"][0]); lon = np.asarray(ds["XLONG"][0])
    A = {v: (np.stack([x if x is not None else np.zeros_like(hgt) for x in D[v]]).astype("f8")) for v in V2}
    tot = A["RAINNC"] + A["RAINC"]; frz = A["SNOWNC"] + A["GRAUPELNC"] + A["HAILNC"]; liq = tot - frz
    T = np.array(T); i0 = int(np.argmin(np.abs(T - t_ev0))); i1 = int(np.argmin(np.abs(T - t_end)))
    complete = abs((T[i1] - t_end).total_seconds()) < 1800
    def acc(a): return a[i1] - a[i0]
    tix = {t: k for k, t in enumerate(T)}
    def maxwin(a, h):
        """max of a(t + h) - a(t) over output times t in the event window whose t + h is also an output time in the window.
        A missing output hour drops the windows that start or end on it (v1 indexed frames, so such windows spanned h + 1 h)."""
        best = np.full(a.shape[1:], np.nan)
        for k in range(i0, i1 + 1):
            j = tix.get(T[k] + dt.timedelta(hours=h))
            if j is not None and j <= i1: best = np.fmax(best, a[j] - a[k])
        return best
    hl = np.diff(liq[i0:i1 + 1], axis=0); ht = np.diff(tot[i0:i1 + 1], axis=0)
    # an hourly value whose previous output hour is missing spans 2+ h: set to NaN, list the missing hours in the attributes
    gap = np.array([(T[k] - T[k - 1]).total_seconds() != 3600 for k in range(i0 + 1, i1 + 1)], bool)
    hl[gap] = np.nan; ht[gap] = np.nan
    missing = [str(T[k - 1] + dt.timedelta(hours=m)) for k in range(i0 + 1, i1 + 1) for m in range(1, int((T[k] - T[k - 1]).total_seconds() // 3600))]
    out = xr.Dataset(dict(
        liq_total=(("y", "x"), acc(liq)), precip_total=(("y", "x"), acc(tot)), frozen_total=(("y", "x"), acc(frz)),
        snowfall=(("y", "x"), acc(A["SNOWNC"])), graupel=(("y", "x"), acc(A["GRAUPELNC"])),
        liq_max24h=(("y", "x"), maxwin(liq, 24)), liq_max48h=(("y", "x"), maxwin(liq, 48)),
        precip_max24h=(("y", "x"), maxwin(tot, 24)), precip_max48h=(("y", "x"), maxwin(tot, 48)),
        liq_max1h=(("y", "x"), np.nanmax(hl, 0) if len(hl) else np.full(hgt.shape, np.nan)),
        snowmelt=(("y", "x"), acc(A["ACSNOM"])), runoff_sfc=(("y", "x"), acc(A["SFROFF"])), runoff_sub=(("y", "x"), acc(A["UDROFF"])),
        swe_start=(("y", "x"), A["SNOW"][i0]), swe_end=(("y", "x"), A["SNOW"][i1]), t2_mean=(("y", "x"), A["T2"][i0:i1 + 1].mean(0)),
        hgt=(("y", "x"), hgt), landmask=(("y", "x"), lm), lat=(("y", "x"), lat), lon=(("y", "x"), lon),
        liq_hourly=(("time", "y", "x"), hl.astype("f4")), precip_hourly=(("time", "y", "x"), ht.astype("f4"))),
        coords=dict(time=np.array(T[i0 + 1:i1 + 1], dtype="datetime64[ns]")))
    for v in out.data_vars:
        if out[v].dtype == np.float64: out[v] = out[v].astype("f4")
    out.attrs.update(run=name, units="mm (precip/melt/runoff/SWE), m (hgt), K (t2)", event_start=str(t_ev0), event_end=str(t_end),
                     complete=int(complete), note="liquid = RAINNC+RAINC-SNOWNC-GRAUPELNC-HAILNC; windows inside event window",
                     missing_hours=";".join(missing))
    out.to_netcdf(os.path.join(outd, f"{name}_d02.nc"), encoding={v: enc for v in out.data_vars})
    land = lm > 0.5
    def lm_(a): return float(np.nanmean(a[land]))
    summary["d02"] = dict(hours=len(hl), complete=bool(complete), land_mean_liq=lm_(acc(liq)), land_mean_total=lm_(acc(tot)),
        land_mean_frozen=lm_(acc(frz)), liquid_fraction=lm_(acc(liq)) / max(lm_(acc(tot)), 1e-9), land_mean_snowmelt=lm_(acc(A["ACSNOM"])),
        land_mean_runoff=lm_(acc(A["SFROFF"]) + acc(A["UDROFF"])), land_mean_swe_start=lm_(A["SNOW"][i0]), land_mean_swe_end=lm_(A["SNOW"][i1]),
        land_p99_liq_max24h=float(np.nanpercentile(maxwin(liq, 24)[land], 99)), land_p99_liq_max48h=float(np.nanpercentile(maxwin(liq, 48)[land], 99)),
        land_area_km2_liq48_ge100=float((maxwin(liq, 48)[land] >= 100).sum() * 9), land_area_km2_liq48_ge200=float((maxwin(liq, 48)[land] >= 200).sum() * 9))
# ---------------- d01 ----------------
fs = sorted(glob.glob(os.path.join(run, "wrfaux7_d01_*")))
if fs:
    with netCDF4.Dataset(fs[0]) as ds:
        lat1 = np.asarray(ds["XLAT"][0]); lon1 = np.asarray(ds["XLONG"][0]); hgt1 = np.asarray(ds["HGT"][0]); lm1 = np.asarray(ds["LANDMASK"][0])
        ptop = float(ds.getncattr("P_TOP")) if "P_TOP" in ds.ncattrs() else 5000.0
    with open(os.path.join(run, "namelist.base")) as f: nl = f.read()
    import re
    ip = int(re.search(r"i_parent_start = 1, (\d+)", nl).group(1)); jp = int(re.search(r"j_parent_start = 1, (\d+)", nl).group(1))
    nf = (NEST["nx"] - 1) // NEST["ratio"]; fp = (slice(jp - 1, jp - 1 + nf), slice(ip - 1, ip - 1 + nf))                      # d02 footprint on d01 (150 nest cells / 3)
    box = (np.abs(lat1 - E["tlat"]) <= 1.0) & (np.abs(((lon1 - E["tlon"] + 180) % 360) - 180) <= 1.0 / np.cos(np.radians(E["tlat"])))
    TT, IU, IV, IW, PR = [], [], [], [], []
    for t, f, i in ordered_frames("wrfaux7_d01_*"):
        with netCDF4.Dataset(f) as ds:
            r = np.asarray(ds["QVAPOR"][i], "f8"); q = r / (1 + r)
            u = np.asarray(ds["U"][i], "f8"); v = np.asarray(ds["V"][i], "f8"); u = 0.5 * (u[..., 1:] + u[..., :-1]); v = 0.5 * (v[:, 1:] + v[:, :-1])
            p = np.asarray(ds["P"][i], "f8") + np.asarray(ds["PB"][i], "f8"); ps = np.asarray(ds["PSFC"][i], "f8")
            pi = np.concatenate([ps[None], 0.5 * (p[1:] + p[:-1]), np.full((1,) + ps.shape, ptop)]); dp = pi[:-1] - pi[1:]
            IU.append((q * u * dp).sum(0) / g); IV.append((q * v * dp).sum(0) / g); IW.append((q * dp).sum(0) / g)
            PR.append(np.asarray(ds["RAINNC"][i], "f8") + np.asarray(ds["RAINC"][i], "f8")); TT.append(t)
    IU = np.stack(IU); IV = np.stack(IV); IW = np.stack(IW); PR = np.stack(PR); M = np.hypot(IU, IV); TT = np.array(TT)
    j0 = int(np.argmin(np.abs(TT - t_ev0))); j1 = int(np.argmin(np.abs(TT - t_end))); s = (slice(None), slice(None, None, 3), slice(None, None, 3))
    o1 = xr.Dataset(dict(
        ivt_u=(("time", "y3", "x3"), IU[s].astype("f4")), ivt_v=(("time", "y3", "x3"), IV[s].astype("f4")), iwv=(("time", "y3", "x3"), IW[s].astype("f4")),
        lat3=(("y3", "x3"), lat1[::3, ::3]), lon3=(("y3", "x3"), lon1[::3, ::3]),
        ivt_max_event=(("y", "x"), M[j0:j1 + 1].max(0).astype("f4")), precip_event=(("y", "x"), (PR[j1] - PR[j0]).astype("f4")),
        hgt=(("y", "x"), hgt1.astype("f4")), landmask=(("y", "x"), lm1.astype("f4")), lat=(("y", "x"), lat1), lon=(("y", "x"), lon1),
        ivt_nest_mean=(("time",), M[(slice(None),) + fp].mean((1, 2)).astype("f4")), ivt_nest_max=(("time",), M[(slice(None),) + fp].max((1, 2)).astype("f4")),
        ivt_box_mean=(("time",), M[:, box].mean(1).astype("f4")), iwv_box_mean=(("time",), IW[:, box].mean(1).astype("f4"))),
        coords=dict(time=TT.astype("datetime64[ns]")))
    o1.attrs.update(run=name, units="kg m-1 s-1 (IVT), kg m-2 (IWV), mm (precip)", subsample="every 3rd d01 point (27 km)",
                    box=f"+-1 deg lat around {E['tlat']},{E['tlon']}", nest_footprint=f"d01 i {ip}..{ip+nf-1}, j {jp}..{jp+nf-1}")
    o1.to_netcdf(os.path.join(outd, f"{name}_d01.nc"), encoding={v: enc for v in o1.data_vars})
    summary["d01"] = dict(frames=len(TT), ivt_box_max=float(M[j0:j1 + 1][:, box].mean(1).max()), ivt_nest_mean_event=float(M[j0:j1 + 1][(slice(None),) + fp].mean()),
                          iwv_box_max=float(IW[j0:j1 + 1][:, box].mean(1).max()), land_mean_precip_event=float((PR[j1] - PR[j0])[lm1 > 0.5].mean()))
json.dump(summary, open(os.path.join(outd, f"{name}_summary.json"), "w"), indent=1, default=str); print(json.dumps(summary, default=str))
