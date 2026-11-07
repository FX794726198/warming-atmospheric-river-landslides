"""Post-process one AR PGW case (ctl + pgw585) into small NPZ files and one summary CSV.
usage: post_case.py EVENT [OUTDIR=/scratch/$USER/ar_pgw/post] [ARM ...]   (default arms: ctl pgw585)
Per run -> OUTDIR/EVENT_ARM.npz with
  d02: lat, lon, hgt, landmask, cell_area_km2 (dx^2 / mapfac^2 from geo_em.d02), event totals of total precipitation,
       liquid precipitation (RAINNC + RAINC - SNOWNC - GRAUPELNC - HAILNC; WRF RAINNC holds all phases), snowfall (SNOWNC),
       graupel + hail, snowmelt (ACSNOM), surface and underground runoff (SFROFF, UDROFF); max 24 h and max 48 h liquid and
       total precipitation (running windows inside the event window); SWE at event start and end;
  d01: times and the IVT magnitude maximum over the d01 interior (10 cells from the edges dropped) and over the d02 footprint,
       every 3 h, from QVAPOR, U, V, P+PB, PSFC.
Per case -> OUTDIR/EVENT_summary.csv, rows ctl, pgw585, pgw585_minus_ctl, pgw585_over_ctl, with land-only metrics:
  areas (km2) with max-48 h liquid >= 50/100/150/200 mm and >= the ctl run's land 99th percentile of max-48 h liquid;
  mean and max event liquid and total precipitation; liquid fraction (sum liquid / sum total) above 500, 1000 and 1500 m;
  peak IVT (d01 interior and d02 footprint); and the PGW deltas (domain means of dT2m and dT850 on d01 and d02).
Event window = run start + 12 h spin-up to run end (events.py)."""
import sys, os, glob, json, csv, re, datetime as dt, numpy as np, netCDF4
sys.path.insert(0, os.path.expandvars("/scratch/$USER/ar_pgw/scripts")); from events import EVENTS, NEST
B = os.path.expandvars("/scratch/$USER/ar_pgw")
ev = sys.argv[1]; outd = sys.argv[2] if len(sys.argv) > 2 else f"{B}/post"; arms = sys.argv[3:] or ["ctl", "pgw585"]
os.makedirs(outd, exist_ok=True); E = EVENTS[ev]; g = 9.80665
t_ev0 = dt.datetime.strptime(E["start"], "%Y-%m-%d_%H") + dt.timedelta(hours=12); t_end = dt.datetime.strptime(E["end"], "%Y-%m-%d_%H")

def times(ds): return [dt.datetime.strptime(b"".join(t).decode(), "%Y-%m-%d_%H:%M:%S") for t in ds["Times"][:]]

def ordered_frames(run, pattern):
    """[(time, file, index)] for every output time, in time order. When two segments wrote the same time (an earlier segment
    ran past the restart its successor started from), the frame is taken from the file written last (mtime), i.e. from the
    segment that continued the run, so each run is read as one trajectory. (v1, to 2026-11-05 15Z, kept the first copy.)"""
    pick = {}
    for f in sorted(glob.glob(os.path.join(run, pattern)), key=lambda f: (os.path.getmtime(f), f)):
        with netCDF4.Dataset(f) as ds:
            for i, t in enumerate(times(ds)): pick[t] = (f, i)
    return [(t,) + pick[t] for t in sorted(pick)]

def frames(run, pattern):
    """yield (time, dataset, index), one frame per output time (see ordered_frames)"""
    cur = ds = None
    try:
        for t, f, i in ordered_frames(run, pattern):
            if f != cur:
                if ds is not None: ds.close()
                ds, cur = netCDF4.Dataset(f), f
            yield t, ds, i
    finally:
        if ds is not None: ds.close()

def post_run(arm):
    run = f"{B}/runs/{ev}_{arm}"; R = dict(event=ev, arm=arm)
    V = ["RAINNC", "RAINC", "SNOWNC", "GRAUPELNC", "HAILNC", "ACSNOM", "SFROFF", "UDROFF", "SNOW"]
    T, A = [], {v: [] for v in V}; static = None
    for t, ds, i in frames(run, "wrfaux5_d02_*"):
        if static is None:
            static = {k: np.asarray(ds[v][0]) for k, v in (("hgt", "HGT"), ("landmask", "LANDMASK"), ("lat", "XLAT"), ("lon", "XLONG"))}
        T.append(t)
        for v in V: A[v].append(np.asarray(ds[v][i], "f8") if v in ds.variables else np.zeros(static["hgt"].shape))
    if not T: raise SystemExit(f"no d02 output in {run}")
    T = np.array(T); A = {v: np.stack(a) for v, a in A.items()}
    i0 = int(np.argmin(np.abs(T - t_ev0))); i1 = int(np.argmin(np.abs(T - t_end)))
    R["complete"] = bool(abs((T[i1] - t_end).total_seconds()) < 1800 and abs((T[i0] - t_ev0).total_seconds()) < 1800)
    R["d02_first"], R["d02_last"] = str(T[i0]), str(T[i1])
    tot = A["RAINNC"] + A["RAINC"]; frz = A["SNOWNC"] + A["GRAUPELNC"] + A["HAILNC"]; liq = tot - frz
    def acc(a): return (a[i1] - a[i0]).astype("f4")
    tix = {t: k for k, t in enumerate(T)}
    def maxwin(a, h):
        """max of a(t + h) - a(t) over output times t in the event window whose t + h is also an output time in the window.
        A missing output hour drops the windows that start or end on it (v1 indexed frames, so such windows spanned h + 1 h)."""
        best = np.full(a.shape[1:], np.nan)
        for k in range(i0, i1 + 1):
            j = tix.get(T[k] + dt.timedelta(hours=h))
            if j is not None and j <= i1: best = np.fmax(best, a[j] - a[k])
        return best.astype("f4")
    R["missing_hours"] = ";".join(str(T[k - 1] + dt.timedelta(hours=m)) for k in range(i0 + 1, i1 + 1)
                                  for m in range(1, int((T[k] - T[k - 1]).total_seconds() // 3600)))
    geo = netCDF4.Dataset(f"{B}/wps/{ev}/geo_em.d02.nc"); mf = np.asarray(geo["MAPFAC_M"][0], "f8"); dx = float(geo.getncattr("DX")) / 1000.0; geo.close()
    if mf.shape != static["hgt"].shape: mf = np.ones(static["hgt"].shape); R["note"] = "map factor shape mismatch; nominal area used"
    out = dict(static, cell_area_km2=(dx * dx / mf ** 2).astype("f4"),
               precip_total=acc(tot), liq_total=acc(liq), snowfall=acc(A["SNOWNC"]), graupel_hail=acc(A["GRAUPELNC"] + A["HAILNC"]),
               snowmelt=acc(A["ACSNOM"]), runoff_sfc=acc(A["SFROFF"]), runoff_sub=acc(A["UDROFF"]), runoff_total=acc(A["SFROFF"] + A["UDROFF"]),
               liq_max24h=maxwin(liq, 24), liq_max48h=maxwin(liq, 48), precip_max24h=maxwin(tot, 24), precip_max48h=maxwin(tot, 48),
               swe_start=A["SNOW"][i0].astype("f4"), swe_end=A["SNOW"][i1].astype("f4"))
    # d01 IVT
    nl = open(os.path.join(run, "namelist.base")).read()
    ip = int(re.search(r"i_parent_start = 1, (\d+)", nl).group(1)); jp = int(re.search(r"j_parent_start = 1, (\d+)", nl).group(1))
    nf = (NEST["nx"] - 1) // NEST["ratio"]; fp = (slice(jp - 1, jp - 1 + nf), slice(ip - 1, ip - 1 + nf)); e = 10
    TT, MX, MF = [], [], []
    for t, ds, i in frames(run, "wrfaux7_d01_*"):
        ptop = float(ds.getncattr("P_TOP")) if "P_TOP" in ds.ncattrs() else 5000.0
        r = np.asarray(ds["QVAPOR"][i], "f8"); q = r / (1 + r)
        u = np.asarray(ds["U"][i], "f8"); v = np.asarray(ds["V"][i], "f8"); u = 0.5 * (u[..., 1:] + u[..., :-1]); v = 0.5 * (v[:, 1:] + v[:, :-1])
        p = np.asarray(ds["P"][i], "f8") + np.asarray(ds["PB"][i], "f8"); ps = np.asarray(ds["PSFC"][i], "f8")
        pi = np.concatenate([ps[None], 0.5 * (p[1:] + p[:-1]), np.full((1,) + ps.shape, ptop)]); dp = pi[:-1] - pi[1:]
        M = np.hypot((q * u * dp).sum(0) / g, (q * v * dp).sum(0) / g)
        TT.append(t); MX.append(M[e:-e, e:-e].max()); MF.append(M[fp].max())
    TT = np.array(TT); MX = np.array(MX, "f4"); MF = np.array(MF, "f4")
    out.update(ivt_time=np.array([str(t) for t in TT]), ivt_max_d01=MX, ivt_max_d02footprint=MF)
    sel = (TT >= t_ev0) & (TT <= t_end) if len(TT) else np.zeros(0, bool)
    np.savez_compressed(os.path.join(outd, f"{ev}_{arm}.npz"), **out)
    # metrics
    land = static["landmask"] > 0.5; ar = out["cell_area_km2"]
    R.update(peak_ivt_d01=float(MX[sel].max()) if sel.any() else np.nan, peak_ivt_d02footprint=float(MF[sel].max()) if sel.any() else np.nan)
    for k in ("liq_total", "precip_total"):
        R[f"land_mean_{k}"] = float(np.average(out[k][land], weights=ar[land])); R[f"land_max_{k}"] = float(out[k][land].max())
    for k in ("snowfall", "snowmelt", "runoff_total"): R[f"land_mean_{k}"] = float(np.average(out[k][land], weights=ar[land]))
    for thr in (500, 1000, 1500):
        m = land & (static["hgt"] >= thr)
        R[f"liq_fraction_above_{thr}m"] = float((out["liq_total"][m] * ar[m]).sum() / max((out["precip_total"][m] * ar[m]).sum(), 1e-9)) if m.any() else np.nan
    for thr in (50, 100, 150, 200): R[f"area_km2_liq48h_ge{thr}"] = float(ar[land & (out["liq_max48h"] >= thr)].sum())
    R["land_p99_liq48h"] = float(np.nanpercentile(out["liq_max48h"][land], 99))
    return R, out

rows, outs = {}, {}
for a in arms:
    run = f"{B}/runs/{ev}_{a}"
    if not glob.glob(os.path.join(run, "wrfaux5_d02_*")): print("skip", a, "(no output)"); continue
    rows[a], outs[a] = post_run(a); print(a, "done", flush=True)
if "ctl" in rows:
    p99 = rows["ctl"]["land_p99_liq48h"]
    for a in rows:
        land = outs[a]["landmask"] > 0.5; ar = outs[a]["cell_area_km2"]
        rows[a]["ctl_p99_liq48h"] = p99; rows[a]["area_km2_liq48h_ge_ctl_p99"] = float(ar[land & (outs[a]["liq_max48h"] >= p99)].sum())
for a in rows:   # PGW deltas (domain means over all met_em times)
    for dom, fn in (("d01", "pgw_summary.json"), ("d02", "pgw_summary_d02.json")):
        f = f"{B}/wps/{ev}/pgw585/{fn}"
        if a != "ctl" and os.path.exists(f):
            S = json.load(open(f)).get(dom, {})
            rows[a][f"{dom}_dT2m"] = S.get("dT2m"); rows[a][f"{dom}_dT850"] = S.get("dT850")
if "ctl" in rows and "pgw585" in rows:
    c, p = rows["ctl"], rows["pgw585"]; num = [k for k, v in c.items() if isinstance(v, float)]
    rows["pgw585_minus_ctl"] = dict(event=ev, arm="pgw585_minus_ctl", **{k: p[k] - c[k] for k in num if isinstance(p.get(k), float)})
    rows["pgw585_over_ctl"] = dict(event=ev, arm="pgw585_over_ctl", **{k: (p[k] / c[k] if c[k] else np.nan) for k in num if isinstance(p.get(k), float)})
keys = []
for r in rows.values():
    for k in r:
        if k not in keys: keys.append(k)
with open(os.path.join(outd, f"{ev}_summary.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
    for r in rows.values(): w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})
print(open(os.path.join(outd, f"{ev}_summary.csv")).read())
