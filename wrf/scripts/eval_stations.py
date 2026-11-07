"""Station check of the AR PGW control runs against GHCN-Daily precipitation.
usage: eval_stations.py EVENT [EVENT ...]      -> /scratch/$USER/ar_pgw/eval/EVENT_stations.csv, eval/summary.csv (all events so far)

Stations: GHCN-Daily (ghcnd-stations.txt) whose nearest d02 grid point lies inside the 3 km domain minus the 10-cell edge, and
whose nearest interior LAND grid point (WRF LANDMASK) is within 5 km; PRCP reported on every day of the event window, value not
missing and QFLAG blank (passed all QC checks). Observed total = sum of the daily PRCP values (GHCN tenths of mm -> mm).
Simulated total at that land grid point = accumulated RAINNC + RAINC (d02 hourly output, wrfaux5) at the window end minus at
the window start, for ctl and pgw585. Duplicated output hours are read from the segment that continued the run (as post_case.py).
Windows (UTC): GHCN days are the reporting days of each station (local observing day, e.g. 24 h to the morning reading for US
COOP/CoCoRaHS), so the model window is the same set of UTC days 00Z-00Z, except BC (see WIN).
Metrics (stations with obs >= 1 mm for the ratio metrics): median obs and sim, median of per-station bias (sim/obs - 1),
station-mean bias (mean sim / mean obs - 1, unweighted over stations), Spearman rank correlation (all stations), share of
stations with 0.5 <= sim/obs <= 2, and the median pgw585 total for context."""
import sys, os, glob, gzip, datetime as dt, numpy as np, pandas as pd, netCDF4
from scipy.stats import spearmanr
B = os.path.expandvars("/scratch/$USER/ar_pgw"); G = f"{B}/eval/ghcn"; O = f"{B}/eval"; E = 10
D = dt.datetime
# event: (GHCN reporting days, model window start, model window end, note)
WIN = {"NOR_2011": (pd.date_range("2011-12-24", "2011-12-27"), D(2011, 12, 24, 0), D(2011, 12, 28, 0), "days 24-27 Dec, model 24 Dec 00Z - 28 Dec 00Z"),
       "BC_2021": (pd.date_range("2021-11-13", "2021-11-16"), D(2021, 11, 12, 12), D(2021, 11, 16, 12),
                   "days 13-16 Nov, model 12 Nov 12Z - 16 Nov 12Z (four 24 h periods 12Z-12Z; the run ends 16 Nov 12Z). "
                   "Each 12Z-12Z period ends 3 h before a 07 PST (15Z) morning reading of the same date"),
       "ORE_1996": (pd.date_range("1996-11-16", "1996-11-19"), D(1996, 11, 16, 0), D(1996, 11, 20, 0), "days 16-19 Nov, model 16 Nov 00Z - 20 Nov 00Z"),
       "LIG_2014": (pd.date_range("2014-11-09", "2014-11-16"), D(2014, 11, 9, 0), D(2014, 11, 17, 0), "days 9-16 Nov, model 9 Nov 00Z - 17 Nov 00Z"),
       "RSB_2024": (pd.date_range("2024-04-27", "2024-05-02"), D(2024, 4, 27, 0), D(2024, 5, 3, 0), "days 27 Apr - 2 May, model 27 Apr 00Z - 3 May 00Z")}

def times(ds): return [D.strptime(b"".join(t).decode(), "%Y-%m-%d_%H:%M:%S") for t in ds["Times"][:]]
def frame_at(run, t):
    """(file, index) of output time t; if two segments wrote it, the file written last (the segment that continued the run)"""
    pick = None
    for f in sorted(glob.glob(f"{run}/wrfaux5_d02_*"), key=lambda f: (os.path.getmtime(f), f)):
        with netCDF4.Dataset(f) as ds:
            tt = times(ds)
            if t in tt: pick = (f, tt.index(t))
    if pick is None: raise SystemExit(f"{run}: no output at {t}")
    return pick
def acc_total(run, t0, t1, jj, ii):
    out = []
    for t in (t0, t1):
        f, i = frame_at(run, t)
        with netCDF4.Dataset(f) as ds: out.append(np.asarray(ds["RAINNC"][i], "f8")[jj, ii] + np.asarray(ds["RAINC"][i], "f8")[jj, ii])
    return out[1] - out[0]
def xyz(lat, lon):
    la, lo = np.radians(lat), np.radians(lon); return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], -1)

st = pd.read_fwf(f"{G}/ghcnd-stations.txt", colspecs=[(0, 11), (12, 20), (21, 30), (31, 37), (41, 71)], names=["id", "lat", "lon", "elev", "name"], dtype={"id": str, "name": str})
summ_f = f"{O}/summary.csv"
summ = {r["event"]: r for r in pd.read_csv(summ_f).to_dict("records")} if os.path.exists(summ_f) else {}
for ev in sys.argv[1:]:
    days, t0, t1, note = WIN[ev]; run = f"{B}/runs/{ev}_ctl"
    with netCDF4.Dataset(sorted(glob.glob(f"{run}/wrfaux5_d02_*"))[0]) as ds:
        lat = np.asarray(ds["XLAT"][0], "f8"); lon = np.asarray(ds["XLONG"][0], "f8"); lm = np.asarray(ds["LANDMASK"][0]); hgt = np.asarray(ds["HGT"][0])
    ny, nx = lat.shape; P = xyz(lat, lon).reshape(-1, 3)
    inner = np.zeros(lat.shape, bool); inner[E:ny - E, E:nx - E] = True; land_in = (inner & (lm > 0.5)).ravel()
    # candidate stations: inside the d02 lat/lon range (+0.1 deg), then nearest-point tests on the sphere
    c = st[(st.lat >= lat.min() - .1) & (st.lat <= lat.max() + .1) & (st.lon >= lon.min() - .1) & (st.lon <= lon.max() + .1)].copy()
    S = xyz(c.lat.values, c.lon.values)
    near = np.array([np.argmax(P @ s) for s in S]); ok_in = inner.ravel()[near]
    PL = P[land_in]; idx_land = np.flatnonzero(land_in)
    nl = np.array([idx_land[np.argmax(PL @ s)] for s in S])
    dist = 6371.0 * np.arccos(np.clip((P[nl] * S).sum(1), -1, 1))
    c["j"], c["i"] = np.unravel_index(nl, lat.shape); c["dist_km"] = dist.round(2)
    c = c[ok_in & (dist <= 5.0)].copy()
    # GHCN PRCP for those stations and days
    y = days[0].year; ids = set(c.id); want = set(days.strftime("%Y%m%d").astype(int))
    rows = []
    for ch in pd.read_csv(f"{G}/{y}.csv.gz", header=None, names=["id", "date", "el", "val", "m", "q", "s", "obst"], dtype={"id": str, "el": str, "m": str, "q": str, "s": str, "obst": str}, chunksize=4_000_000):
        ch = ch[(ch.el == "PRCP") & ch.date.isin(want) & ch.id.isin(ids)]
        if len(ch): rows.append(ch)
    pr = pd.concat(rows) if rows else pd.DataFrame(columns=["id", "date", "el", "val", "m", "q", "s", "obst"])
    pr = pr.drop_duplicates(["id", "date"])
    good = pr[(pr.val != -9999) & pr.q.isna()]
    cnt = good.groupby("id").date.nunique(); full = cnt[cnt == len(days)].index
    obs = good[good.id.isin(full)].groupby("id").val.sum() / 10.0
    obst = good[good.id.isin(full)].groupby("id").obst.agg(lambda x: x.dropna().iloc[0] if x.notna().any() else "")
    # astype(float): with no station left the mapped column would be a string column and the ratio below would fail (RSB_2024)
    c = c[c.id.isin(full)].copy(); c["obs_mm"] = c.id.map(obs).values.astype(float); c["obs_time"] = c.id.map(obst).values
    jj, ii = c.j.values, c.i.values
    c["ctl_mm"] = acc_total(run, t0, t1, jj, ii).round(1)
    c["pgw585_mm"] = acc_total(f"{B}/runs/{ev}_pgw585", t0, t1, jj, ii).round(1) if os.path.exists(f"{B}/runs/{ev}_pgw585/DONE") else np.nan
    c["grid_lat"] = lat[jj, ii].round(4); c["grid_lon"] = lon[jj, ii].round(4); c["grid_hgt_m"] = hgt[jj, ii].round(0)
    c = c[["id", "name", "lat", "lon", "elev", "obs_time", "j", "i", "grid_lat", "grid_lon", "grid_hgt_m", "dist_km", "obs_mm", "ctl_mm", "pgw585_mm"]]
    c.to_csv(f"{O}/{ev}_stations.csv", index=False)
    r = c[c.obs_mm >= 1.0]; ratio = r.ctl_mm / r.obs_mm
    R = dict(n_stations=len(c), n_obs_ge_1mm=len(r), window=note, median_obs_mm=round(c.obs_mm.median(), 1), median_ctl_mm=round(c.ctl_mm.median(), 1),
             median_bias_pct=round((ratio.median() - 1) * 100, 1), station_mean_bias_pct=round((r.ctl_mm.mean() / r.obs_mm.mean() - 1) * 100, 1),
             spearman=round(spearmanr(c.obs_mm, c.ctl_mm).statistic, 3) if len(c) > 2 else np.nan,
             share_within_factor2=round(((ratio >= 0.5) & (ratio <= 2)).mean(), 3), median_pgw585_mm=round(c.pgw585_mm.median(), 1),
             candidates_in_domain=int(ok_in.sum()), excluded_far_from_land=int((ok_in & (dist > 5.0)).sum()))
    print(ev, R, flush=True)
    summ[ev] = dict(event=ev, **R)
# keep extra rows written by other checks (e.g. RSB_2024_chirps from eval_chirps.py) right after their event
order = [e for e in WIN if e in summ]
order = [x for e in order for x in [e] + [k for k in summ if k.startswith(e + "_")]]
S = pd.DataFrame([summ[e] for e in order])
for k in ("n_stations", "n_obs_ge_1mm", "candidates_in_domain", "excluded_far_from_land"): S[k] = S[k].astype("Int64")   # NaN in a gridded row
S.to_csv(summ_f, index=False)
