"""Sensitivity of the GHCN station check to the observing time (information only; the main numbers are eval_stations.py).
usage: eval_obstime_shift.py EVENT T0 T1 HHMM_LO HHMM_HI       e.g. ORE_1996 1996-11-15_15 1996-11-19_15 0500 1000
Takes eval/EVENT_stations.csv (stations, grid points, obs totals of eval_stations.py), keeps the stations whose GHCN observing
time (obs_time, local) is within [HHMM_LO, HHMM_HI], and recomputes the ctl and pgw585 totals over the model window T0-T1
(UTC; e.g. GHCN days 16-19 Nov read at 07 PST = 24 h periods ending 15Z, i.e. 15 Nov 15Z - 19 Nov 15Z). Prints the same
metrics for that subset with the default window (from the CSV) and with the shifted window."""
import sys, os, glob, datetime as dt, numpy as np, pandas as pd, netCDF4
from scipy.stats import spearmanr
B = os.path.expandvars("/scratch/$USER/ar_pgw"); D = dt.datetime
ev, t0, t1, lo, hi = sys.argv[1], D.strptime(sys.argv[2], "%Y-%m-%d_%H"), D.strptime(sys.argv[3], "%Y-%m-%d_%H"), int(sys.argv[4]), int(sys.argv[5])
def times(ds): return [D.strptime(b"".join(t).decode(), "%Y-%m-%d_%H:%M:%S") for t in ds["Times"][:]]
def frame_at(run, t):
    pick = None
    for f in sorted(glob.glob(f"{run}/wrfaux5_d02_*"), key=lambda f: (os.path.getmtime(f), f)):
        with netCDF4.Dataset(f) as ds:
            tt = times(ds)
            if t in tt: pick = (f, tt.index(t))
    if pick is None: raise SystemExit(f"{run}: no output at {t}")
    return pick
def acc(run, jj, ii):
    out = []
    for t in (t0, t1):
        f, i = frame_at(run, t)
        with netCDF4.Dataset(f) as ds: out.append(np.asarray(ds["RAINNC"][i], "f8")[jj, ii] + np.asarray(ds["RAINC"][i], "f8")[jj, ii])
    return out[1] - out[0]
def metrics(c, sim):
    r = c[c.obs_mm >= 1.0]; ratio = r[sim] / r.obs_mm
    return dict(n=len(c), n_ge1=len(r), median_obs=round(c.obs_mm.median(), 1), median_sim=round(c[sim].median(), 1),
                median_bias_pct=round((ratio.median() - 1) * 100, 1), station_mean_bias_pct=round((r[sim].mean() / r.obs_mm.mean() - 1) * 100, 1),
                spearman=round(spearmanr(c.obs_mm, c[sim]).statistic, 3), share_within_factor2=round(((ratio >= .5) & (ratio <= 2)).mean(), 3))
c = pd.read_csv(f"{B}/eval/{ev}_stations.csv", dtype={"obs_time": str})
ot = pd.to_numeric(c.obs_time, errors="coerce")
print(ev, "obs_time counts:", c.obs_time.fillna("blank").value_counts().head(12).to_dict())
s = c[(ot >= lo) & (ot <= hi)].copy()
s["ctl_shift"] = acc(f"{B}/runs/{ev}_ctl", s.j.values, s.i.values).round(1)
s["pgw_shift"] = acc(f"{B}/runs/{ev}_pgw585", s.j.values, s.i.values).round(1)
print(f"stations with obs_time {lo:04d}-{hi:04d}: default window ctl", metrics(s, "ctl_mm"))
print(f"  shifted window {t0:%d %b %HZ} - {t1:%d %b %HZ} ctl", metrics(s, "ctl_shift"), "median pgw585", round(s.pgw_shift.median(), 1))
