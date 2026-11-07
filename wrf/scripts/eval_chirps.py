"""Gridded check of the RSB_2024 control run against CHIRPS v2.0 daily 0.05 deg (GHCN has no usable station there, and the
INMET archive did not answer). usage: eval_chirps.py   -> eval/RSB_2024_chirps_cells.csv, eval/RSB_2024_chirps_summary.json,
row RSB_2024_chirps in eval/summary.csv

CHIRPS: data.chc.ucsb.edu/products/CHIRPS/v2.0/global_daily/tifs/p05/2024/chirps-v2.0.2024.MM.DD.tif.gz, 27 Apr - 2 May 2024
(6 days, in eval/chirps/), summed to a 6-day total. CHIRPS days are taken as UTC days, so the model window is 27 Apr 00Z - 3 May 00Z.
Model: RAINNC + RAINC (d02 hourly wrfaux5) at the window end minus at the window start, frame of the segment that continued the
run where a time was written twice (as post_case.py v2). Model d02 points inside the domain minus the 10-cell edge and on land
(LANDMASK) are averaged into the CHIRPS 0.05 deg cell that contains them; a cell is used when it has at least 2 such model points
and a valid CHIRPS value (CHIRPS is land-only, -9999 elsewhere). The same is repeated on 0.25 deg cells (5 x 5 CHIRPS cells,
cells with at least 60 % of their CHIRPS cells used) because CHIRPS has fewer independent inputs than its grid spacing.
Metrics: area-weighted (cos lat) mean of CHIRPS and of ctl and pgw585 over the used cells and the bias of the means; Pearson
and Spearman spatial correlation of the 6-day totals; median of the cell bias (ctl/CHIRPS - 1) and the share of cells with
0.5 <= ctl/CHIRPS <= 2, both over cells with CHIRPS >= 1 mm."""
import os, glob, json, datetime as dt, numpy as np, pandas as pd, netCDF4, rasterio
from rasterio.windows import from_bounds
from scipy.stats import spearmanr, pearsonr
B = os.path.expandvars("/scratch/$USER/ar_pgw"); O = f"{B}/eval"; C = f"{O}/chirps"; E = 10; EV = "RSB_2024"; D = dt.datetime
DAYS = pd.date_range("2024-04-27", "2024-05-02"); T0, T1 = D(2024, 4, 27, 0), D(2024, 5, 3, 0)

def times(ds): return [D.strptime(b"".join(t).decode(), "%Y-%m-%d_%H:%M:%S") for t in ds["Times"][:]]
def frame_at(run, t):
    pick = None
    for f in sorted(glob.glob(f"{run}/wrfaux5_d02_*"), key=lambda f: (os.path.getmtime(f), f)):
        with netCDF4.Dataset(f) as ds:
            tt = times(ds)
            if t in tt: pick = (f, tt.index(t))
    if pick is None: raise SystemExit(f"{run}: no output at {t}")
    return pick
def total(run):
    out = []
    for t in (T0, T1):
        f, i = frame_at(run, t)
        with netCDF4.Dataset(f) as ds: out.append(np.asarray(ds["RAINNC"][i], "f8") + np.asarray(ds["RAINC"][i], "f8"))
    return out[1] - out[0]

run = f"{B}/runs/{EV}_ctl"
with netCDF4.Dataset(sorted(glob.glob(f"{run}/wrfaux5_d02_*"))[0]) as ds:
    lat = np.asarray(ds["XLAT"][0], "f8"); lon = np.asarray(ds["XLONG"][0], "f8"); lm = np.asarray(ds["LANDMASK"][0])
ny, nx = lat.shape
use = np.zeros(lat.shape, bool); use[E:ny - E, E:nx - E] = True; use &= lm > 0.5
ctl = total(run); pgw = total(f"{B}/runs/{EV}_pgw585")

# CHIRPS 6-day total over the d02 box (+0.1 deg)
box = (lon.min() - .1, lat.min() - .1, lon.max() + .1, lat.max() + .1)
acc = None
for d in DAYS:
    with rasterio.open(f"{C}/chirps-v2.0.{d:%Y.%m.%d}.tif") as r:
        w = from_bounds(*box, transform=r.transform).round_offsets().round_lengths()
        a = r.read(1, window=w).astype("f8"); tr = r.window_transform(w); nd = r.nodata
    a[(a < 0) | (a == nd)] = np.nan
    acc = a if acc is None else acc + a                        # NaN if any day is missing
lon0, lat0, dx = tr.c, tr.f, tr.a                               # top-left corner, cell size (dy = -dx)
cj = np.floor((lat0 - lat) / dx).astype(int); ci = np.floor((lon - lon0) / dx).astype(int)

def cells(fac, need_frac=None):
    """bin the used model points into cells of fac x fac CHIRPS cells; return a DataFrame of cells"""
    J, I = cj[use] // fac, ci[use] // fac
    key = pd.DataFrame(dict(J=J, I=I, ctl=ctl[use], pgw=pgw[use]))
    g = key.groupby(["J", "I"]).agg(n_model=("ctl", "size"), ctl_mm=("ctl", "mean"), pgw585_mm=("pgw", "mean")).reset_index()
    if fac == 1:
        g["chirps_mm"] = acc[g.J.clip(0, acc.shape[0] - 1), g.I.clip(0, acc.shape[1] - 1)]
        g = g[(g.n_model >= 2) & g.chirps_mm.notna()].copy()
    else:   # mean of the used 0.05 deg cells inside each coarse cell
        fine = cells(1).assign(J5=lambda x: x.J // fac, I5=lambda x: x.I // fac)
        g = fine.groupby(["J5", "I5"]).agg(n_fine=("chirps_mm", "size"), chirps_mm=("chirps_mm", "mean"), ctl_mm=("ctl_mm", "mean"),
                                           pgw585_mm=("pgw585_mm", "mean")).reset_index().rename(columns=dict(J5="J", I5="I"))
        g = g[g.n_fine >= need_frac * fac * fac].copy()
    g["lat"] = lat0 - (g.J + .5) * dx * fac; g["lon"] = lon0 + (g.I + .5) * dx * fac
    return g

def metrics(g):
    wgt = np.cos(np.radians(g.lat)); r = g[g.chirps_mm >= 1.0]; ratio = r.ctl_mm / r.chirps_mm
    mo, mc, mp = (np.average(g[c], weights=wgt) for c in ("chirps_mm", "ctl_mm", "pgw585_mm"))
    return dict(n_cells=len(g), n_cells_ge_1mm=len(r), area_mean_chirps_mm=round(mo, 1), area_mean_ctl_mm=round(mc, 1),
                area_mean_bias_pct=round((mc / mo - 1) * 100, 1), area_mean_pgw585_mm=round(mp, 1),
                pearson=round(pearsonr(g.chirps_mm, g.ctl_mm).statistic, 3), spearman=round(spearmanr(g.chirps_mm, g.ctl_mm).statistic, 3),
                median_chirps_mm=round(g.chirps_mm.median(), 1), median_ctl_mm=round(g.ctl_mm.median(), 1),
                median_cell_bias_pct=round((ratio.median() - 1) * 100, 1), share_within_factor2=round(((ratio >= .5) & (ratio <= 2)).mean(), 3),
                median_pgw585_mm=round(g.pgw585_mm.median(), 1))

g05 = cells(1); g25 = cells(5, 0.6)
R = {"p05": metrics(g05), "p25": metrics(g25),
     "window": "CHIRPS v2.0 days 27 Apr - 2 May 2024 (UTC days), model 27 Apr 00Z - 3 May 00Z", "model_points_used": int(use.sum())}
print(EV, "chirps", json.dumps(R, indent=1), flush=True)
g05.round(2).to_csv(f"{O}/{EV}_chirps_cells.csv", index=False)
json.dump(R, open(f"{O}/{EV}_chirps_summary.json", "w"), indent=1)
# row in summary.csv (same columns as the station rows; cells instead of stations)
m = R["p05"]; summ_f = f"{O}/summary.csv"; S = pd.read_csv(summ_f)
row = dict(event=f"{EV}_chirps", n_stations=m["n_cells"], n_obs_ge_1mm=m["n_cells_ge_1mm"],
           window="GRIDDED, not stations: CHIRPS v2.0 0.05 deg cells over d02 interior land (n_stations = cells); days 27 Apr - 2 May, "
                  "model 27 Apr 00Z - 3 May 00Z; station_mean_bias_pct = bias of the cos-lat area means "
                  f"({m['area_mean_ctl_mm']} vs {m['area_mean_chirps_mm']} mm); Pearson {m['pearson']}; at 0.25 deg: "
                  f"{R['p25']['n_cells']} cells, Spearman {R['p25']['spearman']}, Pearson {R['p25']['pearson']}",
           median_obs_mm=m["median_chirps_mm"], median_ctl_mm=m["median_ctl_mm"], median_bias_pct=m["median_cell_bias_pct"],
           station_mean_bias_pct=m["area_mean_bias_pct"], spearman=m["spearman"], share_within_factor2=m["share_within_factor2"],
           median_pgw585_mm=m["median_pgw585_mm"], candidates_in_domain=np.nan, excluded_far_from_land=np.nan)
S = S[S.event != row["event"]]
S = pd.concat([S, pd.DataFrame([row])], ignore_index=True)
base = [e.split("_chirps")[0].split("_inmet")[0] for e in S.event]; ev_order = list(dict.fromkeys(base))
S["_k"] = [ev_order.index(b) for b in base]; S["_s"] = [e != b for e, b in zip(S.event, base)]
S = S.sort_values(["_k", "_s"]).drop(columns=["_k", "_s"])
for k in ("n_stations", "n_obs_ge_1mm", "candidates_in_domain", "excluded_far_from_land"): S[k] = S[k].astype("Int64")   # 655, not 655.0
S.to_csv(summ_f, index=False)
