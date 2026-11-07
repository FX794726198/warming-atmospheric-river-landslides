"""One-row-per-run summary of the post-processed AR PGW runs (ctl, pgw585 and the levels/ runs) -> levels/post/levels_summary.csv
Domain: WRF d02 (3 km), 10 edge cells removed, land cells (landmask > 0.5), area weights cell_area_km2 (post npz).
  d02_dT2m: area-weighted land mean of t2_mean(run) - t2_mean(ctl) (post_event d02 file; event-mean simulated 2 m temperature);
  area_liq48_geX: land area (km2) with liq_max48h >= X mm; area_liq48_ge_ctlP99: threshold = ctl 99th percentile of liq_max48h
  over the same land cells (as in exposure_summary.csv);
  precip_total, liq_total, snowfall: area-weighted land means (mm); rain_share_ge1000m = sum(A*liq_total)/sum(A*precip_total)
  over land cells with hgt >= 1000 m;
  rain_line_m: land cells with precip_max48h >= 50 mm binned by hgt (200 m bins from 0 m; bins with >= 20 cells);
  ratio = sum(A*liq_max48h)/sum(A*precip_max48h) per bin; centre of the lowest bin with ratio < 0.5 (NaN if none);
  pop_liq48_ge100, road_km_liq48_ge100: from exposure/exposure_summary.csv (threshold ge100mm)."""
import os, glob, csv, numpy as np, xarray as xr
B = os.path.expandvars("/scratch/$USER/ar_pgw"); L = f"{B}/levels"; E = 10
EVS = ["NOR_2011", "BC_2021", "LIG_2014", "ORE_1996", "RSB_2024"]
ORDER = {"ctl": 0, "pgw585_s025": 1, "pgw585_s050": 2, "pgw585_s075": 3, "pgw585": 4}
EXP = {}
if os.path.exists(f"{B}/exposure/exposure_summary.csv"):
    for r in csv.DictReader(open(f"{B}/exposure/exposure_summary.csv")):
        if r["threshold"] == "ge100mm": EXP[(r["event"], r["run"])] = (r["population"], r["road_km_all"])
rows = []
for ev in EVS:
    runs = {"ctl": (f"{B}/post/{ev}_ctl.npz", f"{B}/post/{ev}_ctl_d02.nc"), "pgw585": (f"{B}/post/{ev}_pgw585.npz", f"{B}/post/{ev}_pgw585_d02.nc")}
    for f in glob.glob(f"{L}/post/{ev}/.posted_*"):
        a = os.path.basename(f)[len(".posted_"):]; runs[a] = (f"{L}/post/{ev}/{ev}_{a}.npz", f"{L}/post/{ev}/{ev}_{a}_d02.nc")
    zc = np.load(runs["ctl"][0]); use = np.zeros(zc["landmask"].shape, bool); use[E:-E, E:-E] = True; use &= zc["landmask"] > 0.5
    A = zc["cell_area_km2"].astype("f8"); hgt = zc["hgt"].astype("f8")
    p99 = float(np.nanpercentile(zc["liq_max48h"][use], 99))
    with xr.open_dataset(runs["ctl"][1]) as d: t2c = d["t2_mean"].values.astype("f8")
    for a in sorted(runs, key=lambda a: (ORDER.get(a, 9), a)):
        z = np.load(runs[a][0])
        with xr.open_dataset(runs[a][1]) as d: t2 = d["t2_mean"].values.astype("f8")
        w = A[use]; mean = lambda x: float(np.sum(x[use] * w) / np.sum(w))
        liq48 = z["liq_max48h"].astype("f8"); pm48 = z["precip_max48h"].astype("f8")
        hi = use & (hgt >= 1000)
        rs = float(np.sum((z["liq_total"] * A)[hi]) / np.sum((z["precip_total"] * A)[hi])) if hi.any() else np.nan
        m = use & (pm48 >= 50); rl = np.nan
        if m.any():
            b = np.floor(hgt[m] / 200).astype(int)
            for k in range(b.max() + 1):
                s = b == k
                if s.sum() < 20: continue
                r = np.sum((liq48[m] * A[m])[s]) / np.sum((pm48[m] * A[m])[s])
                if r < 0.5: rl = k * 200 + 100.0; break
        pop, road = EXP.get((ev, a), ("", ""))
        rows.append(dict(event=ev, run=a, d02_dT2m=f"{mean(t2 - t2c):.3f}",
                         area_liq48_ge50=f"{w[liq48[use] >= 50].sum():.0f}", area_liq48_ge100=f"{w[liq48[use] >= 100].sum():.0f}",
                         area_liq48_ge150=f"{w[liq48[use] >= 150].sum():.0f}", area_liq48_ge_ctlP99=f"{w[liq48[use] >= p99].sum():.0f}",
                         ctlP99_mm=f"{p99:.1f}", precip_total=f"{mean(z['precip_total']):.2f}", liq_total=f"{mean(z['liq_total']):.2f}",
                         snowfall=f"{mean(z['snowfall']):.2f}", rain_share_ge1000m=f"{rs:.3f}", rain_line_m=f"{rl:.0f}" if np.isfinite(rl) else "NaN",
                         pop_liq48_ge100=pop, road_km_liq48_ge100=road))
with open(f"{L}/post/levels_summary.csv", "w", newline="") as f:
    wr = csv.DictWriter(f, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
print(open(f"{L}/post/levels_summary.csv").read())
