"""Check of the extreme-precipitation area at 0.25 deg on the daily files of the analysis-ready ERA5 archive (arco_daily.py),
with the definition of v6_fp48.py. The output is the input of scripts/probability/fp025_prob.py.
usage: fp025_arco.py EVENTS_CSV ARCO_DAILY_DIR OUT_CSV
  cells.npz: idx into the 721 x 1440 ARCO grid (lat 90..-90, lon 0..359.75), land cells (lsm > 0.5) at 20-72 deg latitude;
  daily_<y>.npz: tp (uint16, 0.1 mm), day D = hourly accumulations ending D 01 UTC .. D+1 00 UTC;
  P99 = p99_base7922.npy (snow_counts.py pass 1, BASE=base7922: 99th percentile of 2-day tp over 1979-2022, same cell order).
  r2(d) = tp(d) + tp(d-1) (2-day precipitation, the day and the day before);
  window = land cells whose containing 1 deg cell (floor(lat+0.5), floor(lon+0.5)) lies within 4 index offsets of the event's
  1 deg centre cell (the same 9 x 9 deg area as at 1 deg); cell area (111.2 * 0.25)^2 cos(lat) km2 (same formula as at 1 deg);
  fp_sim_km2 = largest land area with r2 >= P99 on a single day d0..d1; fp_uni_km2 = area of the union over d0..d1.
  Events: centre |lat| >= 25 and d1 <= 2022-12-31. window_frac = share of the window's 36 rows of 0.25 deg inside 20-72 deg
  (cells exist only there; 1.0 = complete window)."""
import sys, numpy as np, pandas as pd, collections, datetime as dt
EVF, D, OUTF = sys.argv[1:4]
C = np.load(f"{D}/cells.npz"); idx = C["idx"]; lat = C["lat"][idx // 1440].astype("f8"); lon = C["lon"][idx % 1440].astype("f8")
P99 = np.load(f"{D}/p99_base7922.npy").astype("f8")
ci = np.floor(lat + 0.5).astype(int); cj = np.floor(lon + 0.5).astype(int) % 360
area = (111.2 * 0.25) ** 2 * np.cos(np.deg2rad(lat))
by1 = collections.defaultdict(list)
for k, key in enumerate(zip(ci, cj)): by1[key].append(k)
by1 = {k: np.array(v) for k, v in by1.items()}
EV = pd.read_csv(EVF); EV = EV[(EV.lat.abs() >= 25) & (EV.d1 <= 16070)].sort_values("d0")
t79 = dt.date(1979, 1, 1); cache = {}
def tp_year(y):
    if y not in cache:
        if len(cache) > 2: cache.pop(min(cache))
        cache[y] = np.load(f"{D}/daily_{y}.npz")["tp"]
    return cache[y]
def tp_day(dn):
    d = t79 + dt.timedelta(days=int(dn)); return tp_year(d.year)[d.timetuple().tm_yday - 1].astype("f8") / 10.0
out = []
for e in EV.itertuples():
    la0 = int(round(e.lat)); lo0 = int(round(e.lon)) % 360
    ks = [by1[(a, b % 360)] for a in range(la0 - 4, la0 + 5) for b in range(lo0 - 4, lo0 + 5) if (a, b % 360) in by1]
    rows = np.arange(la0 - 4.5, la0 + 4.5, 0.25); frac = float(np.mean((np.abs(rows) >= 20) & (np.abs(rows) <= 72)))
    if not ks:
        sim = uni = 0.0; simc = 0; wl = 0.0
    else:
        w = np.concatenate(ks); A = area[w]; q = P99[w]; ex = np.zeros(w.size, bool); sim = 0.0; simc = 0; wl = float(A.sum())
        for dn in range(int(e.d0), int(e.d1) + 1):
            r2 = tp_day(dn)[w] + (tp_day(dn - 1)[w] if dn > 0 else 0.0)
            h = r2 >= q; ex |= h; sim = max(sim, float(A[h].sum())); simc = max(simc, int(h.sum()))
        uni = float(A[ex].sum())
    out.append(dict(ev=e.ev, src=e.src, n=e.n, d0=e.d0, d1=e.d1, lat=e.lat, lon=e.lon, ar_frac=e.ar_frac, pooled=e.pooled, dur=e.dur,
                    box="arco", window_frac=frac, fp_sim_km2_1deg=e.fp_sim_km2, fp_uni_km2_1deg=e.fp_uni_km2, fp_sim_cells_1deg=e.fp_sim_cells,
                    fp_sim_km2_025=sim, fp_uni_km2_025=uni, fp_sim_cells_025=simc, window_land_km2_025=wl))
pd.DataFrame(out).to_csv(OUTF, index=False); print("events written", len(out), OUTF)
