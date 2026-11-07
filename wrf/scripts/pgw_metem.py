"""Apply CMIP6 thermodynamic PGW deltas to WPS met_em files.  Method:
  * air temperature: +dT on every met_em level (surface level k=0 gets the tas delta, pressure levels get ta interpolated in ln p);
  * relative humidity held fixed (RH unchanged; real.exe derives QV from RH and the new T); winds unchanged;
  * geopotential height: hydrostatic integration of the virtual-temperature change from the unchanged surface pressure upward;
  * SKINTEMP, SST, soil temperatures (ST*), deep-soil SOILTEMP: +ts delta.
Further details: global 1-deg monthly deltas with month weights per event; periodic longitude; below-ground CMIP6
NaNs filled from the level above; skin/soil T at snow-covered points not pushed above 273.15 K when they were below it; skin T
at sea-ice points capped at 271.35 K; sea ice, snow and soil moisture unchanged.
usage: pgw_metem.py DELTA_monthly.npz EVENT SRC_DIR DST_DIR"""
import sys, glob, os, shutil, json, numpy as np, netCDF4
from scipy.interpolate import RegularGridInterpolator
sys.path.insert(0, os.path.expandvars("/scratch/$USER/ar_pgw/scripts")); from events import EVENTS
dfile, ev, src, dst = sys.argv[1:5]; os.makedirs(dst, exist_ok=True)
Z = np.load(dfile); lat = Z["lat"].astype(float); lon = Z["lon"].astype(float); plev = Z["plev"].astype(float); months = list(Z["months"])
W = EVENTS[ev]["months"]
def mix(a): return sum(w * a[months.index(m)] for m, w in W.items()) / sum(W.values())
ta = mix(Z["ta"].astype("f8")); tas = mix(Z["tas"].astype("f8")); ts = mix(Z["ts"].astype("f8"))
o = np.argsort(-plev); plev = plev[o]; ta = ta[o]                     # descending pressure (near-surface first)
for k in range(len(plev) - 2, -1, -1):                                 # below-ground NaNs: take the level above
    bad = ~np.isfinite(ta[k]); ta[k][bad] = ta[k + 1][bad]
for a in (tas, ts):
    if np.isnan(a).any(): a[np.isnan(a)] = np.nanmean(a)
lonp = np.concatenate([lon, [lon[0] + 360.0]])
def per(a): return np.concatenate([a, a[..., :1]], axis=-1)
I_tas = RegularGridInterpolator((lat, lonp), per(tas)); I_ts = RegularGridInterpolator((lat, lonp), per(ts))
I_ta = [RegularGridInterpolator((lat, lonp), per(ta[k])) for k in range(len(plev))]
Rd, g = 287.04, 9.80665
def es(Tk): Tc = Tk - 273.15; return 611.2 * np.exp(17.67 * Tc / (Tc + 243.5))
cache = {}; log = []
for f in sorted(glob.glob(os.path.join(src, "met_em.d0*.nc"))):
    out = os.path.join(dst, os.path.basename(f)); shutil.copy(f, out)
    d = netCDF4.Dataset(out, "r+"); dom = os.path.basename(f)[7:10]
    if dom not in cache:
        la2 = d["XLAT_M"][0].astype(float); lo2 = d["XLONG_M"][0].astype(float) % 360.0
        pts = np.stack([la2.ravel(), lo2.ravel()], -1); sh = la2.shape
        cache[dom] = (I_tas(pts).reshape(sh), I_ts(pts).reshape(sh), np.stack([I(pts).reshape(sh) for I in I_ta]))
    dtas, dts, dta = cache[dom]
    P = d["PRES"][0].astype("f8"); TT = d["TT"][0].astype("f8"); RH = d["RH"][0].astype("f8")
    lp = np.log(plev); dT = np.empty_like(TT); dT[0] = dtas
    for k in range(1, TT.shape[0]):                                     # linear in ln p between CMIP6 levels
        x = np.log(np.clip(P[k], 1.0, None)); idx = np.clip(np.searchsorted(-lp, -x), 1, len(lp) - 1)
        x0 = lp[idx - 1]; x1 = lp[idx]; w = np.clip((x - x0) / (x1 - x0), 0, 1)
        a0 = np.take_along_axis(dta, (idx - 1)[None], 0)[0]; a1 = np.take_along_axis(dta, idx[None], 0)[0]
        dT[k] = a0 + (a1 - a0) * w
    TTn = TT + dT
    if "GHT" in d.variables:                                            # hydrostatic dZ with fixed surface pressure
        GHT = d["GHT"][0].astype("f8"); dG = np.zeros_like(GHT)
        q = RH / 100 * 0.622 * es(TT) / np.clip(P - 0.378 * RH / 100 * es(TT), 100, None); dTv = dT * (1 + 0.61 * q)
        ps = P[0]; lev = np.arange(1, TT.shape[0]); plv = np.array([np.median(P[k]) for k in lev]); lev = lev[np.argsort(-plv)]
        cum = np.zeros(ps.shape); prevP = ps.copy(); prevD = dTv[0].copy()
        for k in lev:
            pk = P[k]; above = pk < ps
            seg = Rd / g * 0.5 * (prevD + dTv[k]) * np.log(np.clip(prevP, 1, None) / np.clip(pk, 1, None))
            cum = np.where(above, cum + seg, cum)
            dG[k] = np.where(above, cum, Rd / g * dTv[k] * np.log(ps / np.clip(pk, 1, None)))
            prevP = np.where(above, pk, prevP); prevD = np.where(above, dTv[k], prevD)
        d["GHT"][0] = (GHT + dG).astype("f4")
    d["TT"][0] = TTn.astype("f4")
    if "SPECHUMD" in d.variables:
        e = RH / 100 * es(TTn); d["SPECHUMD"][0] = (0.622 * e / np.clip(P - 0.378 * e, 100, None)).astype("f4")
    snow = d["SNOW"][0] > 0 if "SNOW" in d.variables else np.zeros(dts.shape, bool)
    ice = d["SEAICE"][0] > 0 if "SEAICE" in d.variables else np.zeros(dts.shape, bool)
    def warm(a, ok):
        n = a + dts
        n = np.where(snow & (a <= 273.15), np.minimum(n, 273.15), n)   # do not thaw a snow-covered surface at t=0
        return np.where(ok, n, a)
    if "SKINTEMP" in d.variables:
        a = d["SKINTEMP"][0].astype("f8"); n = warm(a, a > 100); n = np.where(ice, np.minimum(n, 271.35), n); d["SKINTEMP"][0] = n.astype("f4")
    if "SST" in d.variables:
        a = d["SST"][0].astype("f8"); d["SST"][0] = np.where(a > 200, a + dts, a).astype("f4")
    if "SOILTEMP" in d.variables:
        a = d["SOILTEMP"][0].astype("f8"); d["SOILTEMP"][0] = np.where(a > 200, a + dts, a).astype("f4")
    for v in list(d.variables):
        if v == "ST" or (v.startswith("ST") and v[2:].isdigit()):
            a = d[v][0].astype("f8"); dd = dts[None] if a.ndim == 3 else dts; sn = snow[None] if a.ndim == 3 else snow
            n = a + dd; n = np.where(sn & (a <= 273.15), np.minimum(n, 273.15), n)
            d[v][0] = np.where(a > 200, n, a).astype("f4")
    k850 = int(np.argmin(np.abs(np.array([np.median(P[k]) for k in range(P.shape[0])]) - 85000.)))
    land = d["LANDMASK"][0] > 0.5
    log.append(dict(file=os.path.basename(f), dom=dom, dT2m=float(dT[0].mean()), dT850=float(dT[k850].mean()),
                    dT2m_land=float(dT[0][land].mean()) if land.any() else None, dSST_ocean=float(dts[~land].mean()) if (~land).any() else None,
                    dZ500=float(dG[int(np.argmin(np.abs(np.array([np.median(P[k]) for k in range(P.shape[0])]) - 50000.)))].mean()) if "GHT" in d.variables else None))
    d.close()
S = {}
for dom in sorted(set(l["dom"] for l in log)):
    L = [l for l in log if l["dom"] == dom]
    S[dom] = {k: float(np.mean([l[k] for l in L if l[k] is not None])) for k in ("dT2m", "dT850", "dT2m_land", "dSST_ocean", "dZ500") if any(l[k] is not None for l in L)}
    S[dom]["files"] = len(L)
S["delta_file"] = dfile; S["month_weights"] = {str(k): v for k, v in W.items()}
json.dump(S, open(os.path.join(dst, "pgw_summary.json"), "w"), indent=1); print(json.dumps(S))
