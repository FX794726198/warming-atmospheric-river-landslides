"""Print domain corners and lat/lon bounds of geo_em files; write era5_areas.json (d01 bbox + 3 deg margin)."""
import netCDF4, numpy as np, json, os
B = os.path.expandvars("/scratch/$USER/ar_pgw/wps"); A = {}
for e in ["BC_2021", "LIG_2014", "NOR_2011", "ORE_1996", "RSB_2024"]:
    for d in (1, 2):
        f = f"{B}/{e}/geo_em.d0{d}.nc"
        if not os.path.exists(f): print(e, d, "missing"); continue
        g = netCDF4.Dataset(f); la = g["XLAT_M"][0]; lo = g["XLONG_M"][0]; h = g["HGT_M"][0]; lm = g["LANDMASK"][0]
        c = [(float(la[j, i]), float(lo[j, i])) for j, i in ((0, 0), (0, -1), (-1, -1), (-1, 0))]
        print(e, f"d0{d}", la.shape, "SW,SE,NE,NW", ["%.2f,%.2f" % x for x in c],
              "lat %.1f..%.1f lon %.1f..%.1f maxHGT %.0f land %.2f" % (la.min(), la.max(), lo.min(), lo.max(), h.max(), lm.mean()))
        if d == 1:
            m = 3.0
            A[e] = [float(np.ceil(la.max() + m)), float(np.floor(lo.min() - m)), float(np.floor(la.min() - m)), float(np.ceil(lo.max() + m))]
json.dump(A, open(f"{B}/era5_areas.json", "w"), indent=1); print(A)
