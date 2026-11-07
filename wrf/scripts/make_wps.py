"""Write namelist.wps for an event. usage: make_wps.py EVENT OUTFILE [d01only]
With 2 domains, the nest is centred on the landslide target using geo_em.d01.nc in the cwd."""
import os
GEOG = os.environ.get('WPS_GEOG', os.path.expandvars('/scratch/$USER/WPS_GEOG'))   # WPS static geographical data
import sys, numpy as np
sys.path.insert(0, os.path.expandvars("/scratch/$USER/ar_pgw/scripts"))
from events import EVENTS, NEST, DX
ev, out = sys.argv[1], sys.argv[2]; d01only = len(sys.argv) > 3
E = EVENTS[ev]; r = NEST["ratio"]
s = E["start"]; e = E["end"]
if d01only:
    md, ips, jps, we, sn, pid, rat, res = 1, "1", "1", f"{E['nx']}", f"{E['ny']}", "1", "1", '"default"'
else:
    import netCDF4
    g = netCDF4.Dataset("geo_em.d01.nc"); la = g["XLAT_M"][0]; lo = g["XLONG_M"][0]
    d = (la - E["tlat"])**2 + ((lo - E["tlon"]) * np.cos(np.radians(E["tlat"])))**2
    j, i = np.unravel_index(np.argmin(d), d.shape)          # 0-based mass-point indices on d01
    npar = (NEST["nx"] - 1) // r                             # parent cells spanned by the nest
    ip = int(i + 1 - npar // 2); jp = int(j + 1 - (NEST["ny"] - 1) // r // 2)
    md = 2; ips = f"1, {ip}"; jps = f"1, {jp}"; we = f"{E['nx']}, {NEST['nx']}"; sn = f"{E['ny']}, {NEST['ny']}"
    pid = "1, 1"; rat = f"1, {r}"; res = '"default","default"'
    print(f"{ev}: target d01 index i={i+1} j={j+1} -> i_parent_start={ip} j_parent_start={jp}")
sd = ",".join([f'"{s}:00:00"'] * md); ed = ",".join([f'"{e}:00:00"'] + [f'"{s}:00:00"'] * (md - 1))   # nest: initial time only (sst_update = 0)
txt = f"""&share
 wrf_core = "ARW", max_dom = {md},
 start_date = {sd},
 end_date   = {ed},
 interval_seconds = 10800, io_form_geogrid = 2,
/
&geogrid
 parent_id = {pid}, parent_grid_ratio = {rat},
 i_parent_start = {ips}, j_parent_start = {jps},
 e_we = {we}, e_sn = {sn},
 geog_data_res = {res},
 dx = {DX:.0f}, dy = {DX:.0f}, map_proj = "lambert",
 ref_lat = {E['ref_lat']}, ref_lon = {E['ref_lon']}, truelat1 = {E['tl1']}, truelat2 = {E['tl2']}, stand_lon = {E['ref_lon']},
 geog_data_path = "{GEOG}",
/
&ungrib
 out_format = "WPS", prefix = "FILE",
/
&metgrid
 fg_name = "FILE", io_form_metgrid = 2,
/
"""
open(out, "w").write(txt)
