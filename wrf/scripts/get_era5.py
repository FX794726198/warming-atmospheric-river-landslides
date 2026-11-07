"""ERA5 3-hourly pressure-level + single-level GRIB for the AR PGW events (CDS). Area = d01 bbox + 3 deg.
One request per event x month x kind; files era5/EVENT_YYYYMM_{pl,sl}.grib"""
import os, json, datetime as dt
from concurrent.futures import ThreadPoolExecutor
import cdsapi, sys
sys.path.insert(0, os.path.expandvars("/scratch/$USER/ar_pgw/scripts")); from events import EVENTS
OUT = os.path.expandvars("/scratch/$USER/ar_pgw/era5"); AREA = json.load(open(os.path.expandvars("/scratch/$USER/ar_pgw/wps/era5_areas.json")))
TIMES = [f"{h:02d}:00" for h in range(0, 24, 3)]
LEVELS = ["1","2","3","5","7","10","20","30","50","70","100","125","150","175","200","225","250","300","350","400","450","500","550","600","650","700","750","775","800","825","850","875","900","925","950","975","1000"]
PL = ["geopotential","temperature","u_component_of_wind","v_component_of_wind","specific_humidity","relative_humidity"]
SL = ["surface_pressure","mean_sea_level_pressure","2m_temperature","2m_dewpoint_temperature","10m_u_component_of_wind","10m_v_component_of_wind",
      "skin_temperature","sea_surface_temperature","sea_ice_cover","land_sea_mask","geopotential","snow_depth",
      "soil_temperature_level_1","soil_temperature_level_2","soil_temperature_level_3","soil_temperature_level_4",
      "volumetric_soil_water_layer_1","volumetric_soil_water_layer_2","volumetric_soil_water_layer_3","volumetric_soil_water_layer_4"]
def days(ev):
    s = dt.datetime.strptime(EVENTS[ev]["start"], "%Y-%m-%d_%H"); e = dt.datetime.strptime(EVENTS[ev]["end"], "%Y-%m-%d_%H")
    out = {}; d = s.date()
    while d <= e.date():
        out.setdefault((d.year, d.month), []).append(d.day); d += dt.timedelta(days=1)
    return out
JOBS = []
for ev in EVENTS:
    for (y, m), dd in days(ev).items():
        for kind in ("pl", "sl"): JOBS.append((ev, y, m, dd, kind))
def fetch(j):
    ev, y, m, dd, kind = j; t = f"{OUT}/{ev}_{y}{m:02d}_{kind}.grib"
    if os.path.exists(t) and os.path.getsize(t) > 0: return t + " exists"
    req = {"product_type": ["reanalysis"], "year": [str(y)], "month": [f"{m:02d}"], "day": [f"{d:02d}" for d in dd], "time": TIMES,
           "area": AREA[ev], "data_format": "grib", "download_format": "unarchived"}
    if kind == "pl": req.update(variable=PL, pressure_level=LEVELS); ds = "reanalysis-era5-pressure-levels"
    else: req.update(variable=SL); ds = "reanalysis-era5-single-levels"
    t0 = dt.datetime.now(); c = cdsapi.Client(quiet=True, progress=False); c.retrieve(ds, req, t + ".part"); os.replace(t + ".part", t)
    return f"{t} {os.path.getsize(t)/1e6:.0f} MB {(dt.datetime.now()-t0).seconds} s"
if __name__ == "__main__":
    for j in JOBS: print("job", j[:3], len(j[3]), "days", j[4], flush=True)
    with ThreadPoolExecutor(8) as ex:
        for r in ex.map(fetch, JOBS): print(r, flush=True)
    print("DONE", flush=True)
