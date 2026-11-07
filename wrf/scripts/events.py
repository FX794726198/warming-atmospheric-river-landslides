"""Event definitions for the AR landslide PGW WRF experiment (times UTC).
start = event start minus 12 h spin-up; end = event end given in the brief."""
EVENTS = {
 # name: ref_lat, ref_lon, truelat1, truelat2, e_we d01, e_sn d01, nest target lat, lon, start, end, months for CMIP6 delta (weights)
 "BC_2021":  dict(ref_lat=45.0,  ref_lon=-138.0, tl1=35.0,  tl2=55.0,  nx=390, ny=330, tlat=49.6,  tlon=-121.4,
                  start="2021-11-12_00", end="2021-11-16_12", months={11: 1.0}),
 "LIG_2014": dict(ref_lat=40.0,  ref_lon=3.0,    tl1=30.0,  tl2=50.0,  nx=390, ny=330, tlat=44.5,  tlon=8.8,
                  start="2014-11-08_12", end="2014-11-17_00", months={11: 1.0}),
 "NOR_2011": dict(ref_lat=55.0,  ref_lon=-12.0,  tl1=45.0,  tl2=65.0,  nx=390, ny=330, tlat=60.9,  tlon=6.6,
                  start="2011-12-23_12", end="2011-12-28_00", months={12: 1.0}),
 "ORE_1996": dict(ref_lat=41.0,  ref_lon=-138.0, tl1=30.0,  tl2=50.0,  nx=390, ny=330, tlat=43.7,  tlon=-123.9,
                  start="1996-11-15_12", end="1996-11-20_00", months={11: 1.0}),
 "RSB_2024": dict(ref_lat=-24.0, ref_lon=-56.0,  tl1=-15.0, tl2=-35.0, nx=390, ny=390, tlat=-29.1, tlon=-51.9,
                  start="2024-04-26_12", end="2024-05-03_00", months={4: 4.5/6.5, 5: 2.0/6.5}),
}
NEST = dict(nx=301, ny=301, ratio=3)   # d02: 3 km, 900 km x 900 km (was 151 x 151 in v1)
DX = 9000.0
