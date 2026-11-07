"""Shared data and helpers for the AR-landslide figures."""
import os as _os
REPO = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
DATA = _os.path.join(REPO, 'data'); FIG = _os.environ.get('FIGOUT', _os.path.join(REPO, 'fig')); CODE = _os.path.join(REPO, 'fig_code')
import sys, numpy as np, pandas as pd
sys.path.insert(0, f'{CODE}')
from figcommon import *
O = f'{DATA}/derived/g1'; CS = f'{DATA}/inputs/cases'
NAVY, AR_C, TC_C, LS_C = '#08306B', '#2171B5', '#E6550D', '#D7301F'
OT_EDGE = '#333333'; OT_C = '#9A9A9A'
INV = {'NVE': ('Norway', '#0072B2'), 'USGS': ('United States', '#E69F00'), 'ITALICA': ('Italy', '#009E73'),
       'GLC': ('Global (GLC)', '#CC79A7'), 'GFLD': ('Global fatal', '#7F7F7F')}
D0 = pd.Timestamp('1979-01-01')

def wilson(k, n, z=1.96):
    if n == 0: return np.nan, np.nan
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h) * 100, (c + h) * 100

def cold_season(lat, month):
    return np.where(np.asarray(lat) >= 0, np.isin(month, [10, 11, 12, 1, 2, 3]), np.isin(month, [4, 5, 6, 7, 8, 9]))

_NAT = None
def national_mask(lat, lon):
    """True where a point lies in the United States, Italy or Norway (Natural Earth 50m), the domains of the national inventories."""
    global _NAT
    import cartopy.io.shapereader as shp
    from shapely.geometry import Point
    from shapely.ops import unary_union
    from shapely.prepared import prep
    if _NAT is None:
        r = shp.Reader(shp.natural_earth('50m', 'cultural', 'admin_0_countries'))
        g = [x.geometry for x in r.records() if x.attributes['ADMIN'] in ('United States of America', 'Italy', 'Norway')]
        _NAT = prep(unary_union(g).buffer(0.5))
    return np.array([_NAT.contains(Point(float(x), float(y))) for y, x in zip(lat, lon)])

def pool_mask(EV):
    """Rows of events.parquet used in pooled statistics: GLC and GFLD events inside the national inventories' countries are left out,
    because those inventories already hold them."""
    lat = EV.ci - 70; lon = np.where(EV.cj > 180, EV.cj - 360, EV.cj)
    return ~(EV.src.isin(['GLC', 'GFLD']).values & national_mask(lat, lon))

def dedupe(B, days=3, deg=2.0):
    """Drop events that another source also holds (dates within +-days, centres within deg in lat and lon), keeping the larger."""
    B = B.sort_values('n', ascending=False).reset_index(drop=True); keep = np.ones(len(B), bool)
    d = B.date.values.astype('datetime64[D]').astype(int); la = B.lat.values; lo = B.lon.values; sr = B.src.values
    from collections import defaultdict
    idx = defaultdict(list)
    for i in range(len(B)):
        hit = False
        for dd in range(d[i] - days, d[i] + days + 1):
            for j in idx.get(dd, ()):
                if sr[j] != sr[i] and abs(la[j] - la[i]) <= deg and abs(((lo[j] - lo[i] + 180) % 360) - 180) <= deg: hit = True; break
            if hit: break
        if hit: keep[i] = False
        else: idx[d[i]].append(i)
    return B[keep]

def events_all():
    """All landslide events of the five inventories (one row per event) with driver (AR/TC/other), plus the event inventories;
    GLC and GFLD events inside the national inventories' countries are left out and cross-source duplicates are removed."""
    A = pd.read_parquet(f'{O}/outbreaks_all.parquet')
    A['drv'] = np.where(A.tcd2 <= 500, 'TC', np.where(A.ar0, 'AR', 'other'))
    A['date'] = pd.to_datetime(A.date0)
    gl = A.src.isin(['GLC', 'GFLD']).values
    A = A[~(gl & national_mask(A.lat.values, A.lon.values))]
    X = pd.read_parquet(f'{O}/extra_outbreaks.parquet')
    X = pd.DataFrame(dict(src=X.source, n=X.n, lat=X.lat, lon=X.lon, drv=X.drv, date=pd.to_datetime(X.date_peak), name=X.name))
    B = pd.concat([A[['src', 'n', 'lat', 'lon', 'drv', 'date']], X], ignore_index=True)
    B = dedupe(B)
    B['month'] = B.date.dt.month
    B['cold'] = cold_season(B.lat, B.month)
    return B

def regional(B=None, nmin=20):
    B = events_all() if B is None else B
    return B[B.n >= nmin].sort_values('n', ascending=False)


# ---------------------------------------------------------------- v06 palette and terrain texture
# AR = navy, TC = vermilion, other storms = warm grey; sequential AR-share ramp: sand -> teal -> navy (discrete levels)
TC_C = '#D7301F'; OT_C = '#9C9C9C'; LS_C = '#B2182B'
AR_RAMP = ['#F4F1E8', '#D9E6E6', '#A8CFD3', '#6FB0BD', '#3D86A6', '#1F5A8C', '#0B3D73']
RAMP5 = ['#EEEBE1', '#CADCDB', '#9DC3C9', '#6FA5B5', '#4A84A1']   # 0-20 ... 80-100 % of extreme-rain days under an AR
LAND_C, SEA_C = '#F1EFE8', '#FFFFFF'
_SR = {}
def _sr_load(p, mmap_mode=None):
    z = np.load(p, mmap_mode=mmap_mode if p.endswith('.npy') else None)
    return z['sr'] if p.endswith('.npz') else z
def hillshade(ax, extent=None, dark=0.55, light=0.45, zorder=0.4, step=None):
    """Natural Earth shaded relief (SR_50M, 1/30 deg) as a translucent texture: shadowed slopes darken, lit slopes lighten, flat land and
    sea stay untouched, so colours drawn below keep their hue. extent = (lon0, lon1, lat0, lat1); step subsamples the raster."""
    import cartopy.crs as ccrs
    full = extent is None or (extent[1] - extent[0]) > 120
    key = 'small' if full else 'full'
    if key not in _SR: _SR[key] = _sr_load(f'{DATA}/inputs/ne_relief/' + ('sr_3600x1800.npy' if full else 'sr_full.npz'), mmap_mode='r')
    a = _SR[key]; res = 0.1 if full else 1 / 30
    lon0, lon1, lat0, lat1 = (-180, 180, -90, 90) if extent is None else extent
    lon0, lon1 = max(lon0, -180), min(lon1, 180); lat0, lat1 = max(lat0, -90), min(lat1, 90)
    r0, r1 = int((90 - lat1) / res), int(np.ceil((90 - lat0) / res)); c0, c1 = int((lon0 + 180) / res), int(np.ceil((lon1 + 180) / res))
    sub = np.asarray(a[r0:r1, c0:c1], dtype=np.float32)
    if step: sub = sub[::step, ::step]
    ext = [-180 + c0 * res, -180 + c1 * res, 90 - r1 * res, 90 - r0 * res]
    H_, W_ = sub.shape
    dk = np.zeros((H_, W_, 4), np.float32); dk[..., 3] = np.clip((205 - sub) / 80, 0, 1) * dark
    lt = np.ones((H_, W_, 4), np.float32); lt[..., 3] = np.clip((sub - 207) / 50, 0, 1) * light
    kw = dict(extent=ext, transform=ccrs.PlateCarree(), origin='upper', interpolation='bilinear', zorder=zorder, regrid_shape=(1200 if full else 900))
    ax.imshow(dk, **kw); ax.imshow(lt, **kw)
