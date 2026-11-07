import os as _os
REPO = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
DATA = _os.path.join(REPO, 'data'); FIG = _os.environ.get('FIGOUT', _os.path.join(REPO, 'fig')); CODE = _os.path.join(REPO, 'fig_code')
# annotations of fig_schematic_story.py (exec'd in its namespace): numbered markers and short labels in the two blocks, headers,
# bottom strip with a map of a real AR, two zoom circles, a key and the numbered story.
import textwrap, glob
import xarray as xr
import cartopy.crs as ccrs, cartopy.feature as cfeature
FSH, FSB = 9.2, 7.4          # headers; every other text
NB = ' '
ANG = np.degrees(np.arctan2(*(P3(1, 0, ZB, 0) - P3(0, 0, ZB, 0))[0][::-1]))   # slope of the block's front edge on the page
def T(x, y, s, col=DARK, ha='left', va='center', fs=FSB, rot=0, halo=True, **k):
    return OV.text(x, y, s, fontsize=fs, color=col, ha=ha, va=va, zorder=9, rotation=rot, rotation_mode='anchor',
                   path_effects=HALO if halo else [], **k)
def num(p, n, col=DARK, r=1.9):
    OV.add_patch(Circle(p, r, facecolor=col, edgecolor='white', lw=0.7, zorder=9.5))
    OV.text(p[0], p[1] - 0.08, str(n), fontsize=FSB - 0.6, color='white', ha='center', va='center', zorder=9.6)
def cen(mask, k):
    xs, vs, zz = X[mask], V[mask], Z[mask]; vis = visible(xs, vs, zz + 0.02); return P3(xs[vis], vs[vis], zz[vis] + 0.02, k).mean(0)
def face(k, a, off, text, col=DARK):
    """text printed on the block's front face, along its bottom edge, starting at fraction a of the edge and off mm above it"""
    p = P3(a, 0, ZB, k)[0]; T(p[0] - off * np.sin(np.radians(ANG)), p[1] + off * np.cos(np.radians(ANG)), text, col, rot=ANG, halo=False)
# ---- block 0 (today)
p = P3(0.02, 0.30, 0.75, 0)[0]; num(p + [-2.2, -6.0], 1); T(p[0] + 0.6, p[1] - 6.0, 'Atmospheric river')
p = P3(0.42, 0.20, CT, 0)[0]; num(p + [-3.0, 1.5], 2); T(p[0] - 5.6, p[1] + 1.5, 'Air rises, cloud forms', ha='right')
q = P3(0.0, 0.92, ZL[0], 0)[0]; T(q[0] + 0.5, q[1] + 2.6, '0 °C level', NAVY)
p = cen(HR[0], 0); num(p + [0, 0.5], 3)
ps = P3(0.78, 0.10, float(zat(0.78, 0.10)[0]) + 0.05, 0)[0]; num(ps + [1.2, 2.6], 4)
face(0, 0.36, 3.2, 'Heavy rain: landslides can start')
face(0, 0.50, 8.2, 'Higher up: snow is stored')
T(2.0, H - 4.5, 'Today', fs=FSH)
# ---- block 1 (warmer)
q0 = P3(0.0, 0.80, ZL[0], 1)[0]; q1 = P3(0.0, 0.80, ZL[1], 1)[0]
OV.add_patch(FancyArrowPatch(q0, q1, arrowstyle='-|>', color=RED, lw=1.1, mutation_scale=7, shrinkA=0, shrinkB=0, zorder=9))
num((q1[0] - 1.0, q1[1] + 10.0), 5, RED); T(q1[0] + 1.7, q1[1] + 10.0, '0 °C level rises', RED)
p = cen(HR[1] & ~HR[0], 1); num(p + [0, 1.0], 6, RED)
face(1, 0.36, 3.2, 'Snow turns to rain: area grows', RED)
face(1, 0.50, 8.2, 'Heavy rain moves upslope', RED)
T(95.0, H - 4.5, 'Warmer climate, same storm', RED, fs=FSH)
# ---------------------------------------------------------------- bottom strip: real AR map, two zoom circles, key, story
d1 = xr.open_dataset(glob.glob(f'{DATA}/inputs/wrf/post/NOR_*/*ctl_d01_ivt.nc')[0])
PRJ = ccrs.LambertConformal(central_longitude=-10, central_latitude=55, standard_parallels=(47, 63))
am = ax_mm(1.5, 9.0, 40, 34, projection=PRJ); am.set_extent([-31, 15, 44, 66.5], ccrs.PlateCarree())
am.add_feature(cfeature.LAND, facecolor='#D9D9D9', edgecolor='none', zorder=0); am.coastlines(lw=0.3, color='#8A8A8A', zorder=1)
iv = gaussian_filter(d1.ivt_max_event.values.astype(float), 1.0)
_ny, _nx = iv.shape; _jj, _ii = np.meshgrid(np.arange(_ny), np.arange(_nx), indexing='ij')
_edge = np.minimum.reduce([_jj, _ii, _ny - 1 - _jj, _nx - 1 - _ii]).astype(float)
iv = iv * np.clip((_edge - 6) / 18, 0, 1)   # fade out the lateral-boundary zone of the outer domain
am.contourf(d1.lon.values, d1.lat.values, iv, levels=[450, 650, 850, 1050, 5000], colors=['#E4E0F1', '#BDB3DC', '#8B79BF', '#55368F'],
            alpha=0.92, transform=ccrs.PlateCarree(), zorder=2)
lo_, la_ = d1.lon.values[202:302, 254:354], d1.lat.values[202:302, 254:354]
bx = np.r_[lo_[0, :], lo_[:, -1], lo_[-1, ::-1], lo_[::-1, 0]]; by = np.r_[la_[0, :], la_[:, -1], la_[-1, ::-1], la_[::-1, 0]]
am.spines['geo'].set_linewidth(0.4); am.spines['geo'].set_edgecolor('#8A8A8A')
num((4.0, 4.6), 1); T(6.8, 4.6, 'Seen from above', halo=False)
RZ = 16.0; cA = (62.0, 27.0); cB = (97.0, 27.0); SOIL, ROCK = '#CDB596', '#D9D3C8'
exec(open(f'{CODE}/figS01_mechanism_zoom.py').read().replace('FSI = 6.4', 'FSI = 6.9'))
num((49.5, 4.6), 4); T(52.3, 4.6, 'Snow holds the water', halo=False)
num((82.0, 4.6), 3); num((86.4, 4.6), 6, RED); T(89.2, 4.6, 'Rain soaks the slope', halo=False)
# key: two columns of short entries
def kbox(x, y, fc, ec='none', lw=0.0):
    OV.add_patch(Rectangle((x, y - 1.3), 4.2, 2.6, facecolor=fc, edgecolor=ec, lw=lw, zorder=9))
KX = (120.0, 153.0); KY = (40.0, 32.6, 25.2, 17.8, 10.4)
def akey(x, y):
    OV.add_patch(Polygon([(x, y - 1.1), (x + 3.0, y - 1.1), (x + 3.0, y - 2.0), (x + 4.6, y), (x + 3.0, y + 2.0), (x + 3.0, y + 1.1), (x, y + 1.1)],
                         closed=True, facecolor='#7C63B5', edgecolor='none', zorder=9))
def lskey(x, y, fc, ec):
    """the landslide outline of the blocks, drawn flat: crown upper left, toe lower right"""
    d = np.array([0.80, -0.60]); nb = np.array([0.60, 0.80]); L, w = 4.4, 1.35
    u = np.r_[LS_T, LS_T[::-1]] * L; sn = np.r_[-LS_HW, LS_HW[::-1]] * w
    P = np.c_[x + 0.45 + u * d[0] + sn * nb[0], y + 1.6 + u * d[1] + sn * nb[1]]
    OV.add_patch(Polygon(P, closed=True, facecolor=fc, edgecolor=ec, lw=0.6, zorder=9))
    m = LS_T <= 0.16; tt = np.r_[LS_T[m][::-1], LS_T[m]]; hh = np.r_[-LS_HW[m][::-1], LS_HW[m]]
    OV.plot(x + 0.45 + tt * L * d[0] + hh * w * nb[0], y + 1.6 + tt * L * d[1] + hh * w * nb[1], color=ec, lw=1.1, zorder=9.1, solid_capstyle='round')
akey(KX[0], KY[0]); T(KX[0] + 6.0, KY[0], 'Atmospheric river', halo=False)
for j in range(4): OV.plot([KX[1] + 0.6 + j * 1.1, KX[1] + 0.2 + j * 1.1], [KY[0] + 1.3, KY[0] - 1.3], color='#2F67B0', lw=0.8, zorder=9)
T(KX[1] + 6.0, KY[0], 'Rain', halo=False)
flake_mm(np.array([KX[1] + 15.5, KY[0]]), 1.0, z=9); T(KX[1] + 17.6, KY[0], 'Snow', halo=False)
kbox(KX[0], KY[1], to_rgb(NAVY) + (0.22,), NAVY, 0.8); T(KX[0] + 6.0, KY[1], '0 °C level, today', halo=False)
kbox(KX[1], KY[1], to_rgb(RED) + (0.18,), RED, 0.8); T(KX[1] + 6.0, KY[1], '0 °C level, warmer', halo=False)
kbox(KX[0], KY[2], '#A6C9E6', BLUE, 0.9); T(KX[0] + 6.0, KY[2], '≥100 mm in 2 days', halo=False)
OV.plot([KX[1], KX[1] + 4.2], [KY[2]] * 2, color=NAVY, lw=0.9, ls=(0, (2.2, 1.4)), zorder=9); T(KX[1] + 6.0, KY[2], 'Today’s extent', halo=False)
kbox(KX[0], KY[3], 'white', '#9DB0C8', 0.6); T(KX[0] + 6.0, KY[3], 'Snow cover', halo=False)
lskey(KX[0], KY[4] - 1.0, '#B8946A', SCAR); T(KX[0] + 6.0, KY[4], 'Landslide', halo=False)
lskey(KX[1], KY[4] - 1.0, '#E58A80', '#8E1020'); T(KX[1] + 6.0, KY[4], 'Possible new\nlandslide', halo=False)
bump_fonts(fig, f=1.0, floor=6.6)
for r_ in overlap_check(fig): print('QA', r_)
