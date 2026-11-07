"""Supplementary mechanism figure (story version). Two identical solid mountain blocks side by side, left = today, right = the same
storm in a warmer climate. The front face of each block is a cross-section from the sea to the crest; the atmospheric river flows in
from the sea, is lifted by the range into the clouds, and precipitation falls as snow above and rain below the 0 degC level
(translucent plane). On the slopes, the heavy-rain area (blue) is the part of the storm's footprint below the 0 degC level, and the
landslides cluster inside it. Numbered steps are explained in a bottom strip with two zoom circles and a map of a real AR."""
import os as _os
REPO = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
DATA = _os.path.join(REPO, 'data'); FIG = _os.environ.get('FIGOUT', _os.path.join(REPO, 'fig')); CODE = _os.path.join(REPO, 'fig_code')
import sys
sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from arcommon import *
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.patches import FancyArrowPatch, Ellipse, Circle, Polygon, Rectangle
from matplotlib.lines import Line2D
from matplotlib.colors import to_rgb
from matplotlib.collections import LineCollection
from scipy.ndimage import map_coordinates, gaussian_filter
import contourpy
W, H = 183, 106
fig = plt.figure(figsize=(W * MM, H * MM))
def ax_mm(l, b, w, h, **k): return fig.add_axes([l / W, b / H, w / W, h / H], **k)
RED, NAVY, BLUE, DARK, SNOWC, SCAR = '#B2182B', '#1F3B73', '#2F6DB5', '#2B2B2B', '#5F7898', '#4A3424'
FS, FSK = 7.6, 7.0
ZL = {0: 1.3, 1: 2.1}
AZ, EL = -118.0, 20.0
SW = (0.20, 0.74)
# ---------------------------------------------------------------- terrain: x = sea (0) to crest (0.80) and lee (1), v = back along the range
nx, nv = 181, 121
xg = np.linspace(0, 1, nx); vg = np.linspace(0, 1, nv); X, V = np.meshgrid(xg, vg)
rng = np.random.default_rng(11)
VAL = (0.36, 0.86)
def noise(sig, seed):
    r = np.random.default_rng(seed).standard_normal((nv, nx)); r = gaussian_filter(r, sig); return r / r.std()
def yc_(yv, x): return yv + 0.03 * np.sin(8 * x)
def terrain():
    crest = 2.75 + 0.45 * np.sin(2 * np.pi * (V * 0.9 + 0.10)) + 0.25 * np.sin(2 * np.pi * (V * 2.1 + 0.4))
    z = crest * np.exp(-((X - 0.80) / np.where(X < 0.80, 0.25, 0.35)) ** 2)
    z = z + 0.35 * np.exp(-((X - 0.43) / 0.06) ** 2) * (0.8 + 0.2 * np.sin(2 * np.pi * (V * 2.3 + 0.2)))
    for yv in VAL: z = z * (1 - 0.40 * np.exp(-((V - yc_(yv, X)) / 0.06) ** 2) * np.clip((X - 0.34) / 0.22, 0, 1) * np.clip((0.95 - X) / 0.2, 0, 1))
    z = z * (1 + 0.06 * np.sin(2 * np.pi * V * 4.3 + 5 * X)) * (1 + 0.05 * noise(4, 3))
    z = z + 0.05 + 0.08 * np.clip((X - 0.30) / 0.12, 0, 1)
    sea = 1 - np.clip((X - 0.21) / 0.04, 0, 1)
    return z * (1 - sea) - 0.25 * sea
Z = terrain(); ZB = -0.65; ZS = np.maximum(Z, 0.0)
print('zmax', Z.max().round(2))
def zat(x, v):
    x, v = np.atleast_1d(x), np.atleast_1d(v); return map_coordinates(Z, [np.clip(v, 0, 1) * (nv - 1), np.clip(x, 0, 1) * (nx - 1)], order=1)
GX = np.gradient(Z, xg, axis=1); GV = np.gradient(Z, vg, axis=0)
def gat(x, v):
    x, v = np.atleast_1d(x), np.atleast_1d(v); c = [v * (nv - 1), x * (nx - 1)]
    return map_coordinates(GX, c, order=1), map_coordinates(GV, c, order=1)
BOX = (1.0, 0.62, 0.50); ZR = 3.9 - ZB
# visibility: march from a point towards the (orthographic) viewer and test whether the terrain is in the way
_ce, _se, _ca, _sa = np.cos(np.radians(EL)), np.sin(np.radians(EL)), np.cos(np.radians(AZ)), np.sin(np.radians(AZ))
DV = np.array([_ce * _ca / BOX[0], _ce * _sa / BOX[1], _se * ZR / BOX[2]])
TS_ = np.linspace(0.004, 1.6, 320)
def visible(x, v, z):
    shp = np.shape(x); x, v, z = np.ravel(x)[:, None], np.ravel(v)[:, None], np.ravel(z)[:, None]
    xs, vs, zs = x + DV[0] * TS_, v + DV[1] * TS_, z + DV[2] * TS_
    ins = (xs >= 0) & (xs <= 1) & (vs >= 0) & (vs <= 1) & (zs < 4.0)
    zt = map_coordinates(Z, [np.clip(vs, 0, 1).ravel() * (nv - 1), np.clip(xs, 0, 1).ravel() * (nx - 1)], order=1).reshape(xs.shape)
    return (~((zt > zs + 0.01) & ins).any(-1)).reshape(shp)
SNOWN = 0.07 * noise(1.8, 5)
c_plain, c_forest, c_high, c_rock = map(np.array, map(to_rgb, ('#BCD39C', '#7FA878', '#A3AE86', '#B9AE97')))
def lerp(a, b, t): return a * (1 - t[..., None]) + b * t[..., None]
t1 = np.clip((Z - 0.20) / 0.35, 0, 1); t2 = np.clip((Z - 1.25) / 0.6, 0, 1); t3 = np.clip((Z - 2.1) / 0.5, 0, 1)
RGB0 = lerp(lerp(lerp(c_plain, c_forest, t1), c_high, t2), c_rock, t3)
dzdx = np.gradient(Z * BOX[2] / ZR * 2.0, xg, axis=1) / BOX[0]; dzdv = np.gradient(Z * BOX[2] / ZR * 2.0, vg, axis=0) / BOX[1]
nrm = np.dstack([-dzdx, -dzdv, np.ones_like(Z)]); nrm /= np.linalg.norm(nrm, axis=2, keepdims=True)
a_, h_ = np.deg2rad(AZ + 180 - 35), np.deg2rad(42); Lv = np.array([np.cos(h_) * np.cos(a_), np.cos(h_) * np.sin(a_), np.sin(h_)])
F = 0.62 + 0.45 * np.clip(nrm @ Lv, 0, 1)
# ---------------------------------------------------------------- the storm's footprint and the heavy-rain area below the 0 degC level
def swath(x, v): return (v > SW[0] + 0.02 * np.sin(9 * x + 1)) & (v < SW[1] + 0.02 * np.sin(7 * x + 2))
X0HR = 0.37
def heavy(k): return swath(X, V) & (X > X0HR + 0.015 * np.sin(11 * V)) & (Z < ZL[k] + SNOWN) & (X < 0.82)
HR = {k: heavy(k) for k in (0, 1)}
SNOW = {k: Z > ZL[k] + SNOWN for k in (0, 1)}
HRC = np.array(to_rgb('#8EC3EE'))
def colours(k):
    rgb = RGB0.copy(); m = HR[k]; rgb[m] = rgb[m] * 0.42 + HRC * 0.58
    sh = rgb * F[..., None]
    lo, hi = np.array([0.82, 0.87, 0.94]), np.ones(3); s = SNOW[k]
    sh[s] = lo + (hi - lo) * ((F[s] - 0.62) / 0.45)[:, None]
    sh[Z < 0] = np.array(to_rgb('#8DB8DA'))
    return np.clip(sh, 0, 1)
def outline(mask):
    cg = contourpy.contour_generator(xg, vg, gaussian_filter(mask.astype(float), 0.8)); return [l for l in cg.lines(0.5) if len(l) > 12]
HOUSES = []
for yv in VAL:
    for j in range(5): HOUSES.append((0.255 + 0.026 * (j % 3) + rng.uniform(-0.004, 0.004), yc_(yv, 0.27) + (-0.085 if j < 3 else 0.05) + rng.uniform(-0.012, 0.012)))
cand = rng.uniform([0.36, 0.03], [0.76, 0.97], size=(50000, 2)); zc = zat(cand[:, 0], cand[:, 1]); gxc, gvc = gat(cand[:, 0], cand[:, 1])
steep = gxc > 1.6
away = np.ones(len(cand), bool)
for yv in VAL: away &= np.abs(cand[:, 1] - yc_(yv, cand[:, 0])) > 0.045
insw = swath(cand[:, 0], cand[:, 1]) & (cand[:, 1] > SW[0] + 0.04) & (cand[:, 1] < SW[1] - 0.04)
vis_c = visible(cand[:, 0], cand[:, 1], zc + 0.02)
def pick(mask, n, dmin):
    out = []
    for i in rng.permutation(np.where(mask)[0]):
        if all(np.hypot(*(cand[i] - cand[j])) > dmin for j in out): out.append(i)
        if len(out) == n: break
    return out
I_old = pick(steep & away & insw & vis_c & (cand[:, 0] > X0HR + 0.03) & (zc < ZL[0] - 0.10), 18, 0.05)
I_out = pick(steep & away & vis_c & ~swath(cand[:, 0], cand[:, 1]) & (cand[:, 0] > 0.42) & (zc < ZL[0] - 0.1), 3, 0.15)
I_new = pick(steep & away & insw & vis_c & (zc > ZL[0] + 0.12) & (zc < ZL[1] - 0.10), 14, 0.05)
print('scars', len(I_old), len(I_out), len(I_new))
LS_T = np.linspace(0, 1, 36)
LS_HW = np.interp(LS_T, [0, 0.03, 0.10, 0.22, 0.55, 0.78, 0.92, 1.0], [0, 0.72, 0.97, 0.90, 0.55, 0.74, 0.55, 0])
AXS = {}
def block(k, rect):
    ax = ax_mm(*rect, projection='3d'); AXS[k] = ax
    ax.set_proj_type('ortho'); ax.set_box_aspect(BOX, zoom=1.0); ax.view_init(elev=EL, azim=AZ); ax.set_axis_off(); ax.computed_zorder = False
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_zlim(ZB, 3.9)
    ax.plot_surface(X, V, ZS, facecolors=colours(k), rstride=1, cstride=1, linewidth=0, antialiased=False, shade=False, zorder=1.1)
    def side(t_, b_, col, z): ax.add_collection3d(Poly3DCollection([t_ + b_[::-1]], facecolor=col, edgecolor='none', zorder=z))
    for (xs_, vv, zz) in ((xg, np.zeros(nx), ZS[0]), (np.zeros(nv), vg, ZS[:, 0])):
        top = list(zip(xs_, vv, zz)); soil = list(zip(xs_, vv, np.maximum(zz - 0.22, ZB))); base = list(zip(xs_, vv, np.full(len(zz), ZB)))
        side(top, soil, '#C9AE8A', 1.0); side(soil, base, '#DAD2C5', 0.99)
        kk = np.where(zz <= 0.001)[0]
        if len(kk): side(list(zip(xs_[kk], vv[kk], np.zeros(len(kk)))), list(zip(xs_[kk], vv[kk], np.full(len(kk), -0.25))), '#6FA3CF', 1.02)
        ax.plot(xs_, vv, zz, color='#6B6257', lw=0.5, zorder=1.05)
    def vline(x, v, z, **kw):
        m = visible(x, v, z); z = np.where(m, z, np.nan); ax.plot(x, v, z, **kw)
    for yv in VAL:
        xr = np.linspace(0.19, 0.80, 160); vr = yc_(yv, xr); vline(xr, vr, zat(xr, vr) + 0.015, color='#3F7FC0', lw=0.9, zorder=1.5)
    vrd = np.linspace(0.006, 0.994, 220); xrd = 0.31 + 0.006 * np.sin(17 * vrd); vline(xrd, vrd, zat(xrd, vrd) + 0.015, color='#7A7A7A', lw=0.7, zorder=1.5)
    for hx, hv in HOUSES:
        z0 = float(zat(hx, hv)[0]) + 0.005; dx, dv, hh = 0.016, 0.025, 0.10
        base = [(hx, hv, z0), (hx + dx, hv, z0), (hx + dx, hv + dv, z0), (hx, hv + dv, z0)]; top = [(a, b, c + hh) for a, b, c in base]
        ax.add_collection3d(Poly3DCollection([[base[0], base[1], top[1], top[0]], [base[0], base[3], top[3], top[0]], top], facecolors=['#F4F4F4', '#D8D8D8', '#B5524A'], edgecolor='#555555', linewidths=0.25, zorder=1.6))
    for l in outline(HR[k]): vline(l[:, 0], l[:, 1], zat(l[:, 0], l[:, 1]) + 0.02, color=BLUE, lw=1.1, zorder=1.65)
    if k == 1:
        for l in outline(HR[0]): vline(l[:, 0], l[:, 1], zat(l[:, 0], l[:, 1]) + 0.02, color=NAVY, lw=0.8, ls=(0, (2.2, 1.4)), zorder=1.66)
    def scar(x, v, fc, ec, L=0.040, w=0.0125):
        """landslide outline: rounded crown (head scarp) upslope, narrower track, lobate toe downslope"""
        gx, gv = gat(x, v); d = -np.array([gx[0] / BOX[0], gv[0] / BOX[1]]); d /= np.linalg.norm(d) + 1e-9; nb = np.array([-d[1], d[0]])
        t, hw = LS_T, LS_HW
        u = np.r_[t, t[::-1]] * L - 0.25 * L; sn = np.r_[-hw, hw[::-1]] * w
        px = np.clip(x + u * d[0] + sn * nb[0], 0.001, 0.999); pv = np.clip(v + (u * d[1] + sn * nb[1]) * BOX[0] / BOX[1], 0.001, 0.999); pz = zat(px, pv) + 0.03
        ax.add_collection3d(Poly3DCollection([list(zip(px, pv, pz))], facecolor=fc, edgecolor=ec, linewidths=0.4, zorder=1.7))
        m = t <= 0.16; tt = np.r_[t[m][::-1], t[m]]; hh = np.r_[-hw[m][::-1], hw[m]]
        cx = np.clip(x + (tt * L - 0.25 * L) * d[0] + hh * w * nb[0], 0.001, 0.999); cv = np.clip(v + ((tt * L - 0.25 * L) * d[1] + hh * w * nb[1]) * BOX[0] / BOX[1], 0.001, 0.999)
        ax.plot(cx, cv, zat(cx, cv) + 0.035, color=ec, lw=0.9, zorder=1.72, solid_capstyle='round')
    for i in I_old + I_out: scar(cand[i, 0], cand[i, 1], '#B8946A', SCAR)
    if k == 1:
        for i in I_new: scar(cand[i, 0], cand[i, 1], '#E58A80', '#8E1020')
    for ch in ax.get_children(): ch.set_clip_on(False)
DY = 28
RECT = {0: (-15.5, DY, 118, 92), 1: (77.5, DY, 118, 92)}
block(0, RECT[0]); block(1, RECT[1])
fig.canvas.draw()
def P3(x, v, z, k):
    ax = AXS[k]; x2, y2, _ = proj3d.proj_transform(np.ravel(x).astype(float), np.ravel(v).astype(float), np.ravel(z).astype(float), ax.get_proj())
    d = ax.transData.transform(np.c_[x2, y2]); p = fig.transFigure.inverted().transform(d); return p * [W, H]
for k in (0, 1): print(k, [P3(*pt, k).round(1) for pt in [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0), (0.8, 0.35, 3.4), (0.8, 0, 3.0), (0, 0, ZB), (-0.12, 0.47, 0.85), (0.56, 0.47, 4.8)]])
# ---------------------------------------------------------------- overlay rasters (AR tube, clouds) and precipitation shafts
OV = fig.add_axes([0, 0, 1, 1], zorder=5); OV.set_xlim(0, W); OV.set_ylim(0, H); OV.set_axis_off(); OV.patch.set_alpha(0)
PX = 9.0; NXp, NYp = int(W * PX), int(H * PX)
def splat(P, w, img):
    i = np.clip((P[:, 0] * PX).astype(int), 0, NXp - 1); j = np.clip((P[:, 1] * PX).astype(int), 0, NYp - 1); np.add.at(img, (j, i), w)
def show(img, z):
    a = img[..., 3] > 0.004
    if not a.any(): return
    jj, ii = np.where(a.any(1))[0], np.where(a.any(0))[0]; j0, j1, i0, i1 = jj[0], jj[-1] + 1, ii[0], ii[-1] + 1
    OV.imshow(img[j0:j1, i0:i1], extent=(i0 / PX, i1 / PX, j0 / PX, j1 / PX), origin='lower', zorder=z, interpolation='bilinear')
def smooth01(t): t = np.clip(t, 0, 1); return t * t * (3 - 2 * t)
_env = gaussian_filter(np.maximum(Z[(vg > 0.18) & (vg < 0.76)].max(0), 0), 3)
def ar_R(s): return 1.0 - 0.58 * smooth01((s - 0.42) / 0.55)
def ar_path(s):
    xc = -0.13 + 0.74 * s
    zl_ = 0.80 + 1.85 * smooth01((xc - 0.20) / 0.42)
    return xc, np.maximum(zl_, np.interp(xc, xg, _env) + 0.30 * ar_R(s) + 0.10)
TEAL0, TEAL1 = '#C9C3E3', '#4B2C8A'
def ar_tube(k):
    ns, nphi, nrho = 700, 40, 10
    s = np.linspace(0, 1, ns)[:, None, None]; phi = np.linspace(0, 2 * np.pi, nphi, endpoint=False)[None, :, None]; rho = np.linspace(0.05, 1, nrho)[None, None, :]
    xc, zc_ = ar_path(s)
    R = ar_R(s) * (0.80 + 0.20 * smooth01(s / 0.3))
    vv = 0.47 + 0.26 * R * rho * np.cos(phi); zz = zc_ + 0.26 * R * rho * np.sin(phi); xx = xc + 0.0 * vv
    wgt = (1 - rho ** 2) * smooth01(s / 0.10) * (1 - 0.80 * smooth01((s - 0.33) / 0.32)) * (1 - smooth01((s - 0.86) / 0.14)) * np.ones_like(phi) * np.ones_like(xx)
    P = P3(xx, vv, zz, k)
    D = np.zeros((NYp, NXp)); splat(P, wgt.ravel(), D)
    S_ = np.zeros((NYp, NXp)); splat(P, (wgt * s).ravel(), S_)
    D = gaussian_filter(D, 1.0); S_ = gaussian_filter(S_, 1.0); sp = np.where(D > 1e-6, S_ / np.maximum(D, 1e-9), 0)
    A = 1 - np.exp(-D / np.percentile(D[D > 1e-3], 40) * 2.4)
    c0, c1 = np.array(to_rgb(TEAL0)), np.array(to_rgb(TEAL1))
    img = np.zeros((NYp, NXp, 4)); t_ = np.clip(sp / 0.45, 0, 1)[..., None] ** 0.8; img[..., :3] = c0 * (1 - t_) + c1 * t_
    img[..., 3] = np.clip(A, 0, 0.92) * (1 - 0.72 * smooth01((sp - 0.36) / 0.26))   # thin and translucent where it is lifted over the slope
    show(img, 6)
    for r_ in (-0.6, 0.0, 0.6):
        ss = np.linspace(0.06, 0.93, 120); xc, zc_ = ar_path(ss); RR = ar_R(ss)
        q = P3(xc, 0.47 + 0.18 * r_ * RR, zc_ + 0.12 * (1 - abs(r_)) * RR, k)
        seg = np.stack([q[:-1], q[1:]], 1); al = 0.10 + 0.70 * np.sin(np.pi * np.linspace(0, 1, len(seg))) ** 0.7
        OV.add_collection(LineCollection(seg, colors=[(1, 1, 1, a) for a in al], lw=0.55, zorder=6.2))
CB, CT = 2.75, 3.55
def clouds(k):
    rr = np.random.default_rng(7)
    D = np.zeros((NYp, NXp)); Hh = np.zeros((NYp, NXp))
    n = 60; cen = np.c_[rr.uniform(0.42, 0.68, n), rr.uniform(0.20, 0.74, n), rr.uniform(CB + 0.30, CT, n)]
    for (cx, cv, cz) in cen:
        p = P3(cx, cv, cz, k)[0]; rad = rr.uniform(1.6, 3.0) * (0.7 + 0.5 * (cz - CB) / (CT - CB))
        gx_ = np.arange(int((p[0] - 3 * rad) * PX), int((p[0] + 3 * rad) * PX)); gy_ = np.arange(int((p[1] - 3 * rad) * PX), int((p[1] + 3 * rad) * PX))
        gx_ = gx_[(gx_ >= 0) & (gx_ < NXp)]; gy_ = gy_[(gy_ >= 0) & (gy_ < NYp)]
        GXp, GYp = np.meshgrid(gx_ / PX, gy_ / PX); g = np.exp(-((GXp - p[0]) ** 2 + (GYp - p[1]) ** 2) / (2 * rad ** 2))
        D[np.ix_(gy_, gx_)] += g; Hh[np.ix_(gy_, gx_)] += g * (cz - CB) / (CT - CB)
    hf = np.where(D > 1e-6, Hh / np.maximum(D, 1e-9), 0)
    A = np.clip((D - 0.45) / 0.35, 0, 1)
    Dsm = gaussian_filter(D, 6); gy, gx = np.gradient(Dsm); lit = (gx * 0.6 - gy * 0.8); lit = lit / (np.percentile(np.abs(lit[A > 0.1]), 90) + 1e-9)
    sh = np.clip(0.55 + 0.45 * lit, 0, 1) * 0.6 + 0.4 * np.clip(hf, 0, 1)
    c0, c1 = np.array(to_rgb('#8E9AAA')), np.array(to_rgb('#FFFFFF'))
    img = np.zeros((NYp, NXp, 4)); t = np.clip(sh, 0, 1)[..., None]; img[..., :3] = c0 * (1 - t) + c1 * t; img[..., 3] = np.clip(A, 0, 0.96)
    show(img, 6.5)
def curtain(k):
    """dense fine streaks from the cloud base to the ground: rain below the 0 degC level, snow above"""
    rr = np.random.default_rng(21); segs = []
    for x0 in np.linspace(0.41, 0.73, 10):
        for v0 in np.linspace(SW[0] + 0.05, SW[1] - 0.05, 8):
            x, v = x0 + rr.uniform(-0.012, 0.012), v0 + rr.uniform(-0.025, 0.025)
            zg = float(zat(x, v)[0]); ztop = min(ZL[k], CB) - 0.02
            if zg > ztop - 0.25: continue
            off = rr.uniform(0, 0.22); z1 = ztop - off
            while z1 - 0.13 > zg + 0.03:
                z2 = z1 - 0.13; xa_, xb_ = x - 0.05 * (z1 - zg) / 2.0, x - 0.05 * (z2 - zg) / 2.0
                va_ = np.array([v, v]); za_ = np.array([z1, z2]); xs_ = np.array([xa_, xb_])
                if visible(xs_, va_, za_ + 0.01).all(): segs.append(P3(xs_, va_, za_, k))
                z1 = z2 - 0.11
    OV.add_collection(LineCollection(segs, colors='#2F67B0', lw=0.75, alpha=0.9, zorder=6.3, capstyle='round'))
    # snowflakes between the cloud base and the 0 degC level
    fl = []
    for _try in range(4000):
        if len(fl) >= 24: break
        x, v = rr.uniform(0.44, 0.74), rr.uniform(SW[0] + 0.03, SW[1] - 0.03); zg = float(zat(x, v)[0])
        lo_ = max(ZL[k], zg) + 0.12
        if lo_ > CB - 0.1: continue
        z = rr.uniform(lo_, CB - 0.05)
        if not np.ravel(visible(x, v, z))[0]: continue
        p = P3(x, v, z, k)[0]
        if all(np.hypot(*(p - f)) > 3.0 for f in fl): fl.append(p)
    for p in fl: flake_mm(p, 0.75)
def flake_mm(p, r, col='#5E7394', lw=0.5, z=6.4):
    segs = []
    for ang in (90, 30, 150):
        t = np.radians(ang); d = np.array([np.cos(t), np.sin(t)]) * r; segs.append([p - d, p + d])
    OV.add_collection(LineCollection(segs, colors=col, lw=lw, zorder=z, capstyle='round'))
def zero_plane(k, zl, col, alpha, edge_ls='-', fill=True):
    xs = np.linspace(0.0, 1, 340); vs = np.linspace(0, 1, 220); XX, VV = np.meshgrid(xs, vs)
    ZZ = zat(XX.ravel(), VV.ravel()).reshape(XX.shape)
    m = (ZZ < zl - 0.01) & visible(XX, VV, np.full_like(XX, zl))
    if fill:
        q = P3(XX[m], VV[m], np.full(m.sum(), zl), k); D = np.zeros((NYp, NXp)); splat(q, np.ones(len(q)), D)
        D = gaussian_filter(D, 1.2)
        A = np.clip(D / np.percentile(D[D > 1e-3], 50), 0, 1) * alpha
        img = np.zeros((NYp, NXp, 4)); img[..., :3] = to_rgb(col); img[..., 3] = A
        show(img, 5.8)
    xf = np.linspace(0, 1, 500); zf = zat(xf, np.zeros_like(xf)); xe = xf[np.argmax(zf >= zl)]
    q = P3(xf[xf <= xe], np.zeros((xf <= xe).sum()), np.full((xf <= xe).sum(), zl), k)
    OV.plot(q[:, 0], q[:, 1], color=col, lw=1.0, ls=edge_ls, zorder=6.05, solid_capstyle='butt')
    vv = np.linspace(0, 1, 80); q = P3(np.zeros_like(vv), vv, np.full_like(vv, zl), k)
    OV.plot(q[:, 0], q[:, 1], color=col, lw=0.8, ls=edge_ls, zorder=5.9, solid_capstyle='butt')
    return P3(xe, 0.0, zl, k)[0]
for k in (0, 1):
    zero_plane(k, ZL[k], NAVY if k == 0 else RED, 0.17)
    if k == 1: zero_plane(1, ZL[0], NAVY, 0, edge_ls=(0, (2.2, 1.4)), fill=False)
    curtain(k); ar_tube(k); clouds(k)
exec(open(f'{CODE}/figS01_mechanism_labels.py').read())
OUT = f'{FIG}/FigS01_mechanism'
fig.savefig(OUT + '.png', dpi=300); print('saved')
if 'pdf' in sys.argv: fig.savefig(OUT + '.pdf'); print('pdf')
