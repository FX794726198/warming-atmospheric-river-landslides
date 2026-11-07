import os as _os
REPO = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
DATA = _os.path.join(REPO, 'data'); FIG = _os.environ.get('FIGOUT', _os.path.join(REPO, 'fig')); CODE = _os.path.join(REPO, 'fig_code')
def zoom_base(cx_, cy_, r_, col, sky):
    a_ = fig.add_axes([(cx_ - r_) / W, (cy_ - r_) / H, 2 * r_ / W, 2 * r_ / H], zorder=6)
    a_.set_xlim(-1, 1); a_.set_ylim(-1, 1); a_.set_axis_off(); a_.set_aspect('equal'); a_.patch.set_alpha(0)
    clip = Circle((0, 0), 1, transform=a_.transData)
    a_.add_patch(Circle((0, 0), 1, facecolor=sky, edgecolor='none', zorder=0))
    a_.add_patch(Circle((0, 0), 1, facecolor='none', edgecolor=col, lw=1.1, zorder=30))
    return a_, clip
qq = np.linspace(-1.3, 1.3, 240)
def surf(x): return -0.28 - 0.42 * x + 0.04 * np.sin(2.4 * x + 0.4)
SLOPE = np.degrees(np.arctan(-0.42)); TS = 0.30; FSI = 6.4
def band(a_, clip, top, bot, fc, z, ec='none', lw=0, x=None):
    x = qq if x is None else x
    a_.add_patch(Polygon(np.r_[np.c_[x, top], np.c_[x[::-1], bot[::-1]]], closed=True, facecolor=fc, edgecolor=ec, lw=lw, clip_path=clip, zorder=z))
def ground(a_, clip, soil_col, peb='#9C8463'):
    ys = surf(qq)
    a_.add_patch(Polygon(np.r_[np.c_[qq, ys - TS], [[1.3, -1.5], [-1.3, -1.5]]], closed=True, facecolor=ROCK, edgecolor='none', clip_path=clip, zorder=1))
    band(a_, clip, ys, ys - TS, soil_col, 1.2)
    offs = (TS, 0.52, 0.74, 0.96, 1.18, 1.40)
    for o in offs[1:]: a_.plot(qq, ys - o, color='#B5AA99', lw=0.45, clip_path=clip, zorder=1.3)
    for k in range(5):
        for x0 in np.arange(-1.0 + 0.10 * (k % 2), 1.0, 0.20):
            a_.plot([x0, x0 + 0.02], [surf(x0) - offs[k], surf(x0) - offs[k + 1]], color='#B5AA99', lw=0.4, clip_path=clip, zorder=1.3)
    px = rng.uniform(-1, 1, 80); pz = surf(px) - rng.uniform(0.05, TS - 0.04, 80)
    a_.scatter(px, pz, s=rng.uniform(0.5, 2.5, 80), c=peb, linewidths=0, clip_path=clip, zorder=1.4)
    a_.plot(qq, ys - TS, color='#9C8B74', lw=0.5, clip_path=clip, zorder=1.5)
def rtext(a_, x, y, t, col, fs=FSI, rot=SLOPE, **k):
    a_.text(x, y, t, fontsize=fs, color=col, rotation=rot, rotation_mode='anchor', ha='center', va='center', zorder=20, path_effects=HALO, **k)
def conifer(a_, clip, x0, base, h=0.50, w=0.30, snow=False, tilt=0.0, z=3):
    c_, s_ = np.cos(np.radians(tilt)), np.sin(np.radians(tilt)); R = np.array([[c_, -s_], [s_, c_]])
    tr = lambda pts: (np.array(pts) - [x0, base]) @ R.T + [x0, base]
    a_.add_patch(Polygon(tr([(x0 - 0.02, base - 0.05), (x0 + 0.02, base - 0.05), (x0 + 0.02, base + 0.14), (x0 - 0.02, base + 0.14)]), facecolor='#6B4F3A', edgecolor='none', clip_path=clip, zorder=z))
    for yb, wk in ((0.10, 1.0), (0.30, 0.76), (0.48, 0.52)):
        y0 = base + yb * h / 0.5; apex = y0 + 0.42 * h
        a_.add_patch(Polygon(tr([(x0 - w * wk / 2, y0), (x0 + w * wk / 2, y0), (x0, apex)]), facecolor='#2F5D46', edgecolor='none', clip_path=clip, zorder=z + 0.1))
        if snow:
            f_ = 0.42; a_.add_patch(Polygon(tr([(x0 - w * wk / 2 * f_, apex - 0.42 * h * f_), (x0 + w * wk / 2 * f_, apex - 0.42 * h * f_), (x0, apex)]), facecolor='white', edgecolor='none', clip_path=clip, zorder=z + 0.2))
            a_.add_patch(Polygon(tr([(x0 - w * wk / 2, y0), (x0 - w * wk * 0.2, y0 + 0.05), (x0 + w * wk * 0.2, y0 + 0.035), (x0 + w * wk / 2, y0)]), facecolor='white', edgecolor='none', clip_path=clip, zorder=z + 0.2))
def flake(a_, clip, x, y, r, col='white', lw=0.65, z=5):
    for ang in (0, 60, 120):
        t = np.radians(ang + 15)
        for sg in (1, -1):
            ex, ey = x + sg * r * np.cos(t), y + sg * r * np.sin(t); a_.plot([x, ex], [y, ey], color=col, lw=lw, solid_capstyle='round', clip_path=clip, zorder=z)
            bx, by = x + sg * 0.6 * r * np.cos(t), y + sg * 0.6 * r * np.sin(t)
            for b in (0.6, -0.6):
                tb = t + (0 if sg > 0 else np.pi) + b; a_.plot([bx, bx + 0.32 * r * np.cos(tb)], [by, by + 0.32 * r * np.sin(tb)], color=col, lw=lw * 0.8, clip_path=clip, zorder=z)
rz = RZ
# --- today: snow stays on the slope
a_, clip = zoom_base(*cA, rz, NAVY, '#C6D6EA'); ground(a_, clip, SOIL)
ys = surf(qq); SN = 0.32
band(a_, clip, ys + SN, ys, 'white', 2)
band(a_, clip, ys + 0.08, ys, '#DCE6F2', 2.1)
for o in (0.08, 0.16): a_.plot(qq, ys + o, color='#B4C6DC', lw=0.45, clip_path=clip, zorder=2.2)
a_.plot(qq, ys + SN, color='#3F5F8C', lw=1.0, clip_path=clip, zorder=2.3)
a_.plot(qq, ys, color='#7A6A55', lw=0.5, clip_path=clip, zorder=2.3)
TREES_A = ((-0.55, 0.36), (0.45, 0.40), (0.78, 0.30))
for x0, hh in TREES_A: conifer(a_, clip, x0, surf(x0) + SN - 0.04, h=hh, w=0.28, snow=True, z=3)
fx, fy = [], []
for _ in range(6000):
    if len(fx) >= 14: break
    x_, y_ = rng.uniform(-0.92, 0.92), rng.uniform(-0.2, 0.95)
    if x_ ** 2 + y_ ** 2 < 0.84 and y_ > surf(x_) + SN + 0.10 and all(np.hypot(x_ - a, y_ - b) > 0.18 for a, b in zip(fx, fy)) \
       and not (abs(x_ - 0.15) < 0.58 and y_ > 0.60) and not any(abs(x_ - t0) < 0.19 and y_ < surf(t0) + SN + 1.45 * h0 for t0, h0 in TREES_A):
        fx.append(x_); fy.append(y_)
for x_, y_ in zip(fx, fy): flake(a_, clip, x_, y_, rng.uniform(0.055, 0.085), lw=0.7)
rtext(a_, 0.02, surf(0.02) + 0.16, 'Snowpack', NAVY)
rtext(a_, -0.12, surf(-0.12) - 0.15, 'Soil', '#4A3826')
rtext(a_, -0.35, surf(-0.35) - 0.55, 'Bedrock', '#5E564C')
a_.text(0.15, 0.74, 'Below 0 °C', fontsize=FSI, color=NAVY, ha='center', va='center', zorder=20, path_effects=HALO)
# --- warmer: rain soaks in and the slope fails
a_, clip = zoom_base(*cB, rz, BLUE, '#D3DCE6'); ground(a_, clip, '#B99E7B', peb='#8E7558')
ys = surf(qq); WT = 0.13
band(a_, clip, ys - WT, ys - TS, '#9DBEDF', 1.25)
a_.plot(qq, ys - WT, color='#2F64A8', lw=0.6, ls=(0, (2.0, 1.4)), clip_path=clip, zorder=1.6)
qa, qb, dsh = -0.25, 0.58, 0.10
mq = np.linspace(qa, qb, 90); slip = surf(mq) - (TS + 0.03) * np.sin(np.pi * (mq - qa) / (qb - qa)) ** 0.55
band(a_, clip, surf(mq), slip, '#7E6C58', 1.7, x=mq)
dv = np.array([1.0, -0.42]) / np.hypot(1, 0.42) * dsh
top_ = np.c_[mq, surf(mq)] + dv; bot_ = np.c_[mq, slip] + dv
a_.add_patch(Polygon(np.r_[top_, bot_[::-1]], closed=True, facecolor='#B99E7B', edgecolor='none', clip_path=clip, zorder=2))
wt_ = np.c_[mq, np.maximum(surf(mq) - WT, slip)] + dv
a_.add_patch(Polygon(np.r_[wt_, bot_[::-1]], closed=True, facecolor='#9DBEDF', edgecolor='none', clip_path=clip, zorder=2.05))
a_.add_patch(Polygon(np.r_[top_, bot_[::-1]], closed=True, facecolor='none', edgecolor=RED, lw=0.9, clip_path=clip, zorder=2.1))
for x_ in (-0.15, 0.08): a_.plot([x_ + dv[0], x_ + dv[0] + 0.03], [surf(x_) + dv[1], surf(x_) + dv[1] - 0.10], color='#6B4F3A', lw=0.6, clip_path=clip, zorder=2.2)
sl_ = np.linspace(qa - 0.02, qb + 0.02, 80); slp = surf(sl_) - (TS + 0.03) * np.sin(np.pi * np.clip((sl_ - qa) / (qb - qa), 0, 1)) ** 0.55
a_.plot(sl_, slp, color=RED, lw=1.1, ls=(0, (2.4, 1.2)), clip_path=clip, zorder=2.3)
xw = 0.80; a_.add_patch(Polygon([(xw - 0.045, surf(xw) - WT + 0.075), (xw + 0.045, surf(xw) - WT + 0.075), (xw, surf(xw) - WT + 0.005)], facecolor='none', edgecolor='#2F64A8', lw=0.6, clip_path=clip, zorder=1.7))
for x_ in (-0.93, 0.70, 0.88): a_.add_patch(FancyArrowPatch((x_, surf(x_) + 0.02), (x_ + 0.015, surf(x_) - 0.12), arrowstyle='-|>', color='#2F64A8', lw=0.9, mutation_scale=5.5, clip_path=clip, zorder=2.4))
conifer(a_, clip, 0.36 + dv[0], surf(0.36) + dv[1], h=0.36, w=0.26, tilt=-24, z=3)
conifer(a_, clip, -0.48, surf(-0.48), h=0.36, w=0.26, z=3)
for x_ in np.linspace(-0.95, 0.95, 30):
    y0 = rng.uniform(surf(x_) + 0.16, 1.0); L_ = rng.uniform(0.12, 0.22)
    if abs(x_ - 0.15) < 0.58 and y0 > 0.56: continue
    if abs(x_ + 0.48) < 0.14 and y0 - L_ < surf(-0.48) + 0.55: continue
    if abs(x_ - 0.40) < 0.16 and y0 - L_ < surf(0.40) + 0.50: continue
    a_.plot([x_, x_ - 0.05 * L_ / 0.2], [y0, y0 - L_], color='#3F72B8', lw=0.75, clip_path=clip, zorder=4)
for x_ in (-0.80, -0.30, 0.70, 0.88): a_.plot([x_ - 0.035, x_, x_ + 0.035], [surf(x_ - 0.035) + 0.045, surf(x_) + 0.005, surf(x_ + 0.035) + 0.045], color='#3F72B8', lw=0.6, clip_path=clip, zorder=4)
a_.add_patch(FancyArrowPatch((qa + 0.16 + dv[0], surf(qa + 0.16) - 0.09 + dv[1]), (qb - 0.10 + dv[0], surf(qb - 0.10) - 0.09 + dv[1]), arrowstyle='-|>', color=RED, lw=1.6, mutation_scale=9, clip_path=clip, zorder=6))
rtext(a_, -0.52, surf(-0.52) - 0.15, 'Wet soil', '#3A2A1A')
rtext(a_, -0.22, surf(-0.22) - TS - 0.18, 'Slip surface', RED)
a_.text(0.15, 0.74, 'Above 0 °C', fontsize=FSI, color=RED, ha='center', va='center', zorder=20, path_effects=HALO)
