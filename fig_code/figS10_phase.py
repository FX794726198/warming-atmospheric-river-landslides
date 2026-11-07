"""SI figure: phase of cold-season AR extreme precipitation on mountains, 1941-2024 (ERA5 0.25 deg).
a Snow share of the AR extreme precipitation amount, Northern and Southern Hemisphere (annual and 11-year means; least-squares trends).
b Same per region (11-year means). c Rain-snow transition elevation per region where it is defined (annual and 11-year means)."""
import os as _os
REPO = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
DATA = _os.path.join(REPO, 'data'); FIG = _os.environ.get('FIGOUT', _os.path.join(REPO, 'fig')); CODE = _os.path.join(REPO, 'fig_code')
import sys; sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from arcommon import *
from scipy import stats
S = pd.read_csv(f'{DATA}/derived/n7/regional_series.csv'); S = S[S.year >= 1941]
COL = {'SCA': ('Scandinavia', '#0072B2'), 'ALP': ('Alps', '#009E73'), 'CAU': ('Caucasus–Türkiye', '#CC79A7'), 'BCA': ('BC–Alaska coast', '#8C6BB1'),
       'PNW': ('US Northwest', '#E69F00'), 'NEA': ('NE North America', '#D55E00'), 'HKH': ('Hindu Kush–Himalaya', '#7F7F7F'),
       'SAN': ('Southern Andes', '#56B4E9'), 'NH': ('All land 25–72° N', '#000000'), 'SH': ('All land 25–56° S', '#555555')}
W, H = 183, 70
fig = plt.figure(figsize=(W * MM, H * MM))
def ax_mm(l, b, w, h): return fig.add_axes([l / W, b / H, w / W, h / H])
fp = lambda p: f'{p:.3f}' if p < 0.01 else f'{p:.2f}'
roll = lambda y: pd.Series(y).rolling(11, center=True, min_periods=7).mean().values
B0, AH = 10, 45
# a
axa = ax_mm(11, B0, 46, AH)
for k in ('NH', 'SH'):
    q = S[S.region == k].sort_values('year'); y = q.snowfrac_amount.values * 100
    axa.plot(q.year, roll(y), color=COL[k][1], lw=1.4)
    for a0, ls in ((1941, '-'), (1979, (0, (2, 1.2)))):
        z = q[q.year >= a0]; r = stats.linregress(z.year, z.snowfrac_amount * 100)
        axa.plot(z.year, r.intercept + r.slope * z.year, color='#B2182B' if k == 'NH' else '#E08080', lw=0.8, ls=ls)
    r1 = stats.linregress(q.year, q.snowfrac_amount * 100); z = q[q.year >= 1979]; r2 = stats.linregress(z.year, z.snowfrac_amount * 100)
    txt = f'{COL[k][0]}\n1941–2024: {r1.slope*10:+.2f}% per decade (P = {fp(r1.pvalue)})\n1979–2024: {r2.slope*10:+.2f}% per decade (P = {fp(r2.pvalue)})'
    axa.text(1942, 59 if k == 'NH' else 2, txt.replace('+', '+').replace('-0', '−0'), fontsize=5.0, color=COL[k][1], va='top' if k == 'NH' else 'bottom', linespacing=1.05)
axa.set_xlim(1940, 2025); axa.set_ylim(0, 60); axa.set_xticks([1950, 1970, 1990, 2010])
axa.set_ylabel('Snow share of AR extreme\nprecipitation (%)', linespacing=1.0); axa.spines[['top', 'right']].set_visible(False)
# b
axb = ax_mm(72, B0, 46, AH)
for k in ['SCA', 'BCA', 'HKH', 'PNW', 'CAU', 'SAN', 'ALP', 'NEA']:
    q = S[S.region == k].sort_values('year'); axb.plot(q.year, roll(q.snowfrac_amount.values * 100), color=COL[k][1], lw=1.2)
axb.set_xlim(1940, 2025); axb.set_ylim(0, 80); axb.set_xticks([1950, 1970, 1990, 2010])
axb.set_ylabel('Snow share (%)'); axb.spines[['top', 'right']].set_visible(False)
# c
axc = ax_mm(134, B0, 46, AH)
for k in ['ALP', 'CAU', 'HKH', 'SAN', 'PNW', 'SCA', 'NH', 'SH']:
    q = S[S.region == k].sort_values('year'); axc.plot(q.year, roll(q.snowline_m.values), color=COL[k][1], lw=1.2)
axc.set_xlim(1940, 2025); axc.set_xticks([1950, 1970, 1990, 2010]); axc.set_ylim(0, 2600)
axc.set_ylabel('Rain–snow transition (m)'); axc.spines[['top', 'right']].set_visible(False)
from matplotlib.lines import Line2D
fig.legend(handles=[Line2D([], [], color=COL[k][1], lw=1.4, label=COL[k][0]) for k in ['SCA', 'BCA', 'HKH', 'PNW', 'CAU', 'SAN', 'ALP', 'NEA', 'NH', 'SH']],
           loc='upper center', bbox_to_anchor=(0.55, 1.0), ncol=5, frameon=False, fontsize=5.6, handlelength=1.2, columnspacing=1.0, labelspacing=0.2, handletextpad=0.4)
for s_, x in (('a', 0.3), ('b', 61.0), ('c', 123.0)): fig.text(x / W, (B0 + AH + 3.5) / H, s_, fontsize=9, fontweight='bold', va='top')
if __name__ == '__main__':
    bump_fonts(fig); print(overlap_check(fig))
    fig.savefig(f'{FIG}/FigS10_phase.png', dpi=300); print('saved')
