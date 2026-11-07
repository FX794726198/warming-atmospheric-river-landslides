"""Supplementary Figs. S2-S9 and S12: crop and re-letter panels drawn by panels_*.py and figS12_wrf_full.py (output in fig/parts/)."""
import os, numpy as np
from PIL import Image, ImageDraw, ImageFont
from matplotlib import font_manager
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.environ.get('FIGOUT', os.path.join(REPO, 'fig')); P = os.path.join(FIG, 'parts')
_lib = '/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf'
FB = ImageFont.truetype(_lib if os.path.exists(_lib) else font_manager.findfont(font_manager.FontProperties(family='sans-serif', weight='bold')), 37)
def reletter(img, boxes):
    d = ImageDraw.Draw(img)
    for (x0, y0, x1, y1), new, base in boxes:
        d.rectangle((x0 - 1, y0 - 1, x1 + 1, y1 + 1), fill='white'); d.text((x0, base), new, font=FB, fill='black', anchor='ls')
def save(img, name): img.save(os.path.join(FIG, name), dpi=(300, 300)); print('saved', name)
# S2: inventory maps and event-size distribution
_reg = '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf'
FR = ImageFont.truetype(_reg if os.path.exists(_reg) else font_manager.findfont(font_manager.FontProperties(family='sans-serif')), 27)
im = Image.open(f'{P}/Fig1.png').convert('RGB'); A = np.asarray(im).astype(int)
def _darkest(x0, x1): sub = A[1337:1369, x0:x1].reshape(-1, 3); return tuple(int(v) for v in sub[sub.sum(1).argmin()])
c1 = _darkest(112, 230); o = 739
mid = im.crop((0, 739, 2161, 1369)).copy()
reletter(mid, [((5, 747 - o, 22, 766 - o), 'a', 766 - o), ((926, 740 - o, 944, 766 - o), 'b', 766 - o), ((1584, 747 - o, 1600, 766 - o), 'c', 766 - o)])
d = ImageDraw.Draw(mid)
d.rectangle((45, 1335 - o, 360, 1369 - o), fill='white'); d.text((51, 1360 - o), 'Landslides per event', font=FR, fill=c1, anchor='ls')
d.rectangle((1150, 1335 - o, 1500, 1369 - o), fill='white'); d.text((1165, 1360 - o), 'Colours as in Fig. 1a', font=FR, fill=(51, 51, 51), anchor='ls')
g = im.crop((650, 1428, 1270, 2043)).copy(); ImageDraw.Draw(g).rectangle((0, 300, 12, 615), fill='white'); reletter(g, [((31, 8, 48, 35), 'd', 35)])
c = Image.new('RGB', (2161, mid.height + 20 + g.height), 'white'); c.paste(mid, (0, 0)); c.paste(g, ((2161 - g.width) // 2, mid.height + 20))
save(c, 'FigS02_national_inventories.png')
# S4: season
im2 = Image.open(f'{P}/Fig2.png').convert('RGB')
top = im2.crop((0, 0, 2161, 552)); bot = im2.crop((0, 552, 1545, 1464))
c2 = Image.new('RGB', (2161, top.height + bot.height), 'white'); c2.paste(top, (0, 0)); c2.paste(bot, ((2161 - 1545) // 2, top.height))
save(c2, 'FigS04_season.png')
# S3, S5-S9
for src, name in [('FigS1', 'FigS03_cluster_rules'), ('FigS5', 'FigS05_extreme_share'), ('FigS2', 'FigS06_rain_matched'),
                  ('FigS3', 'FigS07_yield'), ('FigS4', 'FigS08_land_surface'), ('Fig4', 'FigS09_subfreezing')]:
    save(Image.open(f'{P}/{src}.png').convert('RGB'), name + '.png')
# S12: WRF maps, bottom rows of the full WRF figure, panels a, b and d (relettered c)
im = Image.open(f'{P}/FigS12_wrf_full.png').convert('RGB'); o = 612
b = im.crop((0, o, 2161, 1535)).copy(); d = ImageDraw.Draw(b)
for (x0, y0, x1, y1), new in [((4, 631, 15, 657), 'a'), ((734, 631, 752, 665), 'b'), ((1160, 631, 1177, 657), 'c'), ((1662, 631, 1666, 657), 'd')]:
    d.rectangle((x0 - 2, y0 - o - 2, x1 + 2, y1 - o + 2), fill='white'); d.text((x0, 657 - o), new, font=FB, fill='black', anchor='ls')
W, H = b.size; new = Image.new('RGB', (1155 + W - 1655, H), 'white'); new.paste(b.crop((0, 0, 1155, H)), (0, 0)); new.paste(b.crop((1655, 0, W, H)), (1155, 0))
a = np.array(new.convert('L'))[0:70, 1155:1230]; ys, xs = np.where(a < 100); dd = ImageDraw.Draw(new)
dd.rectangle((xs.min() + 1155 - 2, ys.min() - 2, xs.max() + 1155 + 2, ys.max() + 2), fill='white')
dd.text((xs.min() + 1155, ys.max() + 1), 'c', font=FB, fill='black', anchor='ls')
save(new, 'FigS12_wrf_maps.png')
