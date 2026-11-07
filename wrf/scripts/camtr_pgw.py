"""Make a CAMtr_volume_mixing_ratio with every year set to the values of one year of the given scenario file (PGW GHG).
usage: camtr_pgw.py SRC(CAMtr_volume_mixing_ratio.SSP585) DST [YEAR=2085]
v2 (2026-11-04): keeps the fixed-width layout of the source (I4 + value columns copied verbatim from the YEAR line);
v1 rewrote lines space-separated, which WRF cannot read ("Not normal ending of CAMtr_volume_mixing_ratio file")."""
import sys
src, dst = sys.argv[1:3]; Y = sys.argv[3] if len(sys.argv) > 3 else "2085"
L = open(src).read().splitlines()
ref = [l for l in L if l[:4] == Y][0]; vals = ref[4:]
out = [(l[:4] + vals if l[:4].strip().isdigit() and len(l) > 4 else l) for l in L]
open(dst, "w").write("\n".join(out) + "\n"); print("GHG set to", Y, vals.split())
