#!/usr/bin/env bash
# Redraw every figure of the paper from data/ into fig/ (or $FIGOUT). Run from the repository root.
set -euo pipefail
cd "$(dirname "$0")/fig_code"
export MPLBACKEND=Agg
mkdir -p "${FIGOUT:-../fig}/parts"
for s in fig1 fig2 fig3 fig4 fig5 figS10_phase figS11_xsec panels_obs1 panels_obs2 panels_obs4 panels_supp figS12_wrf_full; do
  echo "== $s"; python3 "$s.py"
done
python3 figS01_mechanism.py pdf
python3 compose_si.py
