#!/bin/bash
# GHCN-Daily by_year files for the 5 AR events + station list (NOAA NCEI), for the control-run station check
cd /scratch/$USER/ar_pgw/eval/ghcn
for f in daily/ghcnd-stations.txt daily/by_year/1996.csv.gz daily/by_year/2011.csv.gz daily/by_year/2014.csv.gz daily/by_year/2021.csv.gz daily/by_year/2024.csv.gz; do
  b=$(basename $f); curl -sS --retry 3 --max-time 1800 -o $b.part https://www.ncei.noaa.gov/pub/data/ghcn/$f && mv $b.part $b && echo "$(date -u +%TZ) $b $(stat -c %s $b)"
done
echo "all done $(date -u +%TZ)"
