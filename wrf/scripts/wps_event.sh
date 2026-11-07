#!/bin/bash
# usage: wps_event.sh EVENT -> waits for the event's ERA5 GRIB files, runs ungrib + metgrid -> wps/EVENT/ctl,
# then (once cmip6/delta_ssp585_monthly.npz exists) writes the PGW copy wps/EVENT/pgw585.
set -e
EV=$1; B=/scratch/$USER/ar_pgw; D=$B/wps/$EV; cd $D
NEED=$(python3 -c "
import sys;sys.path.insert(0,'$B/scripts');from get_era5 import JOBS
print(' '.join(sorted(set(f'$B/era5/{e}_{y}{m:02d}_{k}.grib' for e,y,m,d,k in JOBS if e=='$EV'))))" 2>/dev/null)
echo "waiting for: $NEED"
have_all() { for f in $NEED; do [ -s $f ] || return 1; done; return 0; }
until have_all; do sleep 30; done
echo "ERA5 present $(date)"
source "${WPS_ENV:-/scratch/$USER/build/wps_env.sh}" >/dev/null 2>&1
rm -f GRIBFILE.* FILE:* met_em.d0*; mkdir -p ctl
./link_grib.csh $NEED
./ungrib.exe > ungrib.log 2>&1; tail -1 ungrib.log
./metgrid.exe > metgrid.log 2>&1; tail -1 metgrid.log
grep -q "Successful completion" metgrid.log || { echo "metgrid failed"; exit 1; }
mv met_em.d0*.nc ctl/; rm -f FILE:* GRIBFILE.*
echo "ctl met_em: $(ls ctl | wc -l) files $(date)"
until [ -s $B/cmip6/delta_ssp585_monthly.npz ]; do sleep 30; done
python3 $B/scripts/pgw_metem.py $B/cmip6/delta_ssp585_monthly.npz $EV ctl pgw585 2>&1 | grep -v -i warn
echo "WPS $EV done $(date)"
