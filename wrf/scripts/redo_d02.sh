#!/bin/bash
# usage: redo_d02.sh EVENT  -> new 301x301 d02 (3 km, 900 km) for an event whose v1 WPS used a 151x151 nest.
# Works in wps/EVENT/d02new (geogrid, ungrib + metgrid for the initial time only), waits for the v1 pipeline
# (wps_event.sh) to finish, moves the v1 d02 files to wps/EVENT/old_d02_151/, puts the new d02 met_em into ctl/ and
# its PGW copy into pgw585/, and touches wps/EVENT/READY. d01 is unchanged.
set -e
EV=$1; B=/scratch/$USER/ar_pgw; W=${WPS_DIR:-/scratch/$USER/build/WPS-4.6.0}; D=$B/wps/$EV; N=$D/d02new; PY=python3
mkdir -p $N/geogrid $N/metgrid; cd $N
source "${WPS_ENV:-/scratch/$USER/build/wps_env.sh}" >/dev/null 2>&1
ln -sf $W/geogrid.exe $W/ungrib.exe $W/metgrid.exe $W/link_grib.csh .
ln -sf $W/geogrid/GEOGRID.TBL.ARW geogrid/GEOGRID.TBL; ln -sf $W/metgrid/METGRID.TBL.ARW metgrid/METGRID.TBL
ln -sf $W/ungrib/Variable_Tables/Vtable.ECMWF Vtable
cp $D/geo_em.d01.nc .
$PY $B/scripts/make_wps.py $EV namelist.wps
./geogrid.exe > geogrid.log 2>&1; tail -1 geogrid.log; grep -q "Successful completion" geogrid.log
cp namelist.wps namelist.wps.full
NEED=$($PY -c "
import sys;sys.path.insert(0,\"$B/scripts\");from get_era5 import JOBS
print(\" \".join(sorted(set(f\"$B/era5/{e}_{y}{m:02d}_{k}.grib\" for e,y,m,d,k in JOBS if e==\"$EV\"))))" 2>/dev/null)
have_all() { for f in $NEED; do [ -s $f ] || return 1; done; return 0; }
until have_all; do sleep 30; done
S=$(grep -o "start_date = \"[^\"]*\"" namelist.wps | head -1 | cut -d\" -f2)
sed -i "s/^ end_date   = .*/ end_date   = \"$S\",\"$S\",/" namelist.wps        # ungrib + metgrid: initial time only
rm -f GRIBFILE.* FILE:* met_em.d0*
./link_grib.csh $NEED
./ungrib.exe > ungrib.log 2>&1; tail -1 ungrib.log
./metgrid.exe > metgrid.log 2>&1; tail -1 metgrid.log; grep -q "Successful completion" metgrid.log
rm -f FILE:* GRIBFILE.*; cp namelist.wps namelist.wps.init; cp namelist.wps.full namelist.wps
echo "new d02 met_em: $(ls met_em.d02.*) $(date)"
until grep -q "WPS $EV done" $B/logs/wps_$EV.log 2>/dev/null; do sleep 30; done
O=$D/old_d02_151; mkdir -p $O/ctl $O/pgw585
mv $D/ctl/met_em.d02.* $O/ctl/ 2>/dev/null || true; mv $D/pgw585/met_em.d02.* $O/pgw585/ 2>/dev/null || true
mv $D/geo_em.d02.nc $D/namelist.wps $O/; cp -p $D/pgw585/pgw_summary.json $O/pgw_summary_v1.json
cp geo_em.d02.nc namelist.wps $D/; mv met_em.d02.* $D/ctl/
rm -rf tmp_ctl tmp_pgw; mkdir -p tmp_ctl; ln -s $D/ctl/met_em.d02.* tmp_ctl/
$PY $B/scripts/pgw_metem.py $B/cmip6/delta_ssp585_monthly.npz $EV tmp_ctl tmp_pgw 2>&1 | grep -v -i warn
mv tmp_pgw/met_em.d02.* $D/pgw585/; mv tmp_pgw/pgw_summary.json $D/pgw585/pgw_summary_d02.json
echo "ctl: $(ls $D/ctl | grep -c d01) d01 + $(ls $D/ctl | grep -c d02) d02; pgw585: $(ls $D/pgw585 | grep -c d01) d01 + $(ls $D/pgw585 | grep -c d02) d02"
touch $D/READY; echo "READY $EV $(date)"
