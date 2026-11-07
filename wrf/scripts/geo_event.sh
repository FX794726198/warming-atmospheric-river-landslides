#!/bin/bash
# usage: geo_event.sh EVENT  -> geo_em.d01/d02 in /scratch/$USER/ar_pgw/wps/EVENT (d01 first, then nest centred on target)
set -e
EV=$1; B=/scratch/$USER/ar_pgw/wps; W=${WPS_DIR:-/scratch/$USER/build/WPS-4.6.0}; D=$B/$EV; mkdir -p $D/geogrid $D/metgrid; cd $D
source "${WPS_ENV:-/scratch/$USER/build/wps_env.sh}" >/dev/null 2>&1
ln -sf $W/geogrid.exe $W/ungrib.exe $W/metgrid.exe $W/link_grib.csh .
ln -sf $W/geogrid/GEOGRID.TBL.ARW geogrid/GEOGRID.TBL; ln -sf $W/metgrid/METGRID.TBL.ARW metgrid/METGRID.TBL
ln -sf $W/ungrib/Variable_Tables/Vtable.ECMWF Vtable
PY=python3
$PY /scratch/$USER/ar_pgw/scripts/make_wps.py $EV namelist.wps d01only
./geogrid.exe > geogrid_d01.log 2>&1; tail -1 geogrid_d01.log
$PY /scratch/$USER/ar_pgw/scripts/make_wps.py $EV namelist.wps
./geogrid.exe > geogrid.log 2>&1; tail -1 geogrid.log
