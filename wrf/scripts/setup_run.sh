#!/bin/bash
# usage: setup_run.sh EVENT ARM(ctl|pgw585) [TAG]  -> runs/EVENT_ARM[_TAG] ready for real.exe + wrf.exe
set -e
EV=$1; A=$2; TAG=${3:+_$3}; B=/scratch/$USER/ar_pgw; R=$B/runs/${EV}_${A}${TAG}; mkdir -p $R; cd $R
W=$(ls -d /apps/arch/2025/software/linux-rhel8-cascadelake/gcc-13.3.0/wrf-4.6.1-*)
for f in $W/run/*; do b=$(basename $f); case $b in namelist*|*.exe|CAMtr_volume_mixing_ratio) ;; *) ln -sf $f .;; esac; done
ln -sf $W/main/real.exe $W/main/wrf.exe .
if [ "$A" = ctl ]; then ln -sf $W/run/CAMtr_volume_mixing_ratio.SSP245 CAMtr_volume_mixing_ratio
else python3 $B/scripts/camtr_pgw.py $W/run/CAMtr_volume_mixing_ratio.SSP585 CAMtr_volume_mixing_ratio 2085; fi
rm -f met_em.d0*; ln -sf $B/wps/$EV/$A/met_em.d0* .
cp $B/templates/iofields_d01.txt $B/templates/iofields_d02.txt .
eval $(python3 - <<PY
import sys,re; sys.path.insert(0,"$B/scripts"); from events import EVENTS, NEST
E=EVENTS["$EV"]; s=E["start"]; e=E["end"]
t=open("$B/wps/$EV/namelist.wps").read()
ip=re.search(r"i_parent_start = 1, (\d+)",t).group(1); jp=re.search(r"j_parent_start = 1, (\d+)",t).group(1)
print(f"SY={s[:4]} SM={s[5:7]} SD={s[8:10]} SH={s[11:13]} EY={e[:4]} EM={e[5:7]} ED={e[8:10]} EH={e[11:13]} NX1={E['nx']} NY1={E['ny']} NX2={NEST['nx']} NY2={NEST['ny']} IPS={ip} JPS={jp}")
PY
)
DT=${DT:-45}
sed -e "s/@SY@/$SY/g;s/@SM@/$SM/g;s/@SD@/$SD/g;s/@SH@/$SH/g;s/@EY@/$EY/g;s/@EM@/$EM/g;s/@ED@/$ED/g;s/@EH@/$EH/g" \
    -e "s/@NX1@/$NX1/g;s/@NY1@/$NY1/g;s/@NX2@/$NX2/g;s/@NY2@/$NY2/g;s/@IPS@/$IPS/g;s/@JPS@/$JPS/g;s/@DT@/$DT/g" $B/templates/namelist.input.tmpl > namelist.base
cp namelist.base namelist.input
echo "run dir $R ready: $SY-$SM-$SD $SH to $EY-$EM-$ED $EH, d02 at i=$IPS j=$JPS, dt=$DT"
