#!/bin/bash
#SBATCH --cpus-per-task=1
#SBATCH -o /scratch/$USER/ar_pgw/logs/wrf_%x_%j.log
# v2. usage: sbatch -J ar_EVENT_ARM -A ACCOUNT -p PARTITION --ntasks=N --mem-per-cpu=M -t HH:MM:SS run_wrf.sh EVENT ARM [MODE]
#   MODE main (default): waits for wps/EVENT/READY, sets up the run dir if needed, real.exe if no wrfinput, then wrf.exe from the
#        newest complete restart. If a "seg" job holds the run dir, it is cancelled and the run continues from its restart.
#   MODE seg: opportunistic segment (e.g. participation-cpu, 1 node, 4 h): exits at once if another job holds the run dir.
#   MODE real: real.exe only.
# wrf.exe is stopped 4 min before the job time limit so the job can clean up. If wrf.exe fails early (e.g. CFL), it is retried
# once from the newest restart with time_step 36 s (d02 12 s). Exit 0 only when the run is complete (DONE), so afterok works.
EV=$1; A=$2; MODE=${3:-main}; B=/scratch/$USER/ar_pgw; R=$B/runs/${EV}_${A}; W=$B/wps/$EV; PY=python3
source "${WPS_ENV:-/scratch/$USER/build/wps_env.sh}" >/dev/null 2>&1
ulimit -s unlimited
echo "job $SLURM_JOB_ID $EV $A mode $MODE ntasks $SLURM_NTASKS partition $SLURM_JOB_PARTITION nodes $SLURM_JOB_NODELIST start $(date)"
END=$(scontrol show job $SLURM_JOB_ID | grep -o "EndTime=[^ ]*" | cut -d= -f2); TEND=$(date -d "$END" +%s)
[ -f $R/DONE ] && { echo "already done"; exit 0; }
for i in $(seq 1 480); do [ -f $W/READY ] && break; sleep 30; done
[ -f $W/READY ] || { echo "inputs not ready"; exit 1; }
mkdir -p $R; L=$R/.lock
alive() { [ -n "$1" ] && squeue -h -j "$1" -o %t 2>/dev/null | grep -q .; }
H=$(cat $L 2>/dev/null)
if [ -n "$H" ] && [ "$H" != "$SLURM_JOB_ID" ] && alive $H; then
  if [ "$MODE" = main ]; then echo "cancelling job $H that holds the run"; scancel $H; while alive $H; do sleep 10; done; sleep 30
  else echo "run held by job $H, exiting"; exit 0; fi
fi
if [ "$MODE" != main ]; then   # v3 guard: a main job (ar_EV_ARM) waits 30 s after cancelling a holder before it locks
  squeue -h -u $USER -t RUNNING,COMPLETING -n ar_${EV}_${A} -o %i 2>/dev/null | grep -q . && { echo "main job of ${EV}_${A} is running, exiting"; exit 0; }
fi
echo $SLURM_JOB_ID > $L
if [ "$MODE" != main ]; then sleep 45; [ "$(cat $L 2>/dev/null)" = "$SLURM_JOB_ID" ] || { echo "lock taken by $(cat $L), exiting"; exit 0; }
  squeue -h -u $USER -t RUNNING,COMPLETING -n ar_${EV}_${A} -o %i 2>/dev/null | grep -q . && { echo "main job of ${EV}_${A} started, exiting"; exit 0; }; fi
trap '[ "$(cat $L 2>/dev/null)" = "$SLURM_JOB_ID" ] && rm -f $L' EXIT
[ -f $R/namelist.base ] || $B/scripts/setup_run.sh $EV $A || { echo "setup failed"; exit 1; }
cd $R
if [ ! -f wrfinput_d01 ] || [ ! -f wrfinput_d02 ] || [ ! -f wrfbdy_d01 ] || [ -f .real_incomplete ]; then
  touch .real_incomplete; cp namelist.base namelist.input; echo "real start $(date)"; srun real.exe; tail -n 1 rsl.error.0000
  mkdir -p logs_real; cp rsl.error.0000 rsl.out.0000 namelist.input logs_real/
  grep -q SUCCESS rsl.error.0000 || { echo "real failed"; exit 1; }; rm -f rsl.* .real_incomplete; echo "real done $(date)"
fi
[ "$MODE" = real ] && exit 0
complete_rst() {   # newest restart pair whose last variable can be read in both files
  for f in $(ls wrfrst_d01_* 2>/dev/null | sort -r); do g=${f/d01/d02}; [ -f $g ] || continue
    $PY $B/scripts/check_nc.py $f $g 2>/dev/null && { echo $f; return; }
  done; }
TRY=0
while :; do
  LAST=$(complete_rst)
  if [ -n "$LAST" ]; then
    T=${LAST#wrfrst_d01_}; Y=${T:0:4}; M=${T:5:2}; D=${T:8:2}; HH=${T:11:2}
    sed -e "s/^ start_year.*/ start_year = $Y, $Y, start_month = $M, $M, start_day = $D, $D, start_hour = $HH, $HH,/" -e "s/restart = .false./restart = .true./" namelist.base > namelist.input
    echo "restart from $T"
  else cp namelist.base namelist.input; echo "cold start"; fi
  LEFT=$(( TEND - $(date +%s) - 240 )); [ $LEFT -lt 600 ] && { echo "not enough time left"; break; }
  SEG=seg_$(date +%Y%m%d%H%M%S)_$SLURM_JOB_ID; T0=$(date +%s); echo "wrf start $(date), stops after $LEFT s at the latest"
  timeout --signal=TERM $LEFT srun wrf.exe; RC=$?; T1=$(date +%s)
  echo "wrf stop $(date) rc $RC wall $((T1-T0)) s"; tail -n 2 rsl.error.0000
  mkdir -p logs_wrf/$SEG; cp rsl.error.0000 rsl.out.0000 namelist.input logs_wrf/$SEG/
  grep -h "Timing for main" rsl.error.0000 | tail -n 1
  grep -il "cfl" rsl.error.* 2>/dev/null | head -n 3 > logs_wrf/$SEG/cfl_files.txt
  if grep -q "SUCCESS COMPLETE WRF" rsl.error.0000; then touch DONE; echo "run complete $(date)"; rm -f rsl.*; break; fi
  [ "$(cat $L 2>/dev/null)" = "$SLURM_JOB_ID" ] || { echo "lock now held by $(cat $L 2>/dev/null); leaving the run to it"; break; }
  rm -f rsl.*
  if [ $RC -eq 124 ] || [ $TRY -ge 1 ]; then break; fi
  TRY=1; echo "wrf failed early; retrying once with time_step 36"; sed -i "s/^ time_step = [0-9]*,/ time_step = 36,/" namelist.base
done
for d in d01 d02; do ls wrfrst_${d}_* 2>/dev/null | sort | head -n -2 | xargs -r rm -f; done   # keep the two newest restarts
[ -f DONE ] && exit 0 || exit 1
