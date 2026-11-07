#!/bin/bash
# usage: submit_segs.sh  -> opportunistic 4 h WRF segments on participation-cpu (1 node, 64 tasks) for all 10 runs.
# Each run gets a chain of segments (afterany). run_wrf.sh MODE seg exits at once if the run is DONE or held by another job,
# and a main compute-p1 job that starts later cancels the segment holding its run and continues from the newest restart.
# Depth-first order by pairs, shortest cases first (NOR, ORE, BC, RSB, LIG): rank r gets nice = r * 4,000,000 so a run's later
# segments still go before the next run's first segment. Appends to runs/jobs.txt.
B=/scratch/$USER/ar_pgw; ACC=${ACCOUNT:?set ACCOUNT to your Slurm account}; P=participation-cpu
declare -A NSEG=( [NOR_2011]=2 [ORE_1996]=2 [BC_2021]=2 [RSB_2024]=3 [LIG_2014]=3 )
r=0
for EV in NOR_2011 ORE_1996 BC_2021 RSB_2024 LIG_2014; do for A in ctl pgw585; do
  dep=""; ids=""
  for s in $(seq 1 ${NSEG[$EV]}); do
    J=$(sbatch --parsable -J ars_${EV}_${A}_$s -A $ACC -p $P --nodes=1 --ntasks=64 --cpus-per-task=1 --mem-per-cpu=3500M -t 03:59:00 \
        --nice=$((r * 4000000)) $dep $B/scripts/run_wrf.sh $EV $A seg 2>/dev/null)
    dep="--dependency=afterany:$J"; ids="$ids $J"
  done
  echo "$(date -u +%F_%TZ) ${EV}_${A} seg jobs:$ids partition=$P ntasks=64 time=03:59:00 nice=$((r * 4000000))" | tee -a $B/runs/jobs.txt
  r=$((r + 1))
done; done
