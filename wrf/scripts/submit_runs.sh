#!/bin/bash
# v2. usage: submit_runs.sh [EVENT ...]  -> submits the main WRF jobs (ctl + pgw585) on partition compute-p1.
# The jobs wait for wps/EVENT/READY and set up their run dirs themselves. Appends to runs/jobs.txt.
B=/scratch/$USER/ar_pgw; ACC=${ACCOUNT:?set ACCOUNT to your Slurm account}; P=compute-p1; MEM=2000M
declare -A TL=( [BC_2021]=10:00:00 [LIG_2014]=15:00:00 [NOR_2011]=10:00:00 [ORE_1996]=10:00:00 [RSB_2024]=13:00:00 )
declare -A NT=( [BC_2021]=192 [LIG_2014]=288 [NOR_2011]=192 [ORE_1996]=192 [RSB_2024]=240 )
EVS=${@:-NOR_2011 ORE_1996 RSB_2024 BC_2021 LIG_2014}
mkdir -p $B/runs
for EV in $EVS; do for A in ctl pgw585; do
  J=$(sbatch --parsable -J ar_${EV}_${A} -A $ACC -p $P --ntasks=${NT[$EV]} --cpus-per-task=1 --mem-per-cpu=$MEM -t ${TL[$EV]} \
      $B/scripts/run_wrf.sh $EV $A main)
  echo "$(date -u +%F_%TZ) $J ${EV}_${A} main account=$ACC partition=$P ntasks=${NT[$EV]} mem-per-cpu=$MEM time=${TL[$EV]}" | tee -a $B/runs/jobs.txt
done; done
