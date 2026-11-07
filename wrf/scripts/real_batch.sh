#!/bin/bash
#SBATCH --cpus-per-task=1
#SBATCH -o /scratch/$USER/ar_pgw/logs/real_%x_%j.log
# usage: sbatch ... real_batch.sh "EV:ARM EV:ARM ..."  -> real.exe for each run in turn (run_wrf.sh MODE real), skipping runs that have it
for x in $1; do bash /scratch/$USER/ar_pgw/scripts/run_wrf.sh ${x%%:*} ${x##*:} real; done
