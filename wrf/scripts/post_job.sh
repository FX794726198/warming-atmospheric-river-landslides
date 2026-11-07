#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=3500M
#SBATCH -o /scratch/$USER/ar_pgw/logs/post_%x_%j.log
# usage: sbatch -J ar_post_EVENT -A ACC -p PART -t 01:30:00 --dependency=afterok:CTLJOB:PGWJOB post_job.sh EVENT
# -> /scratch/$USER/ar_pgw/post/EVENT_{ctl,pgw585}.npz + EVENT_summary.csv (post_case.py), and the NetCDF files of post_event.py
EV=$1; B=/scratch/$USER/ar_pgw; PY=python3
echo "post $EV start $(date) on $SLURM_JOB_NODELIST"
$PY $B/scripts/post_case.py $EV $B/post || echo "post_case failed"
for A in ctl pgw585; do $PY $B/scripts/post_event.py $B/runs/${EV}_$A $B/post > $B/logs/post_event_${EV}_$A.log 2>&1 || echo "post_event $A failed"; done
ls -la $B/post | grep $EV; echo "post $EV done $(date)"; echo "post $EV done $(date)" >> $B/logs/post_ar_post_${EV}_login.log
