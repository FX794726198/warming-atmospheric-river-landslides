#!/bin/bash
# usage: post_all.sh [RUN ...]  (default: every run with DONE) -> /scratch/$USER/ar_pgw/post
B=/scratch/$USER/ar_pgw; source ~/venvs/ff/bin/activate
RUNS=${@:-$(for d in $B/runs/*; do [ -f $d/DONE ] && basename $d; done)}
for r in $RUNS; do echo "== $r"; python $B/scripts/post_event.py $B/runs/$r $B/post > $B/logs/post_$r.log 2>&1 && tail -c 600 $B/logs/post_$r.log || echo "post failed $r"; done
