#!/bin/bash
# INMET automatic-station archive 2024 (public open data), up to 3 tries
cd /scratch/$USER/ar_pgw/eval/inmet
for i in 1 2 3; do
  echo "try $i $(date -u +%TZ)"
  curl -fL --connect-timeout 120 --max-time 1500 -C - -o 2024.zip https://portal.inmet.gov.br/uploads/dadoshistoricos/2024.zip && { echo "ok $(date -u +%TZ) $(stat -c %s 2024.zip)"; exit 0; }
  echo "rc $? $(date -u +%TZ)"; sleep 20
done
echo FAILED
