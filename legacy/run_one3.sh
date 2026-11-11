#!/bin/bash
# helper for run_batch3.sh:  run_one3.sh MODE SEED N ITERS     (rho is always 30)
mode=$1; seed=$2; N=$3; iters=$4
mkdir -p logs3
log="logs3/${mode}_N${N}_it${iters}_seed${seed}.log"
if grep -q "^FINAL" "$log" 2>/dev/null; then echo "skip $mode N=$N it=$iters seed=$seed (already done)"; exit 0; fi
echo "$(date '+%H:%M:%S') start  $mode N=$N it=$iters seed=$seed"
python pinn_eur2.py "$mode" "$seed" "$N" "$iters" 30 > "$log" 2>&1
echo "$(date '+%H:%M:%S') finish $mode N=$N it=$iters seed=$seed"
