#!/bin/bash
# helper for run_batch8.sh:  run_one8.sh MODE SEED ITERS     (N=8, rho 30, rect domain, DIRECT formulation = stretch code 4)
mode=$1; seed=$2; iters=$3
mkdir -p logs8
log="logs8/${mode}_rect_direct_it${iters}_seed${seed}.log"
if grep -q "^FINAL" "$log" 2>/dev/null; then echo "skip $mode it=$iters seed=$seed (already done)"; exit 0; fi
echo "$(date '+%H:%M:%S') start  $mode direct it=$iters seed=$seed"
python pinn_eur2.py "$mode" "$seed" 8 "$iters" 30 rect 4 > "$log" 2>&1
echo "$(date '+%H:%M:%S') finish $mode direct it=$iters seed=$seed"
