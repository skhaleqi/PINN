#!/bin/bash
# helper for run_batch6.sh:  run_one6.sh MODE SEED ITERS RHO     (N=8, domain rect)
mode=$1; seed=$2; iters=$3; rho=$4
mkdir -p logs6
log="logs6/${mode}_rect_it${iters}_rho${rho}_seed${seed}.log"
if grep -q "^FINAL" "$log" 2>/dev/null; then echo "skip $mode it=$iters rho=$rho seed=$seed (already done)"; exit 0; fi
echo "$(date '+%H:%M:%S') start  $mode it=$iters rho=$rho seed=$seed"
python pinn_eur2.py "$mode" "$seed" 8 "$iters" "$rho" rect 0 > "$log" 2>&1
echo "$(date '+%H:%M:%S') finish $mode it=$iters rho=$rho seed=$seed"
