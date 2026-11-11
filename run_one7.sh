#!/bin/bash
# helper for run_batch7.sh:  run_one7.sh MODE SEED     (120000 steps, N=8, rho 30, rect domain)
mode=$1; seed=$2
mkdir -p logs7
log="logs7/${mode}_rect_it120000_seed${seed}.log"
if grep -q "^FINAL" "$log" 2>/dev/null; then echo "skip $mode seed=$seed (already done)"; exit 0; fi
echo "$(date '+%H:%M:%S') start  $mode seed=$seed (120000 steps)"
python pinn_eur2.py "$mode" "$seed" 8 120000 30 rect 0 > "$log" 2>&1
echo "$(date '+%H:%M:%S') finish $mode seed=$seed"
