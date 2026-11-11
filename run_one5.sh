#!/bin/bash
# helper for run_batch5.sh:  run_one5.sh MODE SEED N ITERS STRETCHCODE     (domain rect, rho 30)
mode=$1; seed=$2; N=$3; iters=$4; st=$5
mkdir -p logs5
log="logs5/${mode}_rect_N${N}_it${iters}_st${st}_seed${seed}.log"
if grep -q "^FINAL" "$log" 2>/dev/null; then echo "skip $mode N=$N it=$iters st=$st seed=$seed (already done)"; exit 0; fi
echo "$(date '+%H:%M:%S') start  $mode N=$N it=$iters st=$st seed=$seed"
python pinn_eur2.py "$mode" "$seed" "$N" "$iters" 30 rect "$st" > "$log" 2>&1
echo "$(date '+%H:%M:%S') finish $mode N=$N it=$iters st=$st seed=$seed"
