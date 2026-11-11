#!/bin/bash
# helper for run_batch4.sh:  run_one4.sh MODE SEED DOMAIN STRETCH     (N=8, 30000 steps, rho=30)
mode=$1; seed=$2; dom=$3; st=$4
mkdir -p logs4
log="logs4/${mode}_dom${dom}_st${st}_seed${seed}.log"
if grep -q "^FINAL" "$log" 2>/dev/null; then echo "skip $mode dom=$dom st=$st seed=$seed (already done)"; exit 0; fi
echo "$(date '+%H:%M:%S') start  $mode dom=$dom st=$st seed=$seed"
python pinn_eur2.py "$mode" "$seed" 8 30000 30 "$dom" "$st" > "$log" 2>&1
echo "$(date '+%H:%M:%S') finish $mode dom=$dom st=$st seed=$seed"
