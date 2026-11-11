#!/bin/bash
# SECOND batch (pinn_eur2.py: payoff clipping, realistic carry, rho = 30).
# 3 queues in parallel (plain, cheb, fourier); each runs seeds 0,1,2,3,4 in this order, so you can stop at any time and
# keep complete seeds. 15 runs in total, roughly 8-9 hours.
#
# HOW TO USE (Mac):   conda activate pinn
#                     cd /Users/grayman/Documents/PhD/PINN
#                     caffeinate -i bash run_batch2.sh
# If interrupted, run the same command again: finished runs are skipped. Logs go to logs2/.
# Results accumulate in results_eur2.csv / results_eur2_cases.csv.
mkdir -p logs2
export MPLBACKEND=Agg

if [ ! -f smoke2_ok ]; then
  rm -rf smoke2; mkdir smoke2
  cp reference_solver.py param_refs_eur.npz pinn_eur2.py smoke2/
  if (cd smoke2 && python pinn_eur2.py cheb 0 8 200 30) > logs2/smoke.log 2>&1; then
    touch smoke2_ok; rm -rf smoke2; echo "smoke test OK"
  else
    echo "SMOKE TEST FAILED - see logs2/smoke.log and send me the last lines"; exit 1
  fi
fi

run() {  # run MODE SEED
  log="logs2/${1}_rho30_seed${2}.log"
  if grep -q "^FINAL" "$log" 2>/dev/null; then echo "skip $1 seed=$2 (already done)"; return; fi
  echo "$(date '+%H:%M:%S') start  $1 seed=$2"
  python pinn_eur2.py "$1" "$2" 8 30000 30 > "$log" 2>&1
  echo "$(date '+%H:%M:%S') finish $1 seed=$2"
}
queue() { for s in 0 1 2 3 4; do run "$1" "$s"; done; }
queue plain &
queue cheb &
queue fourier &
wait
echo "ALL DONE. Now run:  python analyze_eur.py results_eur2_cases.csv"
