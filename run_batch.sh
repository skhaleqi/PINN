#!/bin/bash
# Runs the whole EUR/USD experiment unattended: 3 queues in parallel (plain, cheb, fourier), each doing
#   1) rho=100 seed 0   2) rho=30 seed 0   3) rho=300 seed 0   4) rho=100 seed 1   5) rho=100 seed 2
# in that order of importance, so you can stop at any time and still have the most useful runs.
#
# HOW TO USE (Mac):   conda activate pinn
#                     cd /Users/grayman/Documents/PhD/PINN
#                     caffeinate -i bash run_batch.sh
# (caffeinate keeps the Mac awake; keep the lid open and the charger plugged in.)
# If it is interrupted, run the same command again: finished runs are skipped.
# One log per run is written to logs/ ; results accumulate in results_eur.csv / results_eur_cases.csv.
mkdir -p logs
export MPLBACKEND=Agg

# 1) quick smoke test (200 steps, in a temporary folder so no result files are polluted)
if [ ! -f smoke_ok ]; then
  rm -rf smoke; mkdir smoke
  cp reference_solver.py param_refs_eur.npz pinn_eur.py smoke/
  if (cd smoke && python pinn_eur.py cheb 0 8 200 100) > logs/smoke.log 2>&1; then
    touch smoke_ok; rm -rf smoke; echo "smoke test OK"
  else
    echo "SMOKE TEST FAILED - see logs/smoke.log and send me the last lines"; exit 1
  fi
fi

run() {  # run MODE SEED RHO
  log="logs/${1}_rho${3}_seed${2}.log"
  if grep -q "^FINAL" "$log" 2>/dev/null; then echo "skip $1 rho=$3 seed=$2 (already done)"; return; fi
  echo "$(date '+%H:%M:%S') start  $1 rho=$3 seed=$2"
  python pinn_eur.py "$1" "$2" 8 30000 "$3" > "$log" 2>&1
  echo "$(date '+%H:%M:%S') finish $1 rho=$3 seed=$2"
}
queue() {
  run "$1" 0 100; run "$1" 0 30; run "$1" 0 300; run "$1" 1 100; run "$1" 2 100
}
queue plain &
queue cheb &
queue fourier &
wait
echo "ALL DONE. Now run:  python analyze_eur.py"
