#!/bin/bash
# SIXTH batch: fair comparison at CONVERGENCE (60000 steps) and a first test of a stiffer penalty. About 7-9 hours.
#   plain and cheb with 60000 steps, rho 30, seeds 0-2   (Fourier with 60000 steps, rho 30 is already done in batch 5)
#   fourier with 60000 steps and rho 100, seeds 0-2      (does a stiffer penalty now train, and does it lower the accuracy floor?)
# 9 jobs of 60000 steps, 3 at a time. Ctrl+C stops; the same command resumes.
#
# HOW TO USE (Mac):   conda activate pinn
#                     cd /Users/grayman/Documents/PhD/PINN
#                     caffeinate -i bash run_batch6.sh
chmod +x run_one6.sh
mkdir -p logs6
export MPLBACKEND=Agg
if [ ! -f smoke6_ok ]; then
  rm -rf smoke6; mkdir smoke6
  cp reference_solver.py param_refs_eur.npz param_refs_val.npz pinn_eur2.py smoke6/
  if (cd smoke6 && python pinn_eur2.py cheb 0 8 200 100 rect 0) > logs6/smoke.log 2>&1; then touch smoke6_ok; rm -rf smoke6; echo "smoke test OK"
  else echo "SMOKE TEST FAILED - see logs6/smoke.log and send me the last lines"; exit 1; fi
fi
{
  for s in 0 1 2; do echo "plain $s 60000 30"; echo "cheb $s 60000 30"; echo "fourier $s 60000 100"; done
} | xargs -P 3 -L 1 ./run_one6.sh
python val_eval.py > logs6/val_eval.log 2>&1
echo "ALL DONE. Now run:   python analyze_eur.py results_eur2_cases.csv   and   python learning_curves.py"
