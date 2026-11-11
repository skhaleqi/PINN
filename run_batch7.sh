#!/bin/bash
# SEVENTH batch (the decisive convergence check): plain and cheb with 120000 steps, rho 30, rect domain, seeds 0-2.
# Question: do plain / Chebyshev reach the accuracy floor (about 0.0157) with enough training, or does the Fourier advantage persist?
# 6 jobs of 120000 steps, 3 at a time: the three plain runs first, then the three cheb runs. Roughly 10 hours.
# You can stop after the plain runs (Ctrl+C) if you are short of time: the plain result alone already answers most of the question.
#
# HOW TO USE (Mac):   conda activate pinn
#                     cd /Users/grayman/Documents/PhD/PINN
#                     caffeinate -i bash run_batch7.sh
chmod +x run_one7.sh
mkdir -p logs7
export MPLBACKEND=Agg
if [ ! -f smoke7_ok ]; then
  rm -rf smoke7; mkdir smoke7
  cp reference_solver.py param_refs_eur.npz param_refs_val.npz pinn_eur2.py smoke7/
  if (cd smoke7 && python pinn_eur2.py plain 0 8 200 30 rect 0) > logs7/smoke.log 2>&1; then touch smoke7_ok; rm -rf smoke7; echo "smoke test OK"
  else echo "SMOKE TEST FAILED - see logs7/smoke.log and send me the last lines"; exit 1; fi
fi
{
  for s in 0 1 2; do echo "plain $s"; done
  for s in 0 1 2; do echo "cheb $s"; done
} | xargs -P 3 -L 1 ./run_one7.sh
python val_eval.py > logs7/val_eval.log 2>&1
echo "ALL DONE. Now run:   python analyze_eur.py results_eur2_cases.csv   and   python learning_curves.py"
