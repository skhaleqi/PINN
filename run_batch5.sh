#!/bin/bash
# FIFTH batch: ablations on the FINAL configuration (Fourier features, rect domain, rho 30, payoff clipping). Roughly 7-9 hours.
#   (a) 3 long runs of 60000 steps, seeds 0-2            -> is the network simply under-trained?
#   (b) low-sigma emphasis in the sampling, seeds 0-2    -> can the weak sigma = 0.03 end be improved?
#   (c) feature count N = 4, 12, 16, seeds 0-2           -> sensitivity to the number of Fourier features (N = 8 is the final config)
# 18 job-units, 3 jobs at a time, the long runs first. Ctrl+C stops; the same command resumes.
#
# HOW TO USE (Mac):   conda activate pinn
#                     cd /Users/grayman/Documents/PhD/PINN
#                     caffeinate -i bash run_batch5.sh
chmod +x run_one5.sh
mkdir -p logs5
export MPLBACKEND=Agg
if [ ! -f smoke5_ok ]; then
  rm -rf smoke5; mkdir smoke5
  cp reference_solver.py param_refs_eur.npz param_refs_val.npz pinn_eur2.py smoke5/
  ok=1
  for cfg in "fourier 0 4 200 2" "fourier 0 16 200 0"; do
    if ! (cd smoke5 && python pinn_eur2.py $(echo $cfg | awk '{print $1, $2, $3, $4}') 30 rect $(echo $cfg | awk '{print $5}')) >> logs5/smoke.log 2>&1; then ok=0; fi
  done
  if [ $ok -eq 1 ]; then touch smoke5_ok; rm -rf smoke5; echo "smoke test OK"; else echo "SMOKE TEST FAILED - see logs5/smoke.log and send me the last lines"; exit 1; fi
fi
{
  for s in 0 1 2; do echo "fourier $s 8 60000 0"; done
  for s in 0 1 2; do echo "fourier $s 8 30000 2"; done
  for s in 0 1 2; do for n in 4 12 16; do echo "fourier $s $n 30000 0"; done; done
} | xargs -P 3 -L 1 ./run_one5.sh
python val_eval.py > logs5/val_eval.log 2>&1
echo "ALL DONE. Now run:   python analyze_eur.py results_eur2_cases.csv"
