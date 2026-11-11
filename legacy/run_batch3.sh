#!/bin/bash
# THIRD batch: ablations on the final configuration (pinn_eur2.py, rho = 30). Run it AFTER batch 2.
#   (a) 3 long runs: plain / cheb / fourier, seed 0, 60000 steps (is the network simply under-trained?)
#   (b) feature-count ablation: N = 4 and N = 12 for cheb and fourier, seeds 0, 1, 2, 30000 steps (N = 8 comes from batch 2)
# 15 jobs, 3 at a time (the long ones first), roughly 10 hours. Stop at any time with Ctrl+C; re-run the same command to resume.
#
# HOW TO USE (Mac):   conda activate pinn
#                     cd /Users/grayman/Documents/PhD/PINN
#                     caffeinate -i bash run_batch3.sh
# Results go to the same results_eur2.csv / results_eur2_cases.csv (the analysis separates the variants), logs to logs3/.
chmod +x run_one3.sh
mkdir -p logs3
export MPLBACKEND=Agg
{
  for m in plain cheb fourier; do echo "$m 0 8 60000"; done
  for s in 0 1 2; do
    for n in 4 12; do
      for m in fourier cheb; do echo "$m $s $n 30000"; done
    done
  done
} | xargs -P 3 -L 1 ./run_one3.sh
echo "ALL DONE. Now run:  python analyze_eur.py results_eur2_cases.csv"
