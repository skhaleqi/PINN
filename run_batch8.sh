#!/bin/bash
# EIGHTH batch (controlled ablation of the European + premium split). About 5-7 hours.
#   plain, cheb, fourier learning the WHOLE price directly, 30000 steps, seeds 0-2   (compare with plain_rect / cheb_rect / fourier_rect)
#   fourier learning the whole price directly, 60000 steps, seeds 0-2                 (compare with fourier_rect_it60000)
# Everything else (domain rect, rho 30, payoff clipping, features, optimizer) is identical to the premium runs.
# 12 jobs, 3 at a time. Ctrl+C stops; the same command resumes.
#
# HOW TO USE (Mac):   conda activate pinn
#                     cd /Users/grayman/Documents/PhD/PINN
#                     caffeinate -i bash run_batch8.sh
chmod +x run_one8.sh
mkdir -p logs8
export MPLBACKEND=Agg
if [ ! -f smoke8_ok ]; then
  rm -rf smoke8; mkdir smoke8
  cp reference_solver.py param_refs_eur.npz param_refs_val.npz pinn_eur2.py smoke8/
  if (cd smoke8 && python pinn_eur2.py fourier 0 8 200 30 rect 4) > logs8/smoke.log 2>&1; then touch smoke8_ok; rm -rf smoke8; echo "smoke test OK"
  else echo "SMOKE TEST FAILED - see logs8/smoke.log and send me the last lines"; exit 1; fi
fi
{
  for s in 0 1 2; do echo "plain $s 30000"; echo "cheb $s 30000"; echo "fourier $s 30000"; done
  for s in 0 1 2; do echo "fourier $s 60000"; done
} | xargs -P 3 -L 1 ./run_one8.sh
python val_eval.py > logs8/val_eval.log 2>&1
echo "ALL DONE. Now run:   python analyze_eur.py results_eur2_cases.csv   (send me eur_summary.txt)"
