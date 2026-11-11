#!/bin/bash
# FOURTH batch - fully automatic, about 12-15 hours:
#   stage 0  smoke test, and validation scores of the batch-2 models
#   stage 1  screening with Fourier features, seeds 0-2: domains box / wide / rect, and "band + stretch"   (12 jobs)
#   stage 2  automatic choice of the configuration from the held-out validation contracts (choose_config.py)
#   stage 3  final runs: plain, cheb, fourier x seeds 0-4 in the chosen configuration (15 jobs, finished ones are skipped)
# 3 jobs run at a time. Ctrl+C stops it; the same command resumes (finished jobs are skipped).
#
# HOW TO USE (Mac):   conda activate pinn
#                     cd /Users/grayman/Documents/PhD/PINN
#                     caffeinate -i bash run_batch4.sh
# To force a configuration instead of the automatic choice, create force_config.txt with one line, e.g.:   box 0
chmod +x run_one4.sh
mkdir -p logs4
export MPLBACKEND=Agg

if [ ! -f smoke4_ok ]; then
  rm -rf smoke4; mkdir smoke4
  cp reference_solver.py param_refs_eur.npz param_refs_val.npz pinn_eur2.py smoke4/
  ok=1
  for cfg in "band 0" "rect 1" "wide 0"; do
    set -- $cfg
    if ! (cd smoke4 && python pinn_eur2.py cheb 0 8 200 30 "$1" "$2") >> logs4/smoke.log 2>&1; then ok=0; fi
  done
  if [ $ok -eq 1 ]; then touch smoke4_ok; rm -rf smoke4; echo "smoke test OK"; else echo "SMOKE TEST FAILED - see logs4/smoke.log and send me the last lines"; exit 1; fi
fi

echo "stage 0: validation scores of existing models"
python val_eval.py > logs4/val_eval_0.log 2>&1 || { echo "val_eval.py failed, see logs4/val_eval_0.log"; exit 1; }

echo "stage 1: screening"
{
  for s in 0 1 2; do
    echo "fourier $s box 0"; echo "fourier $s wide 0"; echo "fourier $s rect 0"; echo "fourier $s band 1"
  done
} | xargs -P 3 -L 1 ./run_one4.sh

echo "stage 2: choosing the configuration"
python val_eval.py > logs4/val_eval_1.log 2>&1 || { echo "val_eval.py failed, see logs4/val_eval_1.log"; exit 1; }
python choose_config.py | tee logs4/choice.log
read DOM ST < chosen_config.txt
echo "chosen: domain=$DOM stretch=$ST"

echo "stage 3: final runs"
if [ "$DOM" = "band" ] && [ "$ST" = "0" ]; then
  echo "the default band configuration was chosen: batch 2 already contains all 15 final runs."
else
  {
    for s in 0 1 2 3 4; do for m in plain cheb fourier; do echo "$m $s $DOM $ST"; done; done
  } | xargs -P 3 -L 1 ./run_one4.sh
fi
python val_eval.py > logs4/val_eval_2.log 2>&1
echo "ALL DONE. Now run:   python analyze_eur.py results_eur2_cases.csv   and send me eur_summary.txt, results_val.csv and logs4/choice.log"
