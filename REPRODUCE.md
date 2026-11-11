# How to reproduce the paper's results

All commands run from the repository root, inside the environment of `README.md`. Run times are those observed on an Apple M1 laptop CPU with three runs in parallel (indicative).

## 0. Reference solutions (shipped; rebuilding is optional)

```bash
python make_refs_eur.py                              # param_refs_eur.npz, about 20 minutes
python make_val_refs.py SwingLabels_EuroUsd2.csv     # param_refs_val.npz, about 8 minutes (needs your price file)
```

## 1. Training batches (each writes to results_eur2_cases.csv; logs go to logs2/ ... logs8/)

| Step | Command | Content | Time |
|---|---|---|---|
| batch 1 | `bash run_batch.sh` | rho = 30/100/300, first version | several hours |
| batch 2 | `bash run_batch2.sh` | plain/Chebyshev/Fourier, band domain, 5 seeds, 30,000 steps | 8–9 h |
| batch 4 | `bash run_batch4.sh` | screening of domains; final rect runs | about 9 h |
| batch 5 | `bash run_batch5.sh` | 60,000 steps (Fourier), low-sigma emphasis, N = 4/12/16 | about 10 h |
| batch 6 | `bash run_batch6.sh` | plain/Chebyshev at 60,000 steps; Fourier with rho = 100 | 7–9 h |
| batch 7 | `bash run_batch7.sh` | plain/Chebyshev at 120,000 steps | about 10 h |
| batch 8 | `bash run_batch8.sh` | direct formulation (no premium split) | 5–7 h |

Each script starts with a 200-step smoke test, runs three jobs at a time, skips finished jobs when restarted, and can be stopped with Ctrl+C.

## 2. Analysis and figures

```bash
python val_eval.py                                  # validation scores -> results_val.csv
python analyze_eur.py results_eur2_cases.csv        # -> eur_summary.txt (Tables 1-4, supplement)
python learning_curves.py                           # -> learning_curves.png / .txt (Fig 4)
python benchmark_speed.py eur2_fourier_N8_rho30_seed0_it60000_domrect.pt    # -> speed_summary.txt (Table 5)
python make_figures.py                              # Figs 1, 2, 5, 6
python plot_examples.py eur2_plain_N8_rho30_seed0_it60000_domrect.pt eur2_fourier_N8_rho30_seed0_it60000_domrect.pt   # Fig 3
```

## 3. Where each paper element comes from

| Paper element | Produced by |
|---|---|
| Table 1 (main comparison) | batches 4–7 -> `analyze_eur.py`, section 1 |
| Table 2 (error by sigma) | `analyze_eur.py`, section 3 |
| Table 3 (ablations) | batches 2, 4, 5 |
| Table 4 (premium vs direct) | batch 8 vs batches 4–5 |
| Penalty floor, clipping | `analyze_eur.py`, section 2 |
| Section 6.6 (rho) | batches 1 and 6 |
| Table 5, Fig 6 (speed) | `benchmark_speed.py` |

Seeds are 0–4 (30,000 steps) and 0–2 (longer runs and ablations); a given seed reproduces the same run on the same machine and library versions.
