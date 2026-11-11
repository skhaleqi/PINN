# What each file is for

"Include" = recommended for the GitHub repository (Y), optional (O), or keep out (N).

## Code

| File | Purpose | Used for | Include |
|---|---|---|---|
| `reference_solver.py` | European formula; exact American (Crank–Nicolson + PSOR); penalised American | all reference solutions, benchmark | Y |
| `pinn_eur2.py` | the PINN: premium/direct formulation, plain/Chebyshev/Fourier features, domains, clipping, validation | Tables 1–4, Figs 2–5 | Y |
| `pinn_eur.py` | first-batch version of the PINN (no clipping, band/box domain) | rho = 100/300 results (Section 6.6), re-evaluation | Y |
| `make_refs_eur.py` | builds `param_refs_eur.npz` (105-case test grid; exact and penalised references for rho = 30, 100, 300) | test set | Y |
| `make_val_refs.py` | builds `param_refs_val.npz` (60 validation contracts) from your EUR/USD file | validation set | Y |
| `val_eval.py`, `choose_config.py` | validation scores of saved models and the automatic choice of the configuration | Section 5, batch 4 | Y |
| `analyze_eur.py` | produces `eur_summary.txt` from `results_eur2_cases.csv` | all tables | Y |
| `learning_curves.py` | learning curves from the training logs | Fig 4 | Y |
| `benchmark_speed.py` | speed and accuracy against finite differences | Table 5, Fig 6 | Y |
| `make_figures.py`, `plot_examples.py` | paper figures 1, 2, 3, 5, 6 | figures | Y |
| `vol_range.py` | realised-volatility range and quantiles of a price file | sigma range, weights | Y |
| `reeval_eur.py` | re-evaluation of first-batch models with clipping | Section 3.3 evidence | O |
| `pinn_american_exp.py` | single-instance study (fixed sigma, r, q), 10 seeds | supplement | O |
| `run_batch.sh` | batch 1 (rho 30/100/300, `pinn_eur.py`) | Section 6.6 | Y |
| `run_batch2.sh` | batch 2 (band domain, 5 seeds) | Tables 1, 3 | Y |
| `run_batch4.sh`, `run_one4.sh` | screening of domains and automatic final runs | Table 3 | Y |
| `run_batch5.sh`, `run_one5.sh` | long runs, low-sigma emphasis, feature count | Tables 1, 3 | Y |
| `run_batch6.sh`, `run_one6.sh` | plain/Chebyshev at 60,000 steps, rho = 100 | Tables 1, 6.6 | Y |
| `run_batch7.sh`, `run_one7.sh` | plain/Chebyshev at 120,000 steps | Table 1 | Y |
| `run_batch8.sh`, `run_one8.sh` | direct-formulation ablation | Table 4 | Y |
| `legacy/` | early exploratory scripts | none | N (optional) |

## Data and results

| File | What | Include |
|---|---|---|
| `param_refs_eur.npz` | reference solutions of the 105-case test grid (and the full 189-case grid) | Y (about 1 MB) |
| `param_refs_val.npz` | 60 validation contracts with exact prices | Y |
| `results/summaries/*.txt` | analysis summary of each batch, learning curves, speed benchmark | Y |
| `results/Supplementary_Tables.xlsx` | supplementary tables | Y |
| `results_eur2_cases.csv`, `results_eur2.csv`, `results_val.csv`, `results_eur_cases.csv` (your machine) | raw per-case and per-run results | Zenodo (or git if small) |
| `logs/`, `logs2/`, ..., `logs8/` (your machine) | training logs (needed by `learning_curves.py`) | Zenodo |
| `eur*_*.pt` (your machine) | trained weights | Zenodo (not git) |
| `SwingLabels_*.csv` (your machine) | broker price data | N (licence); describe in the paper |

Suggested release: a git repository with the code and small data, plus a Zenodo archive (DOI) of the raw CSVs, logs and weights, linked from the paper's data-availability statement.
