# Parametric physics-informed neural network for American FX options

Code and results for the paper **"Spectral input features and early-exercise-premium learning for parametric physics-informed pricing of American FX options"** (working title; [authors, journal, DOI to be added]).

One neural network prices American puts under the Garman–Kohlhagen model for all volatilities σ, domestic rates r and foreign rates q in a realistic EUR/USD range. It is trained without option data through the pricing PDE (penalty method), learns only the early-exercise premium on top of the exact European price, and its output is clipped at the payoff.

## Main results (strike = 100, mean absolute error of the at-the-money price)

| model (rect domain, rho = 30) | 30,000 steps | 60,000 steps | 120,000 steps |
|---|---|---|---|
| plain network | 0.0846 | 0.0404 | 0.0249 |
| Chebyshev features | 0.0519 | 0.0251 | 0.0167 |
| Fourier features | 0.0333 | **0.0156** | – |
| European formula (no network) | 0.1235 | | |

* The penalty method has an accuracy floor of about 0.0157; the Fourier model reaches it at 60,000 steps.
* Learning the whole price directly (instead of the premium) is 9–16 times worse at equal training budget.
* Batch pricing: 0.23 ms per contract, about 60 times faster than the Python finite-difference solver in this repository (indicative; see the paper).

Numbers are copied from `results/summaries/` (latest: `batch8_eur_summary_LATEST_all_variants.txt`). All tables of the paper and the supplement: `results/Supplementary_Tables.xlsx`.

## Repository layout

* `reference_solver.py` – classical solvers (exact European, Crank–Nicolson + PSOR American, penalised American).
* `pinn_eur2.py` – the PINN (training, evaluation, all variants). `pinn_eur.py` – first-batch version.
* `make_refs_eur.py`, `make_val_refs.py` – build the reference solutions (`param_refs_eur.npz`, `param_refs_val.npz`, **already included**).
* `run_batch*.sh`, `run_one*.sh` – the experiment batches in the order they were run.
* `val_eval.py`, `choose_config.py`, `analyze_eur.py`, `learning_curves.py`, `benchmark_speed.py` – evaluation and analysis.
* `make_figures.py`, `plot_examples.py` – paper figures.
* `vol_range.py` – realised-volatility range and weights from a price file.
* `pinn_american_exp.py` – single-instance study (fixed σ, r, q) used in the supplement.
* `results/` – summaries of every batch and the supplementary tables. `legacy/` – early exploratory scripts (not used for any reported number; delete if you prefer a clean tree).

See `FILES.md` for what each file is used for and `REPRODUCE.md` for the commands.

## Installation

Python 3.11 (other recent versions probably work). 

```bash
conda create -n pinn python=3.11 -y
conda activate pinn
pip install -r requirements.txt
```

`requirements.txt` keeps NumPy below 2 because some PyTorch builds fail with NumPy 2. Record your exact versions with `pip freeze > requirements-lock.txt` before archiving.

## Quick test (about one minute)

```bash
python pinn_eur2.py fourier 0 8 300 30 rect 0      # 300 training steps; writes results_eur2.csv, results_eur2_cases.csv and a .pt file
```

## Data

The EUR/USD 4-hour price file used to choose the volatility range and the weights is broker data and is not redistributed. The experiments do **not** need it: the reference solutions and the validation contracts are included. To rebuild them from your own price file see `vol_range.py` and `make_val_refs.py`.

## Compute

One 30,000-step model takes about 1–1.5 hours on a laptop CPU when three runs share the machine; the complete study (eight batches) took several days of laptop time. Trained weights are not stored in git (see `FILES.md`).

## Use of AI assistance

An AI assistant (Claude, Anthropic) was used to help write and debug the code, to run and summarise numerical analyses and to draft text. The author reviewed the material and is responsible for it. [Adapt this paragraph to the final declaration in the paper.]

## License and citation

MIT License, All right recived Citation: [paper reference once available]; archived version: [Zenodo DOI].
