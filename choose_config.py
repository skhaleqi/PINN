"""
Chooses the final training configuration from the VALIDATION results (results_val.csv), using Fourier models (N=8, 30000 steps),
seeds 0-2 only, so that every variant is compared on the same footing.

    python choose_config.py        ->  prints a table and writes chosen_config.txt  ("DOMAIN STRETCH", e.g. "box 0")

Rule: among the screened variants (band, box, rect, wide, and band + stretch) take the one with the smallest mean validation ATM error.
If you create the file force_config.txt (one line "DOMAIN STRETCH") it overrides this choice.
"""
import os
import numpy as np
import pandas as pd

if os.path.exists("force_config.txt"):
    cfg = open("force_config.txt").read().split()
    open("chosen_config.txt", "w").write(" ".join(cfg) + "\n")
    print("force_config.txt found ->", cfg)
    raise SystemExit
v = pd.read_csv("results_val.csv")
f = v[(v["mode"] == "fourier") & (v.N == 8) & (v.iters == 30000) & (v.seed.isin([0, 1, 2]))]
t = f.groupby(["domain", "stretch"]).agg(seeds=("seed", "count"), val_atm_mean=("val_atm_mean", "mean"), val_atm_sd=("val_atm_mean", "std"),
                                         val_rel_L2_pct=("val_rel_ex", lambda x: x.mean() * 100), test_atm_mean=("test_atm_mean", "mean"))
print(t.round(4).to_string())
cand = t[t.seeds >= 3]
if len(cand) == 0:
    raise SystemExit("No complete (3-seed) screening results yet.")
domain, stretch = cand.val_atm_mean.idxmin()
open("chosen_config.txt", "w").write(f"{domain} {stretch}\n")
print(f"\nChosen configuration: domain={domain}, stretch={stretch}  (written to chosen_config.txt)")
