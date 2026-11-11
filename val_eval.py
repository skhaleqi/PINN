"""
Evaluates every saved model (eur2_*.pt) on the held-out VALIDATION contracts (param_refs_val.npz) and writes results_val.csv.
The validation set is what the automatic choice between variants uses; the 105-case test grid stays untouched for reporting.

    python val_eval.py        (needs pinn_eur2.py, reference_solver.py, param_refs_eur.npz, param_refs_val.npz in the folder)
"""
import sys
import re
import csv
import glob
import importlib
import numpy as np
import torch

files = sorted(glob.glob("eur2_*.pt"))
if not files:
    sys.exit("No eur2_*.pt files found.")
rows, mod = [], None
for f in files:
    ck = torch.load(f, map_location="cpu", weights_only=False)
    mode, N, rho = ck["mode"], int(ck["N"]), int(round(float(ck["rho"])))
    domain, stretch, iters = ck.get("domain", "band"), int(ck.get("stretch", 0)), int(ck.get("iters", 30000))
    seed = int(re.search(r"seed(\d+)", f).group(1))
    sys.argv = ["pinn_eur2.py", mode, str(seed), str(N), str(iters), str(rho), domain, str(stretch)]
    if mod is None:
        import pinn_eur2 as mod
    else:
        mod = importlib.reload(mod)
    model = mod.PINN()
    model.load_state_dict(ck["state_dict"])
    model.eval()
    vm = mod.val_metrics(mod.predict_val(model))
    _, tm = mod.compute_metrics(mod.predict_cases(model))
    rows.append(dict(mode=mode, N=(0 if mode == "plain" else N), rho=float(rho), seed=seed, domain=domain, stretch=stretch, iters=iters,
                     **vm, test_atm_mean=float(tm["mean_price_err"])))
    print(f"{f[:-3]:52s} val ATM {vm['val_atm_mean']:.4f} | val rel L2 {vm['val_rel_ex'] * 100:.3f}% | test ATM {tm['mean_price_err']:.4f}")
with open("results_val.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print("Saved results_val.csv")
