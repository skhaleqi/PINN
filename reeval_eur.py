"""
Re-evaluates the models trained in the FIRST EUR/USD batch (files eur_*_N*_rho*_seed*.pt) WITH payoff clipping,
i.e. reported price = max(PINN price, payoff). No retraining; takes about a minute.

    python reeval_eur.py                      (run in the folder that has the .pt files, pinn_eur.py, param_refs_eur.npz)

Writes results_eur_clip_cases.csv (same columns as results_eur_cases.csv) and prints, for every model, the mean error
before and after clipping. Then:   python analyze_eur.py results_eur_clip_cases.csv restricted
"""
import sys
import re
import csv
import glob
import importlib
import numpy as np
import torch

files = sorted(glob.glob("eur_*_N*_rho*_seed*.pt"))
if not files:
    sys.exit("No eur_*.pt files found in this folder.")
rows, mod = [], None
print(f"{'model':36s} {'ATM err raw':>12s} {'ATM err clip':>13s} {'relL2 exact raw':>16s} {'relL2 exact clip':>17s}")
for f in files:
    ck = torch.load(f, map_location="cpu", weights_only=False)
    mode, N, rho = ck["mode"], int(ck["N"]), int(round(float(ck["rho"])))
    seed = int(re.search(r"seed(\d+)", f).group(1))
    sys.argv = ["pinn_eur.py", mode, str(seed), str(N), "1", str(rho)]
    if mod is None:
        import pinn_eur as mod
    else:
        mod = importlib.reload(mod)
    model = mod.PINN()
    model.load_state_dict(ck["state_dict"])
    model.eval()
    pred = mod.predict_cases(model)
    _, raw = mod.compute_metrics(pred)                       # raw, compared with the raw penalized reference
    g = np.maximum(1 - mod.XS, 0)
    mod.V_PEN = np.maximum(mod.V_PEN, g)                     # clipped penalized reference
    cases, clip = mod.compute_metrics(np.maximum(pred, g))   # clipped prediction
    print(f"{f[:-3]:36s} {raw['mean_price_err']:12.4f} {clip['mean_price_err']:13.4f} {raw['mean_rel_ex']*100:15.3f}% {clip['mean_rel_ex']*100:16.3f}%")
    for c in cases:
        rows.append(dict(mode=mode, N=N, rho=float(rho), seed=seed, iters=30000, **{k: float(v) for k, v in c.items()}))
with open("results_eur_clip_cases.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print("\nSaved results_eur_clip_cases.csv")
