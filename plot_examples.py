"""
Figure 3: exact American price and model errors against the spot S for two example contracts.

    python plot_examples.py PLAIN_WEIGHTS.pt FOURIER_WEIGHTS.pt
    e.g.  python plot_examples.py eur2_plain_N8_rho30_seed0_it60000_domrect.pt eur2_fourier_N8_rho30_seed0_it60000_domrect.pt

Needs: torch, numpy, matplotlib, pinn_eur2.py, reference_solver.py, param_refs_eur.npz, param_refs_val.npz in the folder.
Contracts: typical (sigma 0.08, r 0.03, q 0.02, tau 1) and hard (sigma 0.03, r 0.06, q 0.02, tau 1).
Writes fig3_examples.png and .pdf. Look at the picture: edit CONTRACTS or the colours at the top as you wish.
"""
import sys
import importlib
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reference_solver import american_put, european_put

CONTRACTS = [("typical", 0.08, 0.03, 0.02, 1.0), ("hard", 0.03, 0.06, 0.02, 1.0)]   # name, sigma, r, q, tau
COL = {"plain": "#6E6E6E", "cheb": "#E69F00", "fourier": "#0072B2"}
K = 100.0
paths = sys.argv[1:]
if len(paths) < 1:
    sys.exit(__doc__)


def load_model(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    mode, N, rho = ck["mode"], int(ck["N"]), int(round(float(ck["rho"])))
    domain, stretch, iters = ck.get("domain", "band"), int(ck.get("stretch", 0)), int(ck.get("iters", 30000))
    seed = int(path.split("seed")[1].split("_")[0].split(".")[0])
    sys.argv = ["pinn_eur2.py", mode, str(seed), str(N), str(iters), str(rho), domain, str(stretch)]
    import pinn_eur2 as mod
    mod = importlib.reload(mod)               # rebuild the module with this model's settings
    model = mod.PINN()
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return mode, mod, model


def predict(mod, model, sig, r, q, tau):
    nx = len(mod.XS)
    x = torch.tensor(mod.XS, dtype=torch.float32).reshape(-1, 1)
    full = lambda v: torch.full((nx, 1), float(v))
    with torch.no_grad():
        net = model(x, full(tau), full(sig), full(r), full(q))
        tot = net if mod.DIRECT else mod.euro_put(x, full(tau), full(sig), full(r), full(q)) + mod.U_SCALE * net
    return np.maximum(tot.numpy().ravel(), mod.PAYOFF_GRID)


models = [load_model(p) for p in paths]
xs = models[0][1].XS
S = xs * K
fig, axes = plt.subplots(2, len(CONTRACTS), figsize=(7.2, 5.0), sharex=True)
for j, (name, sig, r, q, tau) in enumerate(CONTRACTS):
    Sg, V, _ = american_put(K, r, sig, tau, q=q, S_max=300.0, M=800, N=int(400 * tau))
    exact = np.interp(S, Sg, V) / K
    euro = np.array([european_put(s_, K, r, sig, tau, q) for s_ in S]) / K
    axes[0, j].plot(S, exact * K, "k-", lw=1.5, label="exact American")
    axes[0, j].plot(S, euro * K, "--", color="#D55E00", lw=1, label="European")
    axes[0, j].plot(S, np.maximum(1 - xs, 0) * K, ":", color="#888888", lw=1, label="payoff")
    for mode, mod, model in models:
        pred = predict(mod, model, sig, r, q, tau)
        axes[0, j].plot(S, pred * K, lw=1, color=COL.get(mode, "C0"), alpha=0.9)
        axes[1, j].plot(S, (pred - exact) * K, lw=1.2, color=COL.get(mode, "C0"), label=mode)
    axes[0, j].set_title(f"{name}: σ={sig:g}, r={r:g}, q={q:g}, τ={tau:g}", fontsize=9)
    axes[1, j].axhline(0, color="k", lw=0.6)
    axes[1, j].set_xlabel("spot S (strike = 100)")
axes[0, 0].set_ylabel("price"); axes[1, 0].set_ylabel("model − exact (price units)")
axes[0, 0].legend(fontsize=7, frameon=False); axes[1, 0].legend(fontsize=7, frameon=False)
for a in axes.ravel():
    a.grid(alpha=0.25)
plt.tight_layout()
plt.savefig("fig3_examples.png", dpi=300, bbox_inches="tight"); plt.savefig("fig3_examples.pdf", bbox_inches="tight")
print("saved fig3_examples.png / .pdf")
