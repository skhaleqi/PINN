"""
Experiment script for the thesis chapter: American put PINN with different input features.

    python pinn_american_exp.py MODE SEED [N]

    MODE = plain    : input (x, tau) only                         (baseline)
           cheb     : Chebyshev polynomials T_1..T_N of x and tau (idea B)
           fourier  : N sine/cosine terms of x and tau            (CONTROL: same number of
                      inputs as cheb, so a gain of cheb over plain cannot be explained
                      only by "more inputs")
    SEED = random seed (0, 1, 2, ...)        N = features per variable (default 8)

Everything except the input features is identical in all modes (same network,
same penalty schedule, same optimizer, same number of steps).

Each run appends ONE line to results_exp.csv and saves one figure.
Errors are measured against TWO classical reference solutions:
  * "exact"  : true American constraint (Crank-Nicolson + PSOR)
  * "pen"    : the penalized equation with the same rho as the PINN
               (isolates the network error from the bias of the penalty method)

Needs torch, numpy, scipy, matplotlib; reference_solver.py in the same folder.
Run without a plot window:   MPLBACKEND=Agg python pinn_american_exp.py cheb 0
"""
import os
import sys
import csv
import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from reference_solver import american_put, penalized_american_put

# ---------------- command line ----------------
MODE = sys.argv[1] if len(sys.argv) > 1 else "plain"
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 0
NF = int(sys.argv[3]) if len(sys.argv) > 3 else 8
assert MODE in ("plain", "cheb", "fourier"), "MODE must be plain, cheb or fourier"

# ---------------- settings (same for all modes) ----------------
K, r, sigma, T, q = 100.0, 0.05, 0.2, 1.0, 0.0
X_MAX = 3.0
ITERS = 20000
N_PDE, N_IC, N_BC = 2000, 500, 500
W_IC, W_BC = 10.0, 10.0
RHO_START, RHO_FINAL = 5.0, 30.0
CLIP = 1.0
BOUNDARY_TOL = 1e-3
torch.manual_seed(SEED)
np.random.seed(SEED)


# ---------------- input features ----------------
def cheb(z, n):
    """Chebyshev polynomials T_1..T_n of z in [-1, 1]; T_1 = z is the original input."""
    T_ = [torch.ones_like(z), z]
    for _ in range(2, n + 1):
        T_.append(2 * z * T_[-1] - T_[-2])
    return torch.cat(T_[1:n + 1], dim=1)


def fourier(z, n):
    """Control with exactly n columns: z itself, sines and cosines of increasing frequency."""
    u = (z + 1) / 2
    ns = n // 2
    nc = n - 1 - ns
    cols = [z] + [torch.sin(k * np.pi * u) for k in range(1, ns + 1)] \
               + [torch.cos(k * np.pi * u) for k in range(1, nc + 1)]
    return torch.cat(cols, dim=1)


def expand(z):
    if MODE == "plain":
        return z
    if MODE == "cheb":
        return cheb(z, NF)
    return fourier(z, NF)


N_IN = 2 if MODE == "plain" else 2 * NF


class PINN(nn.Module):
    def __init__(self, hidden=64, layers=4):
        super().__init__()
        dims = [N_IN] + [hidden] * layers + [1]
        mods = []
        for i in range(len(dims) - 1):
            mods.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                mods.append(nn.Tanh())
        self.net = nn.Sequential(*mods)

    def forward(self, x, tau):
        xs, ts = 2 * x / X_MAX - 1, 2 * tau / T - 1          # rescale to [-1, 1]
        return self.net(torch.cat([expand(xs), expand(ts)], dim=1))


def payoff(x):
    return torch.clamp(1 - x, min=0.0)


# ---------------- physics ----------------
def pde_residual(model, x, tau, rho):
    v = model(x, tau)
    ones = torch.ones_like(v)
    v_tau = torch.autograd.grad(v, tau, ones, create_graph=True)[0]
    v_x = torch.autograd.grad(v, x, ones, create_graph=True)[0]
    v_xx = torch.autograd.grad(v_x, x, torch.ones_like(v_x), create_graph=True)[0]
    Lv = 0.5 * sigma**2 * x**2 * v_xx + (r - q) * x * v_x - r * v
    return v_tau - Lv - rho * torch.relu(payoff(x) - v)


def compute_loss(model, rho):
    x = (torch.rand(N_PDE, 1) * X_MAX).requires_grad_(True)
    tau = (torch.rand(N_PDE, 1) * T).requires_grad_(True)
    loss_pde = (pde_residual(model, x, tau, rho) ** 2).mean()
    x0 = torch.rand(N_IC, 1) * X_MAX
    loss_ic = ((model(x0, torch.zeros_like(x0)) - payoff(x0)) ** 2).mean()
    tb = torch.rand(N_BC, 1) * T
    loss_bc = ((model(torch.zeros_like(tb), tb) - 1.0) ** 2).mean() \
              + (model(torch.full_like(tb, X_MAX), tb) ** 2).mean()
    return loss_pde + W_IC * loss_ic + W_BC * loss_bc


# ---------------- reference solutions (computed once) ----------------
S_ex, V_ex, _ = american_put(K, r, sigma, T, q=q, S_max=300.0, M=400, N=200)
S_pen, V_pen = penalized_american_put(K, r, sigma, T, q=q, rho=RHO_FINAL, S_max=300.0)


def rough_boundary(xs, v):
    g = np.maximum(1 - xs, 0)
    idx = np.where(v - g < BOUNDARY_TOL)[0]
    return xs[idx.max()] * K if len(idx) else np.nan


def evaluate(model):
    xs = np.linspace(0.5, 1.5, 201)                          # S from 50 to 150, today
    with torch.no_grad():
        xt = torch.tensor(xs, dtype=torch.float32).reshape(-1, 1)
        v = model(xt, torch.full_like(xt, T)).numpy().ravel()
    out = dict(xs=xs, v=v)
    for name, (S_, V_) in dict(ex=(S_ex, V_ex), pen=(S_pen, V_pen)).items():
        ref = np.interp(xs * K, S_, V_) / K
        out[name + "_ref"] = ref
        out[name + "_rel"] = np.linalg.norm(v - ref) / np.linalg.norm(ref)
        out[name + "_max"] = np.max(np.abs(v - ref)) * K
        out[name + "_p100"] = np.interp(100.0, S_, V_)
    out["p100"] = float(np.interp(1.0, xs, v)) * K
    xb = np.linspace(0.5, 1.0, 501)
    with torch.no_grad():
        xbt = torch.tensor(xb, dtype=torch.float32).reshape(-1, 1)
        vb = model(xbt, torch.full_like(xbt, T)).numpy().ravel()
    out["s_star"] = rough_boundary(xb, vb)
    out["s_star_ex"] = rough_boundary(xb, np.interp(xb * K, S_ex, V_ex) / K)
    return out


# ---------------- training ----------------
if __name__ == "__main__":
    model = PINN()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"MODE={MODE} N={NF if MODE != 'plain' else '-'} SEED={SEED} | inputs {N_IN} | parameters {n_params}")
    print(f"reference price(S=100): exact {np.interp(100.0, S_ex, V_ex):.4f} | penalized(rho={RHO_FINAL:.0f}) {np.interp(100.0, S_pen, V_pen):.4f}")
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=ITERS, eta_min=1e-6)
    t0 = time.time()
    for it in range(1, ITERS + 1):
        rho = RHO_START + (RHO_FINAL - RHO_START) * min(1.0, it / (0.5 * ITERS))
        opt.zero_grad()
        loss = compute_loss(model, rho)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        opt.step()
        sched.step()
        if it % 1000 == 0 or it == 1:
            e = evaluate(model)
            print(f"it {it:5d} | rho {rho:5.1f} | loss {loss.item():.2e} | rel L2 vs exact {e['ex_rel']:.2e}"
                  f" | vs penalized {e['pen_rel']:.2e} | price(S=100) {e['p100']:.4f} | {time.time() - t0:.0f}s")
    secs = time.time() - t0

    e = evaluate(model)
    print(f"\nFINAL {MODE} N={NF} seed={SEED} | vs EXACT: rel L2 {e['ex_rel']:.3e}, max {e['ex_max']:.4f} "
          f"| vs PENALIZED: rel L2 {e['pen_rel']:.3e}, max {e['pen_max']:.4f} "
          f"| price(S=100) {e['p100']:.4f} (exact {e['ex_p100']:.4f}) | boundary {e['s_star']:.1f} (ref {e['s_star_ex']:.1f}) | {secs:.0f}s")

    row = dict(mode=MODE, N=NF if MODE != "plain" else 0, seed=SEED, params=n_params,
               rel_exact=e["ex_rel"], max_exact=e["ex_max"], rel_pen=e["pen_rel"], max_pen=e["pen_max"],
               price_S100=e["p100"], price_exact=e["ex_p100"], boundary=e["s_star"],
               boundary_ref=e["s_star_ex"], seconds=round(secs))
    new_file = not os.path.exists("results_exp.csv")
    with open("results_exp.csv", "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new_file:
            w.writeheader()
        w.writerow(row)

    xs = e["xs"]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(xs * K, e["ex_ref"] * K, "k-", label="reference (exact constraint)")
    ax[0].plot(xs * K, e["v"] * K, "r--", label=f"PINN ({MODE})")
    ax[0].plot(xs * K, np.maximum(1 - xs, 0) * K, "g:", label="payoff")
    ax[0].set_xlabel("S"); ax[0].set_ylabel("American put value today")
    ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].plot(xs * K, (e["v"] - e["ex_ref"]) * K, "b-", label="PINN - exact reference")
    ax[1].plot(xs * K, (e["v"] - e["pen_ref"]) * K, "m-", label="PINN - penalized reference")
    ax[1].axhline(0, color="k", lw=0.5)
    ax[1].set_xlabel("S"); ax[1].set_ylabel("error (price units)")
    ax[1].legend(); ax[1].grid(alpha=.3)
    plt.tight_layout()
    plt.savefig(f"exp_{MODE}_N{NF}_seed{SEED}.png", dpi=150)
    plt.show()
