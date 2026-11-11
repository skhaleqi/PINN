"""
Step 1 of the plan: a PARAMETRIC American-put PINN.

ONE network gets four inputs   (x = S/K, tau, sigma, r)   and returns v = V/K for
ALL volatilities and interest rates in the training range at once. After training,
a price for any (sigma, r) in the range is one forward pass (no new training).
This is what later lets an LSTM volatility forecast be turned into an option price instantly.

    python pinn_param_exp.py MODE SEED [N] [ITERS]

    MODE = plain   : the 4 inputs only                                      (baseline)
           cheb    : Chebyshev polynomials T_1..T_N of each of the 4 inputs
           fourier : N sine/cosine terms of each of the 4 inputs (CONTROL, same input count)
    SEED = random seed         N = features per input (default 8)
    ITERS = training steps (default 30000). Smoke test: python pinn_param_exp.py plain 0 8 300

Everything except the input features is identical in all modes.
Needs param_refs.npz (made once by make_param_refs.py) in the same folder.

Outputs: one line in results_param.csv, 60 lines in results_param_cases.csv (error per test case),
         a figure, and the trained weights (.pt) for later use.
Penalty method as before: v_tau = L v + rho*max(g - v, 0), with L v = 0.5 sigma^2 x^2 v_xx + r x v_x - r v.
Assumptions: q = 0 (no foreign rate yet), strike scaled to 1, maturity covered by tau in [0, 1].
"""
import os
import sys
import csv
import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ---------------- command line ----------------
MODE = sys.argv[1] if len(sys.argv) > 1 else "plain"
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 0
NF = int(sys.argv[3]) if len(sys.argv) > 3 else 8
ITERS = int(sys.argv[4]) if len(sys.argv) > 4 else 30000
assert MODE in ("plain", "cheb", "fourier"), "MODE must be plain, cheb or fourier"

# ---------------- settings (same for all modes) ----------------
T = 1.0
X_MAX = 3.0
SIG_LO, SIG_HI = 0.10, 0.40
R_LO, R_HI = 0.01, 0.08
N_PDE, N_IC, N_BC = 3000, 800, 800
W_IC, W_BC = 10.0, 10.0
RHO_START, RHO_FINAL = 5.0, 30.0
CLIP = 1.0
HIDDEN, LAYERS = 96, 4
KP = 100.0                      # only used to express errors in price units (strike = 100)
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
    """Control with exactly n columns: z itself, then sines and cosines of increasing frequency."""
    u = (z + 1) / 2
    ns = n // 2
    nc = n - 1 - ns
    cols = [z] + [torch.sin(k * np.pi * u) for k in range(1, ns + 1)] \
               + [torch.cos(k * np.pi * u) for k in range(1, nc + 1)]
    return torch.cat(cols, dim=1)


def expand(z):
    if MODE == "plain":
        return z
    return cheb(z, NF) if MODE == "cheb" else fourier(z, NF)


N_IN = 4 if MODE == "plain" else 4 * NF


def scale(v, lo, hi):
    return 2 * (v - lo) / (hi - lo) - 1


class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        dims = [N_IN] + [HIDDEN] * LAYERS + [1]
        mods = []
        for i in range(len(dims) - 1):
            mods.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                mods.append(nn.Tanh())
        self.net = nn.Sequential(*mods)

    def forward(self, x, tau, sig, rate):
        z = [scale(x, 0.0, X_MAX), scale(tau, 0.0, T), scale(sig, SIG_LO, SIG_HI), scale(rate, R_LO, R_HI)]
        return self.net(torch.cat([expand(c) for c in z], dim=1))


def payoff(x):
    return torch.clamp(1 - x, min=0.0)


# ---------------- physics ----------------
def pde_residual(model, x, tau, sig, rate, rho):
    v = model(x, tau, sig, rate)
    ones = torch.ones_like(v)
    v_tau = torch.autograd.grad(v, tau, ones, create_graph=True)[0]
    v_x = torch.autograd.grad(v, x, ones, create_graph=True)[0]
    v_xx = torch.autograd.grad(v_x, x, torch.ones_like(v_x), create_graph=True)[0]
    Lv = 0.5 * sig**2 * x**2 * v_xx + rate * x * v_x - rate * v
    return v_tau - Lv - rho * torch.relu(payoff(x) - v)


def rand_params(n):
    return SIG_LO + torch.rand(n, 1) * (SIG_HI - SIG_LO), R_LO + torch.rand(n, 1) * (R_HI - R_LO)


def compute_loss(model, rho):
    x = (torch.rand(N_PDE, 1) * X_MAX).requires_grad_(True)
    tau = (torch.rand(N_PDE, 1) * T).requires_grad_(True)
    sig, rate = rand_params(N_PDE)
    loss_pde = (pde_residual(model, x, tau, sig, rate, rho) ** 2).mean()

    x0 = torch.rand(N_IC, 1) * X_MAX
    s0, r0 = rand_params(N_IC)
    loss_ic = ((model(x0, torch.zeros_like(x0), s0, r0) - payoff(x0)) ** 2).mean()

    tb = torch.rand(N_BC, 1) * T
    sb, rb = rand_params(N_BC)
    loss_bc = ((model(torch.zeros_like(tb), tb, sb, rb) - 1.0) ** 2).mean() \
              + (model(torch.full_like(tb, X_MAX), tb, sb, rb) ** 2).mean()
    return loss_pde + W_IC * loss_ic + W_BC * loss_bc


# ---------------- reference solutions and metrics ----------------
if not os.path.exists("param_refs.npz"):
    sys.exit("param_refs.npz not found. Put it in this folder (or run: python make_param_refs.py).")
REF = np.load("param_refs.npz")
SIGMAS, RATES, TAUS, XS = REF["sigmas"], REF["rates"], REF["taus"], REF["xs"]
V_PEN, V_EX = REF["v_pen"], REF["v_ex"]            # shape (n_sigma, n_rate, n_tau, n_x)
CASES = [(i, j, k) for i in range(len(SIGMAS)) for j in range(len(RATES)) for k in range(len(TAUS))]


def compute_metrics(v_pred):
    """v_pred: array (n_cases, n_x) in the same order as CASES. Returns per-case metrics and summary."""
    rows = []
    for n, (i, j, k) in enumerate(CASES):
        vp, ve, v = V_PEN[i, j, k], V_EX[i, j, k], v_pred[n]
        rows.append(dict(sigma=SIGMAS[i], r=RATES[j], tau=TAUS[k],
                         rel_pen=np.linalg.norm(v - vp) / np.linalg.norm(vp),
                         max_pen=np.max(np.abs(v - vp)) * KP,
                         rel_ex=np.linalg.norm(v - ve) / np.linalg.norm(ve),
                         max_ex=np.max(np.abs(v - ve)) * KP,
                         price_err=abs(v[len(XS) // 2] - ve[len(XS) // 2]) * KP))
    summ = dict(mean_rel_pen=np.mean([c["rel_pen"] for c in rows]), worst_rel_pen=np.max([c["rel_pen"] for c in rows]),
                mean_rel_ex=np.mean([c["rel_ex"] for c in rows]), worst_max_pen=np.max([c["max_pen"] for c in rows]),
                mean_price_err=np.mean([c["price_err"] for c in rows]))
    return rows, summ


def predict_cases(model):
    nx = len(XS)
    x = torch.tensor(np.tile(XS, len(CASES)), dtype=torch.float32).reshape(-1, 1)
    tau = torch.tensor(np.repeat([TAUS[k] for (_, _, k) in CASES], nx), dtype=torch.float32).reshape(-1, 1)
    sig = torch.tensor(np.repeat([SIGMAS[i] for (i, _, _) in CASES], nx), dtype=torch.float32).reshape(-1, 1)
    rate = torch.tensor(np.repeat([RATES[j] for (_, j, _) in CASES], nx), dtype=torch.float32).reshape(-1, 1)
    with torch.no_grad():
        return model(x, tau, sig, rate).numpy().reshape(len(CASES), nx)


# ---------------- training ----------------
if __name__ == "__main__":
    model = PINN()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"MODE={MODE} N={NF if MODE != 'plain' else '-'} SEED={SEED} | inputs {N_IN} | parameters {n_params} | steps {ITERS}")
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=ITERS, eta_min=1e-6)
    t0 = time.time()
    every = max(1, ITERS // 15)
    for it in range(1, ITERS + 1):
        rho = RHO_START + (RHO_FINAL - RHO_START) * min(1.0, it / (0.5 * ITERS))
        opt.zero_grad()
        loss = compute_loss(model, rho)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        opt.step()
        sched.step()
        if it % every == 0 or it == 1:
            _, s = compute_metrics(predict_cases(model))
            print(f"it {it:6d} | rho {rho:5.1f} | loss {loss.item():.2e} | mean rel L2 vs penalized {s['mean_rel_pen']:.2e}"
                  f" | worst case {s['worst_rel_pen']:.2e} | {time.time() - t0:.0f}s")
    secs = time.time() - t0

    cases, s = compute_metrics(predict_cases(model))
    print(f"\nFINAL {MODE} N={NF} seed={SEED} | mean rel L2 vs penalized {s['mean_rel_pen']:.3e} | worst case {s['worst_rel_pen']:.3e}"
          f" | mean rel L2 vs exact {s['mean_rel_ex']:.3e} | worst max err {s['worst_max_pen']:.3f} | mean price err S=100 {s['mean_price_err']:.3f} | {secs:.0f}s")

    nN = NF if MODE != "plain" else 0
    row = dict(mode=MODE, N=nN, seed=SEED, params=n_params, iters=ITERS, **{k: float(v) for k, v in s.items()}, seconds=round(secs))
    for fname, rows_, extra in (("results_param.csv", [row], {}),
                                ("results_param_cases.csv", cases, dict(mode=MODE, N=nN, seed=SEED))):
        new = not os.path.exists(fname)
        with open(fname, "a", newline="") as f:
            data = [{**extra, **{k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in r_.items()}} for r_ in rows_]
            w = csv.DictWriter(f, fieldnames=list(data[0].keys()))
            if new:
                w.writeheader()
            w.writerows(data)
    torch.save(model.state_dict(), f"param_{MODE}_N{NF}_seed{SEED}.pt")

    # figure: heat map of the error at tau = 1 over (sigma, r), and the worst case
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    k1 = list(TAUS).index(1.0)
    grid = np.array([[next(c["rel_pen"] for c, (i2, j2, k2) in zip(cases, CASES) if (i2, j2, k2) == (i, j, k1))
                      for j in range(len(RATES))] for i in range(len(SIGMAS))]) * 100
    im = ax[0].imshow(grid, origin="lower", aspect="auto", cmap="viridis")
    ax[0].set_xticks(range(len(RATES))); ax[0].set_xticklabels([f"{r:.2f}" for r in RATES])
    ax[0].set_yticks(range(len(SIGMAS))); ax[0].set_yticklabels([f"{s_:.3f}" for s_ in SIGMAS])
    ax[0].set_xlabel("r"); ax[0].set_ylabel("sigma"); ax[0].set_title("rel L2 error vs penalized reference (%), tau = 1")
    for i in range(len(SIGMAS)):
        for j in range(len(RATES)):
            ax[0].text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center", color="w", fontsize=8)
    plt.colorbar(im, ax=ax[0])
    nw = int(np.argmax([c["rel_pen"] for c in cases]))
    i, j, k = CASES[nw]
    vpred = predict_cases(model)[nw]
    ax[1].plot(XS * KP, V_EX[i, j, k] * KP, "k-", label="exact reference")
    ax[1].plot(XS * KP, V_PEN[i, j, k] * KP, "g:", label="penalized reference")
    ax[1].plot(XS * KP, vpred * KP, "r--", label=f"PINN ({MODE})")
    ax[1].set_title(f"worst case: sigma={SIGMAS[i]:.3f}, r={RATES[j]:.2f}, tau={TAUS[k]}")
    ax[1].set_xlabel("S"); ax[1].set_ylabel("American put value"); ax[1].legend(); ax[1].grid(alpha=.3)
    plt.tight_layout(); plt.savefig(f"param_{MODE}_N{NF}_seed{SEED}.png", dpi=150); plt.show()
