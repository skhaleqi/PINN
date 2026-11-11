"""
FX-oriented PARAMETRIC American-put PINN, PREMIUM version (Garman-Kohlhagen style).

NEW vs pinn_param_fx.py: the network does NOT learn the whole price. The price is split as
      v = v_euro(x, tau, sigma, r, q) + U * w(x, tau, sigma, r, q)
where v_euro is the EXACT European (Garman-Kohlhagen) formula and the network w learns only the small
early-exercise premium (U = 0.01 just scales it to order 1). Reason: with seed 0 the plain direct PINN had a mean
ATM error of 0.5-1.0 price units, while the European formula alone (no network) has 0.107, so the network was wasting
its capacity on re-learning the European price. Because v_euro solves the Black-Scholes equation exactly, the PDE for
the premium is the same operator with the penalty evaluated on the total price:
      u_tau = L u + rho * max(g - v_euro - u, 0),   u = U*w,   u(tau=0) = 0,
      u(x=0) = 1 - v_euro(0, tau),   u(x=X_MAX) = -v_euro(X_MAX, tau).

ONE network with five inputs  (x = S/K, tau, sigma, r, q)  returns v = V/K for all volatilities,
domestic rates r and foreign rates / dividend yields q in the training ranges.
   EUR/USD put: r = USD rate, q = EUR rate.   Dow Jones: q = dividend yield.   Gold: q close to 0.

    python pinn_param_fx_prem.py MODE SEED [N] [ITERS]

    MODE = plain   : the 5 inputs only                                        (baseline)
           cheb    : Chebyshev polynomials T_1..T_N of each of the 5 inputs
           fourier : N sine/cosine terms of each of the 5 inputs (CONTROL, same input count)
    SEED = random seed      N = features per input (default 8)
    ITERS = training steps (default 30000). Smoke test: python pinn_param_fx_prem.py plain 0 8 300

Differences from pinn_param_exp.py (identical for all three modes):
  * sigma range 0.03 - 0.25 (EUR/USD realized vol in your data was 3-14%), sampled LOG-uniformly and
    fed to the network as ln(sigma), so low volatilities get more resolution and more training points;
  * q is a new input; drift is (r - q);
  * half of the x samples are concentrated in [0.4, 1.6] (around the strike, where the payoff has its kink).
Needs param_refs_fx.npz (made once by make_param_refs_fx.py) in the same folder.
Penalty method as before: v_tau = L v + rho*max(g - v, 0),  L v = 0.5 sigma^2 x^2 v_xx + (r - q) x v_x - r v.
Outputs: results_fxp.csv (1 line per run), results_fxp_cases.csv (162 lines per run), a figure, and the weights (.pt).
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
SIG_LO, SIG_HI = 0.03, 0.25
R_LO, R_HI = 0.0, 0.06
Q_LO, Q_HI = -0.01, 0.05
N_PDE, N_IC, N_BC = 3000, 800, 800
W_IC, W_BC = 10.0, 10.0
RHO_START, RHO_FINAL = 5.0, 30.0
CLIP = 1.0
HIDDEN, LAYERS = 96, 4
U_SCALE = 0.01                 # size of the early-exercise premium (price/strike), only a scaling
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


N_VARS = 5
N_IN = N_VARS if MODE == "plain" else N_VARS * NF


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

    def forward(self, x, tau, sig, rate, q):
        z = [scale(x, 0.0, X_MAX), scale(tau, 0.0, T), scale(torch.log(sig), np.log(SIG_LO), np.log(SIG_HI)),
             scale(rate, R_LO, R_HI), scale(q, Q_LO, Q_HI)]
        return self.net(torch.cat([expand(c) for c in z], dim=1))


def payoff(x):
    return torch.clamp(1 - x, min=0.0)


def euro_put(x, tau, sig, rate, q):
    """Exact European put (Garman-Kohlhagen), strike 1. Values only (no gradients needed)."""
    x, tau = x.detach(), tau.detach()
    t = torch.clamp(tau, min=1e-8)
    xx = torch.clamp(x, min=1e-8)
    sq = sig * torch.sqrt(t)
    d1 = (torch.log(xx) + (rate - q + 0.5 * sig**2) * t) / sq
    d2 = d1 - sq
    return torch.exp(-rate * t) * torch.special.ndtr(-d2) - xx * torch.exp(-q * t) * torch.special.ndtr(-d1)


# ---------------- sampling ----------------
def sample_x(n):
    """Half uniform on [0, X_MAX], half uniform on [0.4, 1.6] (around the strike)."""
    wide = torch.rand(n, 1) * X_MAX
    near = 0.4 + 1.2 * torch.rand(n, 1)
    return torch.where(torch.rand(n, 1) < 0.5, wide, near)


def rand_params(n):
    sig = torch.exp(np.log(SIG_LO) + torch.rand(n, 1) * (np.log(SIG_HI) - np.log(SIG_LO)))
    return sig, R_LO + torch.rand(n, 1) * (R_HI - R_LO), Q_LO + torch.rand(n, 1) * (Q_HI - Q_LO)


# ---------------- physics ----------------
def pde_residual(model, x, tau, sig, rate, q, rho):
    v = model(x, tau, sig, rate, q)
    ones = torch.ones_like(v)
    v_tau = torch.autograd.grad(v, tau, ones, create_graph=True)[0]
    v_x = torch.autograd.grad(v, x, ones, create_graph=True)[0]
    v_xx = torch.autograd.grad(v_x, x, torch.ones_like(v_x), create_graph=True)[0]
    Lv = 0.5 * sig**2 * x**2 * v_xx + (rate - q) * x * v_x - rate * v
    # v is the scaled premium w; the penalty looks at the TOTAL price v_euro + U*w
    deficit = torch.relu(payoff(x) - euro_put(x, tau, sig, rate, q) - U_SCALE * v)
    return v_tau - Lv - (rho / U_SCALE) * deficit


def compute_loss(model, rho):
    x = sample_x(N_PDE).requires_grad_(True)
    tau = (torch.rand(N_PDE, 1) * T).requires_grad_(True)
    sig, rate, q = rand_params(N_PDE)
    loss_pde = (pde_residual(model, x, tau, sig, rate, q, rho) ** 2).mean()

    x0 = sample_x(N_IC)
    s0, r0, q0 = rand_params(N_IC)
    loss_ic = (model(x0, torch.zeros_like(x0), s0, r0, q0) ** 2).mean()          # premium is 0 at maturity

    tb = torch.rand(N_BC, 1) * T
    sb, rb, qb = rand_params(N_BC)
    x_l, x_r = torch.zeros_like(tb), torch.full_like(tb, X_MAX)
    tgt_l = (1.0 - euro_put(x_l, tb, sb, rb, qb)) / U_SCALE
    tgt_r = (0.0 - euro_put(x_r, tb, sb, rb, qb)) / U_SCALE
    loss_bc = ((model(x_l, tb, sb, rb, qb) - tgt_l) ** 2).mean() + ((model(x_r, tb, sb, rb, qb) - tgt_r) ** 2).mean()
    return loss_pde + W_IC * loss_ic + W_BC * loss_bc


# ---------------- reference solutions and metrics ----------------
if not os.path.exists("param_refs_fx.npz"):
    sys.exit("param_refs_fx.npz not found. Put it in this folder (or run: python make_param_refs_fx.py).")
REF = np.load("param_refs_fx.npz")
SIGMAS, RATES, QS, TAUS, XS = REF["sigmas"], REF["rates"], REF["qs"], REF["taus"], REF["xs"]
V_PEN, V_EX = REF["v_pen"], REF["v_ex"]            # shape (n_sigma, n_r, n_q, n_tau, n_x)
CASES = [(i, j, l, k) for i in range(len(SIGMAS)) for j in range(len(RATES))
         for l in range(len(QS)) for k in range(len(TAUS))]


def compute_metrics(v_pred):
    """v_pred: array (n_cases, n_x) in the order of CASES. Returns per-case metrics and a summary."""
    rows = []
    mid = len(XS) // 2
    for n, (i, j, l, k) in enumerate(CASES):
        vp, ve, v = V_PEN[i, j, l, k], V_EX[i, j, l, k], v_pred[n]
        rows.append(dict(sigma=SIGMAS[i], r=RATES[j], q=QS[l], tau=TAUS[k],
                         rel_pen=np.linalg.norm(v - vp) / np.linalg.norm(vp),
                         max_pen=np.max(np.abs(v - vp)) * KP,
                         rel_ex=np.linalg.norm(v - ve) / np.linalg.norm(ve),
                         max_ex=np.max(np.abs(v - ve)) * KP,
                         price_err=abs(v[mid] - ve[mid]) * KP,
                         price_ex=ve[mid] * KP))
    summ = dict(mean_rel_pen=np.mean([c["rel_pen"] for c in rows]), worst_rel_pen=np.max([c["rel_pen"] for c in rows]),
                mean_rel_ex=np.mean([c["rel_ex"] for c in rows]), worst_max_pen=np.max([c["max_pen"] for c in rows]),
                mean_price_err=np.mean([c["price_err"] for c in rows]),
                worst_price_err=np.max([c["price_err"] for c in rows]))
    return rows, summ


def predict_cases(model):
    nx = len(XS)
    x = torch.tensor(np.tile(XS, len(CASES)), dtype=torch.float32).reshape(-1, 1)

    def rep(vals):
        return torch.tensor(np.repeat(vals, nx), dtype=torch.float32).reshape(-1, 1)
    tau = rep([TAUS[k] for (_, _, _, k) in CASES])
    sig = rep([SIGMAS[i] for (i, _, _, _) in CASES])
    rate = rep([RATES[j] for (_, j, _, _) in CASES])
    q = rep([QS[l] for (_, _, l, _) in CASES])
    with torch.no_grad():
        total = euro_put(x, tau, sig, rate, q) + U_SCALE * model(x, tau, sig, rate, q)
        return total.numpy().reshape(len(CASES), nx)


# ---------------- training ----------------
if __name__ == "__main__":
    model = PINN()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"MODE={MODE} N={NF if MODE != 'plain' else '-'} SEED={SEED} | inputs {N_IN} | parameters {n_params} | steps {ITERS} | test cases {len(CASES)}")
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
                  f" | mean price err S=100 {s['mean_price_err']:.3f} | worst {s['worst_price_err']:.3f} | {time.time() - t0:.0f}s")
    secs = time.time() - t0

    cases, s = compute_metrics(predict_cases(model))
    print(f"\nFINAL FX-PREMIUM {MODE} N={NF} seed={SEED} | mean rel L2 vs penalized {s['mean_rel_pen']:.3e} | worst {s['worst_rel_pen']:.3e}"
          f" | mean rel L2 vs exact {s['mean_rel_ex']:.3e} | worst max err {s['worst_max_pen']:.3f}"
          f" | mean price err S=100 {s['mean_price_err']:.3f} (worst {s['worst_price_err']:.3f}) | {secs:.0f}s")

    nN = NF if MODE != "plain" else 0
    row = dict(mode=MODE, N=nN, seed=SEED, params=n_params, iters=ITERS, **{k: float(v) for k, v in s.items()}, seconds=round(secs))
    for fname, rows_, extra in (("results_fxp.csv", [row], {}),
                                ("results_fxp_cases.csv", cases, dict(mode=MODE, N=nN, seed=SEED, iters=ITERS))):
        new = not os.path.exists(fname)
        with open(fname, "a", newline="") as f:
            data = [{**extra, **{k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in r_.items()}} for r_ in rows_]
            w = csv.DictWriter(f, fieldnames=list(data[0].keys()))
            if new:
                w.writeheader()
            w.writerows(data)
    torch.save(dict(state_dict=model.state_dict(), mode=MODE, N=NF, hidden=HIDDEN, layers=LAYERS,
                    u_scale=U_SCALE, formulation="european+premium", ranges=dict(sigma=(SIG_LO, SIG_HI), r=(R_LO, R_HI), q=(Q_LO, Q_HI), x_max=X_MAX, T=T)),
               f"param_fxp_{MODE}_N{NF}_seed{SEED}.pt")

    # figure: mean abs price error at S=100 by (sigma, r), and the worst case
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    grid = np.zeros((len(SIGMAS), len(RATES)))
    for c, (i, j, l, k) in zip(cases, CASES):
        grid[i, j] += c["price_err"] / (len(QS) * len(TAUS))
    im = ax[0].imshow(grid, origin="lower", aspect="auto", cmap="viridis")
    ax[0].set_xticks(range(len(RATES))); ax[0].set_xticklabels([f"{r:.2f}" for r in RATES])
    ax[0].set_yticks(range(len(SIGMAS))); ax[0].set_yticklabels([f"{s_:.3f}" for s_ in SIGMAS])
    ax[0].set_xlabel("r"); ax[0].set_ylabel("sigma"); ax[0].set_title("mean abs price error at S=100 (price units, strike=100)")
    for i in range(len(SIGMAS)):
        for j in range(len(RATES)):
            ax[0].text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center", color="w", fontsize=8)
    plt.colorbar(im, ax=ax[0])
    nw = int(np.argmax([c["price_err"] for c in cases]))
    i, j, l, k = CASES[nw]
    vpred = predict_cases(model)[nw]
    ax[1].plot(XS * KP, V_EX[i, j, l, k] * KP, "k-", label="exact reference")
    ax[1].plot(XS * KP, vpred * KP, "r--", label=f"PINN ({MODE})")
    ax[1].set_title(f"worst case: sigma={SIGMAS[i]:.3f}, r={RATES[j]:.2f}, q={QS[l]:.2f}, tau={TAUS[k]}")
    ax[1].set_xlabel("S"); ax[1].set_ylabel("American put value"); ax[1].legend(); ax[1].grid(alpha=.3)
    plt.tight_layout(); plt.savefig(f"param_fxp_{MODE}_N{NF}_seed{SEED}.png", dpi=150); plt.show()
