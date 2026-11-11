"""
Step 4 (idea B): the SAME American PINN as pinn_american_v2.py, but the network
input is expanded with CHEBYSHEV polynomials T_1..T_N of x and of tau.
Nothing else is changed, so any difference in accuracy comes from the features.

Changes vs pinn_american.py (v1):  RHO_FINAL 100 -> 30 (less stiff),
  gradient clipping, learning rate decays smoothly to ~0 (cosine) so the
  final snapshot is not a random spike.

What is new compared with pinn_european.py (only 3 things):
  1. PENALTY term in the PDE:   v_tau = L v + rho * max(g - v, 0)
        L v = 0.5*sigma^2*x^2*v_xx + (r-q)*x*v_x - r*v      (same operator as before)
        g   = max(1 - x, 0)                                  (payoff)
     Where the option value v falls below the payoff g, the penalty pushes it
     back up. A bigger rho enforces "v >= payoff" more strictly.
     (A classical-solver test showed the bias of this trick at S=100 is about
      -0.11 for rho=10, -0.011 for rho=100, -0.004 for rho=300, in price units.)
  2. Left boundary is v = 1 (at S = 0 the put is exercised at once: worth K).
  3. rho is increased gradually from RHO_START to RHO_FINAL (training is easier).

The reference answer comes from reference_solver.american_put (Crank-Nicolson + PSOR).

Needs: torch, numpy, scipy, matplotlib.   Run: python pinn_american.py
Keep reference_solver.py in the same folder.
"""
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from reference_solver import american_put

# ---------------- settings ----------------
K, r, sigma, T, q = 100.0, 0.05, 0.2, 1.0, 0.0
X_MAX = 3.0
ITERS = 20000
N_PDE, N_IC, N_BC = 2000, 500, 500
W_IC, W_BC = 10.0, 10.0
RHO_START, RHO_FINAL = 5.0, 30.0       # penalty strength (ramped up)
CLIP = 1.0                             # gradient clipping threshold
N_CHEB = 8                             # number of Chebyshev polynomials per variable
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0   # run: python pinn_american_v2.py 1
BOUNDARY_TOL = 1e-3                    # for the rough exercise-boundary estimate
torch.manual_seed(SEED)
np.random.seed(SEED)


def cheb(z, n):
    """Chebyshev polynomials T_1..T_n of z in [-1, 1] (recurrence T_{k+1} = 2 z T_k - T_{k-1}).
    Returns a tensor with n columns. T_1 = z is the original input, so nothing is lost."""
    T = [torch.ones_like(z), z]
    for _ in range(2, n + 1):
        T.append(2 * z * T[-1] - T[-2])
    return torch.cat(T[1:n + 1], dim=1)


# ---------------- the network ----------------
class PINN(nn.Module):
    def __init__(self, hidden=64, layers=4):
        super().__init__()
        dims = [2 * N_CHEB] + [hidden] * layers + [1]   # input size = 2 * N_CHEB
        mods = []
        for i in range(len(dims) - 1):
            mods.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                mods.append(nn.Tanh())
        self.net = nn.Sequential(*mods)

    def features(self, x, tau):
        xs, ts = 2 * x / X_MAX - 1, 2 * tau / T - 1          # rescale to [-1, 1]
        return torch.cat([cheb(xs, N_CHEB), cheb(ts, N_CHEB)], dim=1)

    def forward(self, x, tau):
        return self.net(self.features(x, tau))


def payoff(x):
    return torch.clamp(1 - x, min=0.0)


# ---------------- physics: PDE residual WITH penalty ----------------
def pde_residual(model, x, tau, rho):
    v = model(x, tau)
    ones = torch.ones_like(v)
    v_tau = torch.autograd.grad(v, tau, ones, create_graph=True)[0]
    v_x = torch.autograd.grad(v, x, ones, create_graph=True)[0]
    v_xx = torch.autograd.grad(v_x, x, torch.ones_like(v_x), create_graph=True)[0]
    Lv = 0.5 * sigma**2 * x**2 * v_xx + (r - q) * x * v_x - r * v
    penalty = rho * torch.relu(payoff(x) - v)
    return v_tau - Lv - penalty


def compute_loss(model, rho):
    x = (torch.rand(N_PDE, 1) * X_MAX).requires_grad_(True)
    tau = (torch.rand(N_PDE, 1) * T).requires_grad_(True)
    loss_pde = (pde_residual(model, x, tau, rho) ** 2).mean()

    x0 = torch.rand(N_IC, 1) * X_MAX
    v0 = model(x0, torch.zeros_like(x0))
    loss_ic = ((v0 - payoff(x0)) ** 2).mean()

    tb = torch.rand(N_BC, 1) * T
    v_left = model(torch.zeros_like(tb), tb)                  # American: v(0, tau) = 1
    v_right = model(torch.full_like(tb, X_MAX), tb)           # v(X_MAX, tau) = 0
    loss_bc = ((v_left - 1.0) ** 2).mean() + (v_right ** 2).mean()

    total = loss_pde + W_IC * loss_ic + W_BC * loss_bc
    return total, loss_pde.item(), loss_ic.item(), loss_bc.item()


# ---------------- reference solution (computed once) ----------------
print("Computing the classical reference solution ...")
S_ref, V_ref, S_star_ref = american_put(K, r, sigma, T, q=q, S_max=300.0, M=400, N=200)
price_ref_100 = float(np.interp(100.0, S_ref, V_ref))
print(f"Reference American put at S=100: {price_ref_100:.4f}")


def rough_boundary(xs, v):
    """Largest x (<=1) where v is within BOUNDARY_TOL of the payoff. Rough!
    It is applied in the same way to the reference and to the PINN, so the two
    numbers are comparable, but neither equals the exact boundary."""
    g = np.maximum(1 - xs, 0)
    idx = np.where(v - g < BOUNDARY_TOL)[0]
    return xs[idx.max()] * K if len(idx) else np.nan


def evaluate(model):
    xs = np.linspace(0.5, 1.5, 201)                          # S from 50 to 150, today
    with torch.no_grad():
        xt = torch.tensor(xs, dtype=torch.float32).reshape(-1, 1)
        v_pinn = model(xt, torch.full_like(xt, T)).numpy().ravel()
    v_ref = np.interp(xs * K, S_ref, V_ref) / K
    rel_l2 = np.linalg.norm(v_pinn - v_ref) / np.linalg.norm(v_ref)
    max_abs = np.max(np.abs(v_pinn - v_ref)) * K
    p100 = float(np.interp(1.0, xs, v_pinn)) * K
    xb = np.linspace(0.5, 1.0, 501)
    with torch.no_grad():
        xbt = torch.tensor(xb, dtype=torch.float32).reshape(-1, 1)
        vb = model(xbt, torch.full_like(xbt, T)).numpy().ravel()
    vb_ref = np.interp(xb * K, S_ref, V_ref) / K
    return dict(xs=xs, v_pinn=v_pinn, v_ref=v_ref, rel=rel_l2, mx=max_abs, p100=p100,
                s_star=rough_boundary(xb, vb), s_star_ref=rough_boundary(xb, vb_ref))


# ---------------- training ----------------
if __name__ == "__main__":
    model = PINN()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=ITERS, eta_min=1e-6)
    t0 = time.time()
    for it in range(1, ITERS + 1):
        rho = RHO_START + (RHO_FINAL - RHO_START) * min(1.0, it / (0.5 * ITERS))
        opt.zero_grad()
        loss, lp, li, lb = compute_loss(model, rho)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        opt.step()
        sched.step()
        if it % 500 == 0 or it == 1:
            e = evaluate(model)
            print(f"it {it:5d} | rho {rho:5.1f} | loss {loss.item():.2e} (pde {lp:.1e}, ic {li:.1e}, bc {lb:.1e})"
                  f" | rel L2 {e['rel']:.2e} | max err {e['mx']:.3f} | price(S=100) {e['p100']:.4f}"
                  f" | {time.time() - t0:.0f}s")

    e = evaluate(model)
    summary = (f"FINAL American CHEB{N_CHEB} | rho_final {RHO_FINAL} | seed {SEED} | rel L2 {e['rel']:.3e} | max abs err {e['mx']:.4f} | "
               f"PINN price(S=100) {e['p100']:.4f} vs reference {price_ref_100:.4f} | "
               f"rough boundary PINN {e['s_star']:.1f} vs reference {e['s_star_ref']:.1f}")
    print("\n" + summary)
    with open("results_american.txt", "a") as f:       # keeps a log of every run
        f.write(summary + "\n")

    xs = e["xs"]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(xs * K, e["v_ref"] * K, "k-", label="reference (finite diff.)")
    ax[0].plot(xs * K, e["v_pinn"] * K, "r--", label="PINN")
    ax[0].plot(xs * K, np.maximum(1 - xs, 0) * K, "g:", label="payoff")
    ax[0].set_xlabel("S"); ax[0].set_ylabel("American put value today")
    ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].plot(xs * K, (e["v_pinn"] - e["v_ref"]) * K, "b-")
    ax[1].axhline(0, color="k", lw=0.5)
    ax[1].axvline(e["s_star_ref"], color="gray", ls="--", label="rough boundary (reference)")
    ax[1].set_xlabel("S"); ax[1].set_ylabel("PINN - reference (price units)")
    ax[1].legend(); ax[1].grid(alpha=.3)
    plt.tight_layout(); plt.savefig(f"american_cheb_seed{SEED}.png", dpi=150); plt.show()
