"""
Step 2: a plain PINN for the EUROPEAN put (baseline).
Why European first?  It has an exact answer, so we can check the PINN code
is correct before adding the harder American (early-exercise) part.

Variables (scaled so the numbers are friendly):
    x   = S / K            price relative to the strike, in [0, 3]
    tau = T - t            time to maturity, in [0, T]
    v   = V / K            option value relative to the strike

PDE (Black-Scholes / Garman-Kohlhagen, written in tau):
    v_tau = 0.5*sigma^2*x^2*v_xx + (r - q)*x*v_x - r*v
Conditions:
    tau = 0 :  v = max(1 - x, 0)          (payoff at maturity)
    x   = 0 :  v = exp(-r*tau)            (put is worth the discounted strike)
    x   = 3 :  v = 0                      (put is worthless for a huge price)

Needs:  pip install torch   (CPU version is enough)
Run:    press F5 in Spyder. Keep reference_solver.py in the same folder.
"""
import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from reference_solver import european_put

# ---------------- settings ----------------
K, r, sigma, T, q = 100.0, 0.05, 0.2, 1.0, 0.0
X_MAX = 3.0
ITERS = 6000          # number of training steps (increase for better accuracy)
N_PDE, N_IC, N_BC = 2000, 500, 500   # random points used in every step
W_IC, W_BC = 10.0, 10.0              # loss weights (data-fit terms)
SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)


# ---------------- the network ----------------
class PINN(nn.Module):
    """Input (x, tau) -> output v.  A plain MLP with tanh activations."""
    def __init__(self, hidden=64, layers=4):
        super().__init__()
        dims = [2] + [hidden] * layers + [1]
        mods = []
        for i in range(len(dims) - 1):
            mods.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                mods.append(nn.Tanh())
        self.net = nn.Sequential(*mods)

    def features(self, x, tau):
        # HOOK for the thesis idea: later we replace this with orthogonal
        # (Chebyshev/Legendre) features. For now: just rescale to [-1, 1].
        return torch.cat([2 * x / X_MAX - 1, 2 * tau / T - 1], dim=1)

    def forward(self, x, tau):
        return self.net(self.features(x, tau))


# ---------------- physics: PDE residual via autograd ----------------
def pde_residual(model, x, tau):
    v = model(x, tau)
    ones = torch.ones_like(v)
    v_tau = torch.autograd.grad(v, tau, ones, create_graph=True)[0]
    v_x = torch.autograd.grad(v, x, ones, create_graph=True)[0]
    v_xx = torch.autograd.grad(v_x, x, torch.ones_like(v_x), create_graph=True)[0]
    return v_tau - (0.5 * sigma**2 * x**2 * v_xx + (r - q) * x * v_x - r * v)


def compute_loss(model):
    # 1) PDE term: random points inside the domain
    x = (torch.rand(N_PDE, 1) * X_MAX).requires_grad_(True)
    tau = (torch.rand(N_PDE, 1) * T).requires_grad_(True)
    loss_pde = (pde_residual(model, x, tau) ** 2).mean()

    # 2) initial (payoff) term at tau = 0
    x0 = torch.rand(N_IC, 1) * X_MAX
    v0 = model(x0, torch.zeros_like(x0))
    loss_ic = ((v0 - torch.clamp(1 - x0, min=0.0)) ** 2).mean()

    # 3) boundary terms at x = 0 and x = X_MAX
    tb = torch.rand(N_BC, 1) * T
    v_left = model(torch.zeros_like(tb), tb)
    v_right = model(torch.full_like(tb, X_MAX), tb)
    loss_bc = ((v_left - torch.exp(-r * tb)) ** 2).mean() + (v_right ** 2).mean()

    total = loss_pde + W_IC * loss_ic + W_BC * loss_bc
    return total, loss_pde.item(), loss_ic.item(), loss_bc.item()


# ---------------- evaluation against the exact formula ----------------
def evaluate(model):
    xs = np.linspace(0.5, 1.5, 201)                    # S from 50 to 150
    with torch.no_grad():
        xt = torch.tensor(xs, dtype=torch.float32).reshape(-1, 1)
        v_pinn = model(xt, torch.full_like(xt, T)).numpy().ravel()   # tau = T, i.e. today
    v_exact = european_put(xs * K, K, r, sigma, T, q) / K
    rel_l2 = np.linalg.norm(v_pinn - v_exact) / np.linalg.norm(v_exact)
    max_abs = np.max(np.abs(v_pinn - v_exact)) * K      # in price units
    i100 = np.argmin(abs(xs - 1.0))
    return xs, v_pinn, v_exact, rel_l2, max_abs, v_pinn[i100] * K


# ---------------- training ----------------
if __name__ == "__main__":
    model = PINN()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=2000, gamma=0.5)
    t0 = time.time()
    for it in range(1, ITERS + 1):
        opt.zero_grad()
        loss, lp, li, lb = compute_loss(model)
        loss.backward()
        opt.step()
        sched.step()
        if it % 500 == 0 or it == 1:
            _, _, _, rel, mx, p100 = evaluate(model)
            print(f"it {it:5d} | loss {loss.item():.2e} (pde {lp:.1e}, ic {li:.1e}, bc {lb:.1e})"
                  f" | rel L2 err {rel:.2e} | max err {mx:.4f} | price(S=100) {p100:.4f}"
                  f" | {time.time() - t0:.0f}s")

    xs, v_pinn, v_exact, rel, mx, p100 = evaluate(model)
    print("\nFINAL  relative L2 error:", f"{rel:.3e}", "| max abs error:", f"{mx:.4f}",
          "| PINN price at S=100:", f"{p100:.4f}", "| exact:", f"{european_put(100, K, r, sigma, T, q):.4f}")

    plt.figure(figsize=(7, 4))
    plt.plot(xs * K, v_exact * K, "k-", label="exact")
    plt.plot(xs * K, v_pinn * K, "r--", label="PINN")
    plt.xlabel("S"); plt.ylabel("European put value today"); plt.legend(); plt.grid(alpha=.3)
    plt.tight_layout(); plt.show()
