"""
Speed / accuracy benchmark for the paper: one trained PINN versus classical finite-difference solvers.

    python benchmark_speed.py eur2_fourier_N8_rho30_seed0_domrect.pt          (run AFTER batch 2; needs pinn_eur2.py, reference_solver.py, param_refs_eur.npz)

Part A  PINN: time to price B contracts at once (each contract = a whole price curve on 101 values of S), B = 1, 100, 10000.
Part B  finite differences, one contract at a time: a fast grid, a fine penalized grid, and the PSOR solver of the reference.
Part C  accuracy at S = 100 on 10 random contracts (sigma, r, q, tau drawn from the training domain) against a very fine PSOR
        reference (grid 800), so the table shows error AND time for every method.
Writes speed_summary.txt (send me that small file). Times are on THIS machine, single run of each setting, so treat them as indicative.
"""
import io
import re
import sys
import time
import numpy as np
import torch
from reference_solver import penalized_american_put, american_put

K = 100.0
path = sys.argv[1]
ck = torch.load(path, map_location="cpu", weights_only=False)
mode, N, rho = ck["mode"], int(ck["N"]), int(round(float(ck["rho"])))
seed = int(re.search(r"seed(\d+)", path).group(1))
domain, stretch, iters = ck.get("domain", "band"), int(ck.get("stretch", 0)), int(ck.get("iters", 30000))   # the model must be rebuilt exactly as it was trained
sys.argv = ["pinn_eur2.py", mode, str(seed), str(N), str(iters), str(rho), domain, str(stretch)]
import pinn_eur2 as m
model = m.PINN()
model.load_state_dict(ck["state_dict"])
model.eval()

out = io.StringIO()
def p(*a):
    print(*a); print(*a, file=out)


def pinn_price(sig, rate, q, tau):
    """Prices B contracts on the 101-point S grid in one pass. Inputs are torch tensors of shape (B,1)."""
    B, nx = sig.shape[0], len(m.XS)
    x = torch.tensor(np.tile(m.XS, B), dtype=torch.float32).reshape(-1, 1)
    rep = lambda t: t.repeat_interleave(nx, dim=0)
    with torch.no_grad():
        net = model(x, rep(tau), rep(sig), rep(rate), rep(q))
        tot = net if m.DIRECT else m.euro_put(x, rep(tau), rep(sig), rep(rate), rep(q)) + m.U_SCALE * net
    return np.maximum(tot.numpy().reshape(B, nx), m.PAYOFF_GRID)


def random_contracts(n, rng):
    sig = np.exp(rng.uniform(np.log(m.SIG_LO), np.log(m.SIG_HI), n))
    r = rng.uniform(m.R_LO, m.R_HI, n)
    q = np.empty(n)
    for i in range(n):                                   # realistic carry only, as in training
        while True:
            qi = rng.uniform(m.Q_LO, m.Q_HI)
            if m.CARRY_LO <= r[i] - qi <= m.CARRY_HI:
                q[i] = qi
                break
    tau = rng.uniform(0.1, 1.0, n)
    return sig, r, q, tau


def fd_time(fn, contracts, reps=1):
    t0 = time.perf_counter()
    res = [fn(*c) for c in contracts]
    return (time.perf_counter() - t0) / len(contracts) * 1000, res


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    p(f"Model: {path} (mode={mode}, N={N}, rho={rho}); torch threads: {torch.get_num_threads()}")
    p("\nPart A - PINN, B contracts per call (each a full price curve)")
    for B in (1, 100, 10000):
        s, r, q, t = random_contracts(B, rng)
        args = [torch.tensor(a, dtype=torch.float32).reshape(-1, 1) for a in (s, r, q, t)]
        pinn_price(*args)                                  # warm-up
        times = []
        for _ in range(5):
            t0 = time.perf_counter(); pinn_price(*args); times.append(time.perf_counter() - t0)
        med = float(np.median(times))
        p(f"  B={B:6d}: {med * 1000:9.2f} ms per call  ->  {med / B * 1e6:9.1f} microseconds per contract")

    p("\nPart B/C - finite differences one contract at a time, and accuracy at S=100 (10 random contracts)")
    s, r, q, t = random_contracts(10, rng)
    cons = list(zip(s, r, q, t))
    ref = []
    for (sg, rr, qq, tt) in cons:
        S, V, _ = american_put(K, rr, sg, tt, q=qq, S_max=300.0, M=800, N=int(400 * tt))
        ref.append(float(np.interp(100.0, S, V)))
    ref = np.array(ref)
    pinn_p = pinn_price(*[torch.tensor(a, dtype=torch.float32).reshape(-1, 1) for a in (s, r, q, t)])[:, len(m.XS) // 2] * K
    t0 = time.perf_counter()
    for _ in range(20):
        pinn_price(*[torch.tensor(a[:1], dtype=torch.float32).reshape(-1, 1) for a in (s, r, q, t)])
    pinn_single_ms = (time.perf_counter() - t0) / 20 * 1000
    rows = [("PINN (single contract per call)", pinn_single_ms, np.abs(pinn_p - ref).mean())]
    settings = [("FD penalized, fast grid (M=200, N=200T)", lambda a, b, c, d: penalized_american_put(K, b, a, d, q=c, rho=30.0, S_max=300.0, M=200, N=max(10, int(200 * d))), 1),
                ("FD penalized, fine grid (M=600, N=1200T)", lambda a, b, c, d: penalized_american_put(K, b, a, d, q=c, rho=30.0, S_max=300.0, M=600, N=int(1200 * d)), 1),
                ("PSOR exact constraint (M=400, N=200T)", lambda a, b, c, d: american_put(K, b, a, d, q=c, S_max=300.0, M=400, N=int(200 * d)), 0)]
    for name, fn, penal in settings:
        ms, res = fd_time(fn, cons)
        pr = np.array([float(np.interp(100.0, x_[0], x_[1])) for x_ in res])
        rows.append((name, ms, np.abs(pr - ref).mean()))
    p(f"  {'method':46s} {'ms per contract':>16s} {'mean abs error at S=100':>26s}")
    for name, ms, err in rows:
        p(f"  {name:46s} {ms:16.2f} {err:26.4f}")
    p("\nNote: the PINN is evaluated in a single batch in part A; for a handful of contracts finite differences are competitive, "
      "the PINN advantage appears when thousands of prices (a volatility surface, Monte Carlo of scenarios, an LSTM stream) are needed.")
    open("speed_summary.txt", "w").write(out.getvalue())
    print("\nSaved speed_summary.txt")
