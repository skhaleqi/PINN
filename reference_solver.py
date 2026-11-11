"""
Reference ("ground truth") solver for put options with a continuous
foreign/dividend rate q  (q = foreign interest rate for FX options,
i.e. the Garman-Kohlhagen model; q = 0 gives plain Black-Scholes).

  european_put : closed-form formula (exact)
  american_put : Crank-Nicolson finite differences + PSOR
                 (projected successive over-relaxation) for the early
                 exercise constraint  V >= max(K - S, 0).
Every PINN result in the thesis is compared against these two.
"""
import numpy as np
from math import log, sqrt, exp
from scipy.stats import norm

def european_put(S, K, r, sigma, T, q=0.0):
    S = np.asarray(S, dtype=float)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    return K * exp(-r * T) * norm.cdf(-d2) - S * exp(-q * T) * norm.cdf(-d1)

def american_put(K, r, sigma, T, q=0.0, S_max=None, M=400, N=400,
                 omega=1.3, tol=1e-9, max_iter=2000):
    """Returns S grid, V(S, t=0) and the early-exercise boundary at t=0."""
    S_max = S_max or 4.0 * K
    dS, dt = S_max / M, T / N
    S = np.linspace(0, S_max, M + 1)
    i = np.arange(1, M)
    a = 0.25 * dt * (sigma**2 * i**2 - (r - q) * i)      # lower diag
    b = -0.5 * dt * (sigma**2 * i**2 + r)                # main diag
    c = 0.25 * dt * (sigma**2 * i**2 + (r - q) * i)      # upper diag
    payoff = np.maximum(K - S, 0.0)
    V = payoff.copy()                                     # value at t = T
    for n in range(N):                                    # march backwards in time
        tau = (n + 1) * dt                                # time to maturity
        V_old = V.copy()
        # right-hand side of the Crank-Nicolson scheme
        rhs = (a * V_old[:-2] + (1 + b) * V_old[1:-1] + c * V_old[2:])
        V[0] = K * exp(-r * tau)                          # boundary at S = 0
        V[-1] = 0.0                                       # boundary at S = S_max
        rhs[0] += a[0] * V[0]
        rhs[-1] += c[-1] * V[-1]
        # PSOR: solve (I - A) V_new = rhs  with  V_new >= payoff
        x = np.maximum(V_old[1:-1], payoff[1:-1])
        for _ in range(max_iter):
            err = 0.0
            for j in range(M - 1):
                left = x[j - 1] if j > 0 else V[0]
                right = x[j + 1] if j < M - 2 else V[-1]
                y = (rhs[j] + a[j] * left + c[j] * right) / (1 - b[j])
                new = max(payoff[j + 1], x[j] + omega * (y - x[j]))
                err = max(err, abs(new - x[j]))
                x[j] = new
            if err < tol:
                break
        V[1:-1] = x
    # early-exercise boundary: largest S where V is (numerically) equal to payoff
    ex = np.where((V - payoff < 1e-6) & (payoff > 0))[0]
    S_star = S[ex.max()] if len(ex) else np.nan
    return S, V, S_star

def penalized_american_put(K, r, sigma, T, q=0.0, rho=30.0, S_max=None, M=600, N=1200):
    """The SAME penalized equation the PINN solves:  v_tau = L v + rho*max(g - v, 0),
    solved by implicit Euler + active-set iterations (classical, no neural network).
    Comparing a PINN with THIS reference isolates the network error; comparing it with
    american_put() additionally includes the bias of the penalty method itself.
    Returns the S grid and V(S, today)."""
    from scipy.linalg import solve_banded
    S_max = S_max or 4.0 * K
    x = np.linspace(0, S_max / K, M + 1)
    dx, dt = x[1] - x[0], T / N
    g = np.maximum(1 - x, 0)
    v = g.copy()
    xi = x[1:-1]
    lo = 0.5 * sigma**2 * xi**2 / dx**2 - 0.5 * (r - q) * xi / dx
    mid = -sigma**2 * xi**2 / dx**2 - r
    up = 0.5 * sigma**2 * xi**2 / dx**2 + 0.5 * (r - q) * xi / dx
    for _ in range(N):
        v_old = v.copy()
        v_new = v_old.copy()
        v_new[0], v_new[-1] = 1.0, 0.0
        active = np.zeros(M - 1, dtype=bool)
        for _ in range(50):
            D = rho * active
            ab = np.zeros((3, M - 1))
            ab[0, 1:] = -dt * up[:-1]
            ab[1] = 1 - dt * mid + dt * D
            ab[2, :-1] = -dt * lo[1:]
            rhs = v_old[1:-1] + dt * D * g[1:-1]
            rhs[0] += dt * lo[0] * v_new[0]
            rhs[-1] += dt * up[-1] * v_new[-1]
            sol = solve_banded((1, 1), ab, rhs)
            new_active = sol < g[1:-1]
            v_new[1:-1] = sol
            if (new_active == active).all():
                break
            active = new_active
        v = v_new
    return x * K, v * K

if __name__ == "__main__":
    K, r, sigma, T = 100.0, 0.05, 0.2, 1.0
    S, V, S_star = american_put(K, r, sigma, T, S_max=300.0, M=400, N=200)
    print("European put  S=100 :", round(float(european_put(100, K, r, sigma, T)), 4),
          "(exact, expected 5.5735)")
    print("American put  S=100 :", round(float(np.interp(100, S, V)), 4),
          "(expected about 6.09)")
    print("Early-exercise boundary at t=0 : S* ~", round(float(S_star), 1),
          "(expected about 80-82)")
