"""
Reference solutions for the EUR/USD-focused parametric experiment (run ONCE; about 15-25 minutes).

    python make_refs_eur.py        ->  param_refs_eur.npz

Test grid (all combinations): sigma in {0.03,0.045,0.06,0.08,0.10,0.125,0.15}, r in {0,0.03,0.06},
q in {-0.01,0.02,0.05}, tau in {0.25,0.5,1.0} -> 7*3*3*3 = 189 cases, on S = 50..150 (x = S/K).
r = domestic (USD) rate, q = foreign (EUR) rate (Garman-Kohlhagen).
Stored (values scaled by the strike):
  v_ex       : exact American constraint (Crank-Nicolson + PSOR, grid 800 x 400T)
  v_pen_30, v_pen_100, v_pen_300 : classical solutions of the PENALIZED equation for rho = 30, 100, 300
               (grid 2400 x 2400T). The PINN is compared with the one that has its own rho.
"""
import time
import numpy as np
from reference_solver import american_put, penalized_american_put

K = 100.0
RHOS = (30, 100, 300)
SIGMAS = np.array([0.03, 0.045, 0.06, 0.08, 0.10, 0.125, 0.15])
RATES = np.array([0.0, 0.03, 0.06])
QS = np.array([-0.01, 0.02, 0.05])
TAUS = np.array([0.25, 0.5, 1.0])
XS = np.linspace(0.5, 1.5, 101)


def one_case(args):
    i, j, l, k = args
    s, r, q, T = SIGMAS[i], RATES[j], QS[l], TAUS[k]
    pens = []
    for rho in RHOS:
        S, V = penalized_american_put(K, r, s, T, q=q, rho=float(rho), S_max=300.0, M=2400, N=int(2400 * T))
        pens.append(np.interp(XS * K, S, V) / K)
    S, V, _ = american_put(K, r, s, T, q=q, S_max=300.0, M=800, N=int(400 * T))
    return (i, j, l, k), pens, np.interp(XS * K, S, V) / K


if __name__ == "__main__":
    import multiprocessing as mp
    shape = (len(SIGMAS), len(RATES), len(QS), len(TAUS), len(XS))
    v_pen = {rho: np.zeros(shape) for rho in RHOS}
    v_ex = np.zeros(shape)
    jobs = [(i, j, l, k) for i in range(len(SIGMAS)) for j in range(len(RATES)) for l in range(len(QS)) for k in range(len(TAUS))]
    t0 = time.time()
    with mp.Pool(max(1, mp.cpu_count() - 1)) as pool:
        for n, (idx, pens, ve) in enumerate(pool.imap_unordered(one_case, jobs), start=1):
            v_ex[idx] = ve
            for rho, vp in zip(RHOS, pens):
                v_pen[rho][idx] = vp
            if n % 20 == 0:
                print(f"{n}/{len(jobs)} cases ({time.time() - t0:.0f}s)", flush=True)
    np.savez("param_refs_eur.npz", sigmas=SIGMAS, rates=RATES, qs=QS, taus=TAUS, xs=XS, v_ex=v_ex,
             **{f"v_pen_{rho}": v_pen[rho] for rho in RHOS})
    for rho in RHOS:
        print(f"rho={rho}: max |penalized - exact| = {(np.abs(v_pen[rho] - v_ex).max() * K):.3f} price units")
    print("saved param_refs_eur.npz")
