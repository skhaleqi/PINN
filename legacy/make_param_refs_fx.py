"""
Reference solutions for the FX-oriented parametric experiment (inputs sigma, r, q).
Run ONCE (numpy/scipy only, uses all CPU cores, about 10-20 minutes):   python make_param_refs_fx.py      -> param_refs_fx.npz

Test grid (all combinations):  sigma in {0.03,0.05,0.075,0.10,0.15,0.25}, r in {0,0.03,0.06},
q in {-0.01,0.02,0.05}, tau in {0.25,0.5,1.0}  ->  6*3*3*3 = 162 cases, on S = 50..150 (x = S/K).
r = domestic (USD) rate, q = foreign (EUR) rate or dividend yield (Garman-Kohlhagen).
v_pen: classical solution of the penalized equation (same rho as the PINN) -> isolates network error
v_ex : classical solution with the exact American constraint (PSOR)        -> true answer
Values are scaled by the strike (v = V/K).
Grids are fine (exact: 800 x 400T, penalized: 2400 x 2400T) because at low sigma a coarse grid is visibly wrong
(at sigma = 0.03 the price at S=100 changed by 0.04 between grid sizes 400 and 800).
"""
import time
import numpy as np
from reference_solver import american_put, penalized_american_put

K, RHO = 100.0, 30.0
SIGMAS = np.array([0.03, 0.05, 0.075, 0.10, 0.15, 0.25])
RATES = np.array([0.0, 0.03, 0.06])
QS = np.array([-0.01, 0.02, 0.05])
TAUS = np.array([0.25, 0.5, 1.0])
XS = np.linspace(0.5, 1.5, 101)

def one_case(args):
    i, j, l, k = args
    s, r, q, T = SIGMAS[i], RATES[j], QS[l], TAUS[k]
    S, V = penalized_american_put(K, r, s, T, q=q, rho=RHO, S_max=300.0, M=2400, N=int(2400 * T))
    vp = np.interp(XS * K, S, V) / K
    S, V, _ = american_put(K, r, s, T, q=q, S_max=300.0, M=800, N=int(400 * T))
    ve = np.interp(XS * K, S, V) / K
    return (i, j, l, k), vp, ve


if __name__ == "__main__":
    import multiprocessing as mp
    shape = (len(SIGMAS), len(RATES), len(QS), len(TAUS), len(XS))
    v_pen, v_ex = np.zeros(shape), np.zeros(shape)
    jobs = [(i, j, l, k) for i in range(len(SIGMAS)) for j in range(len(RATES)) for l in range(len(QS)) for k in range(len(TAUS))]
    t0 = time.time()
    with mp.Pool(max(1, mp.cpu_count() - 1)) as pool:
        for n, (idx, vp, ve) in enumerate(pool.imap_unordered(one_case, jobs), start=1):
            v_pen[idx], v_ex[idx] = vp, ve
            if n % 20 == 0:
                print(f"{n}/{len(jobs)} cases ({time.time() - t0:.0f}s)", flush=True)
    np.savez("param_refs_fx.npz", sigmas=SIGMAS, rates=RATES, qs=QS, taus=TAUS, xs=XS, v_pen=v_pen, v_ex=v_ex, rho=RHO)
    print("saved param_refs_fx.npz; max |penalized - exact| (price units):", (np.abs(v_pen - v_ex).max() * K).round(3))
