"""
Builds the classical reference solutions for the PARAMETRIC experiment (step 1 of the plan).
Run ONCE (numpy/scipy only, about 2-5 minutes):   python make_param_refs.py
Creates param_refs.npz, which pinn_param_exp.py loads.

For every test case (sigma, r, maturity tau) it stores, on the grid S = 50..150 (x = S/K):
  v_pen : classical solution of the PENALIZED equation (same rho as the PINN)  -> isolates network error
  v_ex  : classical solution with the exact American constraint (PSOR)          -> true answer
Values are scaled by the strike (v = V/K), like the PINN output.
"""
import time
import numpy as np
from reference_solver import american_put, penalized_american_put

K, Q, RHO = 100.0, 0.0, 30.0
SIGMAS = np.array([0.10, 0.175, 0.25, 0.325, 0.40])     # training range is [0.10, 0.40]
RATES = np.array([0.01, 0.03, 0.05, 0.08])              # training range is [0.01, 0.08]
TAUS = np.array([0.25, 0.5, 1.0])                       # years to maturity
XS = np.linspace(0.5, 1.5, 101)                         # S from 50 to 150

v_pen = np.zeros((len(SIGMAS), len(RATES), len(TAUS), len(XS)))
v_ex = np.zeros_like(v_pen)
t0 = time.time()
for i, s in enumerate(SIGMAS):
    for j, r in enumerate(RATES):
        for k, T in enumerate(TAUS):
            S, V = penalized_american_put(K, r, s, T, q=Q, rho=RHO, S_max=300.0, M=600, N=int(1200 * T))
            v_pen[i, j, k] = np.interp(XS * K, S, V) / K
            S, V, _ = american_put(K, r, s, T, q=Q, S_max=300.0, M=400, N=int(200 * T))
            v_ex[i, j, k] = np.interp(XS * K, S, V) / K
    print(f"sigma={s:.3f} done ({time.time() - t0:.0f}s)")
np.savez("param_refs.npz", sigmas=SIGMAS, rates=RATES, taus=TAUS, xs=XS, v_pen=v_pen, v_ex=v_ex, rho=RHO)
gap = np.abs(v_pen - v_ex).max(axis=-1) * K
print("max |penalized - exact| over S=50..150, price units: overall", gap.max().round(3),
      "| for r=0.01:", gap[:, 0].max().round(3), "| for r=0.08:", gap[:, -1].max().round(3))
print("saved param_refs.npz")
