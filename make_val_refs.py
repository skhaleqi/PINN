"""
Builds a held-out VALIDATION set of random EUR/USD-like contracts, used to choose between model variants
(so the 105-case test grid is not used for choices). Run ONCE (about 8 minutes) - I already ran it and ship param_refs_val.npz.

    python make_val_refs.py SwingLabels_EuroUsd2.csv        ->  param_refs_val.npz

Contract parameters: sigma is drawn from the EMPIRICAL distribution of the 21-day realized volatility of your EUR/USD file
(so the validation weights low and typical volatilities the way the real data does); r ~ U[0, 0.06]; q ~ U[-0.01, 0.05] with
carry r - q in [-0.01, 0.04]; tau ~ U[0.1, 1]. Reference = exact American price (PSOR, grid 800 x 400T) on S = 50..150.
"""
import sys
import numpy as np
import pandas as pd
from reference_solver import american_put

K, NVAL = 100.0, 60
d = pd.read_csv(sys.argv[1], sep=";")
d["dt"] = pd.to_datetime(d["datetime"], format="%Y.%m.%d %H:%M")
daily = d.set_index("dt")["close"].resample("1D").last().dropna()
vols = (np.log(daily).diff().dropna().rolling(21).std() * np.sqrt(252)).dropna().to_numpy()
vols = np.clip(vols, 0.03, 0.15)
rng = np.random.default_rng(123)
sig = rng.choice(vols, NVAL)
r = rng.uniform(0.0, 0.06, NVAL)
q = np.empty(NVAL)
for i in range(NVAL):
    while True:
        qi = rng.uniform(-0.01, 0.05)
        if -0.01 <= r[i] - qi <= 0.04:
            q[i] = qi
            break
tau = rng.uniform(0.1, 1.0, NVAL)
xs = np.linspace(0.5, 1.5, 101)
v_ex = np.zeros((NVAL, len(xs)))
for i in range(NVAL):
    S, V, _ = american_put(K, r[i], sig[i], tau[i], q=q[i], S_max=300.0, M=800, N=max(10, int(400 * tau[i])))
    v_ex[i] = np.interp(xs * K, S, V) / K
    if (i + 1) % 10 == 0:
        print(f"{i + 1}/{NVAL}", flush=True)
np.savez("param_refs_val.npz", sig=sig, r=r, q=q, tau=tau, xs=xs, v_ex=v_ex)
print("saved param_refs_val.npz; sigma quartiles:", np.percentile(sig, [0, 25, 50, 75, 100]).round(3))
